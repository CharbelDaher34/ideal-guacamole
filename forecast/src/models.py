"""Models (project library: python/models.py): interface 2, the model, the placeholder models and the test every
model goes through. Training (train_and_test, champion_challenger) and the lab use it; the nightly forecast only gives
the production model its input rows (forecast.py).

A model (a "forecaster" in the project variable that picks it) is a class with:
    name            its name. The project variable forecaster says which one the scenario Retrain, test and promote
                    trains (default: daily_catboost).
    params          its settings (a dict), recorded in Experiment Tracking. params["input_days"]: how many days of
                    each client it gets (default 365).
    fit(daily, clients, cutoff)
                    learns from the data known at the cut-off day (interface 1, cashflow.py): daily: daily_flows
                    (client_no, day, category, money_in, money_out, postings, pandas), every day up to the cut-off
                    day; clients: the clients dataset; cutoff: the cut-off day (a Timestamp: the last day of data).
    predict_rows(rows)
                    what the nightly forecast calls. rows: a Polars DataFrame (cashflow.model_rows), one row per
                    client, category (each category the client has, and TOTAL: all its categories together) and window
                    (1 = days 1-30 after the cut-off day, 2 = days 31-60, 3 = days 61-90): client_no, category, window,
                    day_365 ... day_1 (the net cash flow of each of the last input_days days, day_1 the cut-off day).
                    Returns a numpy array, one row per input row: P10, EV, P90 of the window's net cash flow (SAR).
    predict(daily, clients, cutoff)
                    the same from daily flows, for the test, champion / challenger and the lab: DayModel gives it.
The data scientist's own models go in forecasters.py: the setup notebook creates that file once and never overwrites
it. lab_compare (notebook forecast_lab) tries them against the production model without touching production.
"""
import numpy as np
import pandas as pd
import polars as pl

from cashflow import (FORECAST_COLUMNS, HORIZONS, INPUT_DAYS, PERIOD_DAYS, cutoff_date, day_matrix, finish,
                      model_rows, period_matrix, period_series, until)

HOLDOUT = 6                 # the last 6 periods of 30 days are kept aside to test the models
OUTPUT = ["p10", "ev", "p90"]


class DayModel:
    """Base of the placeholders: predict (from daily flows) through predict_rows (from the model's input rows), so the
    test and the nightly forecast run the same code."""
    params = {"input_days": INPUT_DAYS}

    def predict(self, daily, clients, cutoff):
        rows = model_rows(*day_matrix(until(daily, cutoff), cutoff, self.params["input_days"]))
        if rows.is_empty():
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        values = self.predict_rows(rows)
        out = rows.select("client_no", "category", "window").to_pandas()
        out["window"] = out["window"].astype(int)
        out[OUTPUT] = values
        return out[FORECAST_COLUMNS]

    @staticmethod
    def days(rows):
        """The day columns of the input rows, as a numpy float32 array (the oldest day first)."""
        return rows.select(pl.col("^day_.*$")).to_numpy().astype(np.float32, copy=False)


class RecentAverage(DayModel):
    """A simple model with no machine learning, to compare others with (and fall back on). Per client and category,
    EV = the average net cash flow of the last 3 periods of 30 days (the same for the 3 windows). P10 and P90 = EV -/+
    width x the series' standard deviation over the last 12 periods, where fit sets width so that, over the 12
    periods before the cut-off, 80% of the actual periods fell inside."""
    name = "recent_average"
    params = {"input_days": INPUT_DAYS, "periods": 3, "spread_periods": 12, "coverage": 0.8}

    def fit(self, daily, clients, cutoff):
        wide = period_matrix(daily, cutoff)
        n, s = self.params["periods"], self.params["spread_periods"]
        z = []
        for k in range(1, min(12, wide.shape[1] - 1) + 1):         # each of the last 12 periods, from the ones before
            past, actual = wide.iloc[:, :-k], wide.iloc[:, -k]
            ev, sd = past.iloc[:, -n:].mean(axis=1), past.iloc[:, -s:].std(axis=1)
            ok = actual.notna() & ev.notna() & (sd > 0)
            z.append(((actual - ev).abs() / sd)[ok])
        z = pd.concat(z) if z else pd.Series(dtype=float)
        self.width = float(z.quantile(self.params["coverage"])) if len(z) else 1.28
        return self

    def predict_rows(self, rows):
        s = self.params["spread_periods"]
        days = self.days(rows)[:, -PERIOD_DAYS * s:]
        periods = days.reshape(len(days), s, PERIOD_DAYS).sum(axis=2)     # the last 12 periods of 30 days
        ev = periods[:, -self.params["periods"]:].mean(axis=1)
        sd = periods.std(axis=1, ddof=1)
        return np.column_stack([ev - self.width * sd, ev, ev + self.width * sd])


class DailyCatBoost(DayModel):
    """The production placeholder (option A), until the data scientist's models replace it: ONE CatBoost model with
    the MultiQuantile loss gives P10, EV (the median, far less thrown off by a few huge days than the mean) and P90
    together, for every category, TOTAL and the 3 windows. Its inputs are the rows of cashflow.model_rows: the 365
    days (divided by the series' scale so small and large clients look alike), the category (as a number) and the
    window. Nothing else: no sums over 30 days, no calendar.
    Target in training: the net cash flow of the window (the sum of its 30 days), divided by the same scale. Trained on
    rolling cut-offs: every 30 days back from the cut-off (up to `cutoffs` of them), each series whose whole year of
    input is known, each window whose 30 days are known; at most max_clients clients and max_rows rows (random
    samples beyond that, the same every run). threads: CatBoost's threads when predicting (the nightly forecast sets
    it from its workers)."""
    name = "daily_catboost"
    params = {"input_days": INPUT_DAYS, "cutoffs": 24, "max_clients": 20_000, "max_rows": 200_000,
              "iterations": 300, "depth": 6, "learning_rate": 0.1}
    QUANTILES = (0.1, 0.5, 0.9)          # the model's 3 outputs, in this order: P10, EV, P90
    threads = -1

    def new_model(self):
        from catboost import CatBoostRegressor
        alphas = ",".join(str(q) for q in self.QUANTILES)
        return CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", iterations=self.params["iterations"],
                                 depth=self.params["depth"], learning_rate=self.params["learning_rate"],
                                 random_seed=0, verbose=False, allow_writing_files=False)

    def _inputs(self, days, codes, window):
        """The model's inputs, and each row's scale."""
        scale = np.maximum(np.abs(days).mean(axis=1) * PERIOD_DAYS, 100.0).astype(np.float32)
        return np.column_stack([days / scale[:, None], codes, window]).astype(np.float32), scale

    def _codes(self, categories):
        """Each category as a number (its place among the categories seen in training; empty if new)."""
        code = {c: i for i, c in enumerate(self.categories)}
        return pl.Series(categories, dtype=pl.Utf8).replace_strict(code, default=None, return_dtype=pl.Float32)\
            .fill_null(np.nan).to_numpy()

    def fit(self, daily, clients, cutoff):
        days, rng = self.params["input_days"], np.random.default_rng(0)
        known = until(daily, cutoff)
        if known.empty:
            raise ValueError(f"{self.name}: no daily flows up to {cutoff:%Y-%m-%d} to train on")
        numbers = known["client_no"].astype(str).unique()
        if len(numbers) > self.params["max_clients"]:
            numbers = rng.choice(np.sort(numbers), self.params["max_clients"], replace=False)
            known = known[known["client_no"].astype(str).isin(set(numbers))]
        first = pd.to_datetime(known.groupby("client_no")["day"].min())
        span = (cutoff - first.min()).days + 1
        keys, matrix = day_matrix(known, cutoff, span)                   # every day, the oldest first
        self.categories = sorted(set(keys["category"]))
        codes = self._codes(keys["category"])
        start = span - 1 - (cutoff - first.reindex(keys["client_no"].to_list())).dt.days.to_numpy()
        blocks = []                      # (the cut-off's column, window, the series whose input year is known)
        for k in range(1, self.params["cutoffs"] + 1):
            end = span - 1 - PERIOD_DAYS * k
            if end - days + 1 < 0:
                break
            rows = np.flatnonzero(start <= end - days + 1)
            blocks += [(end, h, rows) for h in HORIZONS if k >= h and len(rows)]
        total = sum(len(rows) for _, _, rows in blocks)
        if not total:
            raise ValueError(f"{self.name}: not enough history to train: a series needs {days} days of input and the "
                             f"{PERIOD_DAYS} days after them (daily flows from {first.min():%Y-%m-%d} to "
                             f"{cutoff:%Y-%m-%d})")
        keep = min(1.0, self.params["max_rows"] / total)
        Xs, ys = [], []
        for end, h, rows in blocks:
            rows = rows[rng.random(len(rows)) < keep]
            X, scale = self._inputs(matrix[rows, end - days + 1:end + 1], codes[rows], np.full(len(rows), h))
            Xs.append(X)
            ys.append(matrix[rows, end + PERIOD_DAYS * (h - 1) + 1:end + PERIOD_DAYS * h + 1].sum(axis=1) / scale)
        self.model = self.new_model().fit(np.vstack(Xs), np.concatenate(ys))
        return self

    def predict_rows(self, rows):
        X, scale = self._inputs(self.days(rows), self._codes(rows["category"]), rows["window"].to_numpy())
        return self.model.predict(X, thread_count=self.threads) * scale[:, None]


def known_forecasters():
    """The forecasters Retrain can train, by name: the placeholders (daily_catboost, recent_average) and those
    listed in FORECASTERS in forecasters.py. Also returns the problem reading forecasters.py, if any (the placeholders
    still work)."""
    found, problem = {DailyCatBoost.name: DailyCatBoost, RecentAverage.name: RecentAverage}, None
    try:
        import forecasters
        for forecaster in forecasters.FORECASTERS:
            found[forecaster.name] = forecaster
    except Exception as e:
        problem = f"forecasters.py could not be read ({type(e).__name__}: {e})"
    return found, problem


def new_forecaster(name=None):
    """A new, untrained forecaster: the one named by the project variable forecaster (default daily_catboost)."""
    import dataiku
    name = name or dataiku.get_custom_variables().get("forecaster") or DailyCatBoost.name
    found, problem = known_forecasters()
    if name not in found:
        raise ValueError(f"No forecaster named {name!r} (project variable forecaster). Known: {', '.join(found)}. "
                         + (problem if problem else "Add yours to FORECASTERS in forecasters.py."))
    forecaster = found[name]()
    missing = [a for a in ("name", "fit", "predict") if not hasattr(forecaster, a)]
    if missing:
        raise ValueError(f"Forecaster {name} has no {', '.join(missing)} (see models.py)")
    return forecaster


def forecast(forecaster, daily, clients, cutoff):
    """P10, EV and P90 of a trained forecaster (SAR), checked, with P10 <= EV <= P90: one row per client, category and
    window. Everything that predicts goes through here: the test, champion / challenger, the lab and the
    nightly forecast."""
    out = forecaster.predict(daily, clients, cutoff)
    name = getattr(forecaster, "name", type(forecaster).__name__)
    if not isinstance(out, pd.DataFrame) or not set(FORECAST_COLUMNS) <= set(out.columns):
        raise ValueError(f"Forecaster {name}: predict must return a DataFrame with the columns "
                         f"{', '.join(FORECAST_COLUMNS)}")
    empty = [c for c in OUTPUT if out[c].isna().any()]
    if empty:
        raise ValueError(f"Forecaster {name}: predict returned empty values in {', '.join(empty)}")
    out = out[FORECAST_COLUMNS].reset_index(drop=True)
    out["client_no"] = out["client_no"].astype(str)
    out["window"] = out["window"].astype(int)
    out[OUTPUT] = finish(out["ev"], out["p10"], out["p90"], out.index)
    return out.sort_values(["client_no", "category", "window"]).reset_index(drop=True)


def active(daily, clients):
    """The active clients and their daily flows: the clients the test and the forecast are made for."""
    clients = clients[clients["status"] == "ACTIVE"]
    return daily[daily["client_no"].isin(set(clients["client_no"]))], clients


def test_forecasts(fitted, daily, clients, cutoff):
    """The test, the same for every model: a forecaster trained without the last 6 periods of 30 days forecasts the
    windows after each test cut-off (the 6 days, 30 days apart, before the cut-off) from the data known at that day,
    for the active clients. Returns one row per client, category, test cut-off and window whose days are known:
    target (the actual net cash flow), p10, ev, p90 and baseline (the simple guess: the same as the 30 days before the
    test cut-off)."""
    daily, clients = active(daily, clients)
    known = period_series(daily, cutoff)
    actual = known.rename(columns={"period": "target_end", "net": "target"})
    last = known.rename(columns={"period": "test_cutoff", "net": "baseline"})
    parts = []
    for k in range(HOLDOUT):
        t = cutoff - pd.Timedelta(days=PERIOD_DAYS * (HOLDOUT - k))
        f = forecast(fitted, until(daily, t), clients, t).assign(test_cutoff=t)
        f["target_end"] = f["window"].map({h: t + pd.Timedelta(days=PERIOD_DAYS * h) for h in HORIZONS})
        parts.append(f[f["target_end"] <= cutoff])
    keys = ["client_no", "category"]
    tested = (pd.concat(parts, ignore_index=True)
              .merge(actual, on=keys + ["target_end"], how="left")
              .merge(last, on=keys + ["test_cutoff"], how="left"))
    tested[["target", "baseline"]] = tested[["target", "baseline"]].fillna(0.0)
    return tested


def backtest(forecaster, daily, clients, cutoff):
    """Train the forecaster without the last 6 periods of 30 days (the data known at the first test cut-off), then
    test it on them (test_forecasts). Returns the test rows; the forecaster is now the backtest forecaster."""
    first = cutoff - pd.Timedelta(days=PERIOD_DAYS * HOLDOUT)
    forecaster.fit(until(daily, first), clients, first)
    return test_forecasts(forecaster, daily, clients, cutoff)


def scores(d):
    """Test scores of predictions (columns target, p10, ev, p90, baseline): EV error (WAPE) of the model and of the
    simple guess, and the share of actual values below P10, between P10 and P90, and above P90."""
    size = d["target"].abs().sum()
    wape = lambda column: round((d["target"] - d[column]).abs().sum() / size, 4) if size else np.nan
    return pd.Series({"rows": len(d), "actual_total": round(d["target"].sum(), 0), "wape_model": wape("ev"),
                      "wape_baseline": wape("baseline") if "baseline" in d else np.nan,
                      "coverage_p10_p90": round(((d["target"] >= d["p10"]) & (d["target"] <= d["p90"])).mean(), 4),
                      "below_p10": round((d["target"] < d["p10"]).mean(), 4),
                      "above_p90": round((d["target"] > d["p90"]).mean(), 4)})


def test_results(tested):
    """Test scores per window (1, 2, 3) and overall (window ALL), for the TOTAL rows (level TOTAL) and the
    category rows (level CATEGORIES). The overall scores, which promotion uses, are on all rows."""
    scored = ["target", "p10", "ev", "p90", "baseline"]
    level = np.where(tested["category"] == "TOTAL", "TOTAL", "CATEGORIES")
    results = pd.concat([
        tested.assign(level=level).groupby(["level", "window"])[scored].apply(scores).reset_index(),
        tested.assign(level=level).groupby("level")[scored].apply(scores).reset_index().assign(window="ALL"),
        scores(tested).to_frame().T.assign(level="ALL", window="ALL")], ignore_index=True)
    results["window"] = results["window"].astype(str)
    return results[["level", "window"] + [c for c in results if c not in ("level", "window")]]


def run_metrics(results):
    """The scores recorded in Experiment Tracking: overall, and the EV error of the TOTAL rows per window.
    results: from test_results."""
    overall = results[(results["level"] == "ALL")].iloc[0]
    metrics = {k: overall[c] for k, c in (("wape", "wape_model"), ("wape_baseline", "wape_baseline"),
                                          ("coverage_p10_p90", "coverage_p10_p90"), ("below_p10", "below_p10"),
                                          ("above_p90", "above_p90"))}
    for row in results[(results["level"] == "TOTAL") & (results["window"] != "ALL")].itertuples():
        metrics[f"wape_total_w{row.window}"] = row.wape_model
    return metrics


def _lab_test():
    """The lab's data and the production model's test, the same as champion / challenger's: daily flows, clients,
    the production model's ID and its backtest forecaster's test rows (None without a production model)."""
    import tracking
    from cashflow import read_daily, read_dataset
    daily, clients = read_daily(), read_dataset("clients")
    production_id = tracking.production_model_id()
    production = (test_forecasts(tracking.load_backtest(production_id), daily, clients, cutoff_date())
                  if production_id else None)
    return daily, clients, production_id, production


def _by_window(tested):
    """Overall test scores per window (all rows), indexed by window (1, 2, 3, ALL)."""
    scored = ["target", "p10", "ev", "p90", "baseline"]
    per = tested.groupby("window")[scored].apply(scores)
    per.index = per.index.astype(str)
    per.loc["ALL"] = scores(tested)
    return per


def _record_try(forecaster, tested, production_id, daily, clients, purpose):
    """Record a lab try in Experiment Tracking (experiment cashflow_lab), with its fingerprint (tracking.describe).
    Returns the run ID."""
    import tracking
    params = {"test_rows": len(tested), "compared_with": production_id or "none"}
    about = tracking.describe(forecaster, daily, clients, cutoff_date(), tested, "lab notebook (lab_compare)")
    run_name = f"{forecaster.name}_{pd.Timestamp.now():%Y%m%dT%H%M}"
    return tracking.log_lab_run(run_name, params, run_metrics(test_results(tested)), forecaster, about, purpose)


def lab_compare(forecasters, record=True, purpose=""):
    """Try several forecasters at once against the production model, on the same test as champion / challenger,
    without touching production. Each forecaster is trained without the last 6 periods of 30 days; each try is
    recorded in Experiment Tracking (experiment cashflow_lab) unless record is False, with `purpose` (what the try
    is for) as a tag. Returns a dict:
        board       one row per model (production first, simple guess last): EV error, P10-P90 coverage, share below
                    P10 and above P90, the decision champion / challenger would take and its reason, the lab run ID
        errors      EV error per window: the simple guess, the production model and each forecaster
        coverage    P10-P90 coverage per window: the production model and each forecaster
        tested      the test rows with each model's p10, ev, p90, by name, for your own analysis"""
    import tracking
    names = [f.name for f in forecasters]
    if len(set(names)) != len(names) or {"production", "simple_guess_baseline"} & set(names):
        raise ValueError(f"Give each forecaster its own name: {names}")
    daily, clients, production_id, production = _lab_test()
    tested = {} if production is None else {"production": production}
    for forecaster in forecasters:
        tested[forecaster.name] = backtest(forecaster, daily, clients, cutoff_date())
    run_ids = ({f.name: _record_try(f, tested[f.name], production_id, daily, clients, purpose) for f in forecasters}
               if record else {})
    reference = scores(production) if production is not None else None
    board = []
    for name, rows in tested.items():
        s = scores(rows)
        decision, reason = (("", f"the production model, {production_id}") if name == "production"
                            else tracking.decide(s, reference))
        board.append({"model": name, "ev_error": s["wape_model"], "coverage_p10_p90": s["coverage_p10_p90"],
                      "below_p10": s["below_p10"], "above_p90": s["above_p90"], "decision": decision,
                      "reason": reason, "lab_run_id": run_ids.get(name, "")})
    board.append({"model": "simple_guess_baseline", "ev_error": scores(next(iter(tested.values())))["wape_baseline"],
                  "decision": "", "reason": "the same as the 30 days before: the bar to beat", "lab_run_id": ""})
    board = pd.DataFrame(board).set_index("model")
    per = {name: _by_window(rows) for name, rows in tested.items()}
    first = next(iter(per.values()))
    errors = pd.DataFrame({"simple_guess_baseline": first["wape_baseline"],
                           **{name: found["wape_model"] for name, found in per.items()}})
    coverage = pd.DataFrame({name: found["coverage_p10_p90"] for name, found in per.items()})
    return {"board": board, "errors": errors, "coverage": coverage, "tested": tested}
