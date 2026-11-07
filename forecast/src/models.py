"""Models (project library: python/models.py): interface 2, the forecaster, the placeholder forecasters and the test
every model goes through. Training (train_and_test, champion_challenger) and the lab use it; inference only calls the
production model's predict.

A forecaster is a class with:
    name            its name. The project variable forecaster says which one the scenario Retrain, test and promote
                    trains (default: quantile_boosting).
    params          its settings (a dict), recorded in Experiment Tracking
    fit(daily, clients, cutoff)
                    learns from the data known at the cut-off day (interface 1, cashflow.py): daily: daily_flows
                    (client_no, day, category, money_in, money_out, postings), every day up to the cut-off day;
                    clients: the clients dataset; cutoff: the cut-off day (a Timestamp: the last day of data).
                    cashflow.period_matrix turns the days into 30-day periods ending at the cut-off day.
    predict(daily, clients, cutoff)
                    the same inputs, for the clients to forecast. Returns a DataFrame with one row per client,
                    category (each category the client has in daily, and TOTAL: all its categories together) and
                    window (1 = days 1-30 after the cut-off day, 2 = days 31-60, 3 = days 61-90): the columns
                    client_no, category, window, p10, ev, p90 (the net cash flow of that window, SAR). forecast()
                    checks it and applies P10 <= EV <= P90.
The data scientist's own forecasters go in forecasters.py: the setup notebook creates that file once and never
overwrites it. lab_try and lab_compare try forecasters against the production model without touching production.
"""
import numpy as np
import pandas as pd

from cashflow import (FORECAST_COLUMNS, HORIZONS, PERIOD_DAYS, TOTAL, cutoff_date, finish, period_matrix,
                      period_series, until)

HOLDOUT = 6                 # the last 6 periods of 30 days are kept aside to test the models
OUTPUT = ["p10", "ev", "p90"]


class RecentAverage:
    """A simple forecaster with no machine learning, to compare others with (and fall back on). Per client and
    category, EV = the average net cash flow of the last 3 periods of 30 days (the same for the 3 windows). P10 and
    P90 = EV -/+ width x the series' standard deviation over the last 12 periods, where fit sets width so that, over
    the 12 periods before the cut-off, 80% of the actual periods fell inside."""
    name = "recent_average"
    params = {"periods": 3, "spread_periods": 12, "coverage": 0.8}

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

    def predict(self, daily, clients, cutoff):
        wide = period_matrix(daily, cutoff)
        if wide.empty:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        ev = wide.iloc[:, -self.params["periods"]:].mean(axis=1).fillna(0.0)
        sd = wide.iloc[:, -self.params["spread_periods"]:].std(axis=1).fillna(ev.abs())
        one = pd.DataFrame({"p10": ev - self.width * sd, "ev": ev, "p90": ev + self.width * sd}).reset_index()
        return pd.concat([one.assign(window=h) for h in HORIZONS], ignore_index=True)[FORECAST_COLUMNS]


class QuantileBoosting:
    """The production placeholder, until the data scientist's models replace it: 3 small models, one for the EV, one
    for P10 and one for P90 (scikit-learn gradient boosting; P10 and P90 with the quantile loss, the EV with the median,
    which is far less thrown off by a few huge periods than the mean). Each of the 3 is ONE model for every category
    and TOTAL: the category is one of its inputs, so a category with little data learns from the others.
    Inputs per client and category at a cut-off day: the net cash flow of the last 12 periods of 30 days divided by
    the series' scale (so small and large clients look alike), the category, how many periods are known, the window
    and the calendar month of the window's middle (seasonality). Trained on the cut-offs of the last 24 periods, on
    at most max_rows rows (a random sample beyond that, the same every run). Each row is forecast from its own
    client's history only."""
    name = "quantile_boosting"
    params = {"lags": 12, "cutoffs": 24, "max_iter": 200, "learning_rate": 0.05, "max_rows": 1_000_000}
    QUANTILES = {"ev": 0.5, "p10": 0.1, "p90": 0.9}

    def _rows(self, wide, p, h):
        """Inputs at period position p (the cut-off) for window h, and each series' scale."""
        last = wide.iloc[:, max(0, p - self.params["lags"] + 1):p + 1]
        scale = last.abs().mean(axis=1).fillna(0).clip(lower=100.0)
        X = last.div(scale, axis=0).fillna(0.0)
        X.columns = [f"lag_{len(X.columns) - i}" for i in range(len(X.columns))]
        X = X.reindex(columns=[f"lag_{k}" for k in range(self.params["lags"], 0, -1)], fill_value=0.0)
        category = pd.Categorical(wide.index.get_level_values("category"), categories=self.categories)
        X["category"] = np.where(category.codes >= 0, category.codes, np.nan)      # a new category: unknown
        X["periods_known"] = last.notna().sum(axis=1).values
        X["window"] = h
        X["calendar_month"] = (wide.columns[p] + pd.Timedelta(days=PERIOD_DAYS * h - PERIOD_DAYS // 2)).month
        return X, scale

    def fit(self, daily, clients, cutoff):
        return self.fit_periods(period_matrix(daily, cutoff))

    def predict(self, daily, clients, cutoff):
        return self.predict_periods(period_matrix(daily, cutoff))

    def fit_periods(self, wide):
        """fit, from the net cash flow per 30-day period (cashflow.period_matrix)."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.categories = sorted(set(wide.index.get_level_values("category")))
        n = wide.shape[1]
        blocks = [(p, h) for p in range(max(0, n - 1 - self.params["cutoffs"]), n - 1) for h in HORIZONS if p + h < n]
        keep = min(1.0, self.params["max_rows"] / max(1, len(wide) * len(blocks)))
        rng, Xs, ys = np.random.default_rng(0), [], []
        for p, h in blocks:
            X, scale = self._rows(wide, p, h)
            y = wide.iloc[:, p + h] / scale
            ok = (y.notna() & wide.iloc[:, p].notna()).values & (rng.random(len(y)) < keep)
            Xs.append(X[ok])
            ys.append(y[ok])
        X, y = pd.concat(Xs), pd.concat(ys)
        settings = {"max_iter": self.params["max_iter"], "learning_rate": self.params["learning_rate"],
                    "categorical_features": ["category"], "random_state": 0}
        self.models = {column: HistGradientBoostingRegressor(loss="quantile", quantile=q, **settings).fit(X, y)
                       for column, q in self.QUANTILES.items()}
        return self

    def predict_periods(self, wide):
        """predict, from the net cash flow per 30-day period (cashflow.period_matrix)."""
        if wide.empty:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        out = []
        for h in HORIZONS:
            X, scale = self._rows(wide, wide.shape[1] - 1, h)
            rows = wide.index.to_frame(index=False).assign(window=h)
            for column, model in self.models.items():
                rows[column] = model.predict(X) * scale.values
            out.append(rows)
        return pd.concat(out, ignore_index=True)[FORECAST_COLUMNS]


def known_forecasters():
    """The forecasters Retrain can train, by name: the placeholders (quantile_boosting, recent_average) and those
    listed in FORECASTERS in forecasters.py. Also returns the problem reading forecasters.py, if any (the placeholders
    still work)."""
    found, problem = {QuantileBoosting.name: QuantileBoosting, RecentAverage.name: RecentAverage}, None
    try:
        import forecasters
        for forecaster in forecasters.FORECASTERS:
            found[forecaster.name] = forecaster
    except Exception as e:
        problem = f"forecasters.py could not be read ({type(e).__name__}: {e})"
    return found, problem


def new_forecaster(name=None):
    """A new, untrained forecaster: the one named by the project variable forecaster (default quantile_boosting)."""
    import dataiku
    name = name or dataiku.get_custom_variables().get("forecaster") or QuantileBoosting.name
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


def _record_try(forecaster, tested, production_id):
    """Record a lab try in Experiment Tracking (experiment cashflow_lab). Returns the run ID."""
    import tracking
    params = {"forecaster": forecaster.name, **getattr(forecaster, "params", {}),
              "cutoff_date": f"{cutoff_date():%Y-%m-%d}", "test_rows": len(tested),
              "compared_with": production_id or "none"}
    run_name = f"{forecaster.name}_{pd.Timestamp.now():%Y%m%dT%H%M}"
    return tracking.log_lab_run(run_name, params, run_metrics(test_results(tested)), forecaster)


def lab_try(forecaster, record=True):
    """Try a forecaster against the production model without touching production (the lab: recipe try_forecaster).
    It is trained without the last 6 periods of 30 days and tested on them, the way champion / challenger tests a new model, and
    the try is recorded in Experiment Tracking (experiment cashflow_lab, never deployed).
    Returns: per window (1, 2, 3, ALL) the EV error of the simple guess, of the production model and of the
    candidate, and their P10-P90 coverage; the decision champion / challenger would take and its reason; the lab run
    ID (None if not recorded)."""
    import tracking
    daily, clients, production_id, production = _lab_test()
    candidate = backtest(forecaster, daily, clients, cutoff_date())
    mine = _by_window(candidate)
    table = pd.DataFrame({"simple_guess_error": mine["wape_baseline"], "production_error": np.nan,
                          "candidate_error": mine["wape_model"], "production_coverage": np.nan,
                          "candidate_coverage": mine["coverage_p10_p90"]})
    if production is not None:
        theirs = _by_window(production)
        table["production_error"], table["production_coverage"] = theirs["wape_model"], theirs["coverage_p10_p90"]
    decision, reason = tracking.decide(scores(candidate), scores(production) if production is not None else None)
    run_id = _record_try(forecaster, candidate, production_id) if record else None
    return table.rename_axis("window"), decision, reason, run_id


def lab_compare(forecasters, record=True):
    """Try several forecasters at once against the production model, on the same test as lab_try and champion /
    challenger, without touching production. Each forecaster is trained without the last 6 periods of 30 days; each try is
    recorded in Experiment Tracking (experiment cashflow_lab) unless record is False. Returns a dict:
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
    run_ids = {f.name: _record_try(f, tested[f.name], production_id) for f in forecasters} if record else {}
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
