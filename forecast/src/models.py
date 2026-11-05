"""Models (project library: python/models.py): interface 2, the forecaster, the reference forecaster (CatBoost) and the
test every model goes through. Training (train_and_test, champion_challenger) and the lab notebook use it; inference
never imports it.

A forecaster is a class with:
    name            its name. The project variable forecaster says which one the scenario Retrain, test and promote
                    trains (default: catboost).
    params          its settings (a dict), recorded in Experiment Tracking
    inputs          the model-row columns it reads (cashflow.MODEL_ROW_COLUMNS, no dates). Each saved model version
                    is given these columns.
    fit(rows)       learns from training rows. Column target is the amount to predict: the client's net cash flow
                    (money in minus money out) of the predicted month, in SAR.
    predict(rows)   a DataFrame with the index of rows and the columns p10, ev, p90 (SAR). forecast() then applies
                    P10 <= EV <= P90.
The data scientist's own forecasters go in forecasters.py: the setup notebook creates that file once and never
overwrites it. lab_try and lab_compare try forecasters against the production model without touching
production (the Lab zone and the notebook forecast_lab).
"""
import numpy as np
import pandas as pd

from cashflow import add_months, cutoff_month, finish, month_start

HOLDOUT_MONTHS = 6          # the last months kept aside to test the models
OUTPUT = ["p10", "ev", "p90"]

# the reference forecaster's inputs
CAT_FEATURES = ["segment", "sector", "city", "sole_proprietor"]
AMOUNT_FEATURES = ["last_amount", "avg_3m", "same_month_last_year", "client_in_avg_3m", "client_out_avg_3m"]
OTHER_FEATURES = ["months_ahead", "target_month_of_year", "history_months", "log_scale", "log_annual_sales",
                  "employees"]
FEATURES = CAT_FEATURES + AMOUNT_FEATURES + OTHER_FEATURES


def model_matrix(rows):
    """The reference forecaster's inputs: amounts divided by the client's scale (its usual month), categorical inputs
    as text."""
    X = rows[FEATURES].copy()
    for c in AMOUNT_FEATURES:
        X[c] = rows[c] / rows["scale"]
    for c in CAT_FEATURES:
        X[c] = X[c].fillna("NA").astype(str).str.upper()       # TRUE / FALSE however a yes / no column is read
    return X


class CatBoostForecaster:
    """The reference forecaster: one CatBoost model for the EV and one for P10 and P90 together, trained on the net
    cash flow divided by the client's scale (its usual month), so small and large clients look alike."""
    name = "catboost"
    params = {"iterations": 600, "depth": 6, "learning_rate": 0.08, "random_seed": 0, "verbose": 0}
    quantiles = (0.1, 0.9)
    inputs = FEATURES + ["scale"]

    def fit(self, rows):
        from catboost import CatBoostRegressor
        X, y = model_matrix(rows), rows["target"] / rows["scale"]
        self.ev = CatBoostRegressor(loss_function="RMSE", **self.params)
        self.ev.fit(X, y, cat_features=CAT_FEATURES)
        alphas = ",".join(str(q) for q in self.quantiles)
        self.p10_p90 = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", **self.params)
        self.p10_p90.fit(X, y, cat_features=CAT_FEATURES)
        return self

    def predict(self, rows):
        X = model_matrix(rows)
        scale = rows["scale"].values
        q = self.p10_p90.predict(X)
        return pd.DataFrame({"p10": q[:, 0] * scale, "ev": self.ev.predict(X) * scale, "p90": q[:, 1] * scale},
                            index=rows.index)


def known_forecasters():
    """The forecasters Retrain can train, by name: the reference one and those listed in FORECASTERS in
    forecasters.py. Also returns the problem reading forecasters.py, if any (the reference one still works)."""
    found, problem = {CatBoostForecaster.name: CatBoostForecaster}, None
    try:
        import forecasters
        for forecaster in forecasters.FORECASTERS:
            found[forecaster.name] = forecaster
    except Exception as e:
        problem = f"forecasters.py could not be read ({type(e).__name__}: {e})"
    return found, problem


def new_forecaster(name=None):
    """A new, untrained forecaster: the one named by the project variable forecaster (default catboost)."""
    import dataiku
    name = name or dataiku.get_custom_variables().get("forecaster") or CatBoostForecaster.name
    found, problem = known_forecasters()
    if name not in found:
        raise ValueError(f"No forecaster named {name!r} (project variable forecaster). Known: {', '.join(found)}. "
                         + (problem if problem else "Add yours to FORECASTERS in forecasters.py."))
    forecaster = found[name]()
    missing = [a for a in ("name", "inputs", "fit", "predict") if not hasattr(forecaster, a)]
    if missing:
        raise ValueError(f"Forecaster {name} has no {', '.join(missing)} (see models.py)")
    return forecaster


def forecast(forecaster, rows):
    """P10, EV and P90 of a trained forecaster for model rows (SAR), checked, with P10 <= EV <= P90. Everything that
    predicts goes through here: the test, champion / challenger, the lab and the saved models."""
    out = forecaster.predict(rows)
    name = getattr(forecaster, "name", type(forecaster).__name__)
    if not isinstance(out, pd.DataFrame) or not set(OUTPUT) <= set(out.columns):
        raise ValueError(f"Forecaster {name}: predict must return a DataFrame with the columns p10, ev, p90")
    if len(out) != len(rows) or not out.index.equals(rows.index):
        raise ValueError(f"Forecaster {name}: predict must return one row per input row, with the same index")
    empty = [c for c in OUTPUT if out[c].isna().any()]
    if empty:
        raise ValueError(f"Forecaster {name}: predict returned empty values in {', '.join(empty)}")
    return finish(out["ev"], out["p10"], out["p90"], rows.index)


def holdout(rows):
    """The test rows: predicted month in the last 6 months before the cut-off, with the simple guess (baseline: the
    average of the last 3 months)."""
    rows = rows.assign(target_month=month_start(rows["target_month"]))
    test = rows[rows["target_month"] >= add_months(cutoff_month(), -(HOLDOUT_MONTHS - 1))].copy()
    test["baseline"] = test["avg_3m"]
    return test


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
    """Test scores per months ahead (1, 2, 3) and overall (months_ahead ALL). tested: the test rows with p10, ev,
    p90."""
    scored = [c for c in ("target", "p10", "ev", "p90", "baseline") if c in tested]   # the columns scores reads
    results = pd.concat([
        tested.groupby("months_ahead")[scored].apply(scores).reset_index(),
        scores(tested).to_frame().T.assign(months_ahead="ALL")], ignore_index=True)
    results["months_ahead"] = results["months_ahead"].astype(str)
    return results


def run_metrics(results):
    """The scores recorded in Experiment Tracking: overall and the EV error per months ahead. results: from
    test_results."""
    overall = results[results["months_ahead"] == "ALL"].iloc[0]
    metrics = {k: overall[c] for k, c in (("wape", "wape_model"), ("wape_baseline", "wape_baseline"),
                                          ("coverage_p10_p90", "coverage_p10_p90"), ("below_p10", "below_p10"),
                                          ("above_p90", "above_p90"))}
    for row in results[results["months_ahead"] != "ALL"].itertuples():
        metrics[f"wape_{row.months_ahead}m"] = row.wape_model
    return metrics


def _lab_test():
    """The lab's test, the same as champion / challenger's: the training rows without the last 6 months (train), those
    months (test), the production model's ID and its backtest forecast on the test (None without a production model)."""
    import tracking
    from cashflow import read_dataset
    rows = read_dataset("training_rows")
    test = holdout(rows)
    production_id = tracking.production_model_id()
    production = test.join(forecast(tracking.load_backtest(production_id), test)) if production_id else None
    return rows.drop(test.index), test, production_id, production


def _by_months_ahead(tested):
    """Test scores per months ahead and overall (ALL), indexed by months ahead."""
    return test_results(tested).set_index("months_ahead")


def _record_try(forecaster, train, test, tested, production_id):
    """Record a lab try in Experiment Tracking (experiment cashflow_lab). Returns the run ID."""
    import tracking
    params = {"forecaster": forecaster.name, **getattr(forecaster, "params", {}),
              "cutoff_month": f"{cutoff_month():%Y-%m}", "training_rows": len(train), "test_rows": len(test),
              "compared_with": production_id or "none", "inputs": ",".join(forecaster.inputs)}
    run_name = f"{forecaster.name}_{pd.Timestamp.now():%Y%m%dT%H%M}"
    return tracking.log_lab_run(run_name, params, run_metrics(test_results(tested)), forecaster)


def lab_try(forecaster, record=True):
    """Try a forecaster against the production model without touching production (the lab: recipe try_forecaster;
    the notebook forecast_lab uses lab_compare). It is trained without the last 6 months and tested on them, the way
    champion / challenger tests a new model, and the try is recorded in Experiment Tracking (experiment cashflow_lab,
    never deployed).
    Returns: per months ahead (1, 2, 3, ALL) the EV error of the simple guess, of the production model and of the
    candidate, and their P10-P90 coverage; the decision champion / challenger would take and its reason; the lab run
    ID (None if not recorded)."""
    import tracking
    train, test, production_id, production = _lab_test()
    forecaster.fit(train)
    candidate = test.join(forecast(forecaster, test))
    mine = _by_months_ahead(candidate)
    table = pd.DataFrame({"simple_guess_error": mine["wape_baseline"], "production_error": np.nan,
                          "candidate_error": mine["wape_model"], "production_coverage": np.nan,
                          "candidate_coverage": mine["coverage_p10_p90"]})
    if production is not None:
        theirs = _by_months_ahead(production)
        table["production_error"], table["production_coverage"] = theirs["wape_model"], theirs["coverage_p10_p90"]
    decision, reason = tracking.decide(scores(candidate), scores(production) if production is not None else None)
    run_id = _record_try(forecaster, train, test, candidate, production_id) if record else None
    return table, decision, reason, run_id


def lab_compare(forecasters, record=True):
    """Try several forecasters at once against the production model, on the same test as lab_try and champion /
    challenger, without touching production. Each forecaster is trained without the last 6 months; each try is
    recorded in Experiment Tracking (experiment cashflow_lab) unless record is False. Returns a dict:
        board       one row per model (production first, simple guess last): EV error, P10-P90 coverage, share below
                    P10 and above P90, the decision champion / challenger would take and its reason, the lab run ID
        errors      EV error per months ahead: the simple guess, the production model and each forecaster
        coverage    P10-P90 coverage per months ahead: the production model and each forecaster
        tested      the test rows with each model's p10, ev, p90, by name, for your own analysis"""
    import tracking
    names = [f.name for f in forecasters]
    if len(set(names)) != len(names) or {"production", "simple_guess_baseline"} & set(names):
        raise ValueError(f"Give each forecaster its own name: {names}")
    train, test, production_id, production = _lab_test()
    tested = {} if production is None else {"production": production}
    for forecaster in forecasters:
        forecaster.fit(train)
        tested[forecaster.name] = test.join(forecast(forecaster, test))
    run_ids = {f.name: _record_try(f, train, test, tested[f.name], production_id) for f in forecasters} if record else {}
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
                  "decision": "", "reason": "average of the last 3 months: the bar to beat", "lab_run_id": ""})
    board = pd.DataFrame(board).set_index("model")
    per = {name: _by_months_ahead(rows) for name, rows in tested.items()}
    first = next(iter(per.values()))
    errors = pd.DataFrame({"simple_guess_baseline": first["wape_baseline"],
                           **{name: found["wape_model"] for name, found in per.items()}})
    coverage = pd.DataFrame({name: found["coverage_p10_p90"] for name, found in per.items()})
    return {"board": board, "errors": errors, "coverage": coverage, "tested": tested}
