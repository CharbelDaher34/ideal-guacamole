"""Models (project library: python/models.py): interface 2, the forecaster, the reference forecaster (CatBoost) and the
test every model goes through. Training (train_and_test, champion_challenger) and the lab notebook use it; inference
never imports it.

A forecaster is a class with:
    name            its name. The project variable forecaster says which one the scenario Retrain, test and promote
                    trains (default: catboost).
    params          its settings (a dict), recorded in Experiment Tracking
    inputs          the model-row columns it reads (cashflow.MODEL_ROW_COLUMNS, no dates). Each saved model version
                    is given these columns, and category.
    fit(rows)       learns from training rows. Column target is the amount to predict, in SAR: the amount of the
                    predicted month minus the obligations already known at the cut-off (the forecast adds those back).
    predict(rows)   a DataFrame with the index of rows and the columns p10, ev, p90 (SAR). forecast() then applies
                    P10 <= EV <= P90 and no negative amounts except the net cash flow.
The data scientist's own forecasters go in forecasters.py: the setup notebook creates that file once and never
overwrites it. lab_try tries one against the production model without touching production (the Lab zone).
"""
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, add_months, cutoff_month, finish, month_start

HOLDOUT_MONTHS = 6          # the last months kept aside to test the models
OUTPUT = ["p10", "ev", "p90"]

# the reference forecaster's inputs (series_share is computed from scale and series_scale)
CAT_FEATURES = ["category", "direction", "segment", "sector", "region"]
SERIES_FEATURES = ["last_amount", "avg_3m", "same_month_last_year", "free_last", "free_avg_3m", "oblig_target_month"]
CLIENT_FEATURES = ["client_in_avg_3m", "client_out_avg_3m", "balance_end", "oblig_in_next_3m", "oblig_out_next_3m"]
OTHER_FEATURES = ["months_ahead", "target_month_of_year", "history_months", "log_scale", "series_share",
                  "log_annual_sales", "employees"]
FEATURES = CAT_FEATURES + SERIES_FEATURES + CLIENT_FEATURES + OTHER_FEATURES


def model_matrix(rows):
    """The reference forecaster's inputs: series amounts divided by the series scale, client amounts by the client
    scale, categories as text."""
    rows = rows.assign(series_share=rows["series_scale"] / rows["scale"])
    X = rows[FEATURES].copy()
    for c in SERIES_FEATURES:
        X[c] = rows[c] / rows["series_scale"]
    for c in CLIENT_FEATURES:
        X[c] = rows[c] / rows["scale"]
    for c in CAT_FEATURES:
        X[c] = X[c].fillna("NA").astype(str)
    return X


class CatBoostForecaster:
    """The reference forecaster: one CatBoost model for the EV and one for P10 and P90 together, trained on amounts
    divided by the series scale (the client's usual amount in that category)."""
    name = "catboost"
    params = {"iterations": 600, "depth": 6, "learning_rate": 0.08, "random_seed": 0, "verbose": 0}
    quantiles = (0.1, 0.9)
    inputs = [f for f in FEATURES if f != "series_share"] + ["scale", "series_scale"]

    def fit(self, rows):
        from catboost import CatBoostRegressor
        X, y = model_matrix(rows), rows["target"] / rows["series_scale"]
        self.ev = CatBoostRegressor(loss_function="RMSE", **self.params)
        self.ev.fit(X, y, cat_features=CAT_FEATURES)
        alphas = ",".join(str(q) for q in self.quantiles)
        self.p10_p90 = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", **self.params)
        self.p10_p90.fit(X, y, cat_features=CAT_FEATURES)
        return self

    def predict(self, rows):
        X = model_matrix(rows)
        scale = rows["series_scale"].values
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
    """P10, EV and P90 of a trained forecaster for model rows (SAR), checked, with P10 <= EV <= P90 and no negative
    amounts except the net cash flow. Everything that predicts goes through here: the test, champion / challenger, the
    lab and the saved models."""
    out = forecaster.predict(rows)
    name = getattr(forecaster, "name", type(forecaster).__name__)
    if not isinstance(out, pd.DataFrame) or not set(OUTPUT) <= set(out.columns):
        raise ValueError(f"Forecaster {name}: predict must return a DataFrame with the columns p10, ev, p90")
    if len(out) != len(rows) or not out.index.equals(rows.index):
        raise ValueError(f"Forecaster {name}: predict must return one row per input row, with the same index")
    empty = [c for c in OUTPUT if out[c].isna().any()]
    if empty:
        raise ValueError(f"Forecaster {name}: predict returned empty values in {', '.join(empty)}")
    return finish(out["ev"], out["p10"], out["p90"], rows["category"], rows.index)


def holdout(rows):
    """The test rows: predicted month in the last 6 months before the cut-off, with the simple guess (baseline: the
    average of the last 3 months, without the scheduled part)."""
    rows = rows.assign(target_month=month_start(rows["target_month"]))
    test = rows[rows["target_month"] >= add_months(cutoff_month(), -(HOLDOUT_MONTHS - 1))].copy()
    test["baseline"] = np.where(test["category"] == "NET", test["free_avg_3m"], test["free_avg_3m"].clip(lower=0))
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
    """Test scores per category and months ahead, per category (months_ahead ALL), per months ahead (category ALL)
    and overall. The ALL rows count the categories only, not the net cash flow. tested: the test rows with p10, ev,
    p90."""
    categories_only = tested[tested["category"] != "NET"]
    results = pd.concat([
        tested.groupby(["category", "months_ahead"]).apply(scores).reset_index(),
        tested.groupby("category").apply(scores).reset_index().assign(months_ahead="ALL"),
        categories_only.groupby("months_ahead").apply(scores).reset_index().assign(category="ALL"),
        scores(categories_only).to_frame().T.assign(category="ALL", months_ahead="ALL")], ignore_index=True)
    results["months_ahead"] = results["months_ahead"].astype(str)
    results["category_name"] = results["category"].map({c: v[0] for c, v in CATEGORIES.items()}).fillna(
        "All categories")
    return results


def run_metrics(results):
    """The scores recorded in Experiment Tracking: overall (all categories and months ahead) and the EV error per
    category. results: from test_results."""
    overall = results[(results["category"] == "ALL") & (results["months_ahead"] == "ALL")].iloc[0]
    metrics = {k: overall[c] for k, c in (("wape", "wape_model"), ("wape_baseline", "wape_baseline"),
                                          ("coverage_p10_p90", "coverage_p10_p90"), ("below_p10", "below_p10"),
                                          ("above_p90", "above_p90"))}
    for row in results[(results["months_ahead"] == "ALL") & (results["category"] != "ALL")].itertuples():
        metrics[f"wape_{row.category}"] = row.wape_model
    return metrics


def lab_try(forecaster, record=True):
    """Try a forecaster against the production model without touching production (the lab: notebook forecast_lab and
    recipe try_forecaster). It is trained without the last 6 months and tested on them, the way champion / challenger
    tests a new model, and the try is recorded in Experiment Tracking (experiment cashflow_lab, never deployed).
    Returns: per category (all months ahead together) the EV error of the simple guess, of the production model and of
    the candidate, and their P10-P90 coverage; the decision champion / challenger would take and its reason; the lab
    run ID (None if not recorded)."""
    import tracking
    from cashflow import read_dataset
    rows = read_dataset("training_rows")
    test = holdout(rows)
    train = rows.drop(test.index)
    forecaster.fit(train)
    candidate = test.join(forecast(forecaster, test))
    production_id = tracking.production_model_id()
    production = test.join(forecast(tracking.load_backtest(production_id), test)) if production_id else None

    def by_category(tested):
        found = test_results(tested)
        return found[found["months_ahead"] == "ALL"].set_index("category_name")

    mine = by_category(candidate)
    table = pd.DataFrame({"simple_guess_error": mine["wape_baseline"], "production_error": np.nan,
                          "candidate_error": mine["wape_model"], "production_coverage": np.nan,
                          "candidate_coverage": mine["coverage_p10_p90"]})
    if production is not None:
        theirs = by_category(production)
        table["production_error"], table["production_coverage"] = theirs["wape_model"], theirs["coverage_p10_p90"]

    def overall(tested):            # champion / challenger scores the categories, not the net cash flow
        return scores(tested[tested["category"] != "NET"])

    decision, reason = tracking.decide(overall(candidate), overall(production) if production is not None else None)
    run_id = None
    if record:
        params = {"forecaster": forecaster.name, **getattr(forecaster, "params", {}),
                  "cutoff_month": f"{cutoff_month():%Y-%m}", "training_rows": len(train), "test_rows": len(test),
                  "compared_with": production_id or "none", "inputs": ",".join(forecaster.inputs)}
        run_name = f"{forecaster.name}_{pd.Timestamp.now():%Y%m%dT%H%M}"
        run_id = tracking.log_lab_run(run_name, params, run_metrics(test_results(candidate)), forecaster)
    return table, decision, reason, run_id
