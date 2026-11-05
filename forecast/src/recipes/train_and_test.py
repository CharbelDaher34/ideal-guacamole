# Train and test one model (one model ID, tracked in Experiment Tracking with its settings and scores):
#   backtest models: trained without the last 6 months, then tested on them (test_results)
#   final models: trained on all rows, deployed as version <model ID> of the saved models forecast_ev,
#   forecast_p10 and forecast_p90 (not active: the promotion activates it). model_reference: a sample of the
#   model's inputs (no date columns), evaluated with each version so the version knows its input columns.
import dataiku
import pandas as pd

from cashflow import (CATBOOST, CATEGORIES, FEATURES, QUANTILES, cutoff_month, fit_models, holdout, month_start,
                      predict, read_dataset, scores)
from tracking import log_model, new_model_id

cutoff = cutoff_month()
rows = read_dataset("training_rows")
rows["target_month"] = month_start(rows["target_month"])
test = holdout(rows)
backtest = fit_models(rows.drop(test.index))
test = test.join(predict(backtest, test))
final = fit_models(rows)
model_id = new_model_id(cutoff)

categories_only = test[test["category"] != "NET"]
results = pd.concat([
    test.groupby(["category", "months_ahead"]).apply(scores).reset_index(),
    test.groupby("category").apply(scores).reset_index().assign(months_ahead="ALL"),
    categories_only.groupby("months_ahead").apply(scores).reset_index().assign(category="ALL"),
    scores(categories_only).to_frame().T.assign(category="ALL", months_ahead="ALL")], ignore_index=True)
results["months_ahead"] = results["months_ahead"].astype(str)
results["category_name"] = results["category"].map({c: v[0] for c, v in CATEGORIES.items()}).fillna("All categories")
results.insert(0, "model_id", model_id)

overall = scores(categories_only)
metrics = {"wape": overall["wape_model"], "wape_baseline": overall["wape_baseline"],
           "coverage_p10_p90": overall["coverage_p10_p90"], "below_p10": overall["below_p10"],
           "above_p90": overall["above_p90"]}
for row in results[results["months_ahead"] == "ALL"].itertuples():
    if row.category != "ALL":
        metrics[f"wape_{row.category}"] = row.wape_model
params = {**CATBOOST, "quantiles": QUANTILES, "cutoff_month": f"{cutoff:%Y-%m}",
          "load_id": dataiku.get_custom_variables().get("load_id", ""), "training_rows": len(rows),
          "test_rows": len(test), "clients": rows["client_no"].nunique(), "features": ",".join(FEATURES)}
inputs = [f for f in FEATURES if f != "series_share"] + ["scale", "series_scale"]   # series_share is computed
reference = rows[inputs + ["target"]].sample(min(len(rows), 20000), random_state=0)
dataiku.Dataset("model_reference").write_with_schema(reference.reset_index(drop=True))
log_model(model_id, params, metrics, backtest, final, "model_reference")
print(f"Model {model_id}: EV error {overall['wape_model']:.1%}, coverage {overall['coverage_p10_p90']:.1%}")

dataiku.Dataset("test_results").write_with_schema(results)
