# Test: score the last 6 months with the backtest models (trained without them) and compare with what happened.
#   EV error (WAPE) = sum |actual - EV| / sum |actual|, next to a simple guess (average of the last 3 months)
#   coverage = share of actual months between P10 and P90 (aim: about 80%)
import pickle

import dataiku
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, HOLDOUT_MONTHS, add_months, cutoff_month, month_start, predict, read_dataset

last_month = cutoff_month()
holdout_start = add_months(last_month, -(HOLDOUT_MONTHS - 1))
rows = read_dataset("training_rows")
rows["target_month"] = month_start(rows["target_month"])
test = rows[rows["target_month"] >= holdout_start].copy()

folder = dataiku.Folder("models")
models = {}
for kind in ("ev", "p10_p90"):
    with folder.get_download_stream(f"backtest_{kind}.pkl") as s:
        models[kind] = pickle.loads(s.read())

test = test.join(predict(models, test))
test["baseline"] = np.where(test["category"] == "NET", test["free_avg_3m"], test["free_avg_3m"].clip(lower=0))
ids = ["client_no", "category", "direction", "cutoff_month", "months_ahead", "target_month", "actual",
       "obligations_known", "target", "p10", "ev", "p90", "baseline"]
dataiku.Dataset("test_predictions").write_with_schema(test[ids].reset_index(drop=True))


def scores(d):
    size = d["target"].abs().sum()
    inside = (d["target"] >= d["p10"]) & (d["target"] <= d["p90"])
    return pd.Series({"rows": len(d), "actual_total": round(d["target"].sum(), 0),
                      "wape_model": round((d["target"] - d["ev"]).abs().sum() / size, 4) if size else np.nan,
                      "wape_baseline": round((d["target"] - d["baseline"]).abs().sum() / size, 4) if size else np.nan,
                      "coverage_p10_p90": round(inside.mean(), 4),
                      "below_p10": round((d["target"] < d["p10"]).mean(), 4),
                      "above_p90": round((d["target"] > d["p90"]).mean(), 4)})


results = [test.groupby(["category", "months_ahead"]).apply(scores).reset_index(),
           test.groupby("category").apply(scores).reset_index().assign(months_ahead="ALL")]
categories_only = test[test["category"] != "NET"]
results.append(categories_only.groupby("months_ahead").apply(scores).reset_index().assign(category="ALL"))
results.append(scores(categories_only).to_frame().T.assign(category="ALL", months_ahead="ALL"))
results = pd.concat(results, ignore_index=True)
results["months_ahead"] = results["months_ahead"].astype(str)
results["category_name"] = results["category"].map({c: v[0] for c, v in CATEGORIES.items()}).fillna("All categories")
order = ["category", "category_name", "months_ahead", "rows", "actual_total", "wape_model", "wape_baseline",
         "coverage_p10_p90", "below_p10", "above_p90"]
results = results[order].sort_values(["category", "months_ahead"]).reset_index(drop=True)
results["test_months"] = f"{holdout_start:%Y-%m} to {last_month:%Y-%m}"
dataiku.Dataset("test_results").write_with_schema(results)
