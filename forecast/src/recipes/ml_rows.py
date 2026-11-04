# Model-ready rows for the visual models: the same inputs the code models see (amounts divided by their usual
# size), and the answer divided the same way (target_scaled). ml_train: rows whose predicted month is before the
# last 6 months; ml_test: the last 6 months; ml_scoring: the latest cut-off, for clients and categories to forecast.
import dataiku
import pandas as pd

from cashflow import HOLDOUT_MONTHS, add_months, cutoff_month, model_matrix, month_start, read_dataset

last_month = cutoff_month()
holdout_start = add_months(last_month, -(HOLDOUT_MONTHS - 1))


def model_ready(rows, with_target=True):
    out = rows[["client_no", "cutoff_month", "target_month", "series_scale"]].join(model_matrix(rows))
    if with_target:
        out["target"] = rows["target"]
        out["target_scaled"] = rows["target"] / rows["series_scale"]
    return out.reset_index(drop=True)


training = read_dataset("training_rows")
training["target_month"] = month_start(training["target_month"])
scoring = read_dataset("scoring_rows")
scoring = scoring[scoring["enough_history"].astype(bool) & scoring["active"].astype(bool)]

dataiku.Dataset("ml_train").write_with_schema(model_ready(training[training["target_month"] < holdout_start]))
dataiku.Dataset("ml_test").write_with_schema(model_ready(training[training["target_month"] >= holdout_start]))
dataiku.Dataset("ml_scoring").write_with_schema(model_ready(scoring, with_target=False))
