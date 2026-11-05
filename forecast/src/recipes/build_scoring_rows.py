# Scoring rows: one row per client, category and months ahead (1, 2, 3) at the latest cut-off, to forecast with the
# production model. Inference only. The rows are made by model_rows in the project library (cashflow.py), the same
# code that makes training_rows, so the model sees the same inputs it was trained on.
import dataiku

from cashflow import cutoff_month, model_rows, read_dataset

last_month = cutoff_month()
rows = model_rows(read_dataset("monthly_flows"), read_dataset("monthly_obligations"), read_dataset("clients"),
                  read_dataset("monthly_balances"), last_month)
scoring = rows[rows["cutoff_month"] == last_month].drop(columns=["actual", "obligations_known", "target"])
dataiku.Dataset("scoring_rows").write_with_schema(scoring.round(4).reset_index(drop=True))
