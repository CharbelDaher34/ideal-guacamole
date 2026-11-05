# Training rows: one row per client, past cut-off month and months ahead (1, 2, 3), with its answer (the client's
# net cash flow in the predicted month). Training only: the nightly batch does not build it. The rows are made by model_rows in the project library (cashflow.py, interface 1),
# the same code that makes scoring_rows.
import dataiku

from cashflow import cutoff_month, model_rows, read_dataset

rows = model_rows(read_dataset("monthly_flows"), read_dataset("clients"), cutoff_month())
training = rows[rows["enough_history"] & rows["target"].notna()]
dataiku.Dataset("training_rows").write_with_schema(training.round(4).reset_index(drop=True))
