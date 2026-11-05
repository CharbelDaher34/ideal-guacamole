# Archive and log each forecast run:
#   forecast_archive: one CSV file per cut-off month and model (forecast_<cut-off>_<model ID>.csv)
#   inference_log: one row per run (appended): when it ran, which model and load it used, what it produced
import dataiku
import pandas as pd

from cashflow import read_dataset

forecast = read_dataset("final_forecast")
first = forecast.iloc[0]
name = f"forecast_{first['cutoff_month']:%Y-%m}_{first['model_id']}.csv"
with dataiku.Folder("forecast_archive").get_writer(name) as w:
    w.write(forecast.to_csv(index=False).encode())

variables = dataiku.get_custom_variables()
clients = forecast.drop_duplicates("client_no")
log = pd.DataFrame([{
    "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "load_id": variables.get("load_id", ""),
    "as_of_date": variables.get("as_of_date", ""),
    "cutoff_month": f"{first['cutoff_month']:%Y-%m}",
    "model_id": first["model_id"],
    "clients": len(clients),
    "clients_not_enough_history": int((clients["status"] == "NOT_ENOUGH_HISTORY").sum()),
    "forecast_rows": len(forecast),
    "archive_file": name}])
dataiku.Dataset("inference_log").write_with_schema(log)
print("archived", name, "| logged run with model", first["model_id"])
