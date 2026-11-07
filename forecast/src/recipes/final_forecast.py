# Final forecast: the production model (the active version of the saved model cash_forecast) forecasts, for each
# active business client, the net cash flow (money in minus money out) of each of the next 3 months, per category and
# in TOTAL, as P10, EV and P90 in SAR, from the client's daily flows (interface 1). Every row says which model
# (model_id: the active version) and which data (load_id) produced it. Inference reads nothing else from training.
# Built to scale: the daily flows are read in chunks and spread over files on disk by client, then forecast one file
# at a time (about inference_clients_per_chunk clients each, project variable), so memory holds one chunk of clients
# whatever their number. Each chunk's rows are written at once to:
#   final_forecast: the forecast
#   forecast_archive: one compressed CSV file per cut-off month and model (forecast_<cut-off>_<model ID>.csv.gz)
#   inference_log: one row per run (appended): which model and load, what it produced, how long each step took, and
#   the most memory the run used
import gzip
import math
import resource
import tempfile
import time

import dataiku
import pandas as pd

from cashflow import Spread, cutoff_month, iter_dataset, read_dataset, history_months, until
from forecast import forecast_table, production_forecaster
from models import active, forecast

COLUMNS = ["client_no", "category", "cutoff_month", "months_ahead", "target_month", "p10", "ev", "p90",
           "history_months", "status", "model_id", "load_id"]
started = time.time()
variables = dataiku.get_custom_variables()
last_month = cutoff_month()
forecaster, model_id = production_forecaster()
_, clients = active(pd.DataFrame(columns=["client_no"]), read_dataset("clients"))     # the active clients
numbers = set(clients["client_no"])
per_chunk = int(variables.get("inference_clients_per_chunk") or 5000)
chunks = max(1, math.ceil(len(numbers) / per_chunk))
name = f"forecast_{last_month:%Y-%m}_{model_id}.csv.gz"

with tempfile.TemporaryDirectory() as folder:
    spread = Spread(folder, chunks)
    daily_rows = 0
    for daily in iter_dataset("daily_flows"):
        daily = until(daily, last_month)
        daily = daily[daily["client_no"].isin(numbers)]
        daily_rows += len(daily)
        spread.add(daily, "client_no")
    read_seconds = time.time() - started

    out = dataiku.Dataset("final_forecast")
    writer, rows, without_history, forecast_clients = None, 0, 0, 0
    with dataiku.Folder("forecast_archive").get_writer(name) as archive_file, \
            gzip.GzipFile(fileobj=archive_file, mode="wb") as archive:
        for daily in spread.parts():
            group = clients[clients["client_no"].isin(set(daily["client_no"]))]
            pred = forecast(forecaster, daily, group, last_month)
            frame = forecast_table(pred, history_months(daily, last_month), last_month)
            frame["model_id"] = model_id
            frame["load_id"] = variables.get("load_id", "")
            frame = frame[COLUMNS].round(2)
            if writer is None:
                out.write_schema_from_dataframe(frame)
                writer = out.get_writer()
            writer.write_dataframe(frame)
            archive.write(frame.to_csv(index=False, header=rows == 0).encode())
            rows += len(frame)
            first = frame.drop_duplicates("client_no")
            forecast_clients += len(first)
            without_history += int((first["status"] != "OK").sum())
    if writer is None:
        raise ValueError("No active client with daily flows to forecast: check daily_flows and clients")
    writer.close()

total_seconds = time.time() - started
dataiku.Dataset("inference_log").write_with_schema(pd.DataFrame([{
    "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "load_id": variables.get("load_id", ""),
    "as_of_date": variables.get("as_of_date", ""),
    "cutoff_month": f"{last_month:%Y-%m}",
    "model_id": model_id,
    "clients": forecast_clients,
    "clients_not_enough_history": without_history,
    "forecast_rows": rows,
    "daily_rows_read": daily_rows,
    "chunks": chunks,
    "seconds_reading": round(read_seconds, 1),
    "seconds_forecasting": round(total_seconds - read_seconds, 1),
    "seconds_total": round(total_seconds, 1),
    "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    "archive_file": name}]))
print(f"Forecast with {model_id}: {forecast_clients:,} clients in {chunks} chunks, {rows:,} rows "
      f"({without_history:,} clients without enough history), {total_seconds:.0f} s; archived {name}")
