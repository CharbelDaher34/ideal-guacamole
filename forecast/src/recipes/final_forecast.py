# Final forecast: the production model (the active version of the saved model cash_forecast) forecasts, for each
# active business client, the net cash flow (money in minus money out) of each of the next 3 months, per category and
# in TOTAL, as P10, EV and P90 in SAR, from the client's daily flows (interface 1). Inference reads nothing else from
# training. final_forecast has ONE row per active client (forecast.one_row_per_client): status OK, NOT_ENOUGH_HISTORY
# (no values) or NO_POSTINGS (no posting at all, no values), the values as columns (TOTAL_m1_p10, TOTAL_m1_ev, ...),
# and which model (model_id, forecaster) and which data (load_id) produced it.
# Built to scale: the daily flows are read in chunks and spread over files on disk by client, then forecast one file
# at a time (about inference_clients_per_chunk clients each, project variable), so memory holds one chunk of clients
# whatever their number. Each chunk's rows are written at once to:
#   final_forecast: the forecast
#   forecast_archive: one compressed CSV file per cut-off month and model (forecast_<cut-off>_<model ID>.csv.gz)
#   inference_log: one row per run (appended), also when the run fails (status FAILED and the error): which model and
#   load, how many active clients got a forecast or not and why, how long each step took, the most memory used
import gzip
import math
import os
import resource
import tempfile
import time

import dataiku
import pandas as pd

from cashflow import Spread, append_row, cutoff_month, history_months, iter_dataset, read_dataset, until
from forecast import forecast_table, one_row_per_client, production_forecaster
from models import active, forecast

LOG = {"run_at": "string", "status": "string", "error": "string", "load_id": "string", "as_of_date": "string",
       "cutoff_month": "string", "model_id": "string", "forecaster": "string", "active_clients": "bigint",
       "clients_ok": "bigint", "clients_not_enough_history": "bigint", "clients_no_postings": "bigint",
       "daily_rows_read": "bigint", "chunks": "bigint", "seconds_reading": "double", "seconds_forecasting": "double",
       "seconds_total": "double", "peak_memory_mb": "bigint", "archive_file": "string"}
NO_ROWS = ["client_no", "category", "months_ahead", "p10", "ev", "p90", "status", "history_months"]
started = time.time()
variables = dataiku.get_custom_variables()
log = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "load_id": variables.get("load_id", ""),
       "as_of_date": variables.get("as_of_date", ""), "cutoff_month": variables.get("cutoff_month", "")}


def run():
    """The forecast; fills log as it goes, so a failed run still logs what it got to."""
    last_month = cutoff_month()
    forecaster, model_id = production_forecaster()
    log.update(model_id=model_id, forecaster=getattr(forecaster, "name", type(forecaster).__name__))
    _, clients = active(pd.DataFrame(columns=["client_no"]), read_dataset("clients"))     # the active clients
    numbers = set(clients["client_no"])
    if not numbers:
        raise ValueError("No active client in clients: check the staging and STS_FLG in warehouse.py")
    chunks = max(1, math.ceil(len(numbers) / int(variables.get("inference_clients_per_chunk") or 5000)))
    name = f"forecast_{last_month:%Y-%m}_{model_id}.csv.gz"
    log.update(active_clients=len(numbers), chunks=chunks, archive_file=name)

    with tempfile.TemporaryDirectory() as folder:
        spread = Spread(folder, chunks)
        daily_rows = 0
        for daily in iter_dataset("daily_flows"):
            daily = until(daily, last_month)
            daily = daily[daily["client_no"].isin(numbers)]
            daily_rows += len(daily)
            spread.add(daily, "client_no")
        read_seconds = time.time() - started
        log.update(daily_rows_read=daily_rows, seconds_reading=round(read_seconds, 1))

        # forecast every chunk first, kept on disk: final_forecast is only written once all of them worked, so a
        # failed run leaves the previous forecast as it was
        done, seen = [], set()
        for daily in spread.parts():
            group = clients[clients["client_no"].isin(set(daily["client_no"]))]
            pred = forecast(forecaster, daily, group, last_month)
            table = forecast_table(pred, history_months(daily, last_month), last_month)
            done.append(keep(one_row_per_client(table, last_month), folder, len(done)))
            seen |= set(group["client_no"])
        missing = numbers - seen                     # active clients with no posting at all: one row each
        if missing:
            done.append(keep(one_row_per_client(pd.DataFrame(columns=NO_ROWS), last_month, missing), folder,
                             len(done)))

        out, writer, counts = dataiku.Dataset("final_forecast"), None, {}
        with dataiku.Folder("forecast_archive").get_writer(name) as archive_file, \
                gzip.GzipFile(fileobj=archive_file, mode="wb") as archive:
            for path in done:
                frame = pd.read_pickle(path)
                frame["model_id"], frame["forecaster"] = model_id, log["forecaster"]
                frame["load_id"] = variables.get("load_id", "")
                frame = frame.round(2)
                if writer is None:
                    out.write_schema_from_dataframe(frame)
                    writer = out.get_writer()
                writer.write_dataframe(frame)
                archive.write(frame.to_csv(index=False, header=path == done[0]).encode())
                for status, n in frame["status"].value_counts().items():
                    counts[status] = counts.get(status, 0) + int(n)
        writer.close()
    log.update(clients_ok=counts.get("OK", 0), clients_not_enough_history=counts.get("NOT_ENOUGH_HISTORY", 0),
               clients_no_postings=counts.get("NO_POSTINGS", 0),
               seconds_forecasting=round(time.time() - started - read_seconds, 1))


def keep(frame, folder, n):
    """A chunk's forecast, kept on local disk until every chunk is done. Returns its path."""
    path = os.path.join(folder, f"forecast_{n}.pkl")
    frame.to_pickle(path)
    return path


def write_log(status, error=""):
    log.update(status=status, error=error[:1000], seconds_total=round(time.time() - started, 1),
               peak_memory_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    append_row("inference_log", LOG, log)


try:
    run()
except Exception as e:
    write_log("FAILED", f"{type(e).__name__}: {e}")
    raise
write_log("SUCCESS")
print(f"Forecast with {log['model_id']} ({log['forecaster']}): {log['active_clients']:,} active clients in "
      f"{log['chunks']} chunks: {log['clients_ok']:,} OK, {log['clients_not_enough_history']:,} without enough "
      f"history, {log['clients_no_postings']:,} without postings; {log['seconds_total']:.0f} s; "
      f"archived {log['archive_file']}")
