# Final forecast: the production model (the active version of the saved model cash_forecast) forecasts, for each
# active business client, the net cash flow (money in minus money out) of each of the next 3 windows of 30 days after
# the cut-off day (yesterday): days 1-30, 31-60 and 61-90, per category and in TOTAL, as P10, EV and P90 in SAR, from the client's daily flows (interface 1). Inference reads nothing else from
# training. final_forecast has ONE row per active client (forecast.one_row_per_client): status OK, NOT_ENOUGH_HISTORY
# (no values) or NO_POSTINGS (no posting at all, no values), the values as columns (TOTAL_w1_p10, TOTAL_w1_ev, ...),
# and which model (model_id, forecaster) and which data (load_id) produced it.
# Built to scale: staging keeps daily_flows grouped by client, so it is read straight through in chunks of complete
# clients (cashflow.iter_clients: about inference_clients_per_chunk clients each, project variable) and each chunk is
# forecast as soon as it is read. Memory holds one chunk of clients, whatever their number. The chunks' forecasts are
# kept on local disk, then, once every chunk has worked, written to:
#   final_forecast: the forecast
#   forecast_archive: one compressed CSV file per cut-off day and model (forecast_<cut-off>_<model ID>.csv.gz)
#   inference_log: one row per run (appended), also when the run fails (status FAILED and the error): which model and
#   load, how many active clients got a forecast or not and why, how long each step took, the most memory used
import gzip
import os
import resource
import tempfile
import time

import dataiku
import pandas as pd

from cashflow import append_row, cutoff_date, history_days, iter_clients, read_dataset, until
from forecast import forecast_table, one_row_per_client, production_forecaster
from models import active, forecast

LOG = {"run_at": "string", "status": "string", "error": "string", "load_id": "string", "as_of_date": "string",
       "cutoff_date": "string", "model_id": "string", "forecaster": "string", "active_clients": "bigint",
       "clients_ok": "bigint", "clients_not_enough_history": "bigint", "clients_no_postings": "bigint",
       "daily_rows_read": "bigint", "chunks": "bigint", "seconds_reading": "double", "seconds_forecasting": "double",
       "seconds_writing": "double", "seconds_total": "double", "peak_memory_mb": "bigint", "archive_file": "string"}
NO_ROWS = ["client_no", "category", "window", "p10", "ev", "p90", "status", "history_days"]
started = time.time()
variables = dataiku.get_custom_variables()
log = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "load_id": variables.get("load_id", ""),
       "as_of_date": variables.get("as_of_date", ""), "cutoff_date": variables.get("cutoff_date", "")}


def run():
    """The forecast; fills log as it goes, so a failed run still logs what it got to."""
    cutoff = cutoff_date()
    forecaster, model_id = production_forecaster()
    log.update(model_id=model_id, forecaster=getattr(forecaster, "name", type(forecaster).__name__))
    _, clients = active(pd.DataFrame(columns=["client_no"]), read_dataset("clients"))     # the active clients
    numbers = set(clients["client_no"])
    if not numbers:
        raise ValueError("No active client in clients: check the staging and STS_FLG in warehouse.py")
    per_chunk = int(variables.get("inference_clients_per_chunk") or 5000)
    name = f"forecast_{cutoff:%Y-%m-%d}_{model_id}.csv.gz"
    log.update(active_clients=len(numbers), archive_file=name)

    with tempfile.TemporaryDirectory() as folder:
        # read and forecast chunk by chunk; the forecasts are kept on disk: final_forecast is only written once every
        # chunk worked, so a failed run leaves the previous forecast as it was
        done, seen, daily_rows, forecasting = [], set(), 0, 0.0
        for daily in iter_clients("daily_flows", per_chunk,
                                  keep=lambda d: until(d[d["client_no"].isin(numbers)], cutoff)):
            ticked = time.time()
            group = clients[clients["client_no"].isin(set(daily["client_no"]))]
            pred = forecast(forecaster, daily, group, cutoff)
            table = forecast_table(pred, history_days(daily, cutoff), cutoff)
            done.append(keep(one_row_per_client(table, cutoff), folder, len(done)))
            seen |= set(group["client_no"])
            daily_rows += len(daily)
            forecasting += time.time() - ticked
        missing = numbers - seen                     # active clients with no posting at all: one row each
        if missing:
            done.append(keep(one_row_per_client(pd.DataFrame(columns=NO_ROWS), cutoff, missing), folder,
                             len(done)))
        read_done = time.time()
        log.update(daily_rows_read=daily_rows, chunks=len(done) - bool(missing), seconds_forecasting=round(forecasting, 1),
                   seconds_reading=round(read_done - started - forecasting, 1))

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
               clients_no_postings=counts.get("NO_POSTINGS", 0), seconds_writing=round(time.time() - read_done, 1))


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
