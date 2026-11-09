# Load test, step 2 of 2: times the nightly forecast (like final_forecast) on the fake data of load_test_data, with
# a dummy model by default. It never reads or writes the real data, the real models or their logs. No real training:
# the model is fitted on 2 clients only (a few seconds), since only the forecast's time is measured:
#   the daily flows read in chunks of complete clients, each chunk forecast (one row per client), and each chunk
#   written to load_test_forecast (the columns of final_forecast)
# load_test_log: one row per run (appended), also when the run fails (status FAILED and the error): the size of the
# data, the seconds of each step (reading, forecasting, writing), the speed, the most memory used.
# Variables (scenario Load test, step 1): loadtest_clients_per_chunk (clients forecast at a time, like the project
# variable inference_clients_per_chunk), loadtest_model: dummy (default), or the name of a real forecaster
# (quantile_boosting, ...) to time it instead.
import json
import resource
import time

import dataiku
import pandas as pd

from cashflow import (FORECAST_COLUMNS, HORIZONS, Progress, append_row, history_days, iter_clients, period_matrix,
                      read_dataset, whole_number)
from forecast import forecast_table, one_row_per_client
from models import forecast, new_forecaster

LOG = {"run_at": "string", "status": "string", "error": "string", "model": "string", "cutoff_date": "string",
       "clients": "bigint", "months": "bigint", "postings_per_month": "bigint", "postings": "bigint",
       "daily_rows": "bigint", "parquet_files": "bigint", "parquet_mb": "double", "clients_per_chunk": "bigint",
       "chunks": "bigint", "clients_ok": "bigint", "clients_not_enough_history": "bigint",
       "seconds_generating": "double", "seconds_training_2_clients": "double", "seconds_reading": "double",
       "seconds_forecasting": "double", "seconds_writing": "double", "seconds_total": "double",
       "clients_per_second": "double", "daily_rows_per_second": "double", "peak_memory_mb": "bigint"}
started = time.time()
variables = dataiku.get_custom_variables()
log = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}


class Dummy:
    """The dummy model: same inputs and outputs as a real forecaster (models.py), almost no computing. EV = the net
    cash flow of the last 30-day period, P10 / P90 = EV -/+ half of it."""
    name = "dummy"
    params = {}

    def fit(self, daily, clients, cutoff):
        period_matrix(daily, cutoff)              # the same data preparation as a real model
        return self

    def predict(self, daily, clients, cutoff):
        wide = period_matrix(daily, cutoff)
        if wide.empty:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        ev = wide.iloc[:, -1].fillna(0.0)
        one = pd.DataFrame({"p10": ev - ev.abs() / 2, "ev": ev, "p90": ev + ev.abs() / 2}).reset_index()
        return pd.concat([one.assign(window=h) for h in HORIZONS], ignore_index=True)[FORECAST_COLUMNS]


def step(what):
    print(f"PROGRESS load test 2/2, {what} ({time.time() - started:.0f} s so far)", flush=True)


def memory():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def run():
    with dataiku.Folder("load_test_flows").get_download_stream("benchmark.json") as stream:
        info = json.loads(stream.read())
    log.update({k: info[k] for k in ("clients", "months", "postings_per_month", "postings", "daily_rows",
                                     "parquet_files", "parquet_mb", "seconds_generating")}, cutoff_date=info["cutoff"])
    cutoff = pd.Timestamp(info["cutoff"])
    name = variables.get("loadtest_model") or "dummy"
    make = Dummy if name == "dummy" else lambda: new_forecaster(name)
    per_chunk = whole_number("loadtest_clients_per_chunk", variables.get("loadtest_clients_per_chunk", 5000)) or 5000
    log.update(model=name, clients_per_chunk=per_chunk)

    step(f"fitting {name} on 2 clients only (no real training: only the forecast is timed)")
    ticked = time.time()
    clients = read_dataset("load_test_clients")
    first = next(iter_clients("load_test_flows", 2))
    model = make().fit(first, clients[clients["client_no"].isin(set(first["client_no"]))], cutoff)
    log["seconds_training_2_clients"] = round(time.time() - ticked, 1)
    del first

    forecasting, writing, chunks, counts, writer = 0.0, 0.0, 0, {}, None
    out, looping = dataiku.Dataset("load_test_forecast"), time.time()
    progress = Progress(f"load test 2/2, forecasting and writing load_test_forecast ({per_chunk:,} clients per "
                        "chunk)", len(clients))
    for daily in iter_clients("load_test_flows", per_chunk):
        ticked = time.time()
        group = clients[clients["client_no"].isin(set(daily["client_no"]))]
        table = forecast_table(forecast(model, daily, group, cutoff), history_days(daily, cutoff), cutoff)
        frame = one_row_per_client(table, cutoff).round(2)
        frame["model_id"], frame["forecaster"], frame["load_id"] = "load_test", name, "load_test"
        forecasting += time.time() - ticked
        ticked = time.time()
        if writer is None:
            out.write_schema_from_dataframe(frame)
            writer = out.get_writer()
        writer.write_dataframe(frame)
        writing += time.time() - ticked
        for status, n in frame["status"].value_counts().items():
            counts[status] = counts.get(status, 0) + int(n)
        chunks += 1
        progress.add(len(group))
    if writer is not None:
        writer.close()
    progress.finish()
    log.update(seconds_reading=round(time.time() - looping - forecasting - writing, 1),
               seconds_forecasting=round(forecasting, 1), seconds_writing=round(writing, 1), chunks=chunks,
               clients_ok=counts.get("OK", 0), clients_not_enough_history=counts.get("NOT_ENOUGH_HISTORY", 0))
    seconds = max(time.time() - looping, 0.1)                   # reading, forecasting and writing
    log.update(clients_per_second=round(sum(counts.values()) / seconds, 1),
               daily_rows_per_second=round(log["daily_rows"] / seconds, 1))


def write_log(status, error=""):
    log.update(status=status, error=error[:1000], seconds_total=round(time.time() - started, 1),
               peak_memory_mb=memory())
    append_row("load_test_log", LOG, log)


try:
    run()
except Exception as e:
    write_log("FAILED", f"{type(e).__name__}: {e}")
    raise
write_log("SUCCESS")
print("PROGRESS load test 2/2: done. Summary (also in load_test_log):", flush=True)
for key in ("model", "clients", "daily_rows", "parquet_mb", "chunks", "seconds_generating", "seconds_reading",
            "seconds_forecasting", "seconds_writing", "seconds_total",
            "clients_per_second", "peak_memory_mb"):
    print(f"PROGRESS    {key:<30} {log.get(key)}", flush=True)
