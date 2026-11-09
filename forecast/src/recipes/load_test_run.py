# Load test, step 2 of 2: times the nightly forecast (like final_forecast) on the fake data of load_test_data. It
# never reads or writes the real data, the real models or their logs. The model is real CatBoost (by default) but
# trained on 2 clients only: its forecasts mean nothing, its prediction time is real. Each chunk's time is split:
#   reading      the daily flows, in chunks of complete clients
#   preparing    each client's days summed into 30-day periods, and the model's inputs
#   model        the model's predictions (CatBoost: 3 models, P10, EV, P90)
#   formatting   one row per client (final_forecast's columns)
#   1. read, forecast each chunk and keep it as a Parquet file on local disk (as final_forecast does)
#   2. write the whole forecast twice, to compare: as a CSV dataset (load_test_forecast, compressed, like
#      final_forecast) and as Parquet files (folder load_test_output)
#   3. read both back, to compare
# load_test_log: one row per run (appended), also when the run fails (status FAILED and the error): the size of the
# data, the seconds of each step, the speed, the most memory used.
# Variables (scenario Load test, step 1): loadtest_clients_per_chunk (clients forecast at a time, like the project
# variable inference_clients_per_chunk); loadtest_model: catboost (default: CatBoost trained on 2 clients), dummy
# (no model: the last period again), or a forecaster's name (quantile_boosting, ...).
import io
import json
import os
import resource
import tempfile
import time

import dataiku
import pandas as pd

from cashflow import (FORECAST_COLUMNS, HORIZONS, Progress, append_row, history_days, iter_clients, period_matrix,
                      read_dataset, whole_number)
from forecast import forecast_table, one_row_per_client
from models import QuantileBoosting, forecast, new_forecaster

LOG = {"run_at": "string", "status": "string", "error": "string", "model": "string",
       "cutoff_date": "string", "clients": "bigint", "months": "bigint", "postings_per_month": "bigint",
       "postings": "bigint", "daily_rows": "bigint", "parquet_files": "bigint", "parquet_mb": "double",
       "clients_per_chunk": "bigint", "chunks": "bigint", "clients_ok": "bigint",
       "clients_not_enough_history": "bigint", "seconds_generating": "double", "seconds_training_2_clients": "double",
       "seconds_reading": "double", "seconds_preparing": "double", "seconds_model": "double",
       "seconds_formatting": "double", "seconds_writing_csv": "double",
       "seconds_writing_parquet": "double", "seconds_reading_back_csv": "double",
       "seconds_reading_back_parquet": "double", "output_parquet_mb": "double",
       "seconds_total": "double", "clients_per_second": "double", "daily_rows_per_second": "double",
       "peak_memory_mb": "bigint"}
started = time.time()
variables = dataiku.get_custom_variables()
log = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
TIMES = {"preparing": 0.0, "model": 0.0}      # seconds, added up over the chunks by the models below


class Dummy:
    """No model: same inputs and outputs as a forecaster (models.py). EV = the net cash flow of the last 30-day
    period, P10 / P90 = EV -/+ half of it."""
    name = "dummy"
    params = {}

    def fit(self, daily, clients, cutoff):
        return self

    def predict(self, daily, clients, cutoff):
        ticked = time.time()
        wide = period_matrix(daily, cutoff)
        TIMES["preparing"] += time.time() - ticked
        if wide.empty:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        ev = wide.iloc[:, -1].fillna(0.0)
        one = pd.DataFrame({"p10": ev - ev.abs() / 2, "ev": ev, "p90": ev + ev.abs() / 2}).reset_index()
        return pd.concat([one.assign(window=h) for h in HORIZONS], ignore_index=True)[FORECAST_COLUMNS]


class CatBoostTiny(QuantileBoosting):
    """Real CatBoost (3 models: P10, EV, P90 with the quantile loss), with the inputs of quantile_boosting (the last 12
    periods of 30 days, the category, the window, the month), trained on 2 clients only: the forecasts mean nothing,
    the prediction time is real."""
    name = "catboost"
    params = {**QuantileBoosting.params, "iterations": 300, "depth": 6}

    def new_model(self, quantile):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function=f"Quantile:alpha={quantile}", iterations=self.params["iterations"],
                                 depth=self.params["depth"], random_seed=0, verbose=False, allow_writing_files=False)

    def predict(self, daily, clients, cutoff):
        ticked = time.time()
        wide = period_matrix(daily, cutoff)
        TIMES["preparing"] += time.time() - ticked
        if wide.empty:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        out = []
        for h in HORIZONS:
            ticked = time.time()
            X, scale = self._rows(wide, wide.shape[1] - 1, h)
            rows = wide.index.to_frame(index=False).assign(window=h)
            TIMES["preparing"] += time.time() - ticked
            ticked = time.time()
            for column, model in self.models.items():
                rows[column] = model.predict(X) * scale.values
            TIMES["model"] += time.time() - ticked
            out.append(rows)
        return pd.concat(out, ignore_index=True)[FORECAST_COLUMNS]


MODELS = {"dummy": Dummy, "catboost": CatBoostTiny}


def step(what):
    print(f"PROGRESS load test 2/2, {what} ({time.time() - started:.0f} s so far)", flush=True)


def memory():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def folder_mb(folder):
    return round(sum(folder.get_path_details(p).get("size", 0) for p in folder.list_paths_in_partition()) / 1e6, 1)


def run(local):
    with dataiku.Folder("load_test_flows").get_download_stream("benchmark.json") as stream:
        info = json.loads(stream.read())
    log.update({k: info[k] for k in ("clients", "months", "postings_per_month", "postings", "daily_rows",
                                     "parquet_files", "parquet_mb", "seconds_generating")}, cutoff_date=info["cutoff"])
    cutoff = pd.Timestamp(info["cutoff"])
    name = variables.get("loadtest_model") or "catboost"
    make = MODELS.get(name) or (lambda: new_forecaster(name))
    per_chunk = whole_number("loadtest_clients_per_chunk", variables.get("loadtest_clients_per_chunk", 5000)) or 5000
    log.update(model=name, clients_per_chunk=per_chunk)

    step(f"fitting {name} on 2 clients only (no real training: only the forecast is timed)")
    ticked = time.time()
    clients = read_dataset("load_test_clients")
    first = next(iter_clients("load_test_flows", 2))
    model = make().fit(first, clients[clients["client_no"].isin(set(first["client_no"]))], cutoff)
    log["seconds_training_2_clients"] = round(time.time() - ticked, 1)
    TIMES.update(preparing=0.0, model=0.0)
    del first

    # 1. read, forecast, keep each chunk as Parquet on local disk
    forecasting, kept, counts, looping = 0.0, [], {}, time.time()
    progress = Progress(f"load test 2/2, 1/3 reading and forecasting ({per_chunk:,} clients per chunk)", len(clients))
    for daily in iter_clients("load_test_flows", per_chunk):
        ticked = time.time()
        group = clients[clients["client_no"].isin(set(daily["client_no"]))]
        table = forecast_table(forecast(model, daily, group, cutoff), history_days(daily, cutoff), cutoff)
        frame = one_row_per_client(table, cutoff).round(2)
        frame["model_id"], frame["forecaster"], frame["load_id"] = "load_test", name, "load_test"
        kept.append(os.path.join(local, f"chunk_{len(kept):05d}.parquet"))
        frame.to_parquet(kept[-1], index=False)
        forecasting += time.time() - ticked
        for status, n in frame["status"].value_counts().items():
            counts[status] = counts.get(status, 0) + int(n)
        progress.add(len(group))
    progress.finish()
    log.update(seconds_reading=round(time.time() - looping - forecasting, 1),
               seconds_preparing=round(TIMES["preparing"], 1), seconds_model=round(TIMES["model"], 1),
               seconds_formatting=round(forecasting - TIMES["preparing"] - TIMES["model"], 1),
               chunks=len(kept), clients_ok=counts.get("OK", 0),
               clients_not_enough_history=counts.get("NOT_ENOUGH_HISTORY", 0))
    seconds = max(time.time() - looping, 0.1)
    log.update(clients_per_second=round(sum(counts.values()) / seconds, 1),
               daily_rows_per_second=round(log["daily_rows"] / seconds, 1))

    # 2. write the forecast as a CSV dataset (like final_forecast) and as Parquet files, to compare
    step(f"2/3 writing the forecast ({len(kept)} chunks) as a CSV dataset, load_test_forecast")
    ticked, out, writer = time.time(), dataiku.Dataset("load_test_forecast"), None
    for path in kept:
        frame = pd.read_parquet(path)
        if writer is None:
            out.write_schema_from_dataframe(frame)
            writer = out.get_writer()
        writer.write_dataframe(frame)
    if writer is not None:
        writer.close()
    log["seconds_writing_csv"] = round(time.time() - ticked, 2)
    step("2/3 writing the forecast as Parquet files, folder load_test_output")
    ticked, output = time.time(), dataiku.Folder("load_test_output")
    for path in output.list_paths_in_partition():
        output.delete_path(path)
    for path in kept:
        with open(path, "rb") as f, output.get_writer(os.path.basename(path)) as writer:
            writer.write(f.read())
    log["seconds_writing_parquet"] = round(time.time() - ticked, 2)

    # 3. read both back, to compare
    step("3/3 reading the forecast back: the CSV dataset, then the Parquet files")
    ticked = time.time()
    back_csv = len(dataiku.Dataset("load_test_forecast", ignore_flow=True).get_dataframe())   # an output
    log["seconds_reading_back_csv"] = round(time.time() - ticked, 2)
    ticked, back_parquet = time.time(), 0
    for path in output.list_paths_in_partition():
        with output.get_download_stream(path) as stream:
            back_parquet += len(pd.read_parquet(io.BytesIO(stream.read())))
    log["seconds_reading_back_parquet"] = round(time.time() - ticked, 2)
    if back_csv != back_parquet:
        raise ValueError(f"The forecast read back has {back_csv:,} rows from the CSV dataset and {back_parquet:,} "
                         "from the Parquet files")
    log["output_parquet_mb"] = folder_mb(output)


def write_log(status, error=""):
    log.update(status=status, error=error[:1000], seconds_total=round(time.time() - started, 1),
               peak_memory_mb=memory())
    append_row("load_test_log", LOG, log)


try:
    with tempfile.TemporaryDirectory() as local:
        run(local)
except Exception as e:
    write_log("FAILED", f"{type(e).__name__}: {e}")
    raise
write_log("SUCCESS")
print("PROGRESS load test 2/2: done. Summary (also in load_test_log and the dashboard Load test):", flush=True)
for key in ("model", "clients", "daily_rows", "chunks", "seconds_generating", "seconds_reading",
            "seconds_preparing", "seconds_model", "seconds_formatting", "seconds_writing_csv", "seconds_writing_parquet", "seconds_reading_back_csv",
            "seconds_reading_back_parquet", "output_parquet_mb", "seconds_total",
            "clients_per_second", "peak_memory_mb"):
    print(f"PROGRESS    {key:<30} {log.get(key)}", flush=True)
