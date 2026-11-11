"""Inference (project library: python/forecast.py): the production model and the nightly forecast. Inference reads
nothing from training except the active version of the saved model cash_forecast (interface 3, cashflow.SAVED_MODEL).
The load test project runs this same file on fake data.

The nightly forecast (forecast_all), Polars from start to end: the groups of clients of daily_flows run in parallel
(`workers` threads: Polars, the Parquet files and CatBoost work outside Python's lock, so threads run side by side;
processes would not be safe here, as Polars and CatBoost keep their own threads), each group on its own from start
to end (forecast_group):
    reading      the files of its last months only, the last input days (365) of its active clients
    preparing    the model's input rows (cashflow.model_rows: client, category, window, day_365 ... day_1)
    model        the model's predict_rows: P10, EV, P90 of each row
    formatting   one row per client (output_rows)
    writing      one Parquet file of the group, straight into the output folder (one group at a time: Dataiku
                 allows one writer per folder)
The output folder holds one subfolder per night, forecast_<cut-off>_<model ID> (kept: the archive), and current.txt,
which points to the newest one only once every group is written: a failed night leaves the previous forecast current.
The SQL table final_forecast is then loaded from it (load_to_sql)."""
import io
import os
import time

import numpy as np
import pandas as pd
import polars as pl

from cashflow import (DAILY_FLOWS, HORIZONS, MIN_HISTORY_DAYS, SAVED_MODEL, TOTAL, Progress, client_groups,
                      day_matrix, input_days, local_path, model_rows, peak_memory_mb, read_first_days, read_group,
                      read_manifest, read_parquet, window)

VALUES = ("p10", "ev", "p90")
STEPS = ("reading", "preparing", "model", "formatting", "writing")
OUTPUT = "forecast"                 # the output folder
CURRENT = "/current.txt"            # in it: the subfolder of the current forecast


def _saved_model_id():
    import dataiku
    project = dataiku.api_client().get_default_project()
    ids = {m["name"]: m["id"] for m in project.list_saved_models()}
    if SAVED_MODEL not in ids:
        raise ValueError(f"Saved model {SAVED_MODEL} not found. Run the setup notebook.")
    return ids[SAVED_MODEL]


def active_model_id():
    """The production model: the active version of the saved model cash_forecast."""
    import dataiku
    model = dataiku.api_client().get_default_project().get_saved_model(_saved_model_id())
    model_id = (model.get_active_version() or {}).get("id")
    if not model_id:
        raise ValueError("No production model yet: run the scenario Retrain, test and promote")
    return model_id


def production_forecaster():
    """The production model (the active version of cash_forecast), and its model ID. The recipe that calls it must
    have the saved model as an input."""
    import dataiku
    model_id = active_model_id()
    predictor = dataiku.Model(_saved_model_id()).get_predictor(model_id)
    return predictor.model.unwrap_python_model().forecaster, model_id


def value_columns():
    """The value columns of the forecast, always the same and in this order: TOTAL first, then each category of
    warehouse.py (CATEGORY_OF_TYPE and OTHER), each for windows 1, 2, 3 (days 1-30, 31-60, 61-90 after the cut-off
    day) and P10, EV, P90: TOTAL_w1_p10, TOTAL_w1_ev, TOTAL_w1_p90, TOTAL_w2_p10, ..., UTILITIES_w3_p90."""
    from warehouse import CATEGORY_OF_TYPE, OTHER
    categories = [TOTAL] + sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER})
    return [f"{c}_w{h}_{v}" for c in categories for h in HORIZONS for v in VALUES]


def output_rows(pred, clients, cutoff):
    """The forecast of a group: ONE row per client of `clients` (client_no, first_day, recent: whether it has a
    posting in its last input days). pred: one row per client, category and window with p10, ev, p90. Columns:
    client_no, cutoff_date (the last day of data), w1_start ... w3_end (the days of each window), status (OK;
    NOT_ENOUGH_HISTORY: less than MIN_HISTORY_DAYS days since its first posting; NO_POSTINGS: none in its last input
    days), history_days, then value_columns() (empty for a category the client does not have, and when not OK)."""
    columns = value_columns()
    wide = (pred.unpivot(index=["client_no", "category", "window"], on=list(VALUES))
            .with_columns(key=pl.format("{}_w{}_{}", "category", "window", "variable"))
            .pivot(on="key", index="client_no", values="value")) if len(pred) else pl.DataFrame(
        schema={"client_no": pl.Utf8})
    unknown = sorted({c.split("_w")[0] for c in wide.columns[1:] if c not in columns})
    if unknown:
        raise ValueError(f"Forecast for categories not in warehouse.py: {', '.join(unknown)}")
    end = cutoff.to_pydatetime().date()
    history = (pl.lit(end) - pl.col("first_day").dt.date()).dt.total_days() + 1
    dates = {"cutoff_date": pl.lit(end)}
    for h in HORIZONS:
        first, last = window(cutoff, h)
        dates[f"w{h}_start"], dates[f"w{h}_end"] = pl.lit(first.date()), pl.lit(last.date())
    out = clients.with_columns(history_days=history.fill_null(0).cast(pl.Int64)).join(wide, on="client_no", how="left")
    status = (pl.when(pl.col("client_no").is_in(wide["client_no"].implode())).then(pl.lit("OK"))
              .when(pl.col("recent") & (pl.col("history_days") < MIN_HISTORY_DAYS)).then(pl.lit("NOT_ENOUGH_HISTORY"))
              .otherwise(pl.lit("NO_POSTINGS")))
    return out.select("client_no", *[v.alias(k) for k, v in dates.items()], status.alias("status"), "history_days",
                      *[pl.col(c).cast(pl.Float64) if c in out.columns else pl.lit(None, pl.Float64).alias(c)
                        for c in columns])


_JOB = {}       # what every group needs, set by forecast_all before the workers start


def forecast_group(group):
    """One group of clients from start to end (see the module's text). Returns its seconds per step and counts."""
    job, times = _JOB, {}
    model, cutoff, days = job["model"], job["cutoff"], job["days"]
    ticked = time.time()
    clients = job["clients"][group]
    since = cutoff - pd.Timedelta(days=days - 1)
    daily = read_group(job["folder"], job["manifest"], group, since).filter(
        pl.col("client_no").is_in(clients["client_no"].implode()))
    clients = clients.with_columns(recent=pl.col("client_no").is_in(daily["client_no"].unique().implode()))
    enough = clients.filter(pl.col("recent") & ((pl.lit(cutoff.to_pydatetime()) - pl.col("first_day")).dt.total_days()
                                                >= MIN_HISTORY_DAYS - 1))
    daily = daily.filter(pl.col("client_no").is_in(enough["client_no"].implode()))
    times["reading"], ticked = time.time() - ticked, time.time()
    rows = model_rows(*day_matrix(daily, cutoff, days))
    times["preparing"], ticked = time.time() - ticked, time.time()
    values = np.asarray(model.predict_rows(rows), dtype=np.float64) if len(rows) else np.zeros((0, 3))
    if values.shape != (len(rows), 3) or not np.isfinite(values).all():
        raise ValueError(f"The model's predict_rows must return one finite P10, EV, P90 per input row: got {values.shape}")
    times["model"], ticked = time.time() - ticked, time.time()
    ev = values[:, 1]
    pred = rows.select("client_no", "category", "window").with_columns(
        p10=np.minimum(values[:, 0], ev).round(2), ev=ev.round(2), p90=np.maximum(values[:, 2], ev).round(2))
    out = output_rows(pred, clients, cutoff).with_columns(**{k: pl.lit(v) for k, v in job["columns"].items()})
    times["formatting"], ticked = time.time() - ticked, time.time()
    buffer = io.BytesIO()
    out.write_parquet(buffer)
    with job["writing"], job["output"].get_writer(f"/{job['version']}/part_{group:05d}.parquet") as writer:
        writer.write(buffer.getvalue())             # one writer at a time: Dataiku allows one per folder
    times["writing"] = time.time() - ticked
    counts = out["status"].value_counts()
    return {**times, "daily_rows": len(daily), "clients": len(out),
            **dict(zip(counts["status"].to_list(), counts["count"].to_list()))}


def forecast_all(model, model_id, clients, cutoff, workers, name=DAILY_FLOWS, output=OUTPUT, columns=None,
                 prefix=""):
    """The nightly forecast of `clients` (Polars or pandas, client_no: the active clients) with `model`, into the
    folder `output` (see the module's text). columns: values added to every row (forecaster, load_id, ...). prefix:
    the subfolder of `name` holding the daily flows (cashflow.read_manifest; "" in production). Returns
    the seconds of each step (added up over the groups), the wall time, the counts per status, the most memory used
    and the output's subfolder."""
    from concurrent.futures import ThreadPoolExecutor
    import dataiku
    started = time.time()
    folder, manifest = read_manifest(name, prefix)
    if manifest is None:
        raise ValueError(f"{name} is empty: run the staging first")
    clients = (clients if isinstance(clients, pl.DataFrame) else pl.from_pandas(clients[["client_no"]])).select(
        pl.col("client_no").cast(pl.Utf8)).unique()
    clients = clients.join(read_first_days(folder, manifest), on="client_no", how="left").with_columns(
        group=client_groups(clients["client_no"], manifest["groups"]))
    by_group = {g: part.drop("group") for (g,), part in clients.group_by(["group"])}
    version = f"forecast_{cutoff:%Y-%m-%d}_{model_id}"
    out = dataiku.Folder(output)
    for path in out.list_paths_in_partition():                  # an earlier try of the same night
        if path.startswith(f"/{version}/"):
            out.delete_path(path)
    workers = max(1, min(int(workers), len(by_group)))
    model.threads = max(1, (os.cpu_count() or 1) // workers)    # CatBoost's threads: the cores shared out
    import threading
    _JOB.update(model=model, cutoff=cutoff, days=input_days(model), clients=by_group, folder=folder, writing=threading.Lock(),
                manifest=manifest, output=out, version=version,
                columns={"model_id": model_id, **(columns or {})})
    summary = {k: 0.0 for k in STEPS}
    summary.update(daily_rows=0, groups=len(by_group), workers=workers)
    progress = Progress(f"forecasting {len(clients):,} clients: {len(by_group):,} groups, {workers} at a time",
                        len(clients))
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(forecast_group, sorted(by_group)):
            for k, v in result.items():
                if k != "clients":
                    summary[k] = summary.get(k, 0) + v
            progress.add(result["clients"])
    progress.finish()
    with out.get_writer(CURRENT) as writer:
        writer.write(version.encode())
    summary.update(clients=len(clients), seconds=time.time() - started, output=version, peak_memory_mb=peak_memory_mb())
    return summary


def current_output(output=OUTPUT):
    """The output folder, its current subfolder and that subfolder's Parquet files (one per group)."""
    import dataiku
    folder = dataiku.Folder(output)
    paths = folder.list_paths_in_partition()
    if CURRENT not in paths:
        raise ValueError(f"No forecast in the folder {output} yet: run the nightly forecast first")
    with open(local_path(folder, CURRENT)) as f:
        version = f.read().strip()
    return folder, version, sorted(p for p in paths if p.startswith(f"/{version}/") and p.endswith(".parquet"))


def load_to_sql(dataset, output=OUTPUT):
    """Load the current forecast of the output folder into the SQL dataset `dataset` (replacing what was there), one
    group's file at a time. Returns the rows loaded and the subfolder."""
    import dataiku
    folder, version, paths = current_output(output)
    sql, writer, rows = dataiku.Dataset(dataset), None, 0
    progress = Progress(f"loading {version} into {dataset} (files)", len(paths))
    try:
        for path in paths:
            frame = read_parquet(folder, path).to_pandas()
            if writer is None:
                sql.write_schema_from_dataframe(frame)
                writer = sql.get_writer()
            writer.write_dataframe(frame)
            rows += len(frame)
            progress.add()
    finally:
        if writer is not None:
            writer.close()
    progress.finish()
    return rows, version
