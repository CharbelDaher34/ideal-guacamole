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
The SQL table final_forecast is then loaded from it (load_to_sql, or load_to_table: see SQL tables below)."""
import io
import os
import time

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:         # the setup notebooks run on Dataiku's built-in Python: they only use the SQL table helpers
    pl = None

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


# ── SQL tables written by code ─────────────────────────────────────────────────────────────────────────────────────
# A table made with SQL (create_table_sql, ensure_table), then filled with the current forecast (load_to_table). On
# Oracle (the bank) it is written with Oracle's own driver, oracledb, in batches with bind variables (about 16,000
# rows a second; plain INSERT statements through Dataiku: under 1,000, Oracle parses each one), and no Dataiku dataset
# is needed (the bank's licence caps them). Any other database (PostgreSQL in the sandbox): through Dataiku's writer
# into a Dataiku dataset linked to the table.
SQL_TYPES = {"oracle": {"string": "VARCHAR2(4000)", "bigint": "NUMBER(19)", "double": "NUMBER", "date": "DATE"},
             "other": {"string": "VARCHAR(4000)", "bigint": "BIGINT", "double": "DOUBLE PRECISION", "date": "DATE"}}
BATCH_ROWS = 5000       # rows sent to Oracle at a time


def table_name(schema, table):
    """"SCHEMA.TABLE", or "TABLE" when the schema is empty (the connection's own)."""
    schema, table = str(schema or "").strip(), str(table or "").strip()
    return f"{schema}.{table}" if schema else table


def sql_dialect(connection):
    """"oracle" when the Dataiku connection's database answers SELECT ... FROM DUAL (Oracle), else "other"."""
    import dataiku
    try:
        dataiku.SQLExecutor2(connection=connection).query_to_df("SELECT 1 AS one FROM DUAL")
        return "oracle"
    except Exception:
        return "other"


def run_sql(connection, statements, dialect=None):
    """Runs the statements on the Dataiku connection, then COMMIT (Dataiku runs them before a query that returns rows:
    SELECT 1). dialect: sql_dialect(connection), asked when not given."""
    import dataiku
    dialect = dialect or sql_dialect(connection)
    dataiku.SQLExecutor2(connection=connection).query_to_df(
        "SELECT 1 AS one FROM DUAL" if dialect == "oracle" else "SELECT 1 AS one",
        pre_queries=list(statements), post_queries=["COMMIT"])


def output_columns(extra=None):
    """{column: kind} of the forecast's rows, in their order (kinds: string, bigint, double, date): those of
    output_rows (client_no, cutoff_date, w1_start ... w3_end, status, history_days, value_columns()), then those
    forecast_all adds to every row: model_id, then `extra` (its `columns`, e.g. {"forecaster": "string"})."""
    dates = ["cutoff_date"] + [f"w{h}_{edge}" for h in HORIZONS for edge in ("start", "end")]
    return {"client_no": "string", **{d: "date" for d in dates}, "status": "string", "history_days": "bigint",
            **{c: "double" for c in value_columns()}, "model_id": "string", **(extra or {})}


def create_table_sql(table, columns, dialect="oracle"):
    """The CREATE TABLE statement of `table` ("SCHEMA.TABLE") with `columns` ({name: kind}, see SQL_TYPES)."""
    types = SQL_TYPES[dialect]
    body = ",\n".join(f"    {name} {types[kind]}" for name, kind in columns.items())
    return f"CREATE TABLE {table} (\n{body}\n)"


def _table_columns(connection, table, dialect):
    """The columns of `table` (lower case), read from the database's catalog, or None when it is not there."""
    import dataiku
    schema, _, name = table.rpartition(".")
    quoted = lambda v: "'" + v.replace("'", "''") + "'"  # noqa: E731
    if dialect == "oracle":         # unquoted names are stored in upper case
        query = (f"SELECT column_name FROM all_tab_columns WHERE table_name = UPPER({quoted(name)}) AND owner = "
                 + (f"UPPER({quoted(schema)})" if schema else "USER"))
    else:                           # PostgreSQL: in lower case
        query = (f"SELECT column_name FROM information_schema.columns WHERE LOWER(table_name) = LOWER({quoted(name)}) "
                 + (f"AND LOWER(table_schema) = LOWER({quoted(schema)})" if schema
                    else "AND table_schema = current_schema()"))
    found = dataiku.SQLExecutor2(connection=connection).query_to_df(query)
    names = {str(n).lower() for n in found.iloc[:, 0]} if len(found.columns) else set()
    return names or None


def ensure_table(connection, table, columns):
    """Creates `table` ("SCHEMA.TABLE") on the Dataiku connection when it is not there; when it is, adds the columns
    of `columns` ({name: kind}) it lacks (a newer version of the code). Returns a sentence saying what was done."""
    dialect = sql_dialect(connection)
    have = _table_columns(connection, table, dialect)
    if have is None:
        run_sql(connection, [create_table_sql(table, columns, dialect)], dialect)
        return f"created the table {table} on {connection}"
    missing = [c for c in columns if c.lower() not in have]
    if not missing:
        return f"the table {table} on {connection} is there, with every column"
    add = "ALTER TABLE {} ADD ({} {})" if dialect == "oracle" else "ALTER TABLE {} ADD COLUMN {} {}"
    run_sql(connection, [add.format(table, c, SQL_TYPES[dialect][columns[c]]) for c in missing], dialect)
    return (f"added {len(missing)} column{'s' if len(missing) > 1 else ''} to the table {table} on {connection}: "
            f"{', '.join(missing)}")


def _oracle_dsn(params):
    """The address of an Oracle Dataiku connection for oracledb, from its params: its JDBC URL (useURL:
    jdbc:oracle:thin:@//host:port/service, @host:port/service, @host:port:SID or @(DESCRIPTION=...)), else its host,
    port and db, which Dataiku joins as host:port:db (db: the SID)."""
    if params.get("useURL"):
        dsn = str(params.get("url") or "").strip()
        dsn = dsn.split("@", 1)[1] if "@" in dsn else dsn
    else:
        dsn = f"{params.get('host')}:{params.get('port') or 1521}:{params.get('db')}"
    dsn = dsn[2:] if dsn.startswith("//") else dsn
    parts = dsn.split(":")
    if not dsn.startswith("(") and "/" not in dsn and len(parts) == 3:          # host:port:SID
        return (f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={parts[0]})(PORT={parts[1]}))"
                f"(CONNECT_DATA=(SID={parts[2]})))")
    return dsn


def _oracle_login(connection):
    """(user, password, address) of an Oracle Dataiku connection, read from Dataiku: the connection's details must be
    readable by one of the user's groups."""
    import dataiku
    ask = (f"Ask a Dataiku admin to add your group to 'Details readable by' of the connection {connection} "
           "(Administration > Connections > the connection > Security)")
    try:
        info = dataiku.api_client().get_connection(connection).get_info()
    except Exception as e:
        raise RuntimeError(f"Cannot read the details of the connection {connection} ({e}). {ask}.") from None
    params = info.get("params") or {}
    try:
        login = info.get_basic_credential()
    except Exception:           # not given: the params may hold them
        login = {"user": params.get("user"), "password": params.get("password")}
    if not login.get("user") or not login.get("password"):
        raise RuntimeError(f"Dataiku did not give the user and password of the connection {connection}. {ask}; if it "
                           "uses per-user credentials, enter yours (Profile > Credentials).")
    return login["user"], login["password"], _oracle_dsn(params)


def _oracle_connect(connection):
    """An oracledb connection (not in autocommit) with the login of the Oracle Dataiku connection."""
    import oracledb
    user, password, dsn = _oracle_login(connection)
    return oracledb.connect(user=user, password=password, dsn=dsn)


def check_table_writer(connection, driver=True):
    """For an Oracle connection, checks load_to_table can write there: the package oracledb is in this Python and logs
    in (driver=False: not checked, for a notebook running on another Python than the recipes), and the connection's
    user and password can be read. Raises a RuntimeError saying what to ask the admin. Returns a sentence, or None
    for another database (Dataiku's writer: nothing to check)."""
    if sql_dialect(connection) != "oracle":
        return None
    if driver:
        try:
            import oracledb
        except ImportError:
            raise RuntimeError("The package oracledb (Oracle's Python driver) is not in this code env: ask a Dataiku "
                               "admin to add oracledb to it. The forecast is written into the Oracle table with it "
                               "(no Dataiku dataset).") from None
    user, password, dsn = _oracle_login(connection)
    if not driver:
        return f"the login of {connection} can be read (user {user})"
    try:
        oracledb.connect(user=user, password=password, dsn=dsn).close()
    except Exception as e:
        raise RuntimeError(f"oracledb could not log in to {dsn} as {user}, the login of {connection} ({e}). With "
                           "DPY-3010 or DPY-3015 (an older Oracle), oracledb needs its thick mode: ask the admin for "
                           "the Oracle Instant Client on the Dataiku server.") from None
    return f"oracledb logs in to {connection} (user {user})"


def _kind(dtype):
    """The kind (SQL_TYPES) of a Polars column type."""
    name = str(dtype).lower()
    if name.startswith(("int", "uint")) or name == "boolean":
        return "bigint"
    return "double" if name.startswith("float") else "date" if name.startswith("date") else "string"


def load_to_table(connection, table, dataset=None, output=OUTPUT):
    """Loads the current forecast of the folder `output` (one Parquet file per group: current_output) into `table`
    ("SCHEMA.TABLE") on the Dataiku connection, replacing its rows, one group's file at a time. Returns the rows
    loaded and the subfolder, like load_to_sql.
        Oracle      oracledb (check_table_writer): the table is made or given the columns it lacks first
                    (ensure_table, with the Parquet's columns), then its rows are deleted and the new ones inserted in
                    batches of BATCH_ROWS, all in one transaction: readers see the old rows until the one commit at
                    the end, and a failure leaves them.
        other       Dataiku's writer into `dataset`, a Dataiku dataset linked to the table (needed), which keeps the
                    table made by SQL (its schema is not rewritten)."""
    folder, version, paths = current_output(output)
    if sql_dialect(connection) != "oracle":
        return _load_with_dataset(dataset, table, connection, folder, version, paths)
    import oracledb
    binds = {"string": oracledb.DB_TYPE_VARCHAR, "bigint": oracledb.DB_TYPE_NUMBER,
             "double": oracledb.DB_TYPE_NUMBER, "date": oracledb.DB_TYPE_DATE}
    if not paths:
        print(ensure_table(connection, table, output_columns()))
    rows, progress = 0, Progress(f"loading {version} into {table} on {connection} (files, oracledb)", len(paths))
    db = _oracle_connect(connection)
    try:
        cursor = db.cursor()
        for n, path in enumerate(paths):
            frame = read_parquet(folder, path)
            frame = frame.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))     # NaN: empty
            kinds = {c: _kind(t) for c, t in frame.schema.items()}
            if n == 0:
                print(ensure_table(connection, table, kinds), flush=True)
                cursor.execute(f"DELETE FROM {table}")          # undone if anything below fails
            insert = (f"INSERT INTO {table} ({', '.join(kinds)}) "
                      f"VALUES ({', '.join(f':{i}' for i in range(1, len(kinds) + 1))})")
            cursor.setinputsizes(*[binds[k] for k in kinds.values()])
            data = frame.rows()
            for start in range(0, len(data), BATCH_ROWS):
                cursor.executemany(insert, data[start:start + BATCH_ROWS])
            rows += len(data)
            progress.add()
        if not paths:
            cursor.execute(f"DELETE FROM {table}")
        db.commit()
    finally:
        db.close()                  # without the commit: rolled back
    progress.finish()
    return rows, version


def _load_with_dataset(dataset, table, connection, folder, version, paths):
    """load_to_table on a database other than Oracle: Dataiku's writer into `dataset`, linked to the table (its
    schema is not written: the table keeps the types it was made with)."""
    import dataiku
    if not dataset:
        raise ValueError(f"The table {table} on {connection} is not on Oracle: it is written through a Dataiku dataset "
                         "linked to it, and none was given (dataset=...)")
    # the writer matches columns by their exact name, and the database keeps unquoted names in its own case
    # (PostgreSQL: total_w1_ev for TOTAL_w1_ev): each column is renamed to the dataset's own name, case aside
    names = [c["name"] for c in dataiku.Dataset(dataset).read_schema()]
    by_lower = {n.lower(): n for n in names}
    writer, rows = None, 0
    progress = Progress(f"loading {version} into {table} on {connection} (files, Dataiku's writer)", len(paths))
    try:
        writer = dataiku.Dataset(dataset).get_writer()
        for path in paths:
            frame = read_parquet(folder, path).to_pandas()
            unknown = [c for c in frame.columns if c.lower() not in by_lower]
            if unknown:
                raise ValueError(f"The table {table} (dataset {dataset}) has no column {', '.join(unknown)}: run the "
                                 "setup notebook again (it adds the columns a newer version of the code writes)")
            frame = frame.rename(columns={c: by_lower[c.lower()] for c in frame.columns})
            frame = frame.reindex(columns=names)                # the table's order (a column not written: empty)
            writer.write_dataframe(frame)
            rows += len(frame)
            progress.add()
    finally:
        if writer is not None:
            writer.close()
    progress.finish()
    return rows, version
