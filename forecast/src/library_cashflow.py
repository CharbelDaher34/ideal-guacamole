"""Shared by every zone (project library: python/cashflow.py): dates, the business rule on amounts, and the
interfaces between the zones:
    1. the data (staging -> training, lab and inference): daily_flows (one row per client, day and category:
       money_in, money_out, postings, over each client's whole history; a folder of Parquet files, one per group of
       clients and month, see write_daily) and the dataset clients (one row per business client). Every model gets
       these and nothing else. Inference gives each model the last INPUT_DAYS days of each client (the last year),
       laid out by model_rows: one row per client, category and window, one column per day.
    2. the model (lab -> training): models.py.
    3. the saved model (training -> inference): SAVED_MODEL; inference uses its active version.
The forecast: per client, per category and in TOTAL (all categories together), the net cash flow (money in minus
money out) of each of the next 3 windows of 30 days after the cut-off day (the last day of data: yesterday): days
1-30, 31-60 and 61-90, as P10, EV and P90 in SAR."""
import logging
import warnings

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:         # the setup notebooks run on Dataiku's built-in Python: they only use this file's settings
    pl = None

# Quieter job logs, for every recipe (they all import this file): Dataiku's own writer prints, for each block it
# writes, pandas warnings about its own code (DataFrame is highly fragmented, is_datetime64tz_dtype is deprecated),
# and an INFO line every 10 s while it waits for the next chunk (No data to send, waiting more...). Harmless: hidden.
# Our own warnings and Dataiku's real warnings and errors still show.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning, module=r"dataiku\.")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"dataiku\.")
logging.getLogger("dataiku.core.dataset_write").setLevel(logging.WARNING)
# The same for each file written to a folder (Initializing folder writer, Pipe to generator thread, Sending data, Got
# end mark, ...: several lines per file, hundreds of files): Dataiku logs them on the root logger, so they are dropped
# by where they come from. Only their INFO lines: warnings and errors still show.
_CHATTY = ("dkuio.py", "write_base.py", "managed_folder.py")
logging.getLogger().addFilter(
    lambda record: record.levelno >= logging.WARNING
    or not record.pathname.replace("\\", "/").endswith(tuple(f"dataiku/core/{f}" for f in _CHATTY)))

PERIOD_DAYS = 30            # the forecast's windows, and the periods the models work with, are 30 days long
HORIZONS = (1, 2, 3)        # the windows forecast: 1 = days 1-30 after the cut-off day, 2 = days 31-60, 3 = 61-90
TOTAL = "TOTAL"             # the category of a client's all-categories net cash flow
SAVED_MODEL = "cash_forecast"
INPUT_DAYS = 365            # what a model gets for each client: its daily flows of the last 365 days up to the
                            # cut-off day (a forecaster can ask for another number: its params["input_days"])
MIN_HISTORY_DAYS = INPUT_DAYS   # a client needs this many days of history up to the cut-off to get a forecast (else
                            # status NOT_ENOUGH_HISTORY in the final forecast)
MIN_TRAINING_MONTHS = 21    # history needed to train and test a model: a year of input, the 3 windows after it, and
                            # the 6 periods of 30 days the test keeps aside (about 635 days). Less is fine to test
                            # staging only
FORECAST_COLUMNS = ["client_no", "category", "window", "p10", "ev", "p90"]   # what a forecaster returns


def _plain(df):
    """Dates as plain (time zone free) timestamps, client numbers as text."""
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert(None)
    if "client_no" in df:
        df["client_no"] = df["client_no"].astype(str)
    return df


def read_dataset(name, columns=None):
    """A dataset as a DataFrame, with dates as plain (time zone free) timestamps and client numbers as text."""
    import dataiku
    return _plain(dataiku.Dataset(name).get_dataframe(columns=columns))


DAILY_FLOWS = "daily_flows"     # interface 1: a managed folder of Parquet files, written by staging
DAILY_COLUMNS = ["client_no", "day", "category", "money_in", "money_out", "postings"]
DAILY_SCHEMA = dict(zip(DAILY_COLUMNS, [pl.Utf8, pl.Datetime("us"), pl.Utf8, pl.Float64, pl.Float64, pl.Int64])) if pl \
    else {}
MANIFEST = "/manifest.json"     # in the folder: the current files (which group of clients and month each holds)
CLIENTS_PER_GROUP = 5000        # clients per group when the daily flows are written from nothing


def client_groups(clients, groups):
    """The group of each client number (a Polars Series of numbers 0 .. groups - 1): a client never changes group
    (CRC32 of its number), so its rows are always in the files of the same group."""
    import zlib
    numbers = pl.Series(clients, dtype=pl.Utf8)
    unique = numbers.unique()
    group_of = dict(zip(unique.to_list(), (zlib.crc32(n.encode()) % groups for n in unique.to_list())))
    return numbers.replace_strict(group_of, return_dtype=pl.Int64)


def read_manifest(name=DAILY_FLOWS, **kwargs):
    """The folder and its manifest (None when nothing is written yet): {"groups": number of groups, "files":
    {group: {month: [path, rows]}}, "first_days": path, "rows": rows in all}. kwargs: for dataiku.Folder."""
    import json
    import dataiku
    folder = dataiku.Folder(name, **kwargs)
    if MANIFEST not in folder.list_paths_in_partition():
        return folder, None
    with folder.get_download_stream(MANIFEST) as stream:
        return folder, json.loads(stream.read())


def read_parquet(folder, path, columns=None):
    """One Parquet file of a folder, as a Polars DataFrame."""
    import io
    with folder.get_download_stream(path) as stream:
        return pl.read_parquet(io.BytesIO(stream.read()), columns=columns)


def read_group(folder, manifest, group, since=None):
    """The daily flows of a group of clients (Polars), from the day `since` on (only the files of its months are
    read), or all of them."""
    months = (manifest["files"].get(str(group)) or {})
    first = None if since is None else f"{since:%Y-%m}"
    parts = [read_parquet(folder, path) for month, (path, _) in sorted(months.items()) if first is None or month >= first]
    daily = pl.concat(parts) if parts else pl.DataFrame(schema=DAILY_SCHEMA)
    return daily if since is None else daily.filter(pl.col("day") >= since)


def read_first_days(folder, manifest):
    """Each client's first posting day: client_no, first_day (Polars)."""
    return read_parquet(folder, manifest["first_days"])


def read_daily(name=DAILY_FLOWS, columns=None, **kwargs):
    """All of daily_flows as one pandas DataFrame (for training, the lab, the sample data; the nightly forecast reads
    one group of clients at a time, read_group)."""
    folder, manifest = read_manifest(name, **kwargs)
    files = [path for months in (manifest or {}).get("files", {}).values() for path, _ in months.values()]
    daily = pl.concat([read_parquet(folder, p, columns) for p in files]) if files else pl.DataFrame(schema=DAILY_SCHEMA)
    return _plain(daily.to_pandas())


def write_daily(parts, groups, name=DAILY_FLOWS, base=None):
    """Write daily flows: each part (pandas or Polars, DAILY_COLUMNS) holds rows of some groups of clients; its rows of
    each group and calendar month go to one Parquet file, group_0007/2026-11_<run>.parquet, added to the rows already
    in that file (base: the current manifest, when adding new days; None: from nothing). Files are never changed in
    place: new ones are written, then the manifest points to them (a failed run leaves the daily flows as they
    were), then the files it no longer lists are removed. Also each client's first posting day (first_days). Returns
    the new manifest."""
    import io
    import json
    import uuid
    import dataiku
    folder = dataiku.Folder(name)
    run = f"{pd.Timestamp.now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    files = {g: dict(months) for g, months in (base or {}).get("files", {}).items()}
    firsts = [read_first_days(folder, base)] if base else []

    def write(path, frame):
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        with folder.get_writer(path) as writer:
            writer.write(buffer.getvalue())

    for part in parts:
        part = (part if isinstance(part, pl.DataFrame) else pl.from_pandas(part[DAILY_COLUMNS])).select(
            [pl.col(c).cast(k) for c, k in DAILY_SCHEMA.items()])
        if part.is_empty():
            continue
        firsts.append(part.group_by("client_no").agg(first_day=pl.col("day").min()))
        part = part.with_columns(group=client_groups(part["client_no"], groups), month=pl.col("day").dt.strftime("%Y-%m"))
        for (group, month), rows in part.group_by(["group", "month"]):
            group, rows = str(group), rows.drop(["group", "month"])
            before = files.get(group, {}).get(month)
            if before:
                rows = pl.concat([read_parquet(folder, before[0]), rows])
            path = f"/group_{int(group):04d}/{month}_{run}.parquet"
            write(path, rows.sort(["client_no", "day", "category"]))
            files.setdefault(group, {})[month] = [path, len(rows)]
    first_days = f"/first_days_{run}.parquet"
    write(first_days, pl.concat(firsts).group_by("client_no").agg(pl.col("first_day").min()).sort("client_no")
          if firsts else pl.DataFrame(schema={"client_no": pl.Utf8, "first_day": pl.Datetime("us")}))
    manifest = {"groups": groups, "files": files, "first_days": first_days,
                "rows": sum(rows for months in files.values() for _, rows in months.values())}
    with folder.get_writer(MANIFEST) as writer:
        writer.write(json.dumps(manifest).encode())
    kept = {first_days, MANIFEST} | {path for months in files.values() for path, _ in months.values()}
    for path in folder.list_paths_in_partition():
        if path not in kept:
            folder.delete_path(path)
    return manifest


def clear_daily(name=DAILY_FLOWS):
    """Remove every file of daily_flows (the scenario Full rebuild)."""
    import dataiku
    folder = dataiku.Folder(name)
    for path in folder.list_paths_in_partition():
        folder.delete_path(path)


def peak_memory_mb():
    """The most memory this process used, MB (Linux: VmHWM)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024)
    except OSError:
        pass
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def append_row(name, columns, row):
    """Add one row to a log dataset (an appended output of the recipe) with fixed column types, so that rows of
    failed runs (with empty values) never change how the earlier rows are read. columns: {name: Dataiku type}, one of
    string, bigint, double, boolean. When the columns changed since the last row (a new version of the code), the
    earlier rows are first read with the old columns and written again with the new ones, by name (a new column is
    empty in them), so they never end up under the wrong column."""
    import dataiku
    kinds = {"string": "string", "bigint": "Int64", "double": "float64", "boolean": "boolean"}
    frame = pd.DataFrame([{c: row.get(c) for c in columns}])
    dataset = dataiku.Dataset(name)
    before = [c["name"] for c in dataset.read_schema(raise_if_empty=False)]
    if before and before != list(columns):
        earlier = dataiku.Dataset(name, ignore_flow=True).get_dataframe()      # read with the old columns
        print(f"{name}: its columns changed; its {len(earlier)} earlier rows are written again with the new columns")
        frame = pd.concat([earlier.reindex(columns=list(columns)), frame], ignore_index=True)
        dataset.spec_item["appendMode"] = False                                 # this time: all the rows again
    for c, kind in columns.items():
        frame[c] = (frame[c].where(frame[c].notna(), "").astype(str) if kind == "string"
                    else pd.to_numeric(frame[c], errors="coerce").round().astype(kinds[kind]) if kind == "bigint"
                    else pd.to_numeric(frame[c], errors="coerce").astype(kinds[kind]) if kind == "double"
                    else frame[c].astype("boolean"))
    dataset.write_schema([{"name": c, "type": kind} for c, kind in columns.items()])
    with dataset.get_writer() as writer:
        writer.write_dataframe(frame)


class Progress:
    """Progress lines in the job's log, for long steps: how much is done (of how much, when known), the percentage,
    the time so far and about how long is left. One line at most every `every` seconds, and the last one always. They
    start with PROGRESS, so the setup notebook can show them while it waits for the job.
        progress = Progress("reading postings", total=41_000_000); progress.add(200_000) ... progress.finish()"""

    def __init__(self, what, total=None, every=30):
        import time
        self.what, self.total, self.every = what, total, every
        self.started = self.printed = time.time()
        self.done = 0
        print(f"PROGRESS {what}: starting" + (f", {total:,} to do" if total else ""), flush=True)

    def add(self, n=1):
        import time
        self.done += n
        if time.time() - self.printed >= self.every:
            self.show()

    def finish(self):
        self.show(last=True)

    def show(self, last=False):
        import time
        now = time.time()
        self.printed = now
        minutes = (now - self.started) / 60
        if last:
            print(f"PROGRESS {self.what}: done, {self.done:,} in {minutes:.1f} min", flush=True)
            return
        line = f"PROGRESS {self.what}: {self.done:,}"
        if self.total:
            share = min(self.done / self.total, 1.0)
            line += f" of {self.total:,} ({share:.0%})"
        line += f", {minutes:.1f} min so far"
        if self.total and 0 < share < 1:
            line += f", about {minutes * (1 - share) / share:.0f} min left"
        print(line, flush=True)


def waiting(what, every=60):
    """While the warehouse runs a query and no row has come back yet, a PROGRESS line every `every` seconds, so a
    long query never looks stuck. Returns an Event: set it when the first row arrives (or the query ends)."""
    import threading
    import time
    stop, started = threading.Event(), time.time()

    def beat():
        while not stop.wait(every):
            print(f"PROGRESS {what}: no row yet, the warehouse is still running the query "
                  f"({(time.time() - started) / 60:.0f} min)", flush=True)

    threading.Thread(target=beat, daemon=True).start()
    return stop


class Spread:
    """Rows spread over n files on local disk by the hash of a column, then read back one file at a time: all the
    rows with the same value of that column end up in the same file, and memory holds one file, not the whole table.
    Used by staging (postings by reference, then daily flows by client)."""

    def __init__(self, folder, n):
        import os
        os.makedirs(folder, exist_ok=True)
        self.n = max(1, int(n))
        self.paths = [os.path.join(folder, f"part_{i}.pkl") for i in range(self.n)]

    def add(self, df, key):
        """key: a column name, or a Series (same index as df) to spread by."""
        import pickle
        key = df[key] if isinstance(key, str) else key
        part = pd.util.hash_pandas_object(key.astype(str), index=False).to_numpy() % self.n
        for i, rows in df.groupby(part):
            with open(self.paths[i], "ab") as f:
                pickle.dump(rows, f, protocol=4)

    def parts(self):
        """Each file's rows (a DataFrame); files that got no rows are skipped."""
        import os
        import pickle
        for path in self.paths:
            if not os.path.exists(path):
                continue
            frames = []
            with open(path, "rb") as f:
                while True:
                    try:
                        frames.append(pickle.load(f))
                    except EOFError:
                        break
            yield pd.concat(frames, ignore_index=True)


def month_start(dates):
    """First day of the month, as a plain (time zone free) timestamp."""
    d = pd.to_datetime(pd.Series(dates))
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert(None)
    return d.dt.to_period("M").dt.to_timestamp()


def cutoff_date():
    """The forecast's cut-off: the last day of data, by default yesterday (project variable cutoff_date, set by the
    first step of each scenario: warehouse.set_cutoff)."""
    import dataiku
    return pd.Timestamp(dataiku.get_custom_variables()["cutoff_date"])


def whole_number(name, value):
    """A setting that is a whole number, or None for "all" (empty, None, -1 or "all"). Stops with a clear message
    otherwise."""
    value = "" if value is None else str(value).strip()
    if value.lower() in ("", "none", "all", "-1"):
        return None
    try:
        number = float(value)
    except ValueError:
        number = 0.5
    if number != int(number) or number < 1:
        raise ValueError(f"{name} is {value!r}: it must be a whole number (1 or more), or -1 / None for all")
    return int(number)


def history_start(cutoff):
    """The first day staging reads when it reads everything (the first run, or the scenario Full rebuild): the
    project variable history_months months before the cut-off day, or None for the whole history. With less than
    about 18 months, few clients reach the history a forecast and the model test need (MIN_HISTORY_DAYS, and the
    last 6 periods of 30 days kept aside): fine to test the pipeline, not to judge a model."""
    import dataiku
    months = whole_number("history_months", dataiku.get_custom_variables().get("history_months"))
    return None if months is None else cutoff - pd.DateOffset(months=months) + pd.Timedelta(days=1)


def clients_limit():
    """How many business clients staging reads (project variable clients_limit): None for all of them."""
    import dataiku
    return whole_number("clients_limit", dataiku.get_custom_variables().get("clients_limit"))


def until(daily, cutoff):
    """The daily flows known at the cut-off: the days up to the cut-off day (included)."""
    return daily[pd.to_datetime(daily["day"]) <= cutoff]


def window(cutoff, h):
    """The first and last day of window h after the cut-off day (h = 1: days 1-30)."""
    return cutoff + pd.Timedelta(days=PERIOD_DAYS * (h - 1) + 1), cutoff + pd.Timedelta(days=PERIOD_DAYS * h)


def finish(ev, p10, p90, index=None):
    """P10 <= EV <= P90 (SAR). The net cash flow can be below zero."""
    ev, p10, p90 = (np.asarray(v, dtype=float) for v in (ev, p10, p90))
    return pd.DataFrame({"p10": np.minimum(p10, ev), "ev": ev, "p90": np.maximum(p90, ev)}, index=index).round(2)


def _sums(d, key):
    """Net cash flow per client, category (and TOTAL) and key."""
    by_category = d.groupby(["client_no", "category", key])["net"].sum()
    total = d.groupby(["client_no", key])["net"].sum().to_frame().assign(category=TOTAL)
    total = total.set_index("category", append=True).reorder_levels(["client_no", "category", key])["net"]
    return pd.concat([by_category, total])


def _wide(sums, columns):
    """One row per client and category, the given columns; 0 after a series' first value, empty before it."""
    if sums.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["client_no", "category"]))
    wide = sums.unstack(sums.index.names[-1]).reindex(columns=columns)
    started = wide.notna().cumsum(axis=1) > 0
    return wide.fillna(0.0).where(started).sort_index()


def period_matrix(daily, cutoff):
    """The net cash flow (money in minus money out) per 30-day period ending at the cut-off day: one row per client
    and category (and per client in TOTAL), one column per period, named by its last day (the last column is the 30
    days up to the cut-off, the one before the 30 days before, ...). 0 for a period with no posting after the
    series' first period, empty before it. The window forecast h is the period ending cutoff + 30 x h days."""
    d = until(daily, cutoff)
    back = (cutoff - pd.to_datetime(d["day"])).dt.days // PERIOD_DAYS
    d = d.assign(period=(cutoff - pd.to_timedelta(back * PERIOD_DAYS, unit="D")).values,
                 net=(d["money_in"] - d["money_out"]).values)
    if d.empty:
        return _wide(pd.Series(dtype=float), [])
    oldest = int(back.max())
    return _wide(_sums(d, "period"), [cutoff - pd.Timedelta(days=PERIOD_DAYS * k) for k in range(oldest, -1, -1)])


def period_series(daily, cutoff):
    """period_matrix as rows: client_no, category, period (its last day), net (no row before a series' first)."""
    long = period_matrix(daily, cutoff).reset_index().melt(id_vars=["client_no", "category"], var_name="period",
                                                           value_name="net")
    long["period"] = pd.to_datetime(long["period"])
    return long.dropna(subset=["net"]).sort_values(["client_no", "category", "period"]).reset_index(drop=True)


def monthly_series(daily, cutoff):
    """For the EDA: the net cash flow per client, category (and TOTAL) and calendar month, up to the cut-off day (its
    month is partial): client_no, category, month, net (no row before a series' first month)."""
    d = until(daily, cutoff)
    d = d.assign(month=month_start(d["day"]).values, net=(d["money_in"] - d["money_out"]).values)
    if d.empty:
        return pd.DataFrame(columns=["client_no", "category", "month", "net"])
    months = list(pd.date_range(d["month"].min(), month_start([cutoff])[0], freq="MS"))
    long = _wide(_sums(d, "month"), months).reset_index().melt(id_vars=["client_no", "category"], var_name="month",
                                                                value_name="net")
    long["month"] = pd.to_datetime(long["month"])
    return long.dropna(subset=["net"]).sort_values(["client_no", "category", "month"]).reset_index(drop=True)


def input_days(model=None):
    """How many days of daily flows a model gets for each client: its params["input_days"], else INPUT_DAYS."""
    return int((getattr(model, "params", None) or {}).get("input_days") or INPUT_DAYS)


def day_columns(days=INPUT_DAYS):
    """The names of the model's day columns, the oldest first: day_365, ..., day_1 (day_1: the cut-off day)."""
    return [f"day_{days - i}" for i in range(days)]


def day_matrix(daily, cutoff, days=INPUT_DAYS):
    """The net cash flow (money in minus money out) of each of the last `days` days up to the cut-off day, per client
    and category (and per client in TOTAL: all its categories together); 0 for a day with no posting. Returns the
    keys (Polars: client_no, category, sorted) and the matrix (numpy float32, one row per key, the oldest day first).
    daily: Polars or pandas."""
    d = daily if isinstance(daily, pl.DataFrame) else pl.from_pandas(daily[DAILY_COLUMNS])
    end = pd.Timestamp(cutoff).to_pydatetime()
    d = d.filter(pl.col("day").is_between(end - pd.Timedelta(days=days - 1), end)).select(
        pl.col("client_no").cast(pl.Utf8), pl.col("category").cast(pl.Utf8),
        (days - 1 - (pl.lit(end) - pl.col("day")).dt.total_days()).alias("column"),
        (pl.col("money_in") - pl.col("money_out")).alias("net"))
    d = pl.concat([d, d.with_columns(category=pl.lit(TOTAL))])
    keys = d.select("client_no", "category").unique().sort("client_no", "category")
    d = d.join(keys.with_row_index("series"), on=["client_no", "category"])
    flat = d["series"].to_numpy().astype(np.int64) * days + d["column"].to_numpy()
    matrix = np.bincount(flat, weights=d["net"].to_numpy(), minlength=len(keys) * days)
    return keys, matrix.reshape(len(keys), days).astype(np.float32)


def model_rows(keys, matrix):
    """The model's input (interface 2): one row per client, category and window (1, 2, 3: days 1-30, 31-60, 61-90
    after the cut-off day), its columns client_no, category, window, day_365 ... day_1 (the net cash flow of each
    day, SAR). keys, matrix: from day_matrix. Polars."""
    days = pl.DataFrame(matrix, schema=day_columns(matrix.shape[1]), orient="row")
    return pl.concat([keys.with_columns(window=pl.lit(h, pl.Int8)).hstack(days) for h in HORIZONS])
