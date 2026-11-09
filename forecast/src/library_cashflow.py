"""Shared by every zone (project library: python/cashflow.py): dates, the business rule on amounts, and the
interfaces between the zones:
    1. the data (staging -> training, lab and inference): daily_flows (one row per client, day and category:
       money_in, money_out, postings, over each client's whole history; a folder of Parquet files, read with
       read_daily or iter_clients) and the dataset clients (one row per business client). Every model gets these
       and nothing else; period_matrix below turns the days into 30-day periods for the models that work that way.
    2. the forecaster (lab -> training): models.py.
    3. the saved model (training -> inference): SAVED_MODEL; inference uses its active version.
The forecast: per client, per category and in TOTAL (all categories together), the net cash flow (money in minus
money out) of each of the next 3 windows of 30 days after the cut-off day (the last day of data: yesterday): days
1-30, 31-60 and 61-90, as P10, EV and P90 in SAR."""
import logging
import warnings

import numpy as np
import pandas as pd

# Quieter job logs, for every recipe (they all import this file): Dataiku's own writer prints, for each block it
# writes, pandas warnings about its own code (DataFrame is highly fragmented, is_datetime64tz_dtype is deprecated),
# and an INFO line every 10 s while it waits for the next chunk (No data to send, waiting more...). Harmless: hidden.
# Our own warnings and Dataiku's real warnings and errors still show.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning, module=r"dataiku\.")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"dataiku\.")
logging.getLogger("dataiku.core.dataset_write").setLevel(logging.WARNING)

PERIOD_DAYS = 30            # the forecast's windows, and the periods the models work with, are 30 days long
HORIZONS = (1, 2, 3)        # the windows forecast: 1 = days 1-30 after the cut-off day, 2 = days 31-60, 3 = 61-90
TOTAL = "TOTAL"             # the category of a client's all-categories net cash flow
SAVED_MODEL = "cash_forecast"
MIN_HISTORY_DAYS = 360      # a client needs this many days of history up to the cut-off to get a forecast (else
                            # status NOT_ENOUGH_HISTORY in the final forecast)
MIN_TRAINING_MONTHS = 18    # history needed to train and test a model: MIN_HISTORY_DAYS, plus the 6 periods of 30
                            # days the test keeps aside (about 540 days). Less is fine to test staging only
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
CURRENT = "/current.txt"        # in the folder: which version of the files is the current one


def empty_daily():
    """daily_flows with no row, with its column types."""
    return pd.DataFrame({"client_no": pd.Series(dtype=str), "day": pd.Series(dtype="datetime64[ns]"),
                         "category": pd.Series(dtype=str), "money_in": pd.Series(dtype=float),
                         "money_out": pd.Series(dtype=float), "postings": pd.Series(dtype="int64")})


def _daily_files(name=DAILY_FLOWS, **kwargs):
    """The folder and the current version's Parquet files, in order ([] when nothing is staged yet).
    kwargs: for dataiku.Folder (ignore_flow=True to read it in the recipe that writes it)."""
    import dataiku
    folder = dataiku.Folder(name, **kwargs)
    paths = folder.list_paths_in_partition()
    if CURRENT not in paths:
        return folder, []
    with folder.get_download_stream(CURRENT) as stream:
        version = stream.read().decode().strip()
    return folder, sorted(p for p in paths if p.startswith(f"/{version}/") and p.endswith(".parquet"))


def iter_daily_parts(name=DAILY_FLOWS, columns=None, **kwargs):
    """daily_flows one Parquet file at a time (DataFrames). Each file holds complete clients."""
    import io
    import pyarrow.parquet as pq
    folder, paths = _daily_files(name, **kwargs)
    for path in paths:
        with folder.get_download_stream(path) as stream:
            data = stream.read()
        yield _plain(pq.read_table(io.BytesIO(data), columns=columns).to_pandas())


def read_daily(name=DAILY_FLOWS, columns=None, **kwargs):
    """All of daily_flows as one DataFrame (for the sample data or a training set that fits in memory; for all the
    bank's clients, use iter_clients)."""
    parts = list(iter_daily_parts(name, columns, **kwargs))
    return pd.concat(parts, ignore_index=True) if parts else empty_daily()


def write_daily(parts, name=DAILY_FLOWS):
    """Write daily_flows again: each part (a DataFrame of complete clients, each client's rows together) becomes one
    Parquet file of a new version; only once all are written does the new version become the current one, and the
    older versions are removed. So a run that fails half way leaves the current version as it was. Returns the rows
    written."""
    import io
    import uuid
    import dataiku
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder = dataiku.Folder(name)
    version = f"v{pd.Timestamp.now():%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    rows = 0
    for i, part in enumerate(parts):
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(part[DAILY_COLUMNS], preserve_index=False), buffer)
        with folder.get_writer(f"/{version}/part_{i:05d}.parquet") as writer:
            writer.write(buffer.getvalue())
        rows += len(part)
    with folder.get_writer(CURRENT) as writer:
        writer.write(version.encode())
    for path in folder.list_paths_in_partition():
        if path != CURRENT and not path.startswith(f"/{version}/"):
            folder.delete_path(path)
    return rows


def clear_daily(name=DAILY_FLOWS):
    """Remove every file of daily_flows (the scenario Full rebuild)."""
    import dataiku
    folder = dataiku.Folder(name)
    for path in folder.list_paths_in_partition():
        folder.delete_path(path)


def iter_clients(name=DAILY_FLOWS, clients=5000, keep=None, **kwargs):
    """daily_flows read in chunks of `clients` COMPLETE clients: every row of a client is in the same chunk, so memory
    holds one chunk of clients, never the whole table. Staging keeps daily_flows grouped by client, so it is read
    straight through, one Parquet file at a time. keep: a function applied to each file's rows first (a filter).
    Stops with an error if a client's rows are not together (then run the scenario Full rebuild)."""
    done, buffer, runs = set(), [], 0

    def batch(out):
        ids = set(out["client_no"])
        if ids & done:
            raise ValueError(f"{name} is not grouped by client (client {next(iter(ids & done))} is in two places): "
                             "run the scenario Full rebuild of the daily flows")
        done.update(ids)
        return out.reset_index(drop=True)

    def starts(frame):
        """The first row of each client (the rows of a client are together)."""
        ids = frame["client_no"].to_numpy()
        return np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]) if len(ids) else np.array([], dtype=int)

    def cut(frame, last):
        """Full chunks of clients from frame; returns the rest. last: no more rows to come, so the last client is
        complete too."""
        first = starts(frame)
        complete = len(first) if last else len(first) - 1      # the last client may go on in the next file
        while complete >= clients:
            end = first[clients] if clients < len(first) else len(frame)
            yield batch(frame.iloc[:end])
            frame, first, complete = frame.iloc[end:], first[clients:] - end, complete - clients
        if last and len(frame):
            yield batch(frame)
        return frame

    # files are joined only when they hold more than a chunk of clients (one join per chunk, not one per file)
    for part in iter_daily_parts(name, **kwargs):
        part = keep(part) if keep else part
        if part.empty:
            continue
        buffer.append(part)
        runs += len(starts(part))
        if runs > clients:
            rest = yield from cut(pd.concat(buffer, ignore_index=True), last=False)
            buffer, runs = [rest], len(starts(rest))
    if buffer:
        yield from cut(pd.concat(buffer, ignore_index=True), last=True)


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


ENGINE = {"name": "pandas"}    # how period_matrix sums the days: pandas (default) or polars (faster; needs the package
                               # polars in the code env). use_engine changes it


def use_engine(name):
    """Sum the days with pandas or polars in period_matrix (same result). Stops with a clear message if polars is
    asked for and not in the code env."""
    name = (name or "pandas").strip().lower()
    if name not in ("pandas", "polars"):
        raise ValueError(f"Unknown engine {name!r}: pandas or polars")
    if name == "polars":
        try:
            import polars  # noqa: F401
        except ImportError:
            raise ValueError("The engine polars needs the package polars in the code env: ask the admin to add it "
                             "(Administration > Code envs > the env > Packages to install: polars > Update)") from None
    ENGINE["name"] = name


def _period_sums_polars(daily, cutoff):
    """_sums(d, "period") of period_matrix, computed with polars, and how many periods back the oldest one is."""
    import polars as pl
    d = pl.from_pandas(daily[["client_no", "day", "category", "money_in", "money_out"]])
    d = d.with_columns(pl.col("day").cast(pl.Datetime("ns"))).filter(pl.col("day") <= cutoff.to_pydatetime())
    if d.height == 0:
        return pd.Series(dtype=float), None
    d = d.with_columns(back=(pl.lit(cutoff.to_pydatetime()) - pl.col("day")).dt.total_days() // PERIOD_DAYS,
                       net=pl.col("money_in") - pl.col("money_out"))
    by_category = d.group_by(["client_no", "category", "back"]).agg(pl.col("net").sum())
    total = (by_category.group_by(["client_no", "back"]).agg(pl.col("net").sum())
             .with_columns(category=pl.lit(TOTAL)).select(by_category.columns))
    sums = pl.concat([by_category, total]).to_pandas()
    sums["period"] = cutoff - pd.to_timedelta(sums["back"] * PERIOD_DAYS, unit="D")
    return sums.set_index(["client_no", "category", "period"])["net"], int(sums["back"].max())


def period_matrix(daily, cutoff):
    """The net cash flow (money in minus money out) per 30-day period ending at the cut-off day: one row per client
    and category (and per client in TOTAL), one column per period, named by its last day (the last column is the 30
    days up to the cut-off, the one before the 30 days before, ...). 0 for a period with no posting after the
    series' first period, empty before it. The window forecast h is the period ending cutoff + 30 x h days.
    The days are summed with ENGINE (pandas, or polars: use_engine)."""
    if ENGINE["name"] == "polars":
        sums, oldest = _period_sums_polars(daily, cutoff)
        if oldest is None:
            return _wide(pd.Series(dtype=float), [])
        return _wide(sums, [cutoff - pd.Timedelta(days=PERIOD_DAYS * k) for k in range(oldest, -1, -1)])
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


def history_days(daily, cutoff):
    """Per client: the days from its first posting to the cut-off day (both counted)."""
    first = pd.to_datetime(until(daily, cutoff).groupby("client_no")["day"].min())
    return ((cutoff - first).dt.days + 1).rename("history_days")
