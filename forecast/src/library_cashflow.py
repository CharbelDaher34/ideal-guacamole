"""Shared by every zone (project library: python/cashflow.py): dates, the business rule on amounts, and the
interfaces between the zones:
    1. the data (staging -> training, lab and inference): the datasets daily_flows (one row per client, day and
       category: money_in, money_out, postings, over each client's whole history) and clients (one row per business
       client). Every model gets these and nothing else; monthly_matrix and monthly_series below turn the days into
       months for the models that work by month.
    2. the forecaster (lab -> training): models.py.
    3. the saved model (training -> inference): SAVED_MODEL; inference uses its active version.
The forecast: per client, per category and in TOTAL (all categories together), the net cash flow (money in minus
money out) of each of the next 3 months, as P10, EV and P90 in SAR."""
import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3)        # months ahead
TOTAL = "TOTAL"             # the category of a client's all-categories net cash flow
SAVED_MODEL = "cash_forecast"
MIN_HISTORY_MONTHS = 12     # a client needs this many months before the cut-off to get a forecast (else status
                            # NOT_ENOUGH_HISTORY in the final forecast)
FORECAST_COLUMNS = ["client_no", "category", "months_ahead", "p10", "ev", "p90"]   # what a forecaster returns


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


def iter_dataset(name, rows=1_000_000, columns=None, **kwargs):
    """A dataset read in chunks of rows (DataFrames, like read_dataset), so a big one never sits in memory whole."""
    import dataiku
    for chunk in dataiku.Dataset(name, **kwargs).iter_dataframes(chunksize=rows, columns=columns):
        yield _plain(chunk)


def iter_clients(name="daily_flows", clients=5000, rows=1_000_000, keep=None):
    """daily_flows read in chunks of about `clients` COMPLETE clients: every row of a client is in the same chunk, so
    memory holds one chunk of clients, never the whole table. Staging keeps daily_flows grouped by client, so it is
    read straight through, rows read at a time. keep: a function applied to each read chunk first (a filter).
    Stops with an error if a client's rows are not together (then run the scenario Full rebuild)."""
    done, pending, tail = set(), None, None

    def batch(out):
        ids = set(out["client_no"])
        if ids & done:
            raise ValueError(f"{name} is not grouped by client (client {next(iter(ids & done))} is in two places): "
                             "run the scenario Full rebuild of the daily flows")
        done.update(ids)
        return out.reset_index(drop=True)

    for chunk in iter_dataset(name, rows=rows):
        chunk = keep(chunk) if keep else chunk
        if tail is not None:
            chunk = pd.concat([tail, chunk], ignore_index=True)
        if chunk.empty:
            tail = None
            continue
        last = chunk["client_no"].iloc[-1]          # its rows may go on in the next chunk: kept for then
        at_end = chunk["client_no"].eq(last)
        tail = chunk[at_end]
        pending = chunk[~at_end] if pending is None else pd.concat([pending, chunk[~at_end]], ignore_index=True)
        ids = pending["client_no"].unique()
        while len(ids) >= clients:                  # every full chunk of clients
            first = pending["client_no"].isin(set(ids[:clients]))
            yield batch(pending[first])
            pending, ids = pending[~first], ids[clients:]
    rest = [f for f in (pending, tail) if f is not None and len(f)]
    if rest:
        rest = pd.concat(rest, ignore_index=True)
        ids = rest["client_no"].unique()
        for i in range(0, len(ids), clients):
            yield batch(rest[rest["client_no"].isin(set(ids[i:i + clients]))])


def append_row(name, columns, row):
    """Add one row to a log dataset (an appended output of the recipe) with fixed column types, so that rows of
    failed runs (with empty values) never change how the earlier rows are read. columns: {name: Dataiku type}, one of
    string, bigint, double, boolean."""
    import dataiku
    kinds = {"string": "string", "bigint": "Int64", "double": "float64", "boolean": "boolean"}
    frame = pd.DataFrame([{c: row.get(c) for c in columns}])
    for c, kind in columns.items():
        frame[c] = (frame[c].fillna("").astype(str) if kind == "string"
                    else pd.to_numeric(frame[c]).astype(kinds[kind]) if kind != "boolean"
                    else frame[c].astype("boolean"))
    dataset = dataiku.Dataset(name)
    dataset.write_schema([{"name": c, "type": kind} for c, kind in columns.items()])
    with dataset.get_writer() as writer:
        writer.write_dataframe(frame)


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


def add_months(month, n):
    return month + pd.offsets.DateOffset(months=n)


def cutoff_month():
    """The forecast's cut-off: the last complete month (project variable
    cutoff_month, set by the first step of each scenario: warehouse.set_cutoff)."""
    import dataiku
    return pd.Timestamp(dataiku.get_custom_variables()["cutoff_month"] + "-01")


def until(daily, last_month):
    """The daily flows known at the end of last_month (the days up to its last day)."""
    return daily[pd.to_datetime(daily["day"]) < add_months(last_month, 1)]


def finish(ev, p10, p90, index=None):
    """P10 <= EV <= P90 (SAR). The net cash flow can be below zero."""
    ev, p10, p90 = (np.asarray(v, dtype=float) for v in (ev, p10, p90))
    return pd.DataFrame({"p10": np.minimum(p10, ev), "ev": ev, "p90": np.maximum(p90, ev)}, index=index).round(2)


def monthly_matrix(daily, last_month):
    """The net cash flow (money in minus money out) per month: one row per client and category (and per client in
    TOTAL), one column per month up to last_month. 0 for a month with no posting after the series' first month,
    empty before it. Only the days up to the end of last_month are used."""
    d = until(daily, last_month)
    d = d.assign(month=month_start(d["day"]).values, net=(d["money_in"] - d["money_out"]).values)
    by_category = d.groupby(["client_no", "category", "month"])["net"].sum()
    total = d.groupby(["client_no", "month"])["net"].sum().to_frame().assign(category=TOTAL)
    total = total.set_index("category", append=True).reorder_levels(["client_no", "category", "month"])["net"]
    sums = pd.concat([by_category, total])
    if sums.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["client_no", "category"]))
    wide = sums.unstack("month")
    months = pd.date_range(wide.columns.min(), last_month, freq="MS")
    wide = wide.reindex(columns=months)
    started = wide.notna().cumsum(axis=1) > 0
    return wide.fillna(0.0).where(started).sort_index()


def monthly_series(daily, last_month):
    """monthly_matrix as rows: client_no, category, month, net (no row before a series' first month)."""
    long = monthly_matrix(daily, last_month).reset_index().melt(id_vars=["client_no", "category"], var_name="month",
                                                                value_name="net")
    long["month"] = pd.to_datetime(long["month"])
    return long.dropna(subset=["net"]).sort_values(["client_no", "category", "month"]).reset_index(drop=True)


def history_months(daily, last_month):
    """Per client: the months from its first posting to last_month (both counted)."""
    first = month_start(until(daily, last_month).groupby("client_no")["day"].min())
    months = (last_month.year - first.dt.year) * 12 + (last_month.month - first.dt.month) + 1
    return months.rename("history_months")
