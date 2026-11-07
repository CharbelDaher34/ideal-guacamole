# Staging: the 2 warehouse tables, read-only, turned into the project's tables, all in Dataiku. This is interface 1,
# what the data scientist's models get:
#   clients: the business clients of DIM_CUSTOMER (companies and sole proprietors), standard names and values. Read
#   whole every run (it is small).
#   daily_flows: the postings of FCT_STATEMENT_ENTRY summed per client, day and category (money in, money out):
#   business clients only, no reversals, no transfers between the client's own accounts. Only the NEW days are read
#   from the warehouse: the days after the last day already staged, up to the end of the cut-off month. When nothing
#   is staged yet (the first run, or after the scenario Full rebuild), the whole history is read.
#   daily_flows is kept GROUPED BY CLIENT: all the rows of a client are next to each other (sorted by client, day,
#   category within each group of clients). So when there are new days, it is written again: the rows already staged
#   and the new ones, a group of clients at a time. Inference (and anyone else) can then read it in chunks of
#   complete clients. When there is nothing new, daily_flows is not touched.
#   staging_log: one row per run (appended), also when the run fails (status FAILED and the error): what was read,
#   and the last day now in daily_flows.
# A big read never sits in memory whole: the postings are spread over files on disk by TRNS_RFR (both sides of a
# reversal or of an own-account transfer in the same file) and summed one file at a time; the new totals and the
# rows already staged are then spread over files by client and written one file (a group of clients) at a time.
import json
import os
import tempfile
import time

import dataiku
import pandas as pd

from cashflow import Spread, append_row, cutoff_month, iter_dataset
from warehouse import (COLUMNS, TABLES, business_clients, daily_flows, to_standard, transactions_query,
                       value_day)

CHUNK, FILES = 200_000, 128
ROWS_PER_GROUP = 1_000_000          # about this many daily_flows rows per group of clients when rewriting
KEYS = ["client_no", "day", "category"]
EMPTY = pd.DataFrame({"client_no": pd.Series(dtype=str), "day": pd.Series(dtype="datetime64[ns]"),
                      "category": pd.Series(dtype=str), "money_in": pd.Series(dtype=float),
                      "money_out": pd.Series(dtype=float), "postings": pd.Series(dtype="int64")})
started = time.time()
cutoff = cutoff_month()
end = cutoff + pd.DateOffset(months=1)             # postings up to the end of the cut-off month


def read_table(table, since=None):
    """The table's chunks with the standard names, only the columns the project uses. Postings: only the value dates
    from since (None: all) to the end of the cut-off month; through the database connection the warehouse filters
    them (transactions_query), from files they are filtered here."""
    dataset = dataiku.Dataset(TABLES[table][0])
    if table == "transactions":
        try:
            name = dataset.get_location_info()["info"].get("quotedResolvedTableName")
        except Exception:
            name = None
        if name:
            reader = dataiku.SQLExecutor2(dataset=dataset).query_to_iter(transactions_query(name, since, end))
            names, rows = [c["name"] for c in reader.get_schema()], []
            for row in reader.iter_tuples():
                rows.append(row)
                if len(rows) == CHUNK:
                    yield to_standard(pd.DataFrame(rows, columns=names), table)
                    rows = []
            if rows:
                yield to_standard(pd.DataFrame(rows, columns=names), table)
            return
    for chunk in dataset.iter_dataframes(chunksize=CHUNK, columns=list(COLUMNS[table].values())):
        chunk = to_standard(chunk, table)
        if table == "transactions":
            day = value_day(chunk["value_date"])
            chunk = chunk[(day < end) & ((day >= since) if since is not None else True)]
        yield chunk


LOG = {"run_at": "string", "status": "string", "error": "string", "mode": "string", "cutoff_month": "string",
       "read_from": "string", "read_to": "string", "postings_read": "bigint", "postings_kept": "bigint",
       "daily_rows_added": "bigint", "daily_rows_total": "bigint", "partial_write": "boolean", "last_day": "string", "postings_per_month": "string",
       "clients": "bigint", "seconds": "double"}
row = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "cutoff_month": f"{cutoff:%Y-%m}",
       "read_to": f"{end - pd.Timedelta(days=1):%Y-%m-%d}", "partial_write": False}


def stage():
    """Reads the new days and adds them to daily_flows; fills row (the staging_log row) as it goes."""
    # what is already staged: staging_log (this recipe adds to it, so it reads it outside the flow's inputs). Failed
    # runs added nothing, unless one failed while writing daily_flows (partial_write): then only a Full rebuild helps.
    try:
        log = dataiku.Dataset("staging_log", ignore_flow=True).get_dataframe()
    except Exception:
        log = pd.DataFrame(columns=list(LOG))
    if len(log) and log["partial_write"].astype(str).str.lower().eq("true").any():
        raise ValueError("A staging run failed while adding to daily_flows (partial_write in staging_log): run the "
                         "scenario Full rebuild of the daily flows")
    log = log[log["status"] == "SUCCESS"] if len(log) else log
    staged = {}                                    # postings per month already in daily_flows
    for months in log["postings_per_month"]:
        for month, n in json.loads(months).items():
            staged[month] = staged.get(month, 0) + n
    last_day = pd.to_datetime(log["last_day"]).max() if len(log) else None
    staged_rows = int(log.sort_values("run_at")["daily_rows_total"].iloc[-1]) if len(log) else 0
    if last_day is None:
        try:
            first = next(iter_dataset("daily_flows", rows=1, ignore_flow=True), EMPTY)
        except Exception:
            first = EMPTY
        if len(first):
            raise ValueError("staging_log is empty but daily_flows is not: run the scenario Full rebuild")
    since = None if last_day is None else last_day + pd.Timedelta(days=1)
    mode = "FULL" if since is None else "NOTHING_NEW" if since >= end else "NEW_DAYS"
    row.update(mode=mode, read_from=f"{since:%Y-%m-%d}" if mode == "NEW_DAYS" else "")

    clients = business_clients(pd.concat([business_clients(chunk) for chunk in read_table("clients")],
                                         ignore_index=True))
    if clients.empty:
        raise ValueError("No business client in DIM_CUSTOMER: check the codes RETAIL and BUSINESS_ID in "
                         "warehouse.py")
    numbers = set(clients["client_no"])
    dataiku.Dataset("clients").write_with_schema(clients)
    row["clients"] = len(clients)

    read_rows = kept = added = total = 0
    new = {}                                       # postings per month read now
    with tempfile.TemporaryDirectory() as spread:
        if mode != "NOTHING_NEW":
            postings = Spread(os.path.join(spread, "postings"), FILES)
            for chunk in read_table("transactions", since):
                read_rows += len(chunk)
                chunk = chunk[chunk["client_no"].isin(numbers)]
                kept += len(chunk)
                if len(chunk):
                    last_day = max(value_day(chunk["value_date"]).max(), last_day or pd.Timestamp.min)
                    postings.add(chunk, chunk["txn_ref"].fillna(chunk["entry_id"].astype(str)))
            row.update(postings_read=read_rows, postings_kept=kept)
            groups = Spread(os.path.join(spread, "groups"),
                            max(16, -(-(staged_rows + kept) // ROWS_PER_GROUP)))   # groups of clients
            for part in postings.parts():
                flows = daily_flows(part, numbers, cutoff)
                for month, n in flows.groupby(flows["day"].dt.strftime("%Y-%m"))["postings"].sum().items():
                    new[month] = new.get(month, 0) + int(n)
                groups.add(flows, "client_no")

        # the postings must reach the cut-off month: none, or far fewer than the month before, means a late load.
        # Checked before anything is added, so a failed run adds nothing and the next run reads the same days again.
        months = {m: staged.get(m, 0) + new.get(m, 0) for m in set(staged) | set(new)}
        last, before = months.get(f"{cutoff:%Y-%m}", 0), months.get(f"{cutoff - pd.DateOffset(months=1):%Y-%m}", 0)
        if last == 0 or last < 0.5 * before:
            raise ValueError(f"FCT_STATEMENT_ENTRY has {last:,} postings of business clients in {cutoff:%Y-%m} "
                             f"({before:,} the month before): the warehouse load for that month looks late or "
                             "partial")

        if mode == "NOTHING_NEW":
            total = staged_rows                    # daily_flows is not touched
        else:
            # the rows already staged join the new ones, by client; then daily_flows is written again, a group of
            # clients at a time, each client's rows together
            old = 0
            for chunk in (iter_dataset("daily_flows", ignore_flow=True) if mode == "NEW_DAYS" else []):
                groups.add(chunk[list(EMPTY.columns)], "client_no")
                old += len(chunk)
            if old != staged_rows:
                raise ValueError(f"daily_flows has {old:,} rows but staging_log says {staged_rows:,}: run the "
                                 "scenario Full rebuild of the daily flows")
            out = dataiku.Dataset("daily_flows")
            out.write_schema_from_dataframe(EMPTY)
            row["partial_write"] = True            # until the writing is done
            with out.get_writer() as writer:
                for part in groups.parts():
                    part = (part.groupby(KEYS, as_index=False)[["money_in", "money_out", "postings"]].sum()
                            .round({"money_in": 2, "money_out": 2}).sort_values(KEYS))
                    writer.write_dataframe(part[list(EMPTY.columns)])
                    total += len(part)
            row["partial_write"] = False
            added = total - old
    row.update(postings_read=read_rows, postings_kept=kept, daily_rows_added=added, daily_rows_total=total,
               last_day=f"{last_day:%Y-%m-%d}", postings_per_month=json.dumps(dict(sorted(new.items()))))


def write_log(status, error=""):
    row.update(status=status, error=error[:1000], seconds=round(time.time() - started, 1))
    append_row("staging_log", LOG, row)


try:
    stage()
except Exception as e:
    write_log("FAILED", f"{type(e).__name__}: {e}")
    raise
write_log("SUCCESS")
print(f"{row['mode']}: {row['clients']:,} business clients; {row['postings_read']:,} postings read"
      + (f" from {row['read_from']}" if row["read_from"] else "") + f", {row['postings_kept']:,} kept; "
      f"{row['daily_rows_added']:,} daily totals added, {row['daily_rows_total']:,} in daily_flows; last day staged "
      f"{row['last_day']}")
