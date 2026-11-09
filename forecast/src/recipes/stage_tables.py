# Staging: the 2 warehouse tables, read-only, turned into the project's tables, all in Dataiku. This is interface 1,
# what the data scientist's models get:
#   clients: the business clients of DIM_CUSTOMER (companies and sole proprietors), standard names and values. Read
#   whole every run (it is small).
#   daily_flows: the postings of FCT_STATEMENT_ENTRY summed per client, day and category (money in, money out):
#   business clients only, no reversals, no transfers between the client's own accounts. Only the NEW days are read
#   from the warehouse: the days after the last day already staged, up to the cut-off day (yesterday). When nothing
#   is staged yet (the first run, or after the scenario Full rebuild), the whole history is read.
#   daily_flows is a folder of Parquet files (read straight from the files, far faster than a dataset), kept GROUPED
#   BY CLIENT: each file holds complete clients (sorted by client, day, category). So when there are new days, it is
#   written again as a new version: the rows already staged and the new ones, a group of clients per file; the new
#   version only becomes the current one once every file is written (cashflow.write_daily), so a failed run leaves
#   daily_flows as it was. Inference (and anyone else) reads it in chunks of complete clients. When there is nothing
#   new, daily_flows is not touched.
#   staging_log: one row per run (appended), also when the run fails (status FAILED and the error): what was read,
#   and the last day now in daily_flows.
# The warehouse does the heavy work (warehouse.daily_totals_query): it sums the postings into daily totals per client,
# day and category, without reversals and own-account transfers, so only those totals travel, not every posting. A
# big read never sits in memory whole: the totals read and the rows already staged are spread over files on disk by
# client and written one file (a group of clients) at a time.
import json
import os
import tempfile
import time

import dataiku
import pandas as pd

from cashflow import (DAILY_COLUMNS, Progress, Spread, append_row, clients_limit, cutoff_date, history_start,
                      iter_daily_parts, waiting, write_daily)
import warehouse
from warehouse import TABLES, daily_totals_query, one_row_per_client, standard_daily, table_query, to_standard

CHUNK = 200_000
STEP = 10_000                       # rows: how often the reading progress is counted (CHUNK is a multiple of it)
ROWS_PER_GROUP = 1_000_000          # about this many daily_flows rows per group of clients when rewriting
GROUPS_FULL = 512                   # groups of clients when everything is read (its size is not known before)
KEYS = ["client_no", "day", "category"]
started = time.time()
cutoff = cutoff_date()
end = cutoff + pd.Timedelta(days=1)                # postings up to the cut-off day (included)


def read(query, progress, convert):
    """Runs a read-only query through the warehouse connection and yields its rows in chunks (DataFrames, converted
    by convert). progress counts the rows as they arrive. The query is printed first, and until its first row
    arrives a line says every minute that the warehouse is still running it."""
    print(f"PROGRESS SQL: {query}", flush=True)
    started, first = time.time(), waiting(progress.what)
    try:
        reader = dataiku.SQLExecutor2(dataset=dataiku.Dataset(TABLES["clients"][0])).query_to_iter(query)
        names, rows = [c["name"] for c in reader.get_schema()], []
        for row in reader.iter_tuples():
            if not first.is_set():
                first.set()
                print(f"PROGRESS {progress.what}: first rows after {(time.time() - started) / 60:.1f} min",
                      flush=True)
            rows.append(row)
            if len(rows) % STEP == 0:
                progress.add(STEP)
            if len(rows) == CHUNK:
                yield convert(pd.DataFrame(rows, columns=names))
                rows = []
        progress.add(len(rows) % STEP)
        if rows:
            yield convert(pd.DataFrame(rows, columns=names))
    finally:
        first.set()


def table_name(table):
    return sql_name(dataiku.Dataset(TABLES[table][0]))


def sql_name(dataset):
    """The table's full quoted name. The project reads the warehouse through SQL only: anything else stops here."""
    try:
        name = dataset.get_location_info()["info"].get("quotedResolvedTableName")
    except Exception:
        name = None
    if not name:
        raise ValueError(f"{dataset.name} is not a database table: the project reads the warehouse through SQL only. "
                         "Run the setup notebook again (it links the warehouse tables)")
    return name


LOG = {"run_at": "string", "status": "string", "error": "string", "mode": "string", "cutoff_date": "string",
       "read_from": "string", "read_to": "string", "postings_read": "bigint", "postings_kept": "bigint",
       "daily_rows_added": "bigint", "daily_rows_total": "bigint", "last_day": "string", "postings_per_month": "string",
       "clients": "bigint", "seconds": "double"}
row = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "cutoff_date": f"{cutoff:%Y-%m-%d}",
       "read_to": f"{cutoff:%Y-%m-%d}"}


def stage():
    """Reads the new days and adds them to daily_flows; fills row (the staging_log row) as it goes."""
    # what is already staged: staging_log (this recipe adds to it, so it reads it outside the flow's inputs). Failed
    # runs changed nothing.
    try:
        log = dataiku.Dataset("staging_log", ignore_flow=True).get_dataframe()
    except Exception:
        log = pd.DataFrame(columns=list(LOG))
    log = log[log["status"] == "SUCCESS"] if len(log) else log
    last_day = pd.to_datetime(log["last_day"]).max() if len(log) else None
    staged_rows = int(log.sort_values("run_at")["daily_rows_total"].iloc[-1]) if len(log) else 0
    if last_day is None:
        if next(iter_daily_parts(ignore_flow=True), None) is not None:
            raise ValueError("staging_log is empty but daily_flows is not: run the scenario Full rebuild")
    since = None if last_day is None else last_day + pd.Timedelta(days=1)
    mode = "FULL" if since is None else "NOTHING_NEW" if since >= end else "NEW_DAYS"
    if mode == "FULL":
        since = history_start(cutoff)              # project variable history_months (None: the whole history)
    row.update(mode=mode, read_from=f"{since:%Y-%m-%d}" if since is not None and mode != "NOTHING_NEW" else "")

    limit = clients_limit()                        # project variable clients_limit (None: all the business clients)
    print(f"PROGRESS staging ({mode})" + (f", the first {limit:,} business clients only" if limit else ""), flush=True)
    progress = Progress("reading the clients from the warehouse (rows)")
    clients = one_row_per_client(pd.concat(
        [one_row_per_client(chunk) for chunk in read(table_query("clients", table_name("clients"), limit=limit), progress,
                                                   lambda df: to_standard(df, "clients"))], ignore_index=True))
    progress.finish()
    print(f"PROGRESS {len(clients):,} business clients", flush=True)
    if progress.done > len(clients):               # DIM_CUSTOMER keeps history: several rows per client
        print(f"WARNING: DIM_CUSTOMER sent {progress.done:,} rows for {len(clients):,} business clients (several "
              "rows per client). " + ("Its condition CLIENTS_CURRENT_ROW does not keep one row per client: check it"
                                      if warehouse.CLIENTS_CURRENT_ROW.strip() else
                                      "Fill in CLIENTS_CURRENT_ROW in warehouse.py (the notebook's WAREHOUSE_CODE "
                                      "cell) to read only each client's current row")
              + ". Until then, the last row received is kept, which may not be the newest.", flush=True)
    if clients.empty:
        raise ValueError("No business client in DIM_CUSTOMER: check CLIENTS_BUSINESS and CLIENTS_CURRENT_ROW in "
                         "warehouse.py (the notebook's WAREHOUSE_CODE cell)")
    numbers = set(clients["client_no"])
    dataiku.Dataset("clients").write_with_schema(clients)
    row["clients"] = len(clients)

    read_rows = kept = added = total = 0          # postings in the daily totals read, and kept
    new = {}                                       # postings per month read now
    with tempfile.TemporaryDirectory() as spread:
        if mode != "NOTHING_NEW":
            groups = Spread(os.path.join(spread, "groups"),     # groups of clients
                            GROUPS_FULL if mode == "FULL" else max(16, -(-staged_rows // ROWS_PER_GROUP) + 1))
            query = daily_totals_query(table_name("transactions"), table_name("clients"), since, end, limit)
            progress = Progress("reading the daily totals from the warehouse (rows: client x day x category)")
            for chunk in read(query, progress, standard_daily):
                read_rows += int(chunk["postings"].sum())
                chunk = chunk[chunk["client_no"].isin(numbers)]
                kept += int(chunk["postings"].sum())
                if len(chunk):
                    last_day = max(chunk["day"].max(), last_day or pd.Timestamp.min)
                    for month, n in chunk.groupby(chunk["day"].dt.strftime("%Y-%m"))["postings"].sum().items():
                        new[month] = new.get(month, 0) + int(n)
                    groups.add(chunk[DAILY_COLUMNS], "client_no")
            progress.finish()
            row.update(postings_read=read_rows, postings_kept=kept)

        # the postings must reach the cut-off day (yesterday): otherwise the warehouse load looks late. Checked before
        # anything is written, so a failed run changes nothing and the next run reads the same days again.
        if last_day is None or last_day < cutoff:
            raise ValueError(f"FCT_STATEMENT_ENTRY has no posting of business clients after "
                             f"{'(none)' if last_day is None else f'{last_day:%Y-%m-%d}'}: the warehouse load of "
                             f"{cutoff:%Y-%m-%d} (the cut-off day) looks late")

        if mode == "NOTHING_NEW":
            total = staged_rows                    # daily_flows is not touched
        else:
            # the rows already staged join the new ones, by client; then daily_flows is written again as a new
            # version, a group of clients per file, each client's rows together
            old = 0
            if mode == "NEW_DAYS":
                progress = Progress("adding the rows already in daily_flows", staged_rows)
                for part in iter_daily_parts(ignore_flow=True):
                    groups.add(part[DAILY_COLUMNS], "client_no")
                    old += len(part)
                    progress.add(len(part))
                progress.finish()
            if old != staged_rows:
                raise ValueError(f"daily_flows has {old:,} rows but staging_log says {staged_rows:,}: run the "
                                 "scenario Full rebuild of the daily flows")
            progress = Progress("writing daily_flows (groups of clients)", groups.n)
            total = write_daily(counted(part.groupby(KEYS, as_index=False)[["money_in", "money_out", "postings"]]
                                        .sum().round({"money_in": 2, "money_out": 2}).sort_values(KEYS), progress)
                                for part in groups.parts())
            progress.finish()
            added = total - old
    row.update(postings_read=read_rows, postings_kept=kept, daily_rows_added=added, daily_rows_total=total,
               last_day=f"{last_day:%Y-%m-%d}", postings_per_month=json.dumps(dict(sorted(new.items()))))


def counted(part, progress):
    progress.add()
    return part


def write_log(status, error=""):
    row.update(status=status, error=error[:1000], seconds=round(time.time() - started, 1))
    append_row("staging_log", LOG, row)


try:
    stage()
except Exception as e:
    write_log("FAILED", f"{type(e).__name__}: {e}")
    raise
write_log("SUCCESS")
print(f"Staging done ({row['mode']}), last day staged {row['last_day']}:\n"
      f"- clients: {row['clients']:,} business clients\n"
      f"- postings read" + (f" since {row['read_from']}" if row["read_from"] else "") + f": {row['postings_read']:,} "
      "(a posting = one debit or credit line of FCT_STATEMENT_ENTRY, reversals and own-account transfers left out)"
      + (f", {row['postings_kept']:,} of them for the clients above" if row["postings_kept"] != row["postings_read"]
         else "") + "\n"
      f"- daily_flows: {row['daily_rows_added']:,} rows added, {row['daily_rows_total']:,} in all (a row = one "
      "client, one day, one category: money in, money out and how many postings, the postings summed)")
