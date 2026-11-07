# Staging: the 2 warehouse tables, read as they are (read-only, in chunks, only the columns used), turned into the
# project's tables, all in Dataiku. This is interface 1, what the data scientist's models get:
#   clients: the business clients of DIM_CUSTOMER (companies and sole proprietors), standard names and values
#   daily_flows: the postings of FCT_STATEMENT_ENTRY summed per client, day and category (money in, money out), over
#   each client's whole history up to the cut-off month: business clients only, no reversals, no transfers between
#   the client's own accounts. The postings are first spread over files on disk by TRNS_RFR, so that both sides of a
#   reversal or of an own-account transfer are in the same file, then summed one file at a time.
import os
import tempfile

import dataiku
import pandas as pd

from cashflow import cutoff_month
from warehouse import COLUMNS, TABLES, business_clients, daily_flows, to_standard

CHUNK, FILES = 200_000, 32
cutoff = cutoff_month()


def read(table):
    """The table's chunks with the standard names, only the columns the project uses."""
    dataset = dataiku.Dataset(TABLES[table][0])
    for chunk in dataset.iter_dataframes(chunksize=CHUNK, columns=list(COLUMNS[table].values())):
        yield to_standard(chunk, table)


clients = business_clients(pd.concat([business_clients(chunk) for chunk in read("clients")], ignore_index=True))
if clients.empty:
    raise ValueError("No business client in DIM_CUSTOMER: check the codes RETAIL and BUSINESS_ID in warehouse.py")
numbers = set(clients["client_no"])
dataiku.Dataset("clients").write_with_schema(clients)

end = cutoff + pd.DateOffset(months=1)
with tempfile.TemporaryDirectory() as spread:
    read_rows = kept = 0
    for postings in read("transactions"):
        read_rows += len(postings)
        postings = postings[postings["client_no"].isin(numbers) & (pd.to_datetime(postings["value_date"]) < end)]
        kept += len(postings)
        part = pd.util.hash_pandas_object(postings["txn_ref"].fillna(postings["entry_id"].astype(str)),
                                          index=False) % FILES
        for n, rows in postings.groupby(part):
            path = os.path.join(spread, f"{n}.csv")
            rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    totals = []
    for name in sorted(os.listdir(spread)):
        rows = pd.read_csv(os.path.join(spread, name), dtype={"client_no": str, "txn_ref": str, "entry_id": str,
                                                              "reversal": str, "txn_type": str})
        totals.append(daily_flows(rows, numbers, cutoff))
keys = ["client_no", "day", "category"]
flows = (pd.concat(totals, ignore_index=True) if totals
         else pd.DataFrame(columns=keys + ["money_in", "money_out", "postings"]))
flows = (flows.groupby(keys, as_index=False)[["money_in", "money_out", "postings"]].sum()
         .round({"money_in": 2, "money_out": 2}).sort_values(keys).reset_index(drop=True))

# the postings must reach the cut-off month: none, or far fewer than the month before, means a late load
per_month = flows.groupby(pd.to_datetime(flows["day"]).dt.to_period("M").dt.to_timestamp())["postings"].sum()
last, before = per_month.get(cutoff, 0), per_month.get(cutoff - pd.DateOffset(months=1), 0)
if last == 0 or last < 0.5 * before:
    raise ValueError(f"FCT_STATEMENT_ENTRY has {last:,} postings of business clients in {cutoff:%Y-%m} "
                     f"({before:,} the month before): the warehouse load for that month looks late or partial")
dataiku.Dataset("daily_flows").write_with_schema(flows)
print(f"{len(clients):,} business clients; {read_rows:,} postings read, {kept:,} kept; {len(flows):,} daily totals "
      f"in {flows['category'].nunique()} categories")
