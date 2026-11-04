# Load check: the latest warehouse load must be complete before anything else runs.
# Reads CFF_LOAD_RECORD; for the latest load, every table must be loaded with status SUCCESS, and each table must
# hold the number of rows the load record says (a partial load stops here). Writes load_status: the load's
# as-of date and the forecast's cut-off (the last complete month on that date). On any problem the recipe fails,
# so nothing downstream runs.
import dataiku
import pandas as pd
from dataiku import SQLExecutor2

from cashflow import read_dataset
from warehouse import LOAD_RECORD, TABLES

record = read_dataset(LOAD_RECORD)
latest = record.sort_values("LOAD_DTTS").iloc[-1]["LOAD_ID"]
load = record[record["LOAD_ID"] == latest].set_index("TRGT_TBL_NM")
problems = []
for table, (warehouse_table, _) in TABLES.items():
    if warehouse_table not in load.index:
        problems.append(f"{warehouse_table}: not in load {latest}")
        continue
    row = load.loc[warehouse_table]
    if row["LOAD_STA_CD"] != "SUCCESS":
        problems.append(f"{warehouse_table}: load status {row['LOAD_STA_CD']}")
    params = dataiku.Dataset(warehouse_table).get_config()["params"]
    count = SQLExecutor2(dataset=dataiku.Dataset(warehouse_table)).query_to_df(
        f'SELECT COUNT(*) AS n FROM "{params["schema"]}"."{params["table"]}"')["n"].iloc[0]
    if count != row["ROW_CNT"]:
        problems.append(f"{warehouse_table}: {count:,} rows, the load record says {row['ROW_CNT']:,}")
if problems:
    raise ValueError("Warehouse load not usable:\n" + "\n".join(problems))

as_of = pd.Timestamp(load["AS_OF_DT"].max()).normalize()
month_end = as_of + pd.offsets.MonthEnd(0)
cutoff = as_of.to_period("M") if as_of == month_end else as_of.to_period("M") - 1
dataiku.Dataset("load_status").write_with_schema(pd.DataFrame([{
    "load_id": latest, "loaded_at": load["LOAD_DTTS"].max(), "as_of_date": as_of,
    "cutoff_month": cutoff.to_timestamp(), "tables_checked": len(TABLES), "checked_at": pd.Timestamp.now().floor("s")}]))
print(f"Load {latest} OK: as of {as_of:%Y-%m-%d}, cut-off month {cutoff}")
