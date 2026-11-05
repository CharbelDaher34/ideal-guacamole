"""Warehouse tables (project library: python/warehouse.py).

The forecast reads 3 warehouse tables and a load record (CFF_LOAD_RECORD), read-only: through the database
connection to the warehouse (Oracle), or as CSV files in the landing folder (EDW_LANDING); the setup notebook's
settings choose. The sandbox's simulated tables are CSV:
    CFF_CLIENT              EDW.DIM_CUSTOMER: one row per business client, companies and sole proprietors
    FCT_STATEMENT_ENTRY     EDW.FCT_STATEMENT_ENTRY itself: the daily postings. The staging recipe sums them per
                            client, month, direction, T24 module (SYS_ID) and transaction type (TRNS_TYP): in the
                            warehouse with one Oracle query (monthly_flows_sql) on a database connection, in Python
                            (monthly_flows) on CSV files. Only the monthly totals come into the project.
    CFF_SCHEDULED_PYMT      the payments already known for the coming months, the sources stacked in one table:
                            financing instalments, deposit maturities, standing orders
The SQL the warehouse team runs for CFF_CLIENT and CFF_SCHEDULED_PYMT is in warehouse_tables.sql (forecast folder of
the repository). Columns keep their EDW
names; the staging recipe renames them to the standard names the recipes use and turns codes into the project's
values (direction IN / OUT, status ACTIVE / CLOSED, sole proprietor true / false). If a name or a code differs at
the bank, only this file changes.
"""
import numpy as np
import pandas as pd

# the tables the project reads. standard name: (warehouse table, what the warehouse team builds it from)
TABLES = {
    "clients": ("CFF_CLIENT", "EDW.DIM_CUSTOMER"),
    "transactions": ("FCT_STATEMENT_ENTRY", "EDW.FCT_STATEMENT_ENTRY"),
    "scheduled_payments": ("CFF_SCHEDULED_PYMT", "financing schedules, deposits, standing orders"),
}
STATEMENT_SCHEMA = "EDW"       # schema of FCT_STATEMENT_ENTRY on the database connection (the other tables:
                               # WAREHOUSE_SQL_SCHEMA in the setup notebook)
STAGED = ["clients", "monthly_flows_by_origin", "scheduled_payments"]     # what the staging recipe writes
HISTORY_MONTHS = 60            # months of postings summed, up to the cut-off month
# their columns. standard name: warehouse (EDW) name
COLUMNS = {
    # the current row of each business client
    "clients": {"client_no": "CSTMR_CIF", "customer_type": "CSTMR_TYP", "line_of_business": "LOB_DSC",
                "legal_entity": "LGL_ENTY_DSC", "sub_legal_entity": "PRTY_SUB_LGL_DSC", "id_type": "PRTY_ID_TYP_DSC",
                "segment": "RM_SGMNT_NM", "sector": "SCTR_DSC", "business_sector": "BUS_SCTR_EN_NM",
                "industry": "INDST_EN_NM", "sub_industry": "SUB_INDST_EN_NM", "city": "CITY", "status": "STS_FLG",
                "last_annual_sales": "ANNL_SLS_AMT", "number_of_employees": "NMBR_OF_EMP"},
    # one row per posting. AMT_LCY: SAR, negative for money out (T24). RVRSL_MRKR: set on a reversing posting, which
    # carries the reversed posting's TRNS_RFR
    "transactions": {"entry_id": "STMT_ENTRY_ID", "client_no": "CSTMR_CIF", "account_no": "ACNT_NMBR",
                     "value_date": "VAL_DT", "direction": "CRDT_DBT_FLG", "amount": "AMT_LCY", "origin_type": "SYS_ID",
                     "txn_type_tag": "TRNS_TYP", "txn_ref": "TRNS_RFR", "reversal": "RVRSL_MRKR"},
    # the monthly totals of the postings (monthly_flows_sql's columns): one row per client, month (of the value date),
    # direction, T24 module and transaction type
    "monthly_flows_by_origin": {"client_no": "CSTMR_CIF", "month": "MNTH", "direction": "CRDT_DBT_FLG",
                                "origin_type": "SYS_ID", "txn_type_tag": "TRNS_TYP", "amount": "AMT_LCY",
                                "postings": "PSTNG_CNT"},
    # one row per scheduled payment, past and future. SYS_ID and TRNS_TYP: what its statement entry carries when it is
    # paid. KNOWN_DT: when the bank first knew of it
    "scheduled_payments": {"client_no": "CSTMR_CIF", "origin_type": "SYS_ID", "origin_ref": "ORGN_RFR",
                           "direction": "CRDT_DBT_FLG", "txn_type_tag": "TRNS_TYP", "due_date": "DUE_DT",
                           "amount": "AMT_LCY", "known_date": "KNOWN_DT"},
}
OPTIONAL = {"ANNL_SLS_AMT", "NMBR_OF_EMP"}    # not in EDW.DIM_CUSTOMER: sent if the warehouse has them, else empty
ACTIVE_STATUS = ("ACTIVE", "A")               # STS_FLG values of an active client (to confirm with the warehouse team)
SOLE_PROPRIETOR = ("SOLE", "ESTABLISHMENT")   # words of LGL_ENTY_DSC or PRTY_SUB_LGL_DSC for a business owned by 1 person
LOAD_RECORD = "CFF_LOAD_RECORD"               # one row per table per load: LOAD_ID, LOAD_DTTS, AS_OF_DT, SRC_TBL_NM,
                                              # TRGT_TBL_NM, ROW_CNT, LOAD_STA_CD
AUDIT_COLUMNS = ["ETL_LOAD_ID", "ETL_LOAD_DTTS"]      # added by the warehouse team to every table


def to_standard(df, table):
    """A warehouse table with the standard column names and the project's values: direction IN / OUT; for clients,
    status ACTIVE / CLOSED and sole_proprietor true / false. An optional column the warehouse does not send stays
    empty; a missing required column stops here with its name."""
    missing = [c for c in COLUMNS[table].values() if c not in df and c not in OPTIONAL]
    if missing:
        raise ValueError(f"{TABLES[table][0]} has no column {', '.join(missing)}: check the warehouse file")
    out = pd.DataFrame({standard: df[column] if column in df else np.nan for standard, column in COLUMNS[table].items()})
    if "direction" in out:
        out["direction"] = out["direction"].astype(str).str.strip().str.upper().str[0].map({"C": "IN", "D": "OUT"})
    if table == "clients":
        active = out["status"].astype(str).str.strip().str.upper().isin(ACTIVE_STATUS)
        out["status"] = np.where(active, "ACTIVE", "CLOSED")
        legal = (out["legal_entity"].fillna("").astype(str) + " " + out["sub_legal_entity"].fillna("").astype(str))
        out["sole_proprietor"] = legal.str.upper().str.contains("|".join(SOLE_PROPRIETOR))
    return out


def is_sql(dataset):
    """A dataset read through a database connection (not CSV files)."""
    return "quotedResolvedTableName" in (dataset.get_location_info().get("info") or {})


def monthly_flows(entries, client_numbers, cutoff):
    """The monthly totals of the postings (standard names), in Python: business clients only, the HISTORY_MONTHS months
    up to the cut-off, without reversed postings (nor what they reverse) and without transfers between the client's
    own accounts (a debit and a credit with the same TRNS_RFR on the same client, netting to zero). The same rules as
    monthly_flows_sql."""
    e = entries[entries["client_no"].isin(set(client_numbers))].copy()
    e["month"] = pd.to_datetime(e["value_date"]).dt.to_period("M").dt.to_timestamp()
    e = e[(e["month"] > cutoff - pd.DateOffset(months=HISTORY_MONTHS)) & (e["month"] <= cutoff)]
    marked = e["reversal"].notna() & (e["reversal"].astype(str).str.strip() != "")
    e = e[~e["txn_ref"].isin(set(e.loc[marked, "txn_ref"].dropna()))]
    legs = e[e["txn_ref"].notna()].groupby(["txn_ref", "client_no"]).agg(sides=("direction", "nunique"),
                                                                         net=("amount", "sum"))
    own = legs[(legs["sides"] == 2) & (legs["net"].abs() < 0.005)].index
    e = e[~pd.MultiIndex.from_frame(e[["txn_ref", "client_no"]]).isin(own)]
    e = e.assign(amount=e["amount"].abs(), txn_type_tag=e["txn_type_tag"].fillna("NONE"))
    return (e.groupby(["client_no", "month", "direction", "origin_type", "txn_type_tag"], as_index=False)
            .agg(amount=("amount", "sum"), postings=("entry_id", "count")).round({"amount": 2}))


def monthly_flows_sql(entries_table, clients_table, cutoff):
    """The same monthly totals as monthly_flows, as one Oracle query run in the warehouse: only the totals come back.
    entries_table, clients_table: the tables' full quoted names; cutoff: the cut-off month (first day)."""
    c = COLUMNS["transactions"]
    first = (cutoff - pd.DateOffset(months=HISTORY_MONTHS - 1)).strftime("%Y-%m-%d")
    end = (cutoff + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    cif, ref, flag, amount = c["client_no"], c["txn_ref"], c["direction"], c["amount"]
    month = f"TRUNC(e.{c['value_date']}, 'MM')"
    kind = f"NVL(e.{c['txn_type_tag']}, 'NONE')"
    return f"""WITH entries AS (
    SELECT s.* FROM {entries_table} s
    WHERE s.{cif} IN (SELECT {COLUMNS['clients']['client_no']} FROM {clients_table})
      AND s.{c['value_date']} >= DATE '{first}' AND s.{c['value_date']} < DATE '{end}'
), reversed AS (
    SELECT DISTINCT {ref} FROM entries WHERE {c['reversal']} IS NOT NULL
), own_transfers AS (
    SELECT {ref}, {cif} FROM entries WHERE {ref} IS NOT NULL GROUP BY {ref}, {cif}
    HAVING COUNT(DISTINCT {flag}) = 2 AND ABS(SUM({amount})) < 0.005
)
SELECT e.{cif} AS CSTMR_CIF, {month} AS MNTH, e.{flag} AS CRDT_DBT_FLG, e.{c['origin_type']} AS SYS_ID,
       {kind} AS TRNS_TYP, SUM(ABS(e.{amount})) AS AMT_LCY, COUNT(*) AS PSTNG_CNT
FROM entries e
WHERE NOT EXISTS (SELECT 1 FROM reversed r WHERE r.{ref} = e.{ref})
  AND NOT EXISTS (SELECT 1 FROM own_transfers o WHERE o.{ref} = e.{ref} AND o.{cif} = e.{cif})
GROUP BY e.{cif}, {month}, e.{flag}, e.{c['origin_type']}, {kind}"""


def check_latest_load():
    """The latest warehouse load must be complete before the forecast runs: every table loaded with status
    SUCCESS and, for CSV files, holding the number of rows the load record says (a partial load stops here; a table
    read through the database connection is not counted: FCT_STATEMENT_ENTRY is far too big). Sets the project
    variables load_id, as_of_date and cutoff_month (the last complete month on the as-of date), which the recipes
    read. Raises an error on any problem, which stops the scenario before anything is built.
    Run by the first step of each scenario: from warehouse import check_latest_load; check_latest_load()"""
    import dataiku

    record = dataiku.Dataset(LOAD_RECORD).get_dataframe()
    latest = record.sort_values("LOAD_DTTS").iloc[-1]["LOAD_ID"]
    load = record[record["LOAD_ID"] == latest].set_index("TRGT_TBL_NM")
    problems = []
    for warehouse_table, _ in TABLES.values():
        if warehouse_table not in load.index:
            problems.append(f"{warehouse_table}: not in load {latest}")
            continue
        row = load.loc[warehouse_table]
        if row["LOAD_STA_CD"] != "SUCCESS":
            problems.append(f"{warehouse_table}: load status {row['LOAD_STA_CD']}")
        dataset = dataiku.Dataset(warehouse_table)
        if is_sql(dataset):
            continue
        first_column = next(iter(COLUMNS[next(t for t, v in TABLES.items() if v[0] == warehouse_table)].values()))
        count = len(dataset.get_dataframe(columns=[first_column]))
        if count != row["ROW_CNT"]:
            problems.append(f"{warehouse_table}: {count:,} rows, the load record says {row['ROW_CNT']:,}")
    if problems:
        raise ValueError("Warehouse load not usable:\n" + "\n".join(problems))

    as_of = pd.Timestamp(load["AS_OF_DT"].max())
    as_of = (as_of.tz_convert(None) if as_of.tzinfo else as_of).normalize()
    cutoff = as_of.to_period("M") if as_of == as_of + pd.offsets.MonthEnd(0) else as_of.to_period("M") - 1
    project = dataiku.api_client().get_default_project()
    variables = project.get_variables()
    variables["standard"].update(load_id=latest, as_of_date=f"{as_of:%Y-%m-%d}", cutoff_month=str(cutoff))
    project.set_variables(variables)
    print(f"Load {latest} OK: as of {as_of:%Y-%m-%d}, cut-off month {cutoff}")
    return str(cutoff)



for _table, _columns in COLUMNS.items():          # warehouse names unique within a table
    assert len(set(_columns.values())) == len(_columns), _table
