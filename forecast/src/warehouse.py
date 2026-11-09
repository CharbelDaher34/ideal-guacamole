"""Warehouse tables (project library: python/warehouse.py).

The forecast reads 2 tables of the bank's warehouse, read-only, and creates or writes nothing there:
    DIM_CUSTOMER          EDW.DIM_CUSTOMER: the clients. The project keeps the business clients (companies and sole
                          proprietors) and only the columns below: personal fields are never read. Through the
                          database connection, a read-only SELECT of those columns (table_query), whole every run.
    FCT_STATEMENT_ENTRY   EDW.FCT_STATEMENT_ENTRY: the daily postings. The staging recipe sums them per client, day
                          and category (money in, money out) in Dataiku and adds the new days to daily_flows. Through
                          the database connection, the only query is a read-only SELECT of the columns below with a
                          date filter on VAL_DT (table_query): the last HISTORY_MONTHS months the first time (or after
                          the scenario Full rebuild), then only the days after the last day already in daily_flows.
Always through the database connection to the warehouse (the setup's EDW_CONNECTION: Oracle at the bank, read-only;
a local PostgreSQL with fake data on the test machine): the same code, only the connection differs. If a name or a
code differs at the bank, only this file changes.
"""
import numpy as np
import pandas as pd

# the tables the project reads. standard name: (table, where it is)
TABLES = {
    "clients": ("DIM_CUSTOMER", "EDW.DIM_CUSTOMER"),
    "transactions": ("FCT_STATEMENT_ENTRY", "EDW.FCT_STATEMENT_ENTRY"),
}
STAGED = ["clients", "daily_flows", "staging_log"]       # what the staging recipe writes

# Categories: each posting gets one from its transaction type (TRNS_TYP); a type not listed goes to OTHER.
# MOCK LIST: these are the sample data's types. At the bank, replace it with the real TRNS_TYP codes (ask the
# warehouse team for the list of codes and what each means); nothing else changes.
CATEGORY_OF_TYPE = {
    "COLLECTION": "CUSTOMER_RECEIPTS", "CARD_SETTLEMENT": "CUSTOMER_RECEIPTS", "CHEQUE": "CUSTOMER_RECEIPTS",
    "SALARY": "SALARIES", "RENT": "RENT", "SUPPLIER": "SUPPLIERS", "FINANCING": "FINANCING", "GOV": "GOVERNMENT",
    "UTILITY": "UTILITIES", "CARD": "CARDS", "FEE": "FEES",
}
OTHER = "OTHER"

# the columns read. standard name: warehouse (EDW) name
COLUMNS = {
    "clients": {"client_no": "CSTMR_CIF", "customer_type": "CSTMR_TYP", "line_of_business": "LOB_DSC",
                "legal_entity": "LGL_ENTY_DSC", "sub_legal_entity": "PRTY_SUB_LGL_DSC", "id_type": "PRTY_ID_TYP_DSC",
                "segment": "RM_SGMNT_NM", "sector": "SCTR_DSC", "business_sector": "BUS_SCTR_EN_NM",
                "industry": "INDST_EN_NM", "sub_industry": "SUB_INDST_EN_NM", "city": "CITY", "status": "STS_FLG"},
    # one row per posting. AMT_LCY: SAR, negative for money out (T24). TRNS_TYP: the transaction type, which gives
    # the category. RVRSL_MRKR: set on a reversing posting, which carries the reversed posting's TRNS_RFR
    "transactions": {"entry_id": "STMT_ENTRY_ID", "client_no": "CSTMR_CIF", "account_no": "ACNT_NMBR",
                     "value_date": "VAL_DT", "direction": "CRDT_DBT_FLG", "amount": "AMT_LCY", "txn_type": "TRNS_TYP",
                     "txn_ref": "TRNS_RFR", "reversal": "RVRSL_MRKR"},
}
# Which rows of DIM_CUSTOMER are read: 2 SQL conditions, applied in the warehouse (the bank's own query, checked on
# 2026-11-08: about 420k rows, one per customer). Column names in double quotes, so they work as they are on Oracle.
# The business clients: relationship manager code starting with 12, or the sector Sole Proprietorships
CLIENTS_BUSINESS = """ "PRIME_RM" LIKE '12%' OR "SCTR_DSC" = 'Sole Proprietorships' """
# DIM_CUSTOMER keeps history (about 520 million rows): each customer's current row is the one still open
CLIENTS_CURRENT_ROW = """ "END_DT" = DATE '2199-12-31' """
# FCT_STATEMENT_ENTRY is partitioned by month on its load date AS_OF_DT (not on the value date VAL_DT). Staging
# filters on it too, so Oracle reads only the load partitions that can hold the days it needs, not the whole table:
# AS_OF_DT from POSTINGS_LOAD_MARGIN_DAYS days before the first value day read. A posting loaded after its value
# date is always read; only one loaded more than that many days BEFORE its value date would be missed.
# None: no load-date filter (every partition is read).
POSTINGS_LOAD_DATE = "AS_OF_DT"
POSTINGS_LOAD_MARGIN_DAYS = 31
ACTIVE_STATUS = ("ACTIVE", "A")               # STS_FLG of an active client; an empty STS_FLG counts as active too
                                              # (an inactive one shows INACTIVE)
SOLE_PROPRIETOR = ("SOLE", "ESTABLISHMENT")   # words of LGL_ENTY_DSC, PRTY_SUB_LGL_DSC or SCTR_DSC for a business owned by
                                              # 1 person (the sector Sole Proprietorships, for example)


def read_columns(table):
    """The warehouse columns the project reads from a table."""
    return list(COLUMNS[table].values())


def to_standard(df, table):
    """A warehouse table with the standard column names and the project's values: client numbers as text, direction
    IN / OUT; for clients, status ACTIVE / CLOSED and sole_proprietor true / false. A missing column stops here
    with its name."""
    missing = [c for c in COLUMNS[table].values() if c not in df]
    if missing:
        raise ValueError(f"{TABLES[table][0]} has no column {', '.join(missing)}: check the warehouse table")
    out = pd.DataFrame({standard: df[column] for standard, column in COLUMNS[table].items()})
    out["client_no"] = out["client_no"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if "direction" in out:
        out["direction"] = out["direction"].astype(str).str.strip().str.upper().str[0].map({"C": "IN", "D": "OUT"})
    if table == "clients":
        status_raw = out["status"].astype(str).str.strip().str.upper().replace(["NAN", "NONE", "NULL", ""], "")
        active = status_raw.isin(ACTIVE_STATUS) | status_raw.eq("")      # no status at all: active too
        out["status"] = np.where(active, "ACTIVE", "CLOSED")
        legal = (out["legal_entity"].fillna("").astype(str) + " " + out["sub_legal_entity"].fillna("").astype(str)
                 + " " + out["sector"].fillna("").astype(str))
        out["sole_proprietor"] = legal.str.upper().str.contains("|".join(SOLE_PROPRIETOR))
    return out


def one_row_per_client(clients):
    """One row per client number (the last one received). The warehouse already sends one per client
    (CLIENTS_CURRENT_ROW); staging warns if it did not."""
    return clients.drop_duplicates("client_no", keep="last").reset_index(drop=True)


def value_day(dates):
    """The day of each value date, as a plain (time zone free) timestamp."""
    d = pd.to_datetime(pd.Series(dates))
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert(None)
    return d.dt.normalize()


def table_query(table, name, since=None, end=None, count=False, limit=None):
    """The read-only query staging runs through the database connection: only the columns read (COLUMNS) of a table,
    named in the query, so it never depends on the column list Dataiku keeps for the dataset. name: the table's full
    quoted name. For the postings, only the value dates from since (None: the whole history) to before end; the value
    date comes back as text YYYY-MM-DD (TO_CHAR: Oracle and PostgreSQL), so no time zone can move it. count: the
    same rows counted (COUNT(*)). For the clients, only the business clients' current rows (clients_filter): the
    warehouse leaves the other customers and the old versions out, so far fewer rows travel."""
    date = COLUMNS["transactions"]["value_date"] if table == "transactions" else None
    columns = [f"TO_CHAR(\"{date}\", 'YYYY-MM-DD') AS \"{date}\"" if c == date else f'"{c}"'
               for c in COLUMNS[table].values()]
    query = f"SELECT {'COUNT(*) AS N' if count else ', '.join(columns)} FROM {name}"
    if table == "clients":
        query += f" WHERE {clients_filter()}"       # business clients' current rows only: the rest stays in Oracle
        if limit:                                   # only the first `limit` business clients (CLIENTS_LIMIT)
            query += f' ORDER BY "{COLUMNS["clients"]["client_no"]}" FETCH FIRST {int(limit)} ROWS ONLY'
    if date:
        where = [f"\"{date}\" < DATE '{end:%Y-%m-%d}'"]
        if since is not None:
            where.insert(0, f"\"{date}\" >= DATE '{since:%Y-%m-%d}'")
        query += f" WHERE {' AND '.join(where)}"
    return query


def clients_filter():
    """The SQL condition on DIM_CUSTOMER: the business clients (CLIENTS_BUSINESS), only each one's current row
    (CLIENTS_CURRENT_ROW). Used for the clients and, as a subquery, for the postings."""
    current = CLIENTS_CURRENT_ROW.strip()
    return f"({CLIENTS_BUSINESS.strip()})" + (f" AND ({current})" if current else "")


def _text(value):
    return "'" + str(value).replace("'", "''") + "'"


def daily_totals_query(postings, clients, since, end, limit=None):
    """The read-only query staging runs for the postings: the warehouse itself sums them into daily totals, so only
    one row per client, day and category comes back (columns client_no, day, category, money_in, money_out,
    postings), not every posting. postings, clients: the 2 tables' full quoted names. The rules (Oracle and
    PostgreSQL; NULL-safe, as Oracle treats '' as NULL):
    - 3 steps: the selected customers (DIM_CUSTOMER with clients_filter, one row each), the postings of the
      selected days (value dates from since, None: no lower bound, to before end), then the postings joined with
      the customers; with since, also only the load dates from
      POSTINGS_LOAD_MARGIN_DAYS before it (POSTINGS_LOAD_DATE: the partitions Oracle reads);
    - without reversed postings: a posting with a reversal marker (RVRSL_MRKR) carries the reversed posting's
      reference (TRNS_RFR), and every posting with that reference is left out;
    - without transfers between the client's own accounts: on the same client and reference, a credit and a debit
      that net to zero;
    - category from the transaction type (CATEGORY_OF_TYPE, else OTHER); money_in, money_out: the credits and the
      debits as positive SAR amounts (AMT_LCY); postings: how many.
    Window functions (one pass, no NOT IN / NOT EXISTS lookups) keep it fast on a big table."""
    c = COLUMNS["transactions"]
    dates = [f"\"{c['value_date']}\" < DATE '{end:%Y-%m-%d}'"]
    if since is not None:
        dates.insert(0, f"\"{c['value_date']}\" >= DATE '{since:%Y-%m-%d}'")
        if POSTINGS_LOAD_DATE and POSTINGS_LOAD_MARGIN_DAYS is not None:     # only the load partitions needed
            loaded = since - pd.Timedelta(days=POSTINGS_LOAD_MARGIN_DAYS)
            dates.append(f"\"{POSTINGS_LOAD_DATE}\" >= DATE '{loaded:%Y-%m-%d}'")
    cif = COLUMNS["clients"]["client_no"]
    # 1. the selected customers: one row per customer (DISTINCT), so the join below can never count a posting twice
    customers = f"SELECT DISTINCT \"{cif}\" AS c_cif FROM {clients} WHERE {clients_filter()}"
    if limit:                                       # the same first `limit` business clients as table_query
        customers += f" ORDER BY c_cif FETCH FIRST {int(limit)} ROWS ONLY"
    categories = " ".join(f"WHEN {_text(t)} THEN {_text(k)}" for t, k in CATEGORY_OF_TYPE.items())
    category = f"CASE UPPER(TRIM(p_typ)) {categories} ELSE {_text(OTHER)} END"
    return f"""WITH customers AS (
  {customers}
), postings AS (
  SELECT "{c['client_no']}" AS p_cif, TO_CHAR("{c['value_date']}", 'YYYY-MM-DD') AS p_day,
         UPPER(SUBSTR(TRIM("{c['direction']}"), 1, 1)) AS p_dir, "{c['amount']}" AS p_amt,
         "{c['txn_type']}" AS p_typ, "{c['txn_ref']}" AS p_ref, "{c['reversal']}" AS p_rev, "{c['entry_id']}" AS p_id
  FROM {postings}
  WHERE {' AND '.join(dates)}
), p AS (
  SELECT postings.* FROM postings JOIN customers ON customers.c_cif = postings.p_cif
), marked AS (
  SELECT p.*, MAX(CASE WHEN p_ref IS NOT NULL AND LENGTH(TRIM(p_rev)) > 0 AND TRIM(p_rev) NOT IN ('nan', 'None')
                       THEN 1 ELSE 0 END) OVER (PARTITION BY p_ref) AS reversed
  FROM p
), kept AS (
  SELECT marked.*,
         MAX(CASE WHEN p_dir = 'C' THEN 1 ELSE 0 END) OVER (PARTITION BY p_ref, p_cif)
         + MAX(CASE WHEN p_dir = 'D' THEN 1 ELSE 0 END) OVER (PARTITION BY p_ref, p_cif) AS sides,
         COALESCE(ABS(SUM(p_amt) OVER (PARTITION BY p_ref, p_cif)), 0) AS net
  FROM marked WHERE reversed = 0
)
SELECT p_cif AS "client_no", p_day AS "day", {category} AS "category",
       SUM(CASE WHEN p_dir = 'C' THEN ABS(p_amt) ELSE 0 END) AS "money_in",
       SUM(CASE WHEN p_dir = 'D' THEN ABS(p_amt) ELSE 0 END) AS "money_out",
       COUNT(p_id) AS "postings"
FROM kept
WHERE p_ref IS NULL OR NOT (sides = 2 AND net < 0.005)
GROUP BY p_cif, p_day, {category}"""


def standard_daily(df):
    """The daily totals as the warehouse returns them, with the project's types: client numbers as text, the day as
    a date, amounts and counts as numbers."""
    out = pd.DataFrame({"client_no": df["client_no"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
                        "day": pd.to_datetime(df["day"]),
                        "category": df["category"].astype(str),
                        "money_in": pd.to_numeric(df["money_in"]).astype(float).fillna(0.0),
                        "money_out": pd.to_numeric(df["money_out"]).astype(float).fillna(0.0),
                        "postings": pd.to_numeric(df["postings"]).astype("int64")})
    return out


def set_cutoff():
    """The forecast's cut-off: the last day of data, by default yesterday. The project variable cutoff_override
    (YYYY-MM-DD) fixes it instead: the sandbox's simulator sets it, and it reruns a past day. Sets the project
    variables cutoff_date, as_of_date and load_id (the data's as-of date), which the recipes read. The staging recipe
    stops if the warehouse has no posting up to the cut-off day.
    Run by the first step of each scenario: from warehouse import set_cutoff; set_cutoff()"""
    import dataiku
    project = dataiku.api_client().get_default_project()
    variables = project.get_variables()
    override = variables["standard"].get("cutoff_override")
    cutoff = pd.Timestamp(override) if override else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    variables["standard"].update(cutoff_date=f"{cutoff:%Y-%m-%d}", as_of_date=f"{cutoff:%Y-%m-%d}",
                                 load_id=f"EDW_{cutoff:%Y%m%d}")
    project.set_variables(variables)
    print(f"Cut-off day {cutoff:%Y-%m-%d}" + (" (cutoff_override)" if override else " (yesterday)"))
    return f"{cutoff:%Y-%m-%d}"


def check_warehouse_loaded():
    """MOCK. Second step of each scenario: stops the scenario if the warehouse's load of the cut-off day is not
    finished (the ETL from the source systems into EDW). It always passes for now. Replace its body with the bank's
    real check, for example a read-only query on the warehouse's load-control table:
        SELECT STATUS FROM EDW.<LOAD_CONTROL_TABLE> WHERE LOAD_DATE = DATE '<cutoff_date>'   -- must say finished
    and raise an error if it is not finished. Whatever it says, staging also stops if the warehouse has no posting
    up to the cut-off day. Run as: from warehouse import check_warehouse_loaded; check_warehouse_loaded()"""
    import dataiku
    cutoff = dataiku.api_client().get_default_project().get_variables()["standard"]["cutoff_date"]
    print(f"MOCK warehouse check: the load of {cutoff} is assumed finished. Replace check_warehouse_loaded in "
          "warehouse.py with the bank's real check.")
    return True


def clear_daily_flows():
    """Step of the scenario Full rebuild (after the cut-off and the warehouse check): empties daily_flows and staging_log, so that the
    staging recipe reads the whole history again. Run as: from warehouse import clear_daily_flows; clear_daily_flows()"""
    import dataiku
    from cashflow import clear_daily
    clear_daily()
    dataiku.api_client().get_default_project().get_dataset("staging_log").clear()
    print("daily_flows and staging_log emptied: the staging recipe will read the whole history")


for _table, _columns in COLUMNS.items():          # warehouse names unique within a table
    assert len(set(_columns.values())) == len(_columns), _table
