"""Warehouse tables (project library: python/warehouse.py).

The forecast reads 2 tables of the bank's warehouse, read-only, and creates or writes nothing there:
    DIM_CUSTOMER          EDW.DIM_CUSTOMER: the clients. The project keeps the business clients (companies and sole
                          proprietors) and only the columns below: personal fields are never read. Through the
                          database connection, a read-only SELECT of those columns (table_query), whole every run.
    FCT_STATEMENT_ENTRY   EDW.FCT_STATEMENT_ENTRY: the daily postings. The staging recipe sums them per client, day
                          and category (money in, money out) in Dataiku and adds the new days to daily_flows. Through
                          the database connection, the only query is a read-only SELECT of the columns below with a
                          date filter on VAL_DT (table_query): the whole history the first time (or after the
                          scenario Full rebuild), then only the days after the last day already in daily_flows.
Through the database connection to the warehouse (Oracle, read-only), or as CSV files in a landing folder; the setup
notebook's settings choose. The sandbox's simulated tables are CSV (read whole, filtered by date in Dataiku). If a
name or a code differs at the bank, only this file changes.
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
# codes in DIM_CUSTOMER (from the bank's data profile; to confirm on business clients)
ACTIVE_STATUS = ("ACTIVE", "A")               # STS_FLG of an active client (an inactive one shows INACTIVE)
RETAIL = ("RETAIL",)                          # words of LOB_DSC for retail (individual) clients: left out...
BUSINESS_ID = ("COMMERCIAL",)                 # ...unless their ID type (PRTY_ID_TYP_DSC) is a commercial registration
SOLE_PROPRIETOR = ("SOLE", "ESTABLISHMENT")   # words of LGL_ENTY_DSC or PRTY_SUB_LGL_DSC for a business owned by 1 person


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
        active = out["status"].astype(str).str.strip().str.upper().isin(ACTIVE_STATUS)
        out["status"] = np.where(active, "ACTIVE", "CLOSED")
        legal = (out["legal_entity"].fillna("").astype(str) + " " + out["sub_legal_entity"].fillna("").astype(str))
        out["sole_proprietor"] = legal.str.upper().str.contains("|".join(SOLE_PROPRIETOR))
    return out


def business_clients(clients):
    """The business clients (standard names): every client whose line of business does not contain RETAIL, plus the retail
    ones registered with a commercial registration (sole proprietors). One row per client number."""
    lob = clients["line_of_business"].fillna("").astype(str).str.upper()
    id_type = clients["id_type"].fillna("").astype(str).str.upper()
    business = ~lob.str.contains("|".join(RETAIL)) | id_type.str.contains("|".join(BUSINESS_ID))
    return clients[business].drop_duplicates("client_no", keep="last").reset_index(drop=True)


def value_day(dates):
    """The day of each value date, as a plain (time zone free) timestamp."""
    d = pd.to_datetime(pd.Series(dates))
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert(None)
    return d.dt.normalize()


def table_query(table, name, since=None, end=None, count=False):
    """The read-only query staging runs through the database connection: only the columns read (COLUMNS) of a table,
    named in the query, so it never depends on the column list Dataiku keeps for the dataset. name: the table's full
    quoted name. For the postings, only the value dates from since (None: the whole history) to before end; the value
    date comes back as text YYYY-MM-DD (TO_CHAR: Oracle and PostgreSQL), so no time zone can move it. count: the
    same rows counted (COUNT(*)), for the progress lines."""
    date = COLUMNS["transactions"]["value_date"] if table == "transactions" else None
    columns = [f"TO_CHAR(\"{date}\", 'YYYY-MM-DD') AS \"{date}\"" if c == date else f'"{c}"'
               for c in COLUMNS[table].values()]
    query = f"SELECT {'COUNT(*) AS N' if count else ', '.join(columns)} FROM {name}"
    if date:
        where = [f"\"{date}\" < DATE '{end:%Y-%m-%d}'"]
        if since is not None:
            where.insert(0, f"\"{date}\" >= DATE '{since:%Y-%m-%d}'")
        query += f" WHERE {' AND '.join(where)}"
    return query


def category(txn_type):
    """The category of each posting, from its transaction type (CATEGORY_OF_TYPE; OTHER if not listed)."""
    return txn_type.fillna("").astype(str).str.strip().str.upper().map(CATEGORY_OF_TYPE).fillna(OTHER)


def daily_flows(entries, client_numbers, cutoff):
    """The daily totals of postings (standard names): one row per client, day and category with money_in, money_out
    (both positive, SAR) and the number of postings. Business clients only, every day up to the cut-off day,
    without reversed postings (nor what they reverse) and without transfers between the client's own accounts
    (a debit and a credit with the same TRNS_RFR on the same client, netting to zero). The staging recipe calls it on
    postings grouped so that all the postings of a TRNS_RFR are together."""
    e = entries[entries["client_no"].isin(set(client_numbers))].copy()
    e["day"] = value_day(e["value_date"])
    e = e[e["day"] <= cutoff]
    marked = e["reversal"].notna() & ~e["reversal"].astype(str).str.strip().isin(["", "nan", "None"])
    e = e[~e["txn_ref"].isin(set(e.loc[marked, "txn_ref"].dropna()))]
    legs = e[e["txn_ref"].notna()].groupby(["txn_ref", "client_no"]).agg(sides=("direction", "nunique"),
                                                                         net=("amount", "sum"))
    own = legs[(legs["sides"] == 2) & (legs["net"].abs() < 0.005)].index
    e = e[~pd.MultiIndex.from_frame(e[["txn_ref", "client_no"]]).isin(own)]
    e = e.assign(category=category(e["txn_type"]), amount=e["amount"].abs(),
                 money_in=lambda d: d["amount"].where(d["direction"] == "IN", 0.0),
                 money_out=lambda d: d["amount"].where(d["direction"] == "OUT", 0.0))
    return (e.groupby(["client_no", "day", "category"], as_index=False)
            .agg(money_in=("money_in", "sum"), money_out=("money_out", "sum"), postings=("entry_id", "count")))


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
