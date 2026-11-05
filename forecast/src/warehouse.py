"""Warehouse tables (project library: python/warehouse.py).

The forecast reads 2 tables of the bank's warehouse as they are, read-only, and creates nothing there:
    DIM_CUSTOMER          EDW.DIM_CUSTOMER: the clients. The project keeps the business clients (companies and sole
                          proprietors) and only the columns below: personal fields are never read.
    FCT_STATEMENT_ENTRY   EDW.FCT_STATEMENT_ENTRY: the daily postings. The staging recipe reads them in chunks and
                          sums them per client, month and direction (money in, money out) in Dataiku: no query
                          runs in the warehouse but reading the rows.
Through the database connection to the warehouse (Oracle, read-only), or as CSV files in a landing folder; the setup
notebook's settings choose. The sandbox's simulated tables are CSV. If a name or a code differs at the bank, only
this file changes.
"""
import numpy as np
import pandas as pd

# the tables the project reads. standard name: (table, where it is)
TABLES = {
    "clients": ("DIM_CUSTOMER", "EDW.DIM_CUSTOMER"),
    "transactions": ("FCT_STATEMENT_ENTRY", "EDW.FCT_STATEMENT_ENTRY"),
}
STAGED = ["clients", "monthly_flows"]     # what the staging recipe writes
HISTORY_MONTHS = 60            # months of postings summed, up to the cut-off month

# the columns read. standard name: warehouse (EDW) name
COLUMNS = {
    "clients": {"client_no": "CSTMR_CIF", "customer_type": "CSTMR_TYP", "line_of_business": "LOB_DSC",
                "legal_entity": "LGL_ENTY_DSC", "sub_legal_entity": "PRTY_SUB_LGL_DSC", "id_type": "PRTY_ID_TYP_DSC",
                "segment": "RM_SGMNT_NM", "sector": "SCTR_DSC", "business_sector": "BUS_SCTR_EN_NM",
                "industry": "INDST_EN_NM", "sub_industry": "SUB_INDST_EN_NM", "city": "CITY", "status": "STS_FLG",
                "last_annual_sales": "ANNL_SLS_AMT", "number_of_employees": "NMBR_OF_EMP"},
    # one row per posting. AMT_LCY: SAR, negative for money out (T24). RVRSL_MRKR: set on a reversing posting, which
    # carries the reversed posting's TRNS_RFR
    "transactions": {"entry_id": "STMT_ENTRY_ID", "client_no": "CSTMR_CIF", "account_no": "ACNT_NMBR",
                     "value_date": "VAL_DT", "direction": "CRDT_DBT_FLG", "amount": "AMT_LCY", "txn_ref": "TRNS_RFR",
                     "reversal": "RVRSL_MRKR"},
}
OPTIONAL = {"ANNL_SLS_AMT", "NMBR_OF_EMP"}    # not in EDW.DIM_CUSTOMER today: read if they appear, else empty
# codes in DIM_CUSTOMER (from the bank's data profile; to confirm on business clients)
ACTIVE_STATUS = ("ACTIVE", "A")               # STS_FLG of an active client (an inactive one shows INACTIVE)
RETAIL = ("RETAIL",)                          # words of LOB_DSC for retail (individual) clients: left out...
BUSINESS_ID = ("COMMERCIAL",)                 # ...unless their ID type (PRTY_ID_TYP_DSC) is a commercial registration
SOLE_PROPRIETOR = ("SOLE", "ESTABLISHMENT")   # words of LGL_ENTY_DSC or PRTY_SUB_LGL_DSC for a business owned by 1 person


def read_columns(table):
    """The warehouse columns the project reads from a table (the optional ones are read only if they exist)."""
    return list(COLUMNS[table].values())


def to_standard(df, table):
    """A warehouse table with the standard column names and the project's values: client numbers as text, direction
    IN / OUT; for clients, status ACTIVE / CLOSED and sole_proprietor true / false. An optional column that is not
    there stays empty; a missing required column stops here with its name."""
    missing = [c for c in COLUMNS[table].values() if c not in df and c not in OPTIONAL]
    if missing:
        raise ValueError(f"{TABLES[table][0]} has no column {', '.join(missing)}: check the warehouse table")
    out = pd.DataFrame({standard: df[column] if column in df else np.nan for standard, column in COLUMNS[table].items()})
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


def monthly_flows(entries, client_numbers, cutoff):
    """The monthly totals of postings (standard names): business clients only, the HISTORY_MONTHS months up to the
    cut-off, without reversed postings (nor what they reverse) and without transfers between the client's own
    accounts (a debit and a credit with the same TRNS_RFR on the same client, netting to zero). The staging recipe
    calls it on postings grouped so that all the postings of a TRNS_RFR are together."""
    e = entries[entries["client_no"].isin(set(client_numbers))].copy()
    e["month"] = pd.to_datetime(e["value_date"]).dt.to_period("M").dt.to_timestamp()
    e = e[(e["month"] > cutoff - pd.DateOffset(months=HISTORY_MONTHS)) & (e["month"] <= cutoff)]
    marked = e["reversal"].notna() & ~e["reversal"].astype(str).str.strip().isin(["", "nan", "None"])
    e = e[~e["txn_ref"].isin(set(e.loc[marked, "txn_ref"].dropna()))]
    legs = e[e["txn_ref"].notna()].groupby(["txn_ref", "client_no"]).agg(sides=("direction", "nunique"),
                                                                         net=("amount", "sum"))
    own = legs[(legs["sides"] == 2) & (legs["net"].abs() < 0.005)].index
    e = e[~pd.MultiIndex.from_frame(e[["txn_ref", "client_no"]]).isin(own)]
    e = e.assign(amount=e["amount"].abs())
    return (e.groupby(["client_no", "month", "direction"], as_index=False)
            .agg(amount=("amount", "sum"), postings=("entry_id", "count")))


def set_cutoff():
    """The forecast's cut-off: the last complete month, by default the month before today. The project variable
    cutoff_override (YYYY-MM) fixes it instead: the sandbox's simulator sets it, and it reruns a past month. Sets the
    project variables cutoff_month, as_of_date and load_id (the data's as-of date), which the recipes read. The
    staging recipe stops if the postings do not reach the cut-off month (a late warehouse load).
    Run by the first step of each scenario: from warehouse import set_cutoff; set_cutoff()"""
    import dataiku
    project = dataiku.api_client().get_default_project()
    variables = project.get_variables()
    override = variables["standard"].get("cutoff_override")
    cutoff = pd.Period(override, "M") if override else pd.Timestamp.now().to_period("M") - 1
    as_of = cutoff.to_timestamp(how="end").normalize()
    variables["standard"].update(cutoff_month=str(cutoff), as_of_date=f"{as_of:%Y-%m-%d}",
                                 load_id=f"EDW_{as_of:%Y%m%d}")
    project.set_variables(variables)
    print(f"Cut-off month {cutoff} (data as of {as_of:%Y-%m-%d})" + (" (cutoff_override)" if override else ""))
    return str(cutoff)


for _table, _columns in COLUMNS.items():          # warehouse names unique within a table
    assert len(set(_columns.values())) == len(_columns), _table
