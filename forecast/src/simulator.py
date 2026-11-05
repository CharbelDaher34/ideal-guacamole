# Plays the warehouse team (Informatica): publishes the sample data as it was known on a date (AS_OF), as the 3
# tables the forecast reads, in their EDW names (warehouse.py), as CSV datasets of this project: the clients
# (EDW.DIM_CUSTOMER), the daily statement entries (EDW.FCT_STATEMENT_ENTRY) and the scheduled payments.
# Records each load in CFF_LOAD_RECORD. The generator's own 17 source tables stay inside the simulator.
# The sample data is generated once up to SIM_END; publishing "as of" a date keeps what had happened or was
# scheduled by then, and resets what was only known later (a bill paid after AS_OF is still unpaid, ...).
import uuid

import numpy as np
import pandas as pd

GENERATED_COLUMNS = {   # the generator's 17 source tables (sandbox only)
    "clients": ["client_no", "party_type", "segment", "sub_segment", "sector", "industry", "region", "city",
                "joining_date", "legal_form", "sole_proprietor_flag", "last_annual_sales", "number_of_employees",
                "parent_client_no", "status", "closure_date"],
    "accounts": ["account_no", "client_no", "product_type", "product_code", "currency", "open_date", "close_date",
                 "status", "restricted_flag", "payroll_account_flag"],
    "transactions": ["txn_id", "account_no", "posting_date", "value_date", "amount_sar", "amount_orig", "currency",
                     "exchange_rate", "debit_credit", "txn_code", "channel", "txn_type_tag", "counterparty_iban_code",
                     "counterparty_bank", "counterparty_name", "own_account_flag", "origin_type", "origin_ref",
                     "reversal_flag", "original_ref"],
    "daily_balances": ["account_no", "balance_date", "ledger_balance", "available_balance", "blocked_amount",
                       "hold_amount", "overdraft_limit"],
    "standing_orders": ["instruction_id", "account_no", "beneficiary_type", "beneficiary_iban_code", "amount",
                        "currency", "frequency", "start_date", "end_date", "next_payment_date", "status",
                        "created_date", "changed_date"],
    "payroll_files": ["FILE_ID", "DBT_CSTMR_CIF", "PAYROLL_ACNT_NMBR", "CRT_DTTS", "SLRY_TRNS_DT", "FND_DT", "TOT_AMT",
                      "NMBR_OF_EMP", "FILE_TYP_EN_NM", "FILE_SUB_TYP_EN_NM", "STA_CD", "DLT_FLG"],
    "financing_schedules": ["loan_no", "client_no", "product", "account_debited", "contract_start_date",
                            "maturity_date", "instalment_no", "due_date", "principal", "profit", "instalment_amount",
                            "paid_date", "paid_amount", "status", "early_settlement_flag", "credit_limit_total",
                            "credit_limit_used", "credit_limit_available"],
    "deposits": ["deposit_id", "client_no", "account_no", "product", "amount", "start_date", "maturity_date",
                 "expected_profit", "rollover_instruction", "status"],
    "sadad_payments": ["payment_id", "account_no", "client_no", "biller_id", "biller_name", "biller_category",
                       "bill_number", "bill_issue_date", "amount", "due_date", "paid_date", "channel"],
    "code_lists": ["code_type", "code", "meaning"],
    "trade_finance": ["instrument_id", "client_no", "instrument_type", "subtype", "amount", "currency", "issue_date",
                      "expiry_date", "payment_date", "cash_margin_account", "cash_margin_amount", "claim_amount",
                      "claim_date", "status"],
    "card_sales": ["merchant_id", "client_no", "sales_date", "settlement_date", "sales_amount", "fees", "chargebacks",
                   "net_amount_paid"],
    "corporate_cards": ["record_id", "record_type", "client_no", "statement_date", "amount_due", "due_date",
                        "paid_date", "paid_amount", "topup_date", "topup_amount"],
    "bills_issued": ["invoice_id", "client_no", "customer_iban_code", "issue_date", "due_date", "amount", "paid_date",
                     "paid_amount"],
    "client_monthly_summary": ["client_no", "month", "txn_count_in", "txn_amount_in", "txn_count_out",
                               "txn_amount_out", "online_amount", "cheque_count", "avg_balance", "payroll_salary",
                               "payroll_bonus"],
    "group_structure": ["parent_client_no", "child_client_no", "relationship_type", "start_date"],
    "cheques": ["cheque_id", "account_no", "direction", "status", "received_date", "cheque_date", "cleared_date",
                "amount", "post_dated_flag"],
}

# T24 module (SYS_ID) a posting or a scheduled payment carries, by the generator's origin type
SYS_ID = {"NONE": "FT", "INVOICE": "FT", "CARD_SETTLEMENT": "FT", "CORP_CARD": "FT", "PAYROLL": "FT", "SADAD": "FT",
          "CHEQUE": "CQ", "LOAN": "AA", "DEPOSIT": "AA", "LC": "LC", "STANDING_ORDER": "STO"}
LEGAL = {"LLC": ("Limited Liability Company", "Company"), "JSC": ("Joint Stock Company", "Company"),
         "PARTNERSHIP": ("Partnership", "Company"), "SOLE_ESTABLISHMENT": ("Sole Proprietorship", "Establishment")}


def as_of(tables, day):
    """The tables as the warehouse would have them at the end of `day`."""
    d = pd.Timestamp(day)
    t = {name: df.copy() for name, df in tables.items()}
    for name, df in t.items():          # a table with no rows (few sample clients) still gets its columns
        if df.empty and name in GENERATED_COLUMNS:
            t[name] = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c.endswith("_date") else "object")
                                    for c in GENERATED_COLUMNS[name]})
    later = lambda s: pd.to_datetime(s) > d

    c = t["clients"][t["clients"]["joining_date"] <= d].copy()
    c.loc[later(c["closure_date"]), ["status", "closure_date"]] = ["ACTIVE", pd.NaT]
    t["clients"] = c
    a = t["accounts"][t["accounts"]["open_date"] <= d].copy()
    a.loc[later(a["close_date"]), ["status", "close_date"]] = ["ACTIVE", pd.NaT]
    t["accounts"] = a
    t["transactions"] = t["transactions"][t["transactions"]["value_date"] <= d]
    t["daily_balances"] = t["daily_balances"][t["daily_balances"]["balance_date"] <= d]
    t["payroll_files"] = t["payroll_files"][t["payroll_files"]["SLRY_TRNS_DT"] <= d]   # history only

    so = t["standing_orders"][t["standing_orders"]["created_date"] <= d].copy()
    so.loc[later(so["end_date"]), ["status", "end_date"]] = ["ACTIVE", pd.NaT]          # cancelled later
    months = so["frequency"].map({"MONTHLY": 1, "QUARTERLY": 3, "SEMI_ANNUAL": 6, "ANNUAL": 12})
    so["next_payment_date"] = [next((x for x in pd.date_range(s, d + pd.DateOffset(months=13),
                                                               freq=pd.DateOffset(months=int(m))) if x > d), pd.NaT)
                               if st == "ACTIVE" else pd.NaT
                               for s, m, st in zip(so["start_date"], months, so["status"])]
    t["standing_orders"] = so

    ln = t["financing_schedules"]
    ln = ln[(ln["contract_start_date"] <= d) & ~(ln["early_settlement_flag"].astype(bool) & later(ln["due_date"]))].copy()
    unpaid = later(ln["paid_date"]) | ln["paid_date"].isna()
    ln.loc[unpaid, ["paid_date", "paid_amount"]] = [pd.NaT, np.nan]
    ln.loc[unpaid, "status"] = np.where(ln.loc[unpaid, "due_date"] <= d, "OVERDUE", "NOT_DUE")
    t["financing_schedules"] = ln

    dep = t["deposits"][t["deposits"]["start_date"] <= d].copy()
    dep["status"] = np.where(dep["maturity_date"] <= d, "MATURED", "ACTIVE")
    t["deposits"] = dep

    bills = t["sadad_payments"][t["sadad_payments"]["bill_issue_date"] <= d].copy()
    bills.loc[later(bills["paid_date"]), ["payment_id", "paid_date", "channel"]] = [None, pd.NaT, None]
    t["sadad_payments"] = bills

    tf = t["trade_finance"][t["trade_finance"]["issue_date"] <= d].copy()
    lc = tf["instrument_type"] == "LC"
    tf.loc[lc, "status"] = np.where(later(tf.loc[lc, "payment_date"]), "OUTSTANDING", "PAID")
    t["trade_finance"] = tf

    t["card_sales"] = t["card_sales"][t["card_sales"]["sales_date"] <= d]
    cc = t["corporate_cards"]
    cc = cc[(cc["statement_date"] <= d) | (cc["topup_date"] <= d)].copy()
    cc.loc[later(cc["paid_date"]), ["paid_date", "paid_amount"]] = [pd.NaT, np.nan]
    t["corporate_cards"] = cc

    inv = t["bills_issued"][t["bills_issued"]["issue_date"] <= d].copy()
    inv.loc[later(inv["paid_date"]), ["paid_date", "paid_amount"]] = [pd.NaT, np.nan]
    t["bills_issued"] = inv

    chq = t["cheques"][t["cheques"]["received_date"] <= d].copy()
    pending = later(chq["cleared_date"]) | ((chq["status"] == "RETURNED") & later(chq["cheque_date"]))
    chq.loc[pending, "cleared_date"] = pd.NaT
    chq.loc[pending, "status"] = np.where(chq.loc[pending, "post_dated_flag"].astype(bool), "HELD_POST_DATED",
                                          "IN_CLEARING")
    t["cheques"] = chq

    t["client_monthly_summary"] = t["client_monthly_summary"][t["client_monthly_summary"]["month"] <= d]
    t["group_structure"] = t["group_structure"][t["group_structure"]["start_date"] <= d]
    return t


def curate(t, day):
    """The 3 tables the forecast reads, built from the source tables as of `day` the way the warehouse team builds
    them from EDW (warehouse_tables.sql), with the standard column names."""
    d = pd.Timestamp(day)
    client_of = t["accounts"].set_index("account_no")["client_no"]
    scheduled = scheduled_payments(t, d, client_of)
    scheduled["origin_type"] = scheduled["origin_type"].map(SYS_ID)

    # the daily postings as T24 statement entries: amount negative for money out; a reversing posting is marked and
    # carries the reference of the posting it reverses; a transfer between the client's own accounts has both legs,
    # debit and credit, under one reference (the staging recipe leaves these out, as it would at the bank)
    tx = t["transactions"]
    ref = "FT" + tx["txn_id"].astype(str)
    ref = ref.where(~tx["reversal_flag"].astype(bool), "FT" + tx["original_ref"].astype(str))
    own = tx["own_account_flag"].astype(bool)
    entries = pd.DataFrame({
        "entry_id": tx["txn_id"], "client_no": tx["account_no"].map(client_of), "account_no": tx["account_no"],
        "value_date": pd.to_datetime(tx["value_date"]),
        "direction": np.where(tx["debit_credit"] == "C", "IN", "OUT"),
        "amount": np.where(tx["debit_credit"] == "C", 1.0, -1.0) * tx["amount_sar"],
        "origin_type": tx["origin_type"].replace("", np.nan).fillna("NONE").map(SYS_ID),
        "txn_type_tag": tx["txn_type_tag"].replace("", np.nan).fillna("NONE"),
        "txn_ref": ref.where(~own, "OWN" + tx["txn_id"].astype(str)),
        "reversal": np.where(tx["reversal_flag"].astype(bool), "R", None)})
    other_leg = entries[own.values].assign(
        entry_id=lambda d: d["entry_id"] + "M", account_no=lambda d: "OWN-" + d["account_no"].astype(str),
        direction=lambda d: np.where(d["direction"] == "IN", "OUT", "IN"), amount=lambda d: -d["amount"])
    entries = pd.concat([entries, other_leg], ignore_index=True).round({"amount": 2})

    c = t["clients"]
    legal = c["legal_form"].map(LEGAL)
    sole = c["legal_form"] == "SOLE_ESTABLISHMENT"
    clients = pd.DataFrame({
        "client_no": c["client_no"], "customer_type": np.where(sole, "Individual", "Corporate"),
        "line_of_business": "Business Banking", "legal_entity": legal.str[0], "sub_legal_entity": legal.str[1],
        "id_type": "Commercial Registration", "segment": c["segment"], "sector": c["sector"],
        "business_sector": c["sector"], "industry": c["industry"], "sub_industry": c["industry"], "city": c["city"],
        "status": c["status"], "last_annual_sales": c["last_annual_sales"],
        "number_of_employees": c["number_of_employees"]})
    return {"clients": clients, "transactions": entries, "scheduled_payments": scheduled}


def scheduled_payments(t, d, client_of):
    """Every scheduled payment, past and future, with the date it became known."""
    parts = []

    def add(df, origin_type, ref, client, direction, tag, due, amount, known):
        parts.append(pd.DataFrame({
            "origin_type": origin_type, "origin_ref": df[ref].values, "client_no": np.asarray(client),
            "direction": direction, "txn_type_tag": np.asarray(tag, dtype=object) if np.ndim(tag) else tag,
            "due_date": pd.to_datetime(df[due]).values, "amount": np.asarray(amount, dtype=float),
            "known_date": pd.to_datetime(df[known]).values}))

    ln = t["financing_schedules"]
    ln = ln.assign(known=np.where(ln["early_settlement_flag"].astype(bool), ln["due_date"], ln["contract_start_date"]))
    add(ln, "LOAN", "loan_no", ln["client_no"], "OUT", "FINANCING", "due_date", ln["instalment_amount"], "known")
    every = {"MONTHLY": 1, "QUARTERLY": 3, "SEMI_ANNUAL": 6, "ANNUAL": 12}
    tag = {"RENT": "RENT", "SUPPLIER": "SUPPLIER"}
    rows = []
    for o in t["standing_orders"].itertuples():
        stop = min(pd.Timestamp(o.end_date) if pd.notna(o.end_date) else d + pd.DateOffset(months=12),
                   d + pd.DateOffset(months=12))
        for day in pd.date_range(pd.Timestamp(o.start_date), stop, freq=pd.DateOffset(months=every[o.frequency])):
            rows.append((o.instruction_id, client_of[o.account_no], tag.get(o.beneficiary_type, "OTHER"), day,
                         o.amount, o.created_date))
    so = pd.DataFrame(rows, columns=["ref", "client", "tag", "due", "amount", "known"])
    add(so, "STANDING_ORDER", "ref", so["client"], "OUT", so["tag"], "due", so["amount"], "known")
    dep = t["deposits"][t["deposits"]["rollover_instruction"] == "CREDIT_ACCOUNT"]
    add(dep, "DEPOSIT", "deposit_id", dep["client_no"], "IN", "DEPOSIT", "maturity_date",
        dep["amount"] + dep["expected_profit"], "start_date")
    lc = t["trade_finance"][(t["trade_finance"]["instrument_type"] == "LC") & t["trade_finance"]["payment_date"].notna()]
    add(lc, "LC", "instrument_id", lc["client_no"], "OUT", "SUPPLIER", "payment_date",
        lc["amount"] * np.where(lc["currency"] == "USD", 3.75, 1.0), "issue_date")
    card = t["corporate_cards"][t["corporate_cards"]["record_type"] == "STATEMENT"]
    add(card, "CORP_CARD", "record_id", card["client_no"], "OUT", "CARD", "due_date", card["amount_due"],
        "statement_date")
    bills = t["sadad_payments"]
    add(bills, "SADAD", "bill_number", bills["client_no"], "OUT",
        np.where(bills["biller_category"] == "GOVERNMENT", "GOV", "UTILITY"), "due_date", bills["amount"],
        "bill_issue_date")
    chq = t["cheques"]
    chq = chq[chq["post_dated_flag"].astype(bool) & (chq["direction"] == "DEPOSITED")]
    chq = chq.assign(due=pd.to_datetime(chq["cheque_date"]) + pd.Timedelta(days=1))
    add(chq, "CHEQUE", "cheque_id", chq["account_no"].map(client_of), "IN", "CHEQUE", "due", chq["amount"],
        "received_date")
    return pd.concat(parts, ignore_index=True).round({"amount": 2})


def to_warehouse(df, table, load_id, loaded_at):
    """A table in the warehouse's column names and codes (direction C / D), with the audit columns."""
    out = pd.DataFrame({column: df[standard] for standard, column in COLUMNS[table].items()})
    if "CRDT_DBT_FLG" in out:
        out["CRDT_DBT_FLG"] = out["CRDT_DBT_FLG"].map({"IN": "C", "OUT": "D"})
    out["ETL_LOAD_ID"], out["ETL_LOAD_DTTS"] = load_id, loaded_at
    return out.reset_index(drop=True)


def load_warehouse(tables, day, connection):
    """Full load of the 3 tables as of `day`, then one row per table in the load record."""
    load_id, loaded_at = f"L{pd.Timestamp.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}", pd.Timestamp.now().floor("s")
    curated = curate(as_of(tables, day), day)
    existing = {d["name"] for d in project.list_datasets()}
    record = []
    for table, (warehouse_table, source) in TABLES.items():
        rows = to_warehouse(curated[table], table, load_id, loaded_at)
        write_table(warehouse_table, rows, connection, existing)
        record.append({"LOAD_ID": load_id, "LOAD_DTTS": loaded_at, "AS_OF_DT": pd.Timestamp(day),
                       "SRC_TBL_NM": source, "TRGT_TBL_NM": warehouse_table, "ROW_CNT": len(rows),
                       "LOAD_STA_CD": "SUCCESS"})
        print(f"{warehouse_table:26s} {len(rows):>9,} rows")
    previous = dataiku.Dataset(LOAD_RECORD).get_dataframe() if LOAD_RECORD in existing else None
    write_table(LOAD_RECORD, pd.concat([previous, pd.DataFrame(record)], ignore_index=True), connection, existing)
    return load_id


CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}


def write_table(name, df, connection, existing):
    """Write a warehouse table: a CSV dataset of this project."""
    if name not in existing:
        dataset = project.new_managed_dataset(name).with_store_into(connection).create()
        settings = dataset.get_settings()
        settings.get_raw()["formatType"] = "csv"
        settings.get_raw()["formatParams"].update(CSV)
        settings.save()
        existing.add(name)
    dataiku.Dataset(name).write_with_schema(df)
