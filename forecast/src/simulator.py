# Plays the warehouse team (Informatica): publishes the sample data as it was known on a date (AS_OF), in the
# warehouse's table and column names: the 17 detailed tables, and the 4 curated tables built from them (the ones
# the forecast project reads). Records each load in CFF_LOAD_RECORD.
# The sample data is generated once up to SIM_END; publishing "as of" a date keeps what had happened or was
# scheduled by then, and resets what was only known later (a bill paid after AS_OF is still unpaid, ...).
import uuid

import numpy as np
import pandas as pd


def as_of(tables, day):
    """The tables as the warehouse would have them at the end of `day`."""
    d = pd.Timestamp(day)
    t = {name: df.copy() for name, df in tables.items()}
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
    """The 4 curated tables, built from the detailed tables as of `day`."""
    d = pd.Timestamp(day)
    client_of = t["accounts"].set_index("account_no")["client_no"]
    month = lambda s: pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()
    scheduled = scheduled_payments(t, d, client_of)

    tx = t["transactions"]
    reversed_ids = set(tx.loc[tx["reversal_flag"].astype(bool), "original_ref"])
    tx = tx[~tx["reversal_flag"].astype(bool) & ~tx["txn_id"].isin(reversed_ids) & ~tx["own_account_flag"].astype(bool)]
    tx = tx.assign(client_no=tx["account_no"].map(client_of), month=month(tx["value_date"]),
                   direction=np.where(tx["debit_credit"] == "C", "IN", "OUT"),
                   origin_type=tx["origin_type"].fillna(""), txn_type_tag=tx["txn_type_tag"].fillna(""))
    known = (scheduled.assign(known_month=month(scheduled["known_date"]))
             .groupby(["origin_type", "origin_ref"], as_index=False)["known_month"].min())
    tx = tx.merge(known, on=["origin_type", "origin_ref"], how="left")
    lead = (tx["month"].dt.year - tx["known_month"].dt.year) * 12 + tx["month"].dt.month - tx["known_month"].dt.month
    for h in (1, 2, 3):
        tx[f"known_{h}"] = np.where(tx["known_month"].notna() & (lead >= h), tx["amount_sar"], 0.0)
    flows = (tx.groupby(["client_no", "month", "direction", "origin_type", "txn_type_tag"], as_index=False)
             .agg(amount=("amount_sar", "sum"), postings=("txn_id", "count"), known_1=("known_1", "sum"),
                  known_2=("known_2", "sum"), known_3=("known_3", "sum")).round(2))

    b = t["daily_balances"].assign(month=lambda x: month(x["balance_date"]))
    b = b.sort_values("balance_date").groupby(["account_no", "month"]).tail(1)
    balances = (b.assign(client_no=b["account_no"].map(client_of))
                .groupby(["client_no", "month"], as_index=False)[["ledger_balance", "overdraft_limit"]].sum()
                .rename(columns={"ledger_balance": "balance_end"}).round(2))

    acc = t["accounts"]
    counts = acc.assign(current=acc["product_type"] == "CURRENT", restricted=acc["restricted_flag"].astype(bool)) \
        .groupby("client_no").agg(current_accounts=("current", "sum"), restricted_accounts=("restricted", "sum"))
    clients = t["clients"].merge(counts.reset_index(), on="client_no", how="left")
    return {"clients": clients, "monthly_flows_by_origin": flows, "scheduled_payments": scheduled,
            "monthly_balances": balances}


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


def to_warehouse(df, columns, load_id, loaded_at):
    """Standard columns -> warehouse columns (flags as Y / N) plus the audit columns."""
    out = pd.DataFrame(index=df.index)
    for column in columns:
        values = df[column]
        out[warehouse_column(column)] = np.where(values.astype(bool), "Y", "N") if is_flag(column) else values
    out["ETL_LOAD_ID"] = load_id
    out["ETL_LOAD_DTTS"] = loaded_at
    return out.reset_index(drop=True)


def load_warehouse(tables, day, connection, schema):
    """Full load of every warehouse table as of `day` (detailed, then curated), then one row per table in the
    load record."""
    load_id, loaded_at = f"L{pd.Timestamp.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}", pd.Timestamp.now().floor("s")
    published = as_of(tables, day)
    curated = curate(published, day)
    existing = {d["name"] for d in project.list_datasets()}
    record = []
    for data, names, columns in ((published, DETAIL_TABLES, DETAIL_COLUMNS), (curated, TABLES, COLUMNS)):
        for table, (warehouse_table, source) in names.items():
            rows = to_warehouse(data[table], columns[table], load_id, loaded_at)
            write_table(warehouse_table, rows, connection, schema, existing)
            record.append({"LOAD_ID": load_id, "LOAD_DTTS": loaded_at, "AS_OF_DT": pd.Timestamp(day),
                           "SRC_TBL_NM": source, "TRGT_TBL_NM": warehouse_table, "ROW_CNT": len(rows),
                           "LOAD_STA_CD": "SUCCESS"})
            print(f"{warehouse_table:26s} {len(rows):>9,} rows")
    previous = dataiku.Dataset(LOAD_RECORD).get_dataframe() if LOAD_RECORD in existing else None
    write_table(LOAD_RECORD, pd.concat([previous, pd.DataFrame(record)], ignore_index=True), connection, schema,
                existing)
    return load_id


def write_table(name, df, connection, schema, existing):
    """Write a warehouse table (a dataset of this project stored as <schema>.<name>)."""
    if name not in existing:
        dataset = project.new_managed_dataset(name).with_store_into(connection).create()
        settings = dataset.get_settings()
        settings.get_raw()["params"].update(table=name, schema=schema)
        settings.save()
        existing.add(name)
    dataiku.Dataset(name).write_with_schema(df, drop_and_create=True)
