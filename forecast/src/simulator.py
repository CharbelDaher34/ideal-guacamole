# Plays the warehouse team (Informatica): publishes the sample data as it was known on a date (AS_OF), in the
# warehouse's table and column names, and records each load in CFF_LOAD_RECORD.
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


def to_warehouse(df, table, load_id, loaded_at):
    """Standard columns -> warehouse columns (flags as Y / N) plus the audit columns."""
    out = pd.DataFrame(index=df.index)
    for column in COLUMNS[table]:
        values = df[column] if column in df else None
        out[warehouse_column(column)] = np.where(values.astype(bool), "Y", "N") if is_flag(column) else values
    out["ETL_LOAD_ID"] = load_id
    out["ETL_LOAD_DTTS"] = loaded_at
    return out.reset_index(drop=True)


def load_warehouse(tables, day, connection, schema):
    """Full load of every warehouse table as of `day`, then one row per table in the load record."""
    load_id, loaded_at = f"L{pd.Timestamp.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}", pd.Timestamp.now().floor("s")
    published = as_of(tables, day)
    existing = {d["name"] for d in project.list_datasets()}
    record = []
    for table, (warehouse_table, source) in TABLES.items():
        data = to_warehouse(published[table], table, load_id, loaded_at)
        write_table(warehouse_table, data, connection, schema, existing)
        record.append({"LOAD_ID": load_id, "LOAD_DTTS": loaded_at, "AS_OF_DT": pd.Timestamp(day),
                       "SRC_TBL_NM": source, "TRGT_TBL_NM": warehouse_table, "ROW_CNT": len(data),
                       "LOAD_STA_CD": "SUCCESS"})
        print(f"{warehouse_table:26s} {len(data):>9,} rows")
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
