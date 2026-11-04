# Obligations: every payment already scheduled, with the date it became known.
# One row per scheduled payment (obligation_items), then per client, category, month due and month known
# (monthly_obligations).
import dataiku
import numpy as np
import pandas as pd

from cashflow import FX_TO_SAR, add_months, cutoff_month, month_start, read_dataset

last_month = cutoff_month()
horizon_end = add_months(last_month, 4)          # schedules are needed up to 3 months after the last full month
client_of = read_dataset("accounts").set_index("account_no")["client_no"]


def items(df, origin_type, ref, client, category, direction, due, amount, known):
    return pd.DataFrame({"origin_type": origin_type, "origin_ref": df[ref].values, "client_no": client.values,
                         "category": category, "direction": direction, "due_date": pd.to_datetime(df[due]).values,
                         "amount": np.asarray(amount, dtype=float), "known_since": pd.to_datetime(df[known]).values})


parts = []

loans = read_dataset("financing_schedules")
known = np.where(loans["early_settlement_flag"].astype(bool), loans["due_date"], loans["contract_start_date"])
parts.append(items(loans.assign(known=known), "LOAN", "loan_no", loans["client_no"], "FINANCING", "OUT", "due_date",
                   loans["instalment_amount"], "known"))

orders = read_dataset("standing_orders")
every = {"MONTHLY": 1, "QUARTERLY": 3, "SEMI_ANNUAL": 6, "ANNUAL": 12}
category = {"RENT": "RENT", "SUPPLIER": "SUPPLIERS"}
rows = []
for o in orders.itertuples():
    stop = pd.Timestamp(o.end_date) if pd.notna(o.end_date) else horizon_end
    for day in pd.date_range(pd.Timestamp(o.start_date), min(stop, horizon_end),
                             freq=pd.DateOffset(months=every[o.frequency])):
        rows.append((o.instruction_id, client_of[o.account_no], category.get(o.beneficiary_type, "OTHER_OUT"), day,
                     o.amount * FX_TO_SAR.get(o.currency, 1.0), o.created_date))
so = pd.DataFrame(rows, columns=["ref", "client", "category", "due", "amount", "known"])
parts.append(pd.DataFrame({"origin_type": "STANDING_ORDER", "origin_ref": so["ref"], "client_no": so["client"],
                           "category": so["category"], "direction": "OUT", "due_date": pd.to_datetime(so["due"]),
                           "amount": so["amount"].astype(float), "known_since": pd.to_datetime(so["known"])}))

dep = read_dataset("deposits")
dep = dep[dep["rollover_instruction"] == "CREDIT_ACCOUNT"]
parts.append(items(dep, "DEPOSIT", "deposit_id", dep["client_no"], "DEPOSIT_MAT", "IN", "maturity_date",
                   dep["amount"] + dep["expected_profit"], "start_date"))

lc = read_dataset("trade_finance")
lc = lc[(lc["instrument_type"] == "LC") & lc["payment_date"].notna()]
parts.append(items(lc, "LC", "instrument_id", lc["client_no"], "SUPPLIERS", "OUT", "payment_date",
                   lc["amount"] * lc["currency"].map(FX_TO_SAR).fillna(1.0), "issue_date"))

card = read_dataset("corporate_cards")
card = card[card["record_type"] == "STATEMENT"]
parts.append(items(card, "CORP_CARD", "record_id", card["client_no"], "CORP_CARD", "OUT", "due_date",
                   card["amount_due"], "statement_date"))

bills = read_dataset("sadad_payments")
parts.append(items(bills, "SADAD", "bill_number", bills["client_no"], "SADAD_GOV", "OUT", "due_date", bills["amount"],
                   "bill_issue_date"))

chq = read_dataset("cheques")
chq = chq[chq["post_dated_flag"].astype(bool) & (chq["direction"] == "DEPOSITED")]
chq = chq.assign(due=pd.to_datetime(chq["cheque_date"]) + pd.Timedelta(days=1))
parts.append(items(chq, "CHEQUE", "cheque_id", chq["account_no"].map(client_of), "CHEQUES_IN", "IN", "due",
                   chq["amount"], "received_date"))

obligation_items = pd.concat(parts, ignore_index=True)
for c in ("due_date", "known_since"):
    obligation_items[c] = pd.to_datetime(obligation_items[c])
obligation_items = obligation_items[obligation_items["due_date"] < horizon_end]
obligation_items["due_month"] = month_start(obligation_items["due_date"])
obligation_items["known_month"] = month_start(obligation_items["known_since"])

monthly_obligations = (obligation_items
                       .groupby(["client_no", "category", "direction", "due_month", "known_month"], as_index=False)
                       .agg(amount=("amount", "sum"), items=("origin_ref", "count")))

dataiku.Dataset("obligation_items").write_with_schema(obligation_items)
dataiku.Dataset("monthly_obligations").write_with_schema(monthly_obligations)
