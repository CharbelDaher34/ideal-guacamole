# Clean the raw uploads: typed dates, de-duplicated transactions, a signed amount
# (+ credit / - debit) and the calendar month each transaction belongs to.
import dataiku
import pandas as pd

tx = dataiku.Dataset("transactions_raw").get_dataframe()
acc = dataiku.Dataset("accounts_raw").get_dataframe()

tx = tx.dropna(subset=["account_id", "booking_date", "amount"]).drop_duplicates("txn_id")
tx["booking_date"] = pd.to_datetime(tx["booking_date"])
tx["month"] = tx["booking_date"].dt.to_period("M").dt.to_timestamp()
tx["direction"] = tx["direction"].str.upper().str.strip()
tx["amount"] = tx["amount"].abs()
tx["signed_amount"] = tx["amount"].where(tx["direction"] == "CREDIT", -tx["amount"])

acc["opening_date"] = pd.to_datetime(acc["opening_date"])

dataiku.Dataset("transactions_prepared").write_with_schema(
    tx[["txn_id", "account_id", "booking_date", "month", "direction", "category",
        "counterparty", "amount", "signed_amount"]])
dataiku.Dataset("accounts").write_with_schema(acc)
