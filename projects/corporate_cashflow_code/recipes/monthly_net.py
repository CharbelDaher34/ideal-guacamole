# Net cash flow per account and month: money in (+) minus money out (-).
import dataiku
import pandas as pd

tx = dataiku.Dataset("transactions").get_dataframe()
tx["month"] = pd.to_datetime(tx["booking_date"]).dt.to_period("M").dt.to_timestamp()
tx["net"] = tx["amount"].where(tx["direction"] == "CREDIT", -tx["amount"])

monthly = tx.groupby(["account_id", "month"], as_index=False)["net"].sum()
dataiku.Dataset("monthly_net").write_with_schema(monthly)
