# The time series: one row per account, flow (IN / EXPENSE), category and day, with the calendar columns.
# amount = the day's total (0 on quiet days) up to yesterday; empty on the HORIZON_DAYS days to forecast.
import dataiku
import pandas as pd
from cashflow_model import HORIZON_DAYS


def day(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()


totals = dataiku.Dataset("daily_totals").get_dataframe()
calendar = dataiku.Dataset("calendar").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
totals["date"], calendar["date"] = day(totals["date"]), day(calendar["date"])
accounts["balance_date"] = day(accounts["balance_date"])

# every series seen in the history x every calendar day up to balance_date + HORIZON_DAYS
series = totals[["account_id", "flow", "category"]].drop_duplicates()
grid = series.merge(accounts[["account_id", "balance_date"]], on="account_id").merge(calendar, how="cross")
grid = grid[grid["date"] <= grid["balance_date"] + pd.Timedelta(days=HORIZON_DAYS)]

daily = grid.merge(totals, on=["account_id", "flow", "category", "date"], how="left")
history = daily["date"] <= daily["balance_date"]
daily.loc[history, "amount"] = daily.loc[history, "amount"].fillna(0)        # quiet day = 0; future stays empty

dataiku.Dataset("daily_series").write_with_schema(
    daily.drop(columns="balance_date").sort_values(["account_id", "flow", "category", "date"]))
