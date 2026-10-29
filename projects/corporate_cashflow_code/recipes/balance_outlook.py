# Follow each account's balance day by day: yesterday's balance + running total of (money in - money out).
# The pessimistic path is the 1-in-10 bad case of that running total: each day's 10%-90% range gives its
# uncertainty (2.563 standard deviations), the uncertainties add up as variances over the days.
# Then one row per account: totals, closing balance, lowest balance and its date, risk level.
import dataiku
import numpy as np
import pandas as pd

forecast = dataiku.Dataset("forecast").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
forecast["date"] = pd.to_datetime(forecast["date"], utc=True).dt.tz_localize(None).dt.normalize()

days = forecast.pivot_table(index=["account_id", "date"], columns="flow",
                            values=["forecast", "forecast_low", "forecast_high"], fill_value=0)
path = pd.DataFrame({
    "inflow": days[("forecast", "IN")], "outflow": days[("forecast", "OUT")],
    "variance": ((days[("forecast_high", "IN")] - days[("forecast_low", "IN")]) / 2.563) ** 2
              + ((days[("forecast_high", "OUT")] - days[("forecast_low", "OUT")]) / 2.563) ** 2,
}).reset_index().sort_values(["account_id", "date"])
path = path.merge(accounts[["account_id", "balance"]].rename(columns={"balance": "opening_balance"}), on="account_id")
running = path.groupby("account_id")
path["balance"] = path["opening_balance"] + running["inflow"].cumsum() - running["outflow"].cumsum()
path["pessimistic_balance"] = path["balance"] - 1.2816 * np.sqrt(running["variance"].cumsum())
path = path[["account_id", "date", "inflow", "outflow", "balance", "pessimistic_balance"]].round(0)

lowest = path.loc[path.groupby("account_id")["balance"].idxmin(), ["account_id", "date", "balance"]]
outlook = (path.groupby("account_id")
           .agg(total_in=("inflow", "sum"), total_out=("outflow", "sum"), closing_balance=("balance", "last"),
                pessimistic_lowest_balance=("pessimistic_balance", "min"))
           .reset_index()
           .merge(lowest.rename(columns={"date": "lowest_balance_date", "balance": "lowest_balance"}), on="account_id")
           .merge(accounts.rename(columns={"balance": "opening_balance"}), on="account_id"))
outlook["risk_level"] = "LOW"
outlook.loc[(outlook["lowest_balance"] < 0) | (outlook["pessimistic_lowest_balance"] < 0), "risk_level"] = "MEDIUM"
outlook.loc[outlook["lowest_balance"] < -outlook["overdraft_limit"], "risk_level"] = "HIGH"

dataiku.Dataset("daily_balance_path").write_with_schema(path)
dataiku.Dataset("account_outlook").write_with_schema(outlook[[
    "account_id", "company_name", "sector", "relationship_manager", "opening_balance", "overdraft_limit",
    "total_in", "total_out", "closing_balance", "lowest_balance", "lowest_balance_date",
    "pessimistic_lowest_balance", "risk_level"]])
