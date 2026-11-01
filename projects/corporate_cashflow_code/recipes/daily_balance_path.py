# Each account's balance day by day, from D (the day after balance_date) to the end of month `horizon_months`:
#   outflows = expenses + obligations, net = inflows - outflows, balance = yesterday's balance + running net.
# Inflows and expenses are forecast; obligations (loan installments) come from the schedule and are certain.
# The bad case: money in bad_case_pct lower, expenses bad_case_pct higher, obligations as scheduled.
import dataiku
import pandas as pd

MONTHS = int(dataiku.get_custom_variables()["horizon_months"])
BAD = float(dataiku.get_custom_variables()["bad_case_pct"])


def day(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()


forecast = dataiku.Dataset("forecast").get_dataframe()
obligations = dataiku.Dataset("daily_obligations").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
forecast["date"], obligations["date"] = day(forecast["date"]), day(obligations["date"])
accounts["first_day"] = day(accounts["balance_date"]) + pd.Timedelta(days=1)                 # D
accounts["last_day"] = (accounts["first_day"].dt.to_period("M") + MONTHS).dt.end_time.dt.normalize()

# one row per account and day of the window (the model's extra days are dropped)
path = accounts[["account_id", "balance", "first_day", "last_day"]].rename(columns={"balance": "yesterday"})
path["date"] = [pd.date_range(a, b) for a, b in zip(path["first_day"], path["last_day"])]
path = path.explode("date").astype({"date": "datetime64[ns]"})

# the day's amounts, rounded to whole units so that the days add up exactly to the months
forecast["inflows"] = forecast["forecast"].where(forecast["flow"] == "IN", 0)
forecast["expenses"] = forecast["forecast"].where(forecast["flow"] == "EXPENSE", 0)
amounts = forecast.groupby(["account_id", "date"], as_index=False)[["inflows", "expenses"]].sum()
installments = obligations.groupby(["account_id", "date"], as_index=False)["amount"].sum()
path = (path.merge(amounts, on=["account_id", "date"], how="left")
        .merge(installments.rename(columns={"amount": "obligations"}), on=["account_id", "date"], how="left")
        .fillna({"inflows": 0, "expenses": 0, "obligations": 0})
        .sort_values(["account_id", "date"]))
path[["inflows", "expenses", "obligations"]] = path[["inflows", "expenses", "obligations"]].round(0)
path["outflows"] = path["expenses"] + path["obligations"]
path["net"] = path["inflows"] - path["outflows"]
path["bad_case_net"] = path["inflows"] * (1 - BAD) - path["expenses"] * (1 + BAD) - path["obligations"]
running = path.groupby("account_id")
path["balance"] = path["yesterday"] + running["net"].cumsum()
path["bad_case_balance"] = (path["yesterday"] + running["bad_case_net"].cumsum()).round(0)

# month = YYYY-MM; month_index 0 = D's month, 1-3 = the next months
path["month"] = path["date"].dt.strftime("%Y-%m")
path["month_index"] = ((path["date"].dt.year - path["first_day"].dt.year) * 12
                       + path["date"].dt.month - path["first_day"].dt.month)

dataiku.Dataset("daily_balance_path").write_with_schema(path[[
    "account_id", "date", "month", "month_index", "inflows", "expenses", "obligations", "outflows", "net",
    "balance", "bad_case_balance"]])
