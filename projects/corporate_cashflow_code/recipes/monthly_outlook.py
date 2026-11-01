# The days of daily_balance_path added up per account and month: totals, opening and closing balance, lowest
# balance and its date, lowest bad-case balance, and the month's risk level:
#   HIGH = the lowest balance goes past the overdraft limit, MEDIUM = it (or the bad case) goes below 0.
import dataiku
import pandas as pd

path = dataiku.Dataset("daily_balance_path").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
path["date"] = pd.to_datetime(path["date"], utc=True).dt.tz_localize(None).dt.normalize()
path = path.sort_values(["account_id", "date"]).reset_index(drop=True)

by_month = path.groupby(["account_id", "month", "month_index"])
monthly = by_month.agg(
    first_day=("date", "min"), last_day=("date", "max"), days=("date", "size"),
    inflows=("inflows", "sum"), expenses=("expenses", "sum"), obligations=("obligations", "sum"),
    outflows=("outflows", "sum"), net=("net", "sum"), closing_balance=("balance", "last"),
    bad_case_lowest_balance=("bad_case_balance", "min")).reset_index()
monthly["opening_balance"] = monthly["closing_balance"] - monthly["net"]
lowest = path.loc[by_month["balance"].idxmin(), ["account_id", "month", "date", "balance"]]
monthly = (monthly.merge(lowest.rename(columns={"date": "lowest_balance_date", "balance": "lowest_balance"}),
                         on=["account_id", "month"])
           .merge(accounts[["account_id", "company_name", "overdraft_limit"]], on="account_id"))

monthly["risk_level"] = "LOW"
monthly.loc[(monthly["lowest_balance"] < 0) | (monthly["bad_case_lowest_balance"] < 0), "risk_level"] = "MEDIUM"
monthly.loc[monthly["lowest_balance"] < -monthly["overdraft_limit"], "risk_level"] = "HIGH"

dataiku.Dataset("monthly_outlook").write_with_schema(monthly[[
    "account_id", "company_name", "month", "month_index", "first_day", "last_day", "days", "inflows", "expenses",
    "obligations", "outflows", "net", "opening_balance", "closing_balance", "lowest_balance", "lowest_balance_date",
    "bad_case_lowest_balance", "overdraft_limit", "risk_level"]].sort_values(["account_id", "month"]))
