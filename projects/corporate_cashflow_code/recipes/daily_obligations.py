# Loan installments to pay from D (the day after balance_date) to the end of month `horizon_months`
# (month 0 = the rest of D's month), straight from the loan schedule: they are known, not forecast.
import dataiku
import pandas as pd

MONTHS = int(dataiku.get_custom_variables()["horizon_months"])


def day(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()


schedule = dataiku.Dataset("loan_schedule").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
schedule["date"] = day(schedule["due_date"])
first = day(accounts["balance_date"]) + pd.Timedelta(days=1)
accounts["first_day"], accounts["last_day"] = first, (first.dt.to_period("M") + MONTHS).dt.end_time.dt.normalize()

due = schedule.merge(accounts[["account_id", "first_day", "last_day"]], on="account_id")
due = due[(due["date"] >= due["first_day"]) & (due["date"] <= due["last_day"])]
due = due.groupby(["account_id", "loan_id", "date"], as_index=False)["amount"].sum()

dataiku.Dataset("daily_obligations").write_with_schema(due.sort_values(["account_id", "date", "loan_id"]))
