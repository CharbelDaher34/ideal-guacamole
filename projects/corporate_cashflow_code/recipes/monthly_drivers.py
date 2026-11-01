# What makes each month: per account, month and flow, every category (IN, EXPENSE) or loan (OBLIGATION), its
# amount and share, and the same month one year earlier (last_year, scaled to the bucket's days, since month 0
# may be partial). change_pct = amount / last_year - 1 (0.25 = +25%; empty when last_year is 0).
import dataiku
import pandas as pd


def day(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()


forecast = dataiku.Dataset("forecast").get_dataframe()
obligations = dataiku.Dataset("daily_obligations").get_dataframe()
history = dataiku.Dataset("transactions_clean").get_dataframe()
months = dataiku.Dataset("monthly_outlook").get_dataframe()[["account_id", "month", "month_index", "first_day", "days"]]
for df in (forecast, obligations, history):
    df["month"] = day(df["date"]).dt.strftime("%Y-%m")
months["first_day"] = day(months["first_day"])

# the months ahead: forecast per category, installments per loan
ahead = pd.concat([
    forecast.groupby(["account_id", "month", "flow", "category"], as_index=False)["forecast"].sum()
    .rename(columns={"category": "driver", "forecast": "amount"}),
    obligations.groupby(["account_id", "month", "loan_id"], as_index=False)["amount"].sum()
    .rename(columns={"loan_id": "driver"}).assign(flow="OBLIGATION"),
])

# the same drivers in the history, per month, moved one year forward to line up with the month ahead
history = history[history["excluded"] == 0].copy()
history["driver"] = history["loan_id"].where(history["flow"] == "OBLIGATION", history["category"])
history["month"] = (pd.PeriodIndex(history["month"], freq="M") + 12).strftime("%Y-%m")
past = (history.assign(amount=history["amount"].abs())
        .groupby(["account_id", "month", "flow", "driver"], as_index=False)["amount"].sum()
        .rename(columns={"amount": "last_year"}))

keys = ["account_id", "month", "flow", "driver"]
drivers = (months.merge(pd.concat([ahead[keys], past[keys]]).drop_duplicates(), on=["account_id", "month"])
           .merge(ahead, on=keys, how="left").merge(past, on=keys, how="left")
           .fillna({"amount": 0, "last_year": 0}))
drivers["amount"] = drivers["amount"].round(0)
drivers["last_year"] = (drivers["last_year"] * drivers["days"] / drivers["first_day"].dt.days_in_month).round(0)
drivers = drivers[(drivers["amount"] > 0) | (drivers["last_year"] > 0)]
total = drivers.groupby(["account_id", "month", "flow"])["amount"].transform("sum")
drivers["share"] = (drivers["amount"] / total.where(total > 0)).fillna(0).round(3)
drivers["change_pct"] = (drivers["amount"] / drivers["last_year"].where(drivers["last_year"] > 0) - 1).round(3)
drivers["rank"] = drivers.groupby(["account_id", "month", "flow"])["amount"].rank(method="first", ascending=False)

dataiku.Dataset("monthly_drivers").write_with_schema(drivers.astype({"rank": int})[[
    "account_id", "month", "month_index", "flow", "driver", "amount", "share", "last_year", "change_pct", "rank"]]
    .sort_values(["account_id", "month", "flow", "rank"]))
