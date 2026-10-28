# Turn the monthly history into a machine-learning table.
# One row = one account "as of" the end of a month (the forecast origin):
#   features  = what we knew at that point (recent months, averages, same month last year)
#   targets   = what actually happened 1, 2 and 3 months later (empty for the latest month)
import dataiku
import pandas as pd


def naive(s):
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


m = dataiku.Dataset("monthly_cashflow").get_dataframe()
acc = dataiku.Dataset("accounts").get_dataframe()[["account_id", "sector", "segment"]]
m["month"] = naive(m["month"])
m = m.sort_values(["account_id", "month"]).reset_index(drop=True)

ts = m[["account_id", "month"]].copy()
ts["month_of_year"] = ts["month"].dt.month
for c in ("inflow", "outflow", "net_cashflow"):
    s = m.groupby("account_id")[c]
    for k in (0, 1, 2):                                   # this month, last month, 2 months ago
        ts[f"{c}_m{k}"] = s.shift(k)
    ts[f"{c}_avg3"] = s.transform(lambda x: x.rolling(3).mean())
    ts[f"{c}_avg12"] = s.transform(lambda x: x.rolling(12).mean())
    for h in (1, 2, 3):
        ts[f"{c}_ly_h{h}"] = s.shift(12 - h)              # same calendar month last year as t+h
        ts[f"{c}_next_{h}"] = s.shift(-h)                 # target: actual value at t+h

ts["scale"] = ts["inflow_avg12"]                          # size of the business, used to normalise
ts = ts.merge(acc, on="account_id", how="left")
ts = ts[ts["inflow_avg12"].notna() & ts["inflow_ly_h1"].notna()]

latest = ts["month"].max()
ts["row_role"] = "TRAIN"
ts.loc[ts["net_cashflow_next_1"].isna() | ts["net_cashflow_next_3"].isna(), "row_role"] = "PARTIAL"
ts.loc[ts["month"] == latest, "row_role"] = "FORECAST_ORIGIN"

dataiku.Dataset("training_set").write_with_schema(ts)
