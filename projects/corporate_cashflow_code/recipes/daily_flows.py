# The time series: one row per account, flow (IN/OUT) and calendar day, 0 on days without transactions,
# up to yesterday (balance_date) + the next `horizon_days` days with an empty amount (to forecast).
import dataiku
import pandas as pd

HORIZON = int(dataiku.get_custom_variables()["horizon_days"])


def day(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()


totals = dataiku.Dataset("daily_totals").get_dataframe()
calendar = dataiku.Dataset("calendar").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
totals["date"], calendar["date"], accounts["balance_date"] = day(totals["date"]), day(calendar["date"]), day(accounts["balance_date"])

grid = (accounts[["account_id", "balance_date"]]
        .merge(pd.DataFrame({"flow": ["IN", "OUT"]}), how="cross")
        .merge(calendar, how="cross"))
grid = grid[grid["date"] <= grid["balance_date"] + pd.Timedelta(days=HORIZON)]
flows = grid.merge(totals, on=["account_id", "flow", "date"], how="left")
history = flows["date"] <= flows["balance_date"]
flows.loc[history, "amount"] = flows.loc[history, "amount"].fillna(0)        # quiet day = 0; future stays empty

dataiku.Dataset("daily_flows").write_with_schema(
    flows.drop(columns="balance_date").sort_values(["account_id", "flow", "date"]))
