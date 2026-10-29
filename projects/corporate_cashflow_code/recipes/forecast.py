# Forecast the daily amount of each series (account x IN/OUT) for the next `horizon_days` days with a
# regression model: gradient boosting on calendar features + past values that are at least `horizon_days`
# old (so they are known for every day to forecast). Tested first on the last `horizon_days` known days
# against three simple baselines, then retrained on the full history. The 10% / 90% quantile models give
# the range.
import math

import dataiku
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = int(dataiku.get_custom_variables()["horizon_days"])
CALENDAR = ["day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end"]
FEATURES = ["series"] + CALENDAR + ["level_28d", "same_weekday_avg", "last_month_same_day"]
WEEKS = range(math.ceil(HORIZON / 7), math.ceil(HORIZON / 7) + 4)       # 4 same weekdays, all >= HORIZON days back


def features(df, origin):
    """Feature columns for every row, using only the amounts known before `origin` (the first day to forecast)."""
    df = df.copy()
    known = df["amount"].where(df["date"] < origin)
    by = known.groupby([df["account_id"], df["flow"]])
    df["level_28d"] = by.transform(lambda s: s.rolling(28).mean().shift(HORIZON))
    df["same_weekday_avg"] = sum(by.shift(7 * k) for k in WEEKS) / len(WEEKS)
    month_ago = (df[["account_id", "flow", "date"]].assign(date=df["date"] + pd.DateOffset(months=1), past=known)
                 .drop_duplicates(["account_id", "flow", "date"], keep="last"))      # 29-31 Jan -> 28 Feb
    df = df.merge(month_ago, on=["account_id", "flow", "date"], how="left").rename(columns={"past": "last_month_same_day"})
    df["series"] = (df["account_id"] + "_" + df["flow"]).astype("category")
    return df


def model(quantile=None):
    loss = {"loss": "quantile", "quantile": quantile} if quantile else {}
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                         random_state=7, **loss)


flows = dataiku.Dataset("daily_flows").get_dataframe()
flows["date"] = pd.to_datetime(flows["date"], utc=True).dt.tz_localize(None).dt.normalize()
flows = flows.sort_values(["account_id", "flow", "date"]).reset_index(drop=True)
history = flows[flows["amount"].notna()]
first_future = history["date"].max() + pd.Timedelta(days=1)

# 1. test: hide the last HORIZON known days, forecast them, compare with what happened
test_start = first_future - pd.Timedelta(days=HORIZON)
df = features(history, test_start)
train = df[(df["date"] < test_start) & df["level_28d"].notna()]
test = df[df["date"] >= test_start].copy()
test["Gradient boosting"] = model().fit(train[FEATURES], train["amount"]).predict(test[FEATURES]).clip(0)

known = df[df["date"] < test_start].set_index(["account_id", "flow", "date"])["amount"]
weeks_back = (test["date"] - test_start).dt.days // 7 + 1              # the latest known same weekday
test["Baseline: last known day"] = test.merge(
    known.groupby(level=[0, 1]).last().rename("v"), on=["account_id", "flow"], how="left")["v"].to_numpy()
test["Baseline: same weekday, last known week"] = known.reindex(pd.MultiIndex.from_arrays(
    [test["account_id"], test["flow"], test["date"] - pd.to_timedelta(7 * weeks_back, unit="D")])).to_numpy()
test["Baseline: same day last month"] = test["last_month_same_day"].fillna(0)

# MASE: error divided by the average error of "same as the day before" on the training days, per series
scale = train.groupby(["account_id", "flow"])["amount"].apply(lambda s: s.diff().abs().mean()).rename("scale")
test = test.merge(scale, on=["account_id", "flow"])
methods = ["Gradient boosting", "Baseline: last known day", "Baseline: same weekday, last known week",
           "Baseline: same day last month"]
evaluation = pd.DataFrame([{
    "method": m,
    "mae": round((test[m] - test["amount"]).abs().mean()),
    "mase": round(((test[m] - test["amount"]).abs() / test["scale"]).groupby([test["account_id"], test["flow"]])
                  .mean().mean(), 3),
} for m in methods])

# 2. forecast: retrain on the full history, forecast the next HORIZON days
df = features(flows, first_future)
train = df[(df["date"] < first_future) & df["level_28d"].notna()]
future = df[df["date"] >= first_future].copy()
future["forecast"] = model().fit(train[FEATURES], train["amount"]).predict(future[FEATURES]).clip(0)
future["forecast_low"] = model(0.1).fit(train[FEATURES], train["amount"]).predict(future[FEATURES]).clip(0)
future["forecast_high"] = model(0.9).fit(train[FEATURES], train["amount"]).predict(future[FEATURES]).clip(0)

dataiku.Dataset("forecast").write_with_schema(
    future[["account_id", "flow", "date", "forecast", "forecast_low", "forecast_high"]].round(2))
dataiku.Dataset("model_evaluation").write_with_schema(evaluation)
