# Shared by the training recipe and the scoring recipe (project library: ... > Libraries), so both read the
# data and compute the features the same way.
# The model is a regression: gradient boosting on calendar features + past values that are at least
# `horizon` days old (so they are known for every day to forecast).
import math

import dataiku
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CALENDAR = ["day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end"]
FEATURES = ["series"] + CALENDAR + ["level_28d", "same_weekday_avg", "last_month_same_day"]


def horizon():
    return int(dataiku.get_custom_variables()["horizon_days"])


def load_flows():
    """daily_flows: history (amount known) + the future days to forecast (amount empty)."""
    flows = dataiku.Dataset("daily_flows").get_dataframe()
    flows["date"] = pd.to_datetime(flows["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return flows.sort_values(["account_id", "flow", "date"]).reset_index(drop=True)


def features(df, origin, horizon):
    """Feature columns for every row, using only the amounts known before `origin` (the first day to forecast)."""
    weeks = range(math.ceil(horizon / 7), math.ceil(horizon / 7) + 4)  # 4 same weekdays, all >= horizon days back
    df = df.copy()
    known = df["amount"].where(df["date"] < origin)
    by = known.groupby([df["account_id"], df["flow"]])
    df["level_28d"] = by.transform(lambda s: s.rolling(28).mean().shift(horizon))
    df["same_weekday_avg"] = sum(by.shift(7 * k) for k in weeks) / len(weeks)
    month_ago = (df[["account_id", "flow", "date"]].assign(date=df["date"] + pd.DateOffset(months=1), past=known)
                 .drop_duplicates(["account_id", "flow", "date"], keep="last"))      # 29-31 Jan -> 28 Feb
    df = df.merge(month_ago, on=["account_id", "flow", "date"], how="left").rename(columns={"past": "last_month_same_day"})
    df["series"] = (df["account_id"] + "_" + df["flow"]).astype("category")
    return df


def model(quantile=None):
    loss = {"loss": "quantile", "quantile": quantile} if quantile else {}
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                         random_state=7, **loss)
