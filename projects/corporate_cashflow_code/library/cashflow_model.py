# Shared by the training and the scoring recipes (project library: ... > Libraries), so both read the series
# and compute the features the same way.
# The model: gradient boosting on the calendar + past values that are at least HORIZON_DAYS old (so they are
# known for every day to forecast). One series = account x flow x category.
import math

import dataiku
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON_DAYS = 123   # days the model forecasts: the longest window, 31 x (horizon_months + 1) - 1 (create_project.py)
CALENDAR = ["day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end"]
FEATURES = ["series", "category"] + CALENDAR + ["month", "level_28d", "same_weekday_avg", "same_day_last_year"]
SERIES = ["account_id", "flow", "category"]


def load_series():
    """daily_series: history (amount known) + the days to forecast (amount empty)."""
    df = dataiku.Dataset("daily_series").get_dataframe()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.sort_values(SERIES + ["date"]).reset_index(drop=True)


def features(df, origin):
    """Feature columns for every row, using only the amounts known before `origin` (the first day to forecast).

    Series go from bank fees (~100 a month) to customer receipts (~100,000 a day), and some accounts grow
    while others shrink. So each series is divided by its level, `scale`: its average day over the last
    known year, moved along its growth (last 365 known days vs the 365 before, x0.5 to x2 a year) to the
    row's date. The model learns amount / scale; multiply back after."""
    weeks = range(math.ceil(HORIZON_DAYS / 7), math.ceil(HORIZON_DAYS / 7) + 4)  # 4 same weekdays, far enough back
    df = df.copy()
    keys = [df[c] for c in SERIES]
    known = df["amount"].where(df["date"] < origin)
    days_back = (origin - df["date"]).dt.days
    last_year = known.where(days_back.between(1, 365)).groupby(keys).transform("mean")
    year_before = known.where(days_back.between(366, 730)).groupby(keys).transform("mean")
    growth = (last_year / year_before).where(year_before > 0).clip(0.5, 2).fillna(1)
    years = ((df["date"] - origin).dt.days + 182.5) / 365                  # from the middle of the last year
    df["scale"] = (last_year * growth ** years).where(last_year > 0, 1)

    by = (known / df["scale"]).groupby(keys)
    df["level_28d"] = by.transform(lambda s: s.rolling(28).mean().shift(HORIZON_DAYS))
    df["same_weekday_avg"] = sum(by.shift(7 * k) for k in weeks) / len(weeks)
    year_ago = (df[SERIES + ["date"]].assign(date=df["date"] + pd.DateOffset(years=1), past=known / df["scale"])
                .drop_duplicates(SERIES + ["date"], keep="last"))                     # 29 Feb -> 28 Feb
    df = df.merge(year_ago, on=SERIES + ["date"], how="left").rename(columns={"past": "same_day_last_year"})
    df["month"] = df["date"].dt.month
    df["series"] = (df["account_id"] + "_" + df["flow"] + "_" + df["category"]).astype("category")
    df["category"] = df["category"].astype("category")
    return df


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                         random_state=7)
