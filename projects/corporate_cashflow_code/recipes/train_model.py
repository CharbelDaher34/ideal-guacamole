# Training, in two parts:
# 1. test: hide the last HORIZON_DAYS known days, forecast them, compare with what happened and with three
#    simple baselines. Two measures: daily MASE (below 1 = better than "same as the day before") and monthly
#    WAPE (the error on the monthly totals, the figure the outlook uses).
# 2. retrain on the full history and save the model (model.pkl) and model_info.json (when, how good, which
#    horizon), read by the model check of the "Daily forecast" scenario.
import pickle
import time

import dataiku
import pandas as pd
from cashflow_model import FEATURES, HORIZON_DAYS, SERIES, features, load_series, model

history = load_series().dropna(subset=["amount"])
first_future = history["date"].max() + pd.Timedelta(days=1)

# 1. test
test_start = first_future - pd.Timedelta(days=HORIZON_DAYS)
df = features(history, test_start)
train = df[(df["date"] < test_start) & df["level_28d"].notna()]
test = df[df["date"] >= test_start].copy()
fitted = model().fit(train[FEATURES], train["amount"] / train["scale"])
test["Gradient boosting"] = fitted.predict(test[FEATURES]).clip(0) * test["scale"]

# baselines, from the raw known amounts
train, test = train.astype({"category": str}), test.astype({"category": str})
known = df[df["date"] < test_start].astype({"category": str}).set_index(SERIES + ["date"])["amount"]


def known_on(dates):
    return known.reindex(pd.MultiIndex.from_arrays(
        [test["account_id"], test["flow"], test["category"], dates])).fillna(0).to_numpy()


weeks_back = (test["date"] - test_start).dt.days // 7 + 1                  # the latest known same weekday
test["Baseline: last known day"] = known_on(pd.Series(test_start - pd.Timedelta(days=1), index=test.index))
test["Baseline: same weekday, last known week"] = known_on(test["date"] - pd.to_timedelta(7 * weeks_back, unit="D"))
test["Baseline: same day last year"] = known_on(test["date"] - pd.DateOffset(years=1))

# MASE: error / average error of "same as the day before" on the training days, per series, then averaged.
# Monthly WAPE: sum of |forecast - actual| / sum of actual, on the totals per account, flow and month.
naive = train.groupby(SERIES)["amount"].apply(lambda s: s.diff().abs().mean()).rename("naive").reset_index()
test = test.merge(naive, on=SERIES)
methods = ["Gradient boosting", "Baseline: last known day", "Baseline: same weekday, last known week",
           "Baseline: same day last year"]
monthly = test.groupby(["account_id", "flow", test["date"].dt.to_period("M")])[["amount"] + methods].sum()
evaluation = pd.DataFrame([{
    "method": m,
    "mae": round((test[m] - test["amount"]).abs().mean()),
    "mase": round(((test[m] - test["amount"]).abs() / test["naive"]).groupby(test["series"], observed=True)
                  .mean().mean(), 3),
    "monthly_wape": round((monthly[m] - monthly["amount"]).abs().sum() / monthly["amount"].abs().sum(), 3),
} for m in methods])
best = evaluation[evaluation["method"] == "Gradient boosting"].iloc[0]

# 2. retrain on the full history and save
df = features(history, first_future)
train = df[df["level_28d"].notna()]
fitted = model().fit(train[FEATURES], train["amount"] / train["scale"])
trained_until = str(history["date"].max().date())
folder = dataiku.Folder("forecast_model")
with folder.get_writer("model.pkl") as w:
    pickle.dump({"model": fitted, "horizon": HORIZON_DAYS, "trained_until": trained_until}, w)
folder.write_json("model_info.json", {
    "trained_at": time.time(), "horizon": HORIZON_DAYS, "trained_until": trained_until,
    "mase": float(best["mase"]), "monthly_wape": float(best["monthly_wape"])})
dataiku.Dataset("model_evaluation").write_with_schema(evaluation)
