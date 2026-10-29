# Training: test the model on the last `horizon_days` known days against three simple baselines, then retrain
# it on the full history and save it to the forecast_model folder, with the 10% / 90% quantile models that
# give the range, and model_info.json (when, how good, for which horizon), read by the daily model check.
# Run by the "Train model" scenario; the daily forecast reuses the saved model.
import pickle
import time

import dataiku
import pandas as pd
from cashflow_model import FEATURES, features, horizon, load_flows, model

HORIZON = horizon()
history = load_flows().dropna(subset=["amount"])
first_future = history["date"].max() + pd.Timedelta(days=1)

# 1. test: hide the last HORIZON known days, forecast them, compare with what happened
test_start = first_future - pd.Timedelta(days=HORIZON)
df = features(history, test_start, HORIZON)
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

# 2. retrain on the full history and save the three models
df = features(history, first_future, HORIZON)
train = df[df["level_28d"].notna()]
models = {column: model(quantile).fit(train[FEATURES], train["amount"])
          for column, quantile in {"forecast": None, "forecast_low": 0.1, "forecast_high": 0.9}.items()}
with dataiku.Folder("forecast_model").get_writer("model.pkl") as w:
    pickle.dump({"models": models, "horizon": HORIZON, "trained_until": str(history["date"].max().date())}, w)
dataiku.Folder("forecast_model").write_json("model_info.json", {
    "trained_at": time.time(), "horizon": HORIZON, "trained_until": str(history["date"].max().date()),
    "mase": float(evaluation.loc[evaluation["method"] == "Gradient boosting", "mase"].iloc[0])})
dataiku.Dataset("model_evaluation").write_with_schema(evaluation)
