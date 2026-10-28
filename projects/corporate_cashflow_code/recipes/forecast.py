# Train a linear regression on the "train" rows, check it on the "test" rows against two simple
# baselines, then forecast next month for every account and compute its risk level.
import dataiku
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

FEATURES = ["net", "net_avg_3m", "net_last_year"]
TARGET = "net_next_month"

rows = dataiku.Dataset("features").get_dataframe()
accounts = dataiku.Dataset("accounts").get_dataframe()
train, test, latest = (rows[rows["role"] == r] for r in ("train", "test", "forecast"))

model = LinearRegression().fit(train[FEATURES], train[TARGET])

evaluation = pd.DataFrame([
    {"method": "Linear regression", "mae": mean_absolute_error(test[TARGET], model.predict(test[FEATURES]))},
    {"method": "Baseline: same month last year", "mae": mean_absolute_error(test[TARGET], test["net_last_year"])},
    {"method": "Baseline: average of last 3 months", "mae": mean_absolute_error(test[TARGET], test["net_avg_3m"])},
]).round(0)

out = latest.merge(accounts, on="account_id")
out["forecast_month"] = pd.to_datetime(out["month"], utc=True).dt.tz_localize(None) + pd.DateOffset(months=1)
out["forecast_net"] = model.predict(out[FEATURES]).round()
out["projected_balance"] = (out["current_balance"] + out["forecast_net"]).round()
out["risk_level"] = "LOW"
out.loc[out["projected_balance"] < 0, "risk_level"] = "MEDIUM"
out.loc[out["projected_balance"] < -out["overdraft_limit"], "risk_level"] = "HIGH"
out = out.rename(columns={"net": "last_month_net", "net_avg_3m": "avg_3m_net"}).round({"last_month_net": 0, "avg_3m_net": 0})

dataiku.Dataset("account_outlook").write_with_schema(out[[
    "account_id", "company_name", "sector", "relationship_manager", "current_balance", "overdraft_limit",
    "last_month_net", "avg_3m_net", "forecast_month", "forecast_net", "projected_balance", "risk_level"]])
dataiku.Dataset("model_evaluation").write_with_schema(evaluation)
