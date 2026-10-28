# Train the forecasting models, measure them honestly, then forecast the next 3 months.
#
# - One LightGBM model per (inflow|outflow) x (horizon 1|2|3 months) = 6 models, shared by all accounts.
# - Money values are divided by each account's average monthly inflow ("scale"), so a small SME and a
#   large corporate look alike to the model; predictions are multiplied back afterwards.
# - Backtest: pretend we are at each of the last 6 month-ends that have 3 known future months, train only
#   on what was known then, and compare to what really happened -- against two simple baselines
#   (same month last year, and the average of the last 3 months).
# - The backtest errors give a 10%-90% range around each net forecast.
import dataiku
import lightgbm as lgb
import numpy as np
import pandas as pd

HORIZONS, FLOWS, N_BACKTEST_ORIGINS = (1, 2, 3), ("inflow", "outflow"), 6


def naive(s):
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


ts = dataiku.Dataset("training_set").get_dataframe()
ts["month"] = naive(ts["month"])
ts = ts.sort_values(["account_id", "month"]).reset_index(drop=True)   # SQL tables have no row order

money = [c for c in ts.columns
         if c.split("_")[0] in ("inflow", "outflow", "net") and "_next_" not in c]
X = pd.concat([ts[money].div(ts["scale"], axis=0),
               ts[["month_of_year"]],
               pd.get_dummies(ts[["sector", "segment"]], dtype=float)], axis=1)
X.columns = [c.replace(" ", "_").replace("-", "_") for c in X.columns]


def fit(mask, flow, h):
    y = ts.loc[mask, f"{flow}_next_{h}"] / ts.loc[mask, "scale"]
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    return model.fit(X[mask], y)


def predict(model, mask):
    return model.predict(X[mask]) * ts.loc[mask, "scale"].values


# ---- backtest -------------------------------------------------------------------------------
last_complete = ts.loc[ts["net_cashflow_next_3"].notna(), "month"].max()
origins = [last_complete - pd.DateOffset(months=k) for k in range(N_BACKTEST_ORIGINS)]
bt = []
for origin in origins:
    at = ts["month"] == origin
    for h in HORIZONS:
        known = ts["month"] <= origin - pd.DateOffset(months=h)       # targets observed by `origin`
        known &= ts[f"net_cashflow_next_{h}"].notna()
        rows = ts.loc[at, ["account_id", "scale"]].assign(origin_month=origin, horizon=h,
                                                          target_month=origin + pd.DateOffset(months=h))
        for flow in FLOWS:
            rows[f"{flow}_actual"] = ts.loc[at, f"{flow}_next_{h}"].values
            rows[f"{flow}_model"] = predict(fit(known, flow, h), at)
            rows[f"{flow}_last_year"] = ts.loc[at, f"{flow}_ly_h{h}"].values
            rows[f"{flow}_avg3"] = ts.loc[at, f"{flow}_avg3"].values
        bt.append(rows)
bt = pd.concat(bt, ignore_index=True)
for method in ("actual", "model", "last_year", "avg3"):
    bt[f"net_{method}"] = bt[f"inflow_{method}"] - bt[f"outflow_{method}"]

evaluation = []
for method, label in (("model", "LightGBM model"), ("last_year", "Baseline: same month last year"),
                      ("avg3", "Baseline: average of last 3 months")):
    for h in HORIZONS:
        b = bt[bt["horizon"] == h]
        wape = {f: (b[f"{f}_{method}"] - b[f"{f}_actual"]).abs().sum() / b[f"{f}_actual"].abs().sum() for f in FLOWS}
        evaluation.append({"method": label, "horizon": h, "n_points": len(b),
                           "inflow_wape": wape["inflow"], "outflow_wape": wape["outflow"],
                           "net_mae": (b[f"net_{method}"] - b["net_actual"]).abs().mean()})
evaluation = pd.DataFrame(evaluation)

# relative error of the model's net forecast -> 10%..90% band per horizon
bt["net_rel_error"] = (bt["net_actual"] - bt["net_model"]) / bt["scale"]
band = bt.groupby("horizon")["net_rel_error"].quantile([0.1, 0.9]).unstack()

# ---- final models on all known history, forecast from the latest month ----------------------
latest = ts["month"].max()
at = ts["month"] == latest
forecast, importance = [], []
for h in HORIZONS:
    out = ts.loc[at, ["account_id", "scale"]].assign(origin_month=latest, horizon=h,
                                                     forecast_month=latest + pd.DateOffset(months=h))
    for flow in FLOWS:
        model = fit(ts[f"{flow}_next_{h}"].notna(), flow, h)
        out[f"{flow}_forecast"] = np.clip(predict(model, at), 0, None)
        gain = model.booster_.feature_importance("gain")
        importance += [{"flow": flow, "horizon": h, "feature": f, "importance": g / gain.sum()}
                       for f, g in zip(X.columns, gain)]
    out["net_forecast"] = out["inflow_forecast"] - out["outflow_forecast"]
    out["net_lower"] = out["net_forecast"] + band.loc[h, 0.1] * out["scale"]
    out["net_upper"] = out["net_forecast"] + band.loc[h, 0.9] * out["scale"]
    forecast.append(out)
forecast = pd.concat(forecast, ignore_index=True).drop(columns="scale").assign(model="LightGBM")

dataiku.Dataset("cashflow_forecast").write_with_schema(forecast)
dataiku.Dataset("model_evaluation").write_with_schema(evaluation)
dataiku.Dataset("feature_importance").write_with_schema(
    pd.DataFrame(importance).sort_values(["flow", "horizon", "importance"], ascending=[True, True, False]))
dataiku.Dataset("backtest_predictions").write_with_schema(bt.drop(columns=["scale", "net_rel_error"]))
