# Inference: forecast the next `horizon_days` days with the model saved by the training recipe. The features
# are computed from the latest data, so a model trained weeks ago still forecasts from yesterday's history.
import pickle

import dataiku
import pandas as pd
from cashflow_model import FEATURES, features, horizon, load_flows

HORIZON = horizon()
with dataiku.Folder("forecast_model").get_download_stream("model.pkl") as s:
    saved = pickle.load(s)
if saved["horizon"] != HORIZON:
    raise ValueError(f"The model was trained for {saved['horizon']} days, horizon_days is {HORIZON}: "
                     "run the \"Train model\" scenario first.")

flows = load_flows()
first_future = flows.dropna(subset=["amount"])["date"].max() + pd.Timedelta(days=1)
future = features(flows, first_future, HORIZON)
future = future[future["date"] >= first_future].copy()
for column, m in saved["models"].items():
    future[column] = m.predict(future[FEATURES]).clip(0)

dataiku.Dataset("forecast").write_with_schema(
    future[["account_id", "flow", "date", "forecast", "forecast_low", "forecast_high"]].round(2))
