# Inference: forecast the next HORIZON_DAYS days of every series with the saved model. The features come from
# the latest data, so a model trained weeks ago still forecasts from yesterday's history.
import pickle

import dataiku
import pandas as pd
from cashflow_model import FEATURES, HORIZON_DAYS, features, load_series

with dataiku.Folder("forecast_model").get_download_stream("model.pkl") as s:
    saved = pickle.load(s)
if saved["horizon"] != HORIZON_DAYS:
    raise ValueError(f"The model was trained for {saved['horizon']} days, not {HORIZON_DAYS}: run \"Train model\".")

series = load_series()
first_future = series.dropna(subset=["amount"])["date"].max() + pd.Timedelta(days=1)
future = features(series, first_future)
future = future[future["date"] >= first_future].copy()
future["forecast"] = saved["model"].predict(future[FEATURES]).clip(0) * future["scale"]

dataiku.Dataset("forecast").write_with_schema(
    future[["account_id", "flow", "category", "date", "forecast"]].astype({"category": str}).round(2))
