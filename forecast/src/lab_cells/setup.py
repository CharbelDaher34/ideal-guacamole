import calendar
import importlib
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google")      # third-party notices on import
logging.getLogger("matplotlib").setLevel(logging.ERROR)
try:
    import mlflow
    import sklearn
except ImportError:
    raise RuntimeError("Use the kernel of the code env cashflow_forecast (Kernel > Change kernel). If it is not in the "
                       "list, ask whoever manages that code env to enable its Jupyter support; meanwhile, set the "
                       "project variable lab_forecaster and build lab_results (Lab zone).")
import dataiku

warnings.filterwarnings("ignore", category=DeprecationWarning)    # third-party notices; after dataiku, which shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cashflow
import forecasters
import models
import tracking

for module in (cashflow, models, tracking, forecasters):     # pick up changes to the project library without a restart
    importlib.reload(module)
from cashflow import FORECAST_COLUMNS, HORIZONS, TOTAL, cutoff_month, monthly_matrix, monthly_series, read_dataset
from forecasters import RecentAverage6
from models import RecentAverage, lab_compare

pd.set_option("display.max_columns", 30)
cutoff = cutoff_month()
daily = read_dataset("daily_flows")          # interface 1: one row per client, day and category
clients = read_dataset("clients")            # one row per business client
monthly = monthly_series(daily, cutoff)      # the net cash flow per client, category (and TOTAL) and month
print(f"daily_flows: {len(daily):,} rows, {daily['client_no'].nunique()} clients, {daily['category'].nunique()} "
      f"categories, {pd.to_datetime(daily['day']).min():%Y-%m-%d} to {pd.to_datetime(daily['day']).max():%Y-%m-%d}; "
      f"cut-off month {cutoff:%Y-%m}")
