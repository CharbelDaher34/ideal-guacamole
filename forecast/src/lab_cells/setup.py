import calendar
import importlib
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google")      # third-party notices on import
logging.getLogger("matplotlib").setLevel(logging.ERROR)
try:
    import catboost
    import mlflow
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
import models
import tracking

for module in (cashflow, models, tracking):     # pick up changes to the project library without a restart
    importlib.reload(module)
from cashflow import CATEGORIES, read_dataset
from models import CAT_FEATURES, CatBoostForecaster, lab_compare, model_matrix

pd.set_option("display.max_columns", 30)
names = {code: name for code, (name, _, _) in CATEGORIES.items()}
rows = read_dataset("training_rows")
print(f"training_rows: {len(rows):,} rows, {rows['client_no'].nunique()} clients, cut-off months "
      f"{rows['cutoff_month'].min():%Y-%m} to {rows['cutoff_month'].max():%Y-%m}")
