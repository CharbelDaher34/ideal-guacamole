import importlib

try:
    import catboost
    import mlflow
except ImportError:
    raise RuntimeError("Use the kernel of the code env cashflow_forecast (Kernel > Change kernel). If it is not in the "
                       "list, ask whoever manages that code env to enable its Jupyter support; meanwhile, set the "
                       "project variable lab_forecaster and build lab_results (Lab zone).")
import cashflow
import models
import tracking

for module in (cashflow, models, tracking):     # pick up changes to the project library without a restart
    importlib.reload(module)
from models import CatBoostForecaster, lab_try
