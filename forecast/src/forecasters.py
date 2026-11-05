"""The data scientist's forecasters (project library: python/forecasters.py). The setup notebook creates this file once
and never overwrites it: it is yours.

Try a forecaster first: in the notebook forecast_lab, or set the project variable lab_forecaster to its name and
build lab_results (Lab zone). To put one in production:
    1. add its class here and to FORECASTERS below,
    2. set the project variable forecaster to its name,
    3. run the scenario Retrain, test and promote: champion / challenger promotes it only if it beats the production
       model on the same last 6 months. To go back, set forecaster to catboost.

A forecaster is a class with (see models.py, and CatBoostForecaster there):
    name            the value of the project variable forecaster that selects it
    params          its settings (a dict), recorded in Experiment Tracking
    inputs          the model-row columns it reads (no dates)
    fit(rows)       learns from training rows (column target: the amount to predict, in SAR)
    predict(rows)   a DataFrame with the index of rows and the columns p10, ev, p90 (SAR)
"""
from models import CatBoostForecaster


class DeeperCatBoost(CatBoostForecaster):
    """Example: the reference CatBoost forecaster with deeper trees."""
    name = "catboost_deeper"
    params = {**CatBoostForecaster.params, "depth": 8}


FORECASTERS = [DeeperCatBoost]
