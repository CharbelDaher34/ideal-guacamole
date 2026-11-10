"""The data scientist's forecasters (project library: python/forecasters.py). The setup notebook creates this file once
and never overwrites it: it is yours.

Try a forecaster first in the notebook forecast_lab (it never touches production). To put one in production:
    1. add its class here and to FORECASTERS below,
    2. set the project variable forecaster to its name,
    3. run the scenario Retrain, test and promote: champion / challenger promotes it only if it passes the promotion
       rules (project variables promotion_*) on the same last 6 periods of 30 days. To go back, set forecaster to daily_catboost.

A model is a class built on models.DayModel with (details in models.py, and the examples DailyCatBoost and
RecentAverage there):
    name            the value of the project variable forecaster that selects it
    params          its settings (a dict), recorded in Experiment Tracking; input_days: the days it gets (365)
    fit(daily, clients, cutoff)
                    learns from the daily flows (client_no, day, category, money_in, money_out, postings) up to the
                    cut-off day (yesterday), and the clients
    predict_rows(rows)
                    rows: one row per client, category (and TOTAL) and window (1, 2, 3: days 1-30, 31-60, 61-90 after
                    the cut-off day) with day_365 ... day_1 (Polars, cashflow.model_rows); returns P10, EV, P90 per row
                    (numpy, the net cash flow of that window, SAR)
"""
from models import RecentAverage


class RecentAverage6(RecentAverage):
    """Example: recent_average (models.py) with the average of the last 6 periods of 30 days instead of 3."""
    name = "recent_average_6m"
    params = {**RecentAverage.params, "periods": 6}


FORECASTERS = [RecentAverage6]
