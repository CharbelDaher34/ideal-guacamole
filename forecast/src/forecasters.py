"""The data scientist's forecasters (project library: python/forecasters.py). The setup notebook creates this file once
and never overwrites it: it is yours.

Try a forecaster first in the notebook forecast_lab (it never touches production). To put one in production:
    1. add its class here and to FORECASTERS below,
    2. set the project variable forecaster to its name,
    3. run the scenario Retrain, test and promote: champion / challenger promotes it only if it passes the promotion
       rules (project variables promotion_*) on the same last 6 periods of 30 days. To go back, set forecaster to quantile_boosting.

A forecaster is a class with (details in models.py, and the examples QuantileBoosting and RecentAverage there):
    name            the value of the project variable forecaster that selects it
    params          its settings (a dict), recorded in Experiment Tracking
    fit(daily, clients, cutoff)
                    learns from the daily flows (client_no, day, category, money_in, money_out, postings) up to the
                    cut-off day (yesterday), and the clients
    predict(daily, clients, cutoff)
                    one row per client, category (and TOTAL) and window (1, 2, 3: days 1-30, 31-60, 61-90 after
                    the cut-off day): client_no, category, window, p10, ev, p90 (the net cash flow of that window, SAR)
cashflow.period_matrix(daily, cutoff) gives the net cash flow per client and category (and TOTAL) and 30-day period
ending at the cut-off day.
"""
from models import RecentAverage


class RecentAverage6(RecentAverage):
    """Example: recent_average (models.py) with the average of the last 6 periods of 30 days instead of 3."""
    name = "recent_average_6m"
    params = {**RecentAverage.params, "periods": 6}


FORECASTERS = [RecentAverage6]
