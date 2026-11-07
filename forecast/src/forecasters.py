"""The data scientist's forecasters (project library: python/forecasters.py). The setup notebook creates this file once
and never overwrites it: it is yours.

Try a forecaster first: in the notebook forecast_lab, or set the project variable lab_forecaster to its name and
build lab_results (Lab zone). To put one in production:
    1. add its class here and to FORECASTERS below,
    2. set the project variable forecaster to its name,
    3. run the scenario Retrain, test and promote: champion / challenger promotes it only if it passes the promotion
       rules (project variables promotion_*) on the same last 6 months. To go back, set forecaster to recent_average.

A forecaster is a class with (details in models.py, and RecentAverage there):
    name            the value of the project variable forecaster that selects it
    params          its settings (a dict), recorded in Experiment Tracking
    fit(daily, clients, cutoff)
                    learns from the daily flows (client_no, day, category, money_in, money_out, postings) up to the
                    end of the cut-off month, and the clients
    predict(daily, clients, cutoff)
                    one row per client, category (and TOTAL) and months ahead (1, 2, 3): client_no, category,
                    months_ahead, p10, ev, p90 (the net cash flow of that month, SAR)
cashflow.monthly_matrix(daily, cutoff) gives the net cash flow per client and category (and TOTAL) and month.
"""
from models import RecentAverage


class RecentAverage6(RecentAverage):
    """Example: the placeholder with the average of the last 6 months instead of 3."""
    name = "recent_average_6m"
    params = {**RecentAverage.params, "months": 6}


FORECASTERS = [RecentAverage6]
