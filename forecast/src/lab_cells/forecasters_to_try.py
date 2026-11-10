class DailyCatBoostDeeper(DailyCatBoost):
    """An example to change: the production placeholder (daily_catboost, models.py) with deeper and more trees. Same
    inputs (the last 365 days, the category, the window), same single model for P10, EV and P90."""
    name = "daily_catboost_deeper"
    params = {**DailyCatBoost.params, "depth": 8, "iterations": 600}
