# Your forecaster: change anything (its settings, its inputs, or fit and predict written from scratch). To try one
# from forecasters.py instead: import forecasters, importlib.reload(forecasters), then use its class below.
class MyForecaster(CatBoostForecaster):
    """Example: the reference CatBoost forecaster with deeper trees."""
    name = "my_forecaster"
    params = {**CatBoostForecaster.params, "depth": 8}


table, decision, reason, run_id = lab_try(MyForecaster())
print("EV error (lower is better) and P10-P90 coverage (aim: 80%), per category, all months ahead together:")
display(table)
print(f"Champion / challenger would decide: {decision} ({reason})")
print("Recorded in Experiment Tracking, experiment cashflow_lab, run", run_id)
