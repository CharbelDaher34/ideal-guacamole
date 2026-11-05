# Train each on the same months, test it on the last 6, next to the production model. catboost is the production
# recipe trained again: it should match the production model's scores (a check that the test is the same).
RECORD = True       # record each try in Experiment Tracking (experiment cashflow_lab)
result = lab_compare([SimpleGuess(), CatBoostForecaster(), CatBoostDepth8(), SklearnBoosting()], record=RECORD)
print("All categories, all months ahead. EV error: lower is better. P10-P90 coverage: aim 80%.")
display(result["board"])
print("EV error per category:")
display(result["errors"])
print("P10-P90 coverage per category:")
display(result["coverage"])
print("EV error per months ahead:")
display(result["by_months_ahead"])
