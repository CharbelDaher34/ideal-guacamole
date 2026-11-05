# Lab (data scientist): try a forecaster against the production model without touching production. The forecaster is
# the one the project variable lab_forecaster names (models.py or forecasters.py). It is trained without the last 6
# months and tested on them, the way champion / challenger tests a new model. The try is recorded in Experiment
# Tracking (experiment cashflow_lab, folder lab_experiments) and added to lab_results: per category, its EV error and
# P10-P90 coverage next to the production model's and the simple guess, and the decision champion / challenger would
# take. Nothing here changes the saved models. Run it by building lab_results (only built when asked by name).
import dataiku
import pandas as pd

from cashflow import cutoff_month
from models import lab_try, new_forecaster

forecaster = new_forecaster(dataiku.get_custom_variables().get("lab_forecaster") or "catboost")
table, decision, reason, run_id = lab_try(forecaster)
print(f"{forecaster.name}: champion / challenger would decide {decision} ({reason})")

table = table.reset_index().assign(tried_at=pd.Timestamp.now().floor("s"), forecaster=forecaster.name,
                                   cutoff_month=f"{cutoff_month():%Y-%m}", decision=decision, reason=reason,
                                   lab_run_id=run_id)
cols = ["tried_at", "forecaster", "cutoff_month", "category_name", "simple_guess_error", "production_error",
        "candidate_error", "production_coverage", "candidate_coverage", "decision", "reason", "lab_run_id"]
dataiku.Dataset("lab_results").write_with_schema(table[cols])
