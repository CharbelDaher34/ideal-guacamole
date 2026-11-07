# Champion / challenger: the latest trained model (candidate) against the production model, both tested on the
# same last 6 periods of 30 days with their backtest forecasters (trained without those periods). The rules are project variables
# (tracking.PROMOTION_RULES, applied by tracking.decide): promoted if its P10-P90 coverage is in the range
# (default 70%-90%), it beats the simple guess (the same as the month before) and its EV error is lower than the
# production model's (by more than promotion_min_gain; a tie keeps the production model). If there is no production
# model to compare with (none yet, or its tracked files were deleted), the candidate is promoted (the forecast needs
# one), with its failed checks as a warning. The production model is the active version of the saved model
# cash_forecast. The decision is added to promotion_log; the next scenario step applies it (activates the
# candidate's version).
import dataiku
import pandas as pd

from cashflow import cutoff_date, read_daily, read_dataset, until
from models import scores, test_forecasts
from tracking import decide, latest_model_id, load_backtest, production_model_id, promotion_rules, runs

cutoff = cutoff_date()
daily, clients = until(read_daily(), cutoff), read_dataset("clients")
candidate = latest_model_id()
production = production_model_id()
tracked = runs()
if production and production not in set(tracked["model_id"]):     # its tracked files were deleted: nothing to test
    print(f"Production model {production} is not in Experiment Tracking: the candidate is judged on its own")
    production = ""
tested = {m: scores(test_forecasts(load_backtest(m), daily, clients, cutoff)) for m in {candidate, production} if m}
c = tested[candidate]
p = tested.get(production) if production and production != candidate else None
if production == candidate:
    decision, reason = "KEEP", "the candidate is already the production model"
else:
    decision, reason = decide(c, p, promotion_rules())
print(decision, "-", reason)

forecaster = dict(zip(tracked["model_id"], tracked["params.forecaster"])) if "params.forecaster" in tracked else {}
dataiku.Dataset("promotion_log").write_with_schema(pd.DataFrame([{
    "decided_at": pd.Timestamp.now().floor("s"), "cutoff_date": dataiku.get_custom_variables()["cutoff_date"],
    "candidate_model_id": candidate, "candidate_forecaster": forecaster.get(candidate),
    "production_model_id": production or None, "production_forecaster": forecaster.get(production),
    "candidate_wape": c["wape_model"], "production_wape": p["wape_model"] if p is not None else None,
    "baseline_wape": c["wape_baseline"], "candidate_coverage": c["coverage_p10_p90"],
    "production_coverage": p["coverage_p10_p90"] if p is not None else None,
    "decision": decision, "reason": reason, "rules": str(promotion_rules())}]))
