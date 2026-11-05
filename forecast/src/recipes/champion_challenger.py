# Champion / challenger: the latest trained model (candidate) against the production model, both tested on the
# same last 6 months with their backtest models (trained without those months). If there is no production model
# yet, the candidate is promoted (the forecast needs one); its checks are noted in the reason. Otherwise it is
# promoted if
#   - its EV error is lower than the production model's (a tie keeps the production model),
#   - its P10-P90 coverage is between 70% and 90%,
#   - and it beats the simple guess (average of the last 3 months).
# The production model is the active version of the saved models. The decision is added to promotion_log; the next
# scenario step applies it (activates the candidate's version).
import dataiku
import pandas as pd

from cashflow import holdout, predict, read_dataset, scores
from tracking import MAX_COVERAGE, MIN_COVERAGE, latest_model_id, load_models, production_model_id

test = holdout(read_dataset("training_rows"))
test = test[test["category"] != "NET"]
candidate = latest_model_id()
production = production_model_id()
tested = {m: scores(test.join(predict(load_models(m, "backtest"), test)))
          for m in {candidate, production} if m}
c = tested[candidate]
p = tested.get(production) if production and production != candidate else None

checks = []
if not MIN_COVERAGE <= c["coverage_p10_p90"] <= MAX_COVERAGE:
    checks.append(f"coverage outside {MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%}")
if c["wape_model"] >= c["wape_baseline"]:
    checks.append(f"does not beat the simple guess ({c['wape_baseline']:.1%})")

if production == candidate:
    decision, reason = "KEEP", "the candidate is already the production model"
elif not production:
    decision = "PROMOTE"
    reason = (f"no production model yet, EV error {c['wape_model']:.2%}, coverage {c['coverage_p10_p90']:.1%}"
              + (f" (warning: {'; '.join(checks)})" if checks else ""))
elif not MIN_COVERAGE <= c["coverage_p10_p90"] <= MAX_COVERAGE:
    decision, reason = "KEEP", f"candidate coverage {c['coverage_p10_p90']:.1%} outside {MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%}"
elif c["wape_model"] >= c["wape_baseline"]:
    decision, reason = "KEEP", f"candidate does not beat the simple guess ({c['wape_model']:.1%} vs {c['wape_baseline']:.1%})"
elif p is not None and c["wape_model"] >= p["wape_model"]:
    decision, reason = "KEEP", f"candidate EV error {c['wape_model']:.2%} not lower than production {p['wape_model']:.2%}"
else:
    decision = "PROMOTE"
    reason = f"candidate EV error {c['wape_model']:.2%} vs production {p['wape_model']:.2%}, coverage {c['coverage_p10_p90']:.1%}"
print(decision, "-", reason)
dataiku.Dataset("promotion_log").write_with_schema(pd.DataFrame([{
    "decided_at": pd.Timestamp.now().floor("s"), "cutoff_month": dataiku.get_custom_variables()["cutoff_month"],
    "candidate_model_id": candidate, "production_model_id": production or None,
    "candidate_wape": c["wape_model"], "production_wape": p["wape_model"] if p is not None else None,
    "baseline_wape": c["wape_baseline"], "candidate_coverage": c["coverage_p10_p90"],
    "production_coverage": p["coverage_p10_p90"] if p is not None else None,
    "decision": decision, "reason": reason}]))
