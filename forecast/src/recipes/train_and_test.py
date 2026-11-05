# Train and test one model, with its own model ID, tracked in Experiment Tracking with its forecaster, settings and
# scores. The forecaster is the one the project variable forecaster names (default catboost, in models.py; the data
# scientist's own are in forecasters.py).
#   backtest: trained without the last 6 months, then tested on them (test_results)
#   final: trained on all rows, deployed as version <model ID> of the saved models forecast_ev, forecast_p10 and
#   forecast_p90 (not active: the promotion activates it). model_reference: a sample of the forecaster's inputs,
#   evaluated with each version so the version knows its input columns.
import dataiku

from cashflow import cutoff_month, read_dataset
from models import forecast, holdout, new_forecaster, run_metrics, test_results
from tracking import log_model, new_model_id

cutoff = cutoff_month()
rows = read_dataset("training_rows")
test = holdout(rows)
backtest, final = new_forecaster(), new_forecaster()
backtest.fit(rows.drop(test.index))
final.fit(rows)
model_id = new_model_id(cutoff)

results = test_results(test.join(forecast(backtest, test)))
results.insert(0, "model_id", model_id)
metrics = run_metrics(results)
inputs = list(final.inputs) + ([] if "category" in final.inputs else ["category"])
params = {"forecaster": final.name, **getattr(final, "params", {}), "cutoff_month": f"{cutoff:%Y-%m}",
          "load_id": dataiku.get_custom_variables().get("load_id", ""), "training_rows": len(rows),
          "test_rows": len(test), "clients": rows["client_no"].nunique(), "inputs": ",".join(inputs)}
reference = rows[inputs + ["target"]].sample(min(len(rows), 20000), random_state=0)
dataiku.Dataset("model_reference").write_with_schema(reference.reset_index(drop=True))
log_model(model_id, params, metrics, backtest, final, "model_reference")
print(f"Model {model_id} ({final.name}): EV error {metrics['wape']:.1%}, coverage {metrics['coverage_p10_p90']:.1%}")

dataiku.Dataset("test_results").write_with_schema(results)
