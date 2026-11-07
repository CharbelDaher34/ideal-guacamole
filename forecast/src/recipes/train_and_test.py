# Train and test one model, with its own model ID, tracked in Experiment Tracking with its forecaster, settings and
# scores. The forecaster is the one the project variable forecaster names (default quantile_boosting, in models.py; the
# data scientist's own are in forecasters.py). It learns from the daily flows and the clients (interface 1).
#   backtest: trained without the last 6 months, then tested on them (models.backtest, test_results)
#   final: trained on everything up to the cut-off, deployed as version <model ID> of the saved model cash_forecast
#   (not active: the promotion activates it)
import dataiku

from cashflow import cutoff_month, read_dataset, until
from models import backtest, new_forecaster, run_metrics, test_results
from tracking import log_model, new_model_id

cutoff = cutoff_month()
daily, clients = until(read_dataset("daily_flows"), cutoff), read_dataset("clients")
backtested, final = new_forecaster(), new_forecaster()
tested = backtest(backtested, daily, clients, cutoff)
final.fit(daily, clients, cutoff)
model_id = new_model_id(cutoff)

results = test_results(tested)
results.insert(0, "model_id", model_id)
metrics = run_metrics(results)
params = {"forecaster": final.name, **getattr(final, "params", {}), "cutoff_month": f"{cutoff:%Y-%m}",
          "load_id": dataiku.get_custom_variables().get("load_id", ""), "daily_rows": len(daily),
          "clients": daily["client_no"].nunique(), "test_rows": len(tested)}
log_model(model_id, params, metrics, backtested, final)
print(f"Model {model_id} ({final.name}): EV error {metrics['wape']:.1%} (simple guess {metrics['wape_baseline']:.1%}), "
      f"coverage {metrics['coverage_p10_p90']:.1%}")

dataiku.Dataset("test_results").write_with_schema(results)
