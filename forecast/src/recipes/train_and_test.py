# Train and test one model, with its own model ID, tracked in Experiment Tracking with its forecaster, settings and
# scores. The forecaster is the one the project variable forecaster names (default daily_catboost, in models.py; the
# data scientist's own are in forecasters.py). It learns from the daily flows and the clients (interface 1).
#   backtest: trained without the last 6 periods of 30 days, then tested on them (models.backtest, test_results)
#   final: trained on everything up to the cut-off, deployed as version <model ID> of the saved model cash_forecast
#   (not active: the promotion activates it)
import time

import dataiku
import pandas as pd

from cashflow import MIN_TRAINING_MONTHS, cutoff_date, read_daily, read_dataset, until
from models import backtest, new_forecaster, run_metrics, test_results
from tracking import log_model, new_model_id

started = time.time()


def step(what):
    """A progress line in the job's log (the setup notebook shows it while it waits)."""
    print(f"PROGRESS training: {what} ({(time.time() - started) / 60:.1f} min so far)", flush=True)


cutoff = cutoff_date()
step("reading daily_flows and clients")
daily, clients = until(read_daily(), cutoff), read_dataset("clients")
first_day = pd.to_datetime(daily["day"]).min() if len(daily) else cutoff
if first_day > cutoff - pd.DateOffset(months=MIN_TRAINING_MONTHS) + pd.Timedelta(days=1):
    raise ValueError(f"Not enough history to train and test a model: daily_flows starts on {first_day:%Y-%m-%d}, and "
                     f"a model needs at least {MIN_TRAINING_MONTHS} months up to the cut-off day ({cutoff:%Y-%m-%d}). "
                     "Run the setup again with more HISTORY_MONTHS, then the scenario Full rebuild")
backtested, final = new_forecaster(), new_forecaster()
step(f"1/2 test: training without the last periods, then testing on them ({len(daily):,} daily rows)")
tested = backtest(backtested, daily, clients, cutoff)
step("2/2 training the final model on everything up to the cut-off")
final.fit(daily, clients, cutoff)
step("saving and deploying the model")
model_id = new_model_id(cutoff)

results = test_results(tested)
results.insert(0, "model_id", model_id)
metrics = run_metrics(results)
params = {"forecaster": final.name, **getattr(final, "params", {}), "cutoff_date": f"{cutoff:%Y-%m-%d}",
          "load_id": dataiku.get_custom_variables().get("load_id", ""), "daily_rows": len(daily),
          "clients": daily["client_no"].nunique(), "test_rows": len(tested)}
log_model(model_id, params, metrics, backtested, final)
print(f"Model {model_id} ({final.name}): EV error {metrics['wape']:.1%} (simple guess {metrics['wape_baseline']:.1%}), "
      f"coverage {metrics['coverage_p10_p90']:.1%}")

dataiku.Dataset("test_results").write_with_schema(results)
