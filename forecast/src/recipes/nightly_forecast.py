# Nightly forecast: the production model (the active version of the saved model cash_forecast) forecasts, for each
# active business client, the net cash flow (money in minus money out) of each of the next 3 windows of 30 days after
# the cut-off day (yesterday): days 1-30, 31-60 and 61-90, per category and in TOTAL, as P10, EV and P90 in SAR, from
# the client's daily flows of the last 365 days (interface 1). Inference reads nothing else from training.
# forecast.forecast_all: the groups of clients of daily_flows in parallel (project variable inference_workers), each
# read, prepared, forecast and written on its own into the folder forecast: one row per client (status OK,
# NOT_ENOUGH_HISTORY or NO_POSTINGS, the values as columns TOTAL_w1_p10, ...), one subfolder per night,
# forecast_<cut-off>_<model ID>, which becomes the current one once every group is written. The next step of the
# Nightly batch loads it into the SQL table final_forecast (recipe load_forecast_sql).
# inference_log: one row per run (appended), also when the run fails (status FAILED and the error): which model and
# load, how many active clients got a forecast or not and why, the seconds of each step (added up over the workers)
# and in all, the most memory used (all the workers together).
import time

import dataiku
import pandas as pd

from cashflow import append_row, cutoff_date, input_days, read_dataset, whole_number
from forecast import STEPS, forecast_all, production_forecaster

LOG = {"run_at": "string", "status": "string", "error": "string", "load_id": "string", "cutoff_date": "string",
       "model_id": "string", "forecaster": "string", "input_days": "bigint", "workers": "bigint", "groups": "bigint",
       "active_clients": "bigint", "clients_ok": "bigint", "clients_not_enough_history": "bigint",
       "clients_no_postings": "bigint", "daily_rows_read": "bigint",
       **{f"seconds_{k}": "double" for k in STEPS}, "seconds_total": "double", "peak_memory_mb": "bigint",
       "output": "string"}
started = time.time()
variables = dataiku.get_custom_variables()
log = {"run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "load_id": variables.get("load_id", ""),
       "cutoff_date": variables.get("cutoff_date", "")}


def run():
    model, model_id = production_forecaster()
    if not hasattr(model, "predict_rows"):
        raise ValueError(f"The production model {model_id} has no predict_rows (models.py): run the scenario Retrain, "
                         "test and promote")
    clients = read_dataset("clients")
    clients = clients[clients["status"] == "ACTIVE"]
    if clients.empty:
        raise ValueError("No active client in clients: check the staging and STS_FLG in warehouse.py")
    name = getattr(model, "name", type(model).__name__)
    log.update(model_id=model_id, forecaster=name, input_days=input_days(model), active_clients=len(clients))
    workers = whole_number("inference_workers", variables.get("inference_workers")) or 4
    done = forecast_all(model, model_id, clients, cutoff_date(), workers,
                        columns={"forecaster": name, "load_id": variables.get("load_id", "")})
    log.update(workers=done["workers"], groups=done["groups"], daily_rows_read=done["daily_rows"],
               clients_ok=done.get("OK", 0), clients_not_enough_history=done.get("NOT_ENOUGH_HISTORY", 0),
               clients_no_postings=done.get("NO_POSTINGS", 0), peak_memory_mb=done["peak_memory_mb"],
               output=done["output"], **{f"seconds_{k}": round(done[k], 1) for k in STEPS})


def write_log(status, error=""):
    log.update(status=status, error=error[:1000], seconds_total=round(time.time() - started, 1))
    append_row("inference_log", LOG, log)


try:
    run()
except Exception as e:
    write_log("FAILED", f"{type(e).__name__}: {e}")
    raise
write_log("SUCCESS")
print(f"Forecast with {log['model_id']} ({log['forecaster']}): {log['active_clients']:,} active clients, "
      f"{log['groups']} groups, {log['workers']} workers: {log['clients_ok']:,} OK, {log['clients_not_enough_history']:,} "
      f"without enough history, {log['clients_no_postings']:,} without postings; {log['seconds_total']:.0f} s; "
      f"folder forecast, {log['output']}")
