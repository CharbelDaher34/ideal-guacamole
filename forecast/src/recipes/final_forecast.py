# Final forecast: the production model (the active version of the saved model cash_forecast) forecasts, for each
# active business client, the net cash flow (money in minus money out) of each of the next 3 months, per category and
# in TOTAL, as P10, EV and P90 in SAR, from the client's daily flows (interface 1). Every row says which model
# (model_id: the active version) and which data (load_id) produced it. Inference reads nothing else from training.
import dataiku

from cashflow import cutoff_month, history_months, read_dataset, until
from forecast import forecast_table, production_forecaster
from models import active, forecast

last_month = cutoff_month()
forecaster, model_id = production_forecaster()
daily, clients = active(until(read_dataset("daily_flows"), last_month), read_dataset("clients"))
pred = forecast(forecaster, daily, clients, last_month)
frame = forecast_table(pred, history_months(daily, last_month), last_month)
frame["model_id"] = model_id
frame["load_id"] = dataiku.get_custom_variables().get("load_id", "")

cols = ["client_no", "category", "cutoff_month", "months_ahead", "target_month", "p10", "ev", "p90", "history_months",
        "status", "model_id", "load_id"]
dataiku.Dataset("final_forecast").write_with_schema(frame[cols].round(2))
print(f"Forecast with {model_id}: {frame['client_no'].nunique():,} clients, {len(frame):,} rows "
      f"({(frame['status'] != 'OK').sum():,} without enough history)")
