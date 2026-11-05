# Final forecast: the production model's prediction (the three Score recipes: EV, P10, P90, with the active version of
# the saved models): each client's net cash flow (money in minus money out) 1, 2 and 3 months ahead. Every row says
# which model (model_id: the active version) and which data (load_id) produced it. Inference reads nothing else from
# training.
import dataiku

from cashflow import cutoff_month, read_dataset
from forecast import active_model_id, forecast_table

keys = ["client_no", "months_ahead"]
pred = read_dataset("scored_ev").rename(columns={"prediction": "ev"})
for q in ("p10", "p90"):
    pred = pred.merge(read_dataset(f"scored_{q}")[keys + ["prediction"]].rename(columns={"prediction": q}), on=keys)
frame = forecast_table(pred, cutoff_month())
frame["model_id"] = active_model_id()
frame["load_id"] = dataiku.get_custom_variables().get("load_id", "")

cols = ["client_no", "cutoff_month", "months_ahead", "target_month", "p10", "ev", "p90", "history_months", "status",
        "model_id", "load_id"]
dataiku.Dataset("final_forecast").write_with_schema(frame[cols].round(2))
