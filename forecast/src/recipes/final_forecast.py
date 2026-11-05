# Final forecast: the production model's prediction (the three Score recipes: EV, P10, P90) plus the obligations due
# that month (already scheduled, so the same amount is added to P10, EV and P90). Financing instalments and deposit
# maturities are obligations only. "NET" is the client total (money in minus money out) with its own P10, EV, P90.
# Every row says which model (model_id, the active version of the saved models), when it was promoted
# (promotion_log), and which load produced it.
import dataiku

from cashflow import cutoff_month, forecast_table, read_dataset
from tracking import production_model_id

keys = ["client_no", "category", "months_ahead"]
pred = read_dataset("scored_ev").rename(columns={"prediction": "ev"})
for q in ("p10", "p90"):
    pred = pred.merge(read_dataset(f"scored_{q}")[keys + ["prediction"]].rename(columns={"prediction": q}), on=keys)
frame = forecast_table(pred, read_dataset("monthly_obligations"), cutoff_month())
frame["model_id"] = production_model_id()
log = read_dataset("promotion_log")
promoted = log[(log["decision"] == "PROMOTE") & (log["candidate_model_id"] == frame["model_id"].iloc[0])]
frame["model_promoted_at"] = promoted["decided_at"].max() if not promoted.empty else None
frame["load_id"] = dataiku.get_custom_variables().get("load_id", "")

cols = ["client_no", "cutoff_month", "months_ahead", "target_month", "category", "category_name", "direction",
        "p10", "ev", "p90", "obligations", "model_p10", "model_ev", "model_p90", "history_months", "status",
        "model_id", "model_promoted_at", "load_id"]
dataiku.Dataset("final_forecast").write_with_schema(frame[cols].round(2))
