# Visual predictions: the three visual models' predictions (EV, P10, P90) back in SAR, in the same shape as the
# code models' predictions.
import dataiku

from cashflow import finish, read_dataset

keys = ["client_no", "category", "cutoff_month", "months_ahead"]
rows = read_dataset("scored_ev_visual")[keys + ["direction", "target_month", "series_scale", "prediction"]]
rows = rows.rename(columns={"prediction": "ev"})
for q in ("p10", "p90"):
    scored = read_dataset(f"scored_{q}_visual")[keys + ["prediction"]].rename(columns={"prediction": q})
    rows = rows.merge(scored, on=keys)
scale = rows["series_scale"]
rows[["p10", "ev", "p90"]] = finish(rows["ev"] * scale, rows["p10"] * scale, rows["p90"] * scale, rows["category"],
                                    rows.index)
cols = ["client_no", "category", "direction", "cutoff_month", "months_ahead", "target_month", "p10", "ev", "p90"]
dataiku.Dataset("predictions_visual").write_with_schema(rows[cols])
