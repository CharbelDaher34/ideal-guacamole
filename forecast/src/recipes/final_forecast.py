# Final forecast: the models' prediction plus the obligations due that month (already scheduled, so the same
# amount is added to P10, EV and P90). Financing instalments and deposit maturities are obligations only.
# "NET" is the client total (money in minus money out) with its own P10, EV and P90.
# The project variable model_source picks the models: "code" (CatBoost recipes) or "visual" (visual ML).
import dataiku
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, HORIZONS, cutoff_month, month_start, read_dataset

last_month = cutoff_month()
source = dataiku.get_custom_variables().get("model_source", "code")
pred = read_dataset("predictions")
pred["target_month"] = month_start(pred["target_month"])
obl = read_dataset("monthly_obligations")
obl["due_month"], obl["known_month"] = month_start(obl["due_month"]), month_start(obl["known_month"])

clients = pred[["client_no", "history_months", "enough_history"]].drop_duplicates("client_no")
frame = clients.merge(pd.DataFrame([(c, h) for c in CATEGORIES for h in HORIZONS], columns=["category", "months_ahead"]),
                      how="cross")
frame["cutoff_month"] = last_month
frame["target_month"] = frame["months_ahead"].apply(lambda h: last_month + pd.offsets.DateOffset(months=h))
frame["direction"] = frame["category"].map({c: v[1] for c, v in CATEGORIES.items()})
frame["category_name"] = frame["category"].map({c: v[0] for c, v in CATEGORIES.items()})

known = obl[obl["known_month"] <= last_month]
by_cat = known.groupby(["client_no", "category", "due_month"], as_index=False)["amount"].sum()
net = (known.assign(amount=np.where(known["direction"] == "IN", 1.0, -1.0) * known["amount"])
       .groupby(["client_no", "due_month"], as_index=False)["amount"].sum().assign(category="NET"))
dues = pd.concat([by_cat, net], ignore_index=True).rename(columns={"due_month": "target_month", "amount": "obligations"})
frame = frame.merge(dues, on=["client_no", "category", "target_month"], how="left")
frame["obligations"] = frame["obligations"].fillna(0.0)

model = pred if source == "code" else read_dataset("predictions_visual")
model = model[["client_no", "category", "months_ahead", "p10", "ev", "p90"]].rename(
    columns={"p10": "model_p10", "ev": "model_ev", "p90": "model_p90"})
frame = frame.merge(model, on=["client_no", "category", "months_ahead"], how="left")
enough = frame["enough_history"].astype(bool)
for q in ("p10", "ev", "p90"):
    # no model value: a category fully scheduled, or one the client did not use in the last 12 months
    frame.loc[enough & frame[f"model_{q}"].isna(), f"model_{q}"] = 0.0
    frame[q] = frame[f"model_{q}"] + frame["obligations"]
frame.loc[~enough, ["p10", "ev", "p90", "model_p10", "model_ev", "model_p90"]] = np.nan
frame["status"] = np.where(enough, "OK", "NOT_ENOUGH_HISTORY")
frame["model_source"] = source

cols = ["client_no", "cutoff_month", "months_ahead", "target_month", "category", "category_name", "direction",
        "p10", "ev", "p90", "obligations", "model_p10", "model_ev", "model_p90", "history_months", "status",
        "model_source"]
order = {c: i for i, c in enumerate(CATEGORIES)}
frame = frame.sort_values(["client_no", "months_ahead"]).sort_values(
    ["client_no", "months_ahead", "category"], key=lambda s: s.map(order) if s.name == "category" else s)
dataiku.Dataset("final_forecast").write_with_schema(frame[cols].round(2).reset_index(drop=True))
