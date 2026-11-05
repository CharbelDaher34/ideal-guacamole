# Final forecast: the production model's prediction (the three Score recipes: EV, P10, P90) plus the obligations due
# that month (already scheduled, so the same amount is added to P10, EV and P90). Financing instalments and deposit
# maturities are obligations only. "NET" is the client total (money in minus money out) with its own P10, EV, P90.
# Every row says which model (model_id, the active version of the saved models), when it was promoted
# (promotion_log), and which load produced it.
import dataiku
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, HORIZONS, cutoff_month, finish, month_start, read_dataset
from tracking import production_model_id

last_month = cutoff_month()
keys = ["client_no", "category", "months_ahead"]
pred = read_dataset("scored_ev").rename(columns={"prediction": "ev"})
for q in ("p10", "p90"):
    pred = pred.merge(read_dataset(f"scored_{q}")[keys + ["prediction"]].rename(columns={"prediction": q}), on=keys)
pred[["p10", "ev", "p90"]] = finish(pred["ev"], pred["p10"], pred["p90"], pred["category"], pred.index)
ok = pred["enough_history"].astype(bool) & pred["active"].astype(bool)
pred.loc[~ok, ["p10", "ev", "p90"]] = np.nan
pred.loc[pred["enough_history"].astype(bool) & ~pred["active"].astype(bool), ["p10", "ev", "p90"]] = 0.0

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

model = pred[keys + ["p10", "ev", "p90"]].rename(columns={"p10": "model_p10", "ev": "model_ev", "p90": "model_p90"})
frame = frame.merge(model, on=keys, how="left")
enough = frame["enough_history"].astype(bool)
for q in ("p10", "ev", "p90"):
    # no model value: a category fully scheduled, or one the client did not use in the last 12 months
    frame.loc[enough & frame[f"model_{q}"].isna(), f"model_{q}"] = 0.0
    frame[q] = frame[f"model_{q}"] + frame["obligations"]
frame.loc[~enough, ["p10", "ev", "p90", "model_p10", "model_ev", "model_p90"]] = np.nan
frame["status"] = np.where(enough, "OK", "NOT_ENOUGH_HISTORY")
frame["model_id"] = production_model_id()
log = read_dataset("promotion_log")
promoted = log[(log["decision"] == "PROMOTE") & (log["candidate_model_id"] == frame["model_id"].iloc[0])]
frame["model_promoted_at"] = promoted["decided_at"].max() if not promoted.empty else None
frame["load_id"] = dataiku.get_custom_variables().get("load_id", "")

cols = ["client_no", "cutoff_month", "months_ahead", "target_month", "category", "category_name", "direction",
        "p10", "ev", "p90", "obligations", "model_p10", "model_ev", "model_p90", "history_months", "status",
        "model_id", "model_promoted_at", "load_id"]
order = {c: i for i, c in enumerate(CATEGORIES)}
frame = frame.sort_values(["client_no", "months_ahead", "category"],
                          key=lambda s: s.map(order) if s.name == "category" else s)
dataiku.Dataset("final_forecast").write_with_schema(frame[cols].round(2).reset_index(drop=True))
