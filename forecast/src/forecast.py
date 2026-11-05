"""Inference (project library: python/forecast.py): the production model's ID and the final forecast table.
Inference reads nothing from training except the active version of the saved models (interface 3,
cashflow.SAVED_MODELS)."""
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, HORIZONS, SAVED_MODELS, finish, month_start


def active_model_id():
    """The production model: the active version of the saved models forecast_ev, forecast_p10 and forecast_p90, the
    one the Score recipes use. All three must have the same."""
    import dataiku
    project = dataiku.api_client().get_default_project()
    ids = {m["name"]: m["id"] for m in project.list_saved_models()}
    missing = [name for name in SAVED_MODELS.values() if name not in ids]
    if missing:
        raise ValueError(f"Saved models not found: {', '.join(missing)}. Run the setup notebook.")
    active = {name: (project.get_saved_model(ids[name]).get_active_version() or {}).get("id")
              for name in SAVED_MODELS.values()}
    if not all(active.values()):
        raise ValueError("No production model yet: run the scenario Retrain, test and promote")
    if len(set(active.values())) != 1:
        raise ValueError(f"The saved models do not have the same active version: {active}. Activate the same model "
                         "ID in all three.")
    return next(iter(active.values()))


def forecast_table(pred, obligations, last_month):
    """The forecast: one row per client, category and month ahead. pred: the model rows of the cut-off with the
    model's ev, p10, p90 (SAR); the obligations due that month and known at the cut-off are added to P10, EV and P90
    (already scheduled, so the same amount). Financing instalments and deposit maturities are obligations only. NET
    is the client total (money in minus money out). A client with less than 12 months of history gets no forecast
    (status NOT_ENOUGH_HISTORY)."""
    keys = ["client_no", "category", "months_ahead"]
    pred = pred.copy()
    pred[["p10", "ev", "p90"]] = finish(pred["ev"], pred["p10"], pred["p90"], pred["category"], pred.index)
    ok = pred["enough_history"].astype(bool) & pred["active"].astype(bool)
    pred.loc[~ok, ["p10", "ev", "p90"]] = np.nan
    pred.loc[pred["enough_history"].astype(bool) & ~pred["active"].astype(bool), ["p10", "ev", "p90"]] = 0.0

    obl = obligations.assign(due_month=month_start(obligations["due_month"]),
                             known_month=month_start(obligations["known_month"]))
    clients = pred[["client_no", "history_months", "enough_history"]].drop_duplicates("client_no")
    frame = clients.merge(pd.DataFrame([(c, h) for c in CATEGORIES for h in HORIZONS],
                                       columns=["category", "months_ahead"]), how="cross")
    frame["cutoff_month"] = last_month
    frame["target_month"] = frame["months_ahead"].apply(lambda h: last_month + pd.offsets.DateOffset(months=h))
    frame["direction"] = frame["category"].map({c: v[1] for c, v in CATEGORIES.items()})
    frame["category_name"] = frame["category"].map({c: v[0] for c, v in CATEGORIES.items()})

    known = obl[obl["known_month"] <= last_month]
    by_cat = known.groupby(["client_no", "category", "due_month"], as_index=False)["amount"].sum()
    net = (known.assign(amount=np.where(known["direction"] == "IN", 1.0, -1.0) * known["amount"])
           .groupby(["client_no", "due_month"], as_index=False)["amount"].sum().assign(category="NET"))
    dues = pd.concat([by_cat, net], ignore_index=True).rename(columns={"due_month": "target_month",
                                                                       "amount": "obligations"})
    dues["target_month"] = pd.to_datetime(dues["target_month"])
    frame = frame.merge(dues, on=["client_no", "category", "target_month"], how="left")
    frame["obligations"] = frame["obligations"].fillna(0.0)

    model = pred[keys + ["p10", "ev", "p90"]].rename(columns={"p10": "model_p10", "ev": "model_ev",
                                                              "p90": "model_p90"})
    frame = frame.merge(model, on=keys, how="left")
    enough = frame["enough_history"].astype(bool)
    for q in ("p10", "ev", "p90"):
        # no model value: a category fully scheduled, or one the client did not use in the last 12 months
        frame.loc[enough & frame[f"model_{q}"].isna(), f"model_{q}"] = 0.0
        frame[q] = frame[f"model_{q}"] + frame["obligations"]
    frame.loc[~enough, ["p10", "ev", "p90", "model_p10", "model_ev", "model_p90"]] = np.nan
    frame["status"] = np.where(enough, "OK", "NOT_ENOUGH_HISTORY")
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return frame.sort_values(["client_no", "months_ahead", "category"],
                             key=lambda s: s.map(order) if s.name == "category" else s).reset_index(drop=True)
