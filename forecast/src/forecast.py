"""Inference (project library: python/forecast.py): the production model's ID and the final forecast table.
Inference reads nothing from training except the active version of the saved models (interface 3,
cashflow.SAVED_MODELS)."""
import numpy as np
import pandas as pd

from cashflow import HORIZONS, SAVED_MODELS, finish


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


def forecast_table(pred, last_month):
    """The forecast: one row per client and month ahead, the net cash flow (money in minus money out) as P10, EV and
    P90 in SAR. pred: the model rows of the cut-off with the model's ev, p10, p90. A client with less than 12 months
    of history gets no forecast (status NOT_ENOUGH_HISTORY)."""
    frame = pred[["client_no", "months_ahead", "history_months", "enough_history", "p10", "ev", "p90"]].copy()
    frame[["p10", "ev", "p90"]] = finish(frame["ev"], frame["p10"], frame["p90"], frame.index)
    enough = frame["enough_history"].astype(bool)
    frame.loc[~enough, ["p10", "ev", "p90"]] = np.nan
    frame["cutoff_month"] = last_month
    frame["target_month"] = frame["months_ahead"].apply(lambda h: last_month + pd.offsets.DateOffset(months=h))
    frame["status"] = np.where(enough, "OK", "NOT_ENOUGH_HISTORY")
    return frame.sort_values(["client_no", "months_ahead"]).reset_index(drop=True)
