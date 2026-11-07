"""Inference (project library: python/forecast.py): the production model and the final forecast table. Inference reads
nothing from training except the active version of the saved model cash_forecast (interface 3, cashflow.SAVED_MODEL).
"""
import numpy as np
import pandas as pd

from cashflow import MIN_HISTORY_MONTHS, SAVED_MODEL, add_months


def _saved_model_id():
    import dataiku
    project = dataiku.api_client().get_default_project()
    ids = {m["name"]: m["id"] for m in project.list_saved_models()}
    if SAVED_MODEL not in ids:
        raise ValueError(f"Saved model {SAVED_MODEL} not found. Run the setup notebook.")
    return ids[SAVED_MODEL]


def active_model_id():
    """The production model: the active version of the saved model cash_forecast."""
    import dataiku
    model = dataiku.api_client().get_default_project().get_saved_model(_saved_model_id())
    model_id = (model.get_active_version() or {}).get("id")
    if not model_id:
        raise ValueError("No production model yet: run the scenario Retrain, test and promote")
    return model_id


def production_forecaster():
    """The production model's trained forecaster (the active version of cash_forecast), and its model ID. The recipe
    that calls it must have the saved model as an input."""
    import dataiku
    model_id = active_model_id()
    predictor = dataiku.Model(_saved_model_id()).get_predictor(model_id)
    return predictor.model.unwrap_python_model().forecaster, model_id


def forecast_table(pred, history, last_month):
    """The forecast: one row per client, category (and TOTAL) and month ahead, the net cash flow (money in minus money
    out) as P10, EV and P90 in SAR. pred: the production model's forecast (models.forecast); history: months of
    history per client (cashflow.history_months). A client with less than MIN_HISTORY_MONTHS months of history gets
    no values (status NOT_ENOUGH_HISTORY)."""
    frame = pred.merge(history.reset_index(), on="client_no", how="left")
    enough = frame["history_months"].fillna(0) >= MIN_HISTORY_MONTHS
    frame.loc[~enough, ["p10", "ev", "p90"]] = np.nan
    frame["cutoff_month"] = last_month
    frame["target_month"] = frame["months_ahead"].map({h: add_months(last_month, h) for h in (1, 2, 3)})
    frame["status"] = np.where(enough, "OK", "NOT_ENOUGH_HISTORY")
    return frame.sort_values(["client_no", "category", "months_ahead"]).reset_index(drop=True)
