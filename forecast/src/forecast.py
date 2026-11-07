"""Inference (project library: python/forecast.py): the production model and the final forecast table. Inference reads
nothing from training except the active version of the saved model cash_forecast (interface 3, cashflow.SAVED_MODEL).
"""
import numpy as np
import pandas as pd

from cashflow import HORIZONS, MIN_HISTORY_MONTHS, SAVED_MODEL, TOTAL, add_months

VALUES = ("p10", "ev", "p90")


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


def value_columns():
    """The value columns of the final forecast, always the same and in this order: TOTAL first, then each category of
    warehouse.py (CATEGORY_OF_TYPE and OTHER), each for months ahead 1, 2, 3 and P10, EV, P90:
    TOTAL_m1_p10, TOTAL_m1_ev, TOTAL_m1_p90, TOTAL_m2_p10, ..., UTILITIES_m3_p90."""
    from warehouse import CATEGORY_OF_TYPE, OTHER
    categories = [TOTAL] + sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER})
    return [f"{c}_m{h}_{v}" for c in categories for h in HORIZONS for v in VALUES]


def one_row_per_client(table, last_month, no_postings=()):
    """The final forecast: ONE row per client. table: forecast_table (one row per client, category and month ahead).
    Columns: client_no, cutoff_month, m1_month, m2_month, m3_month (the months forecast), status, history_months,
    then value_columns() (empty for a category the client does not have, and for a client without enough history).
    no_postings: active clients with no posting at all, added with status NO_POSTINGS and no values."""
    columns = value_columns()
    keys = ["client_no", "category", "months_ahead"]
    values = pd.DataFrame(index=pd.Index([], name="client_no"))
    if len(table):
        values = table.set_index(keys)[list(VALUES)].unstack(["category", "months_ahead"])
        values.columns = [f"{c}_m{h}_{v}" for v, c, h in values.columns]
    unknown = sorted(set(values.columns) - set(columns))
    if unknown:
        raise ValueError(f"Forecast for categories not in warehouse.py: {', '.join(sorted({u.split('_m')[0] for u in unknown}))}")
    info = table.groupby("client_no")[["status", "history_months"]].first()
    missing = pd.DataFrame({"status": "NO_POSTINGS", "history_months": 0},
                           index=pd.Index(sorted(no_postings), name="client_no"))
    info = pd.concat([info, missing])
    info["history_months"] = pd.to_numeric(info["history_months"]).fillna(0).astype(int)
    months = pd.DataFrame({"cutoff_month": last_month, **{f"m{h}_month": add_months(last_month, h) for h in HORIZONS}},
                          index=info.index)
    return pd.concat([months, info, values.reindex(index=info.index, columns=columns).astype(float)],
                     axis=1).reset_index()
