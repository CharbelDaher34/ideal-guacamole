"""Inference (project library: python/forecast.py): the production model and the final forecast table. Inference reads
nothing from training except the active version of the saved model cash_forecast (interface 3, cashflow.SAVED_MODEL).
"""
import numpy as np
import pandas as pd

from cashflow import HORIZONS, MIN_HISTORY_DAYS, SAVED_MODEL, TOTAL, window

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


def forecast_table(pred, history, cutoff):
    """The forecast: one row per client, category (and TOTAL) and window (1, 2, 3: days 1-30, 31-60, 61-90 after the
    cut-off day), the net cash flow (money in minus money out) as P10, EV and P90 in SAR. pred: the production model's
    forecast (models.forecast); history: days of history per client (cashflow.history_days). A client with less than
    MIN_HISTORY_DAYS days of history gets no values (status NOT_ENOUGH_HISTORY)."""
    frame = pred.merge(history.reset_index(), on="client_no", how="left")
    enough = frame["history_days"].fillna(0) >= MIN_HISTORY_DAYS
    frame.loc[~enough, ["p10", "ev", "p90"]] = np.nan
    frame["status"] = np.where(enough, "OK", "NOT_ENOUGH_HISTORY")
    return frame.sort_values(["client_no", "category", "window"]).reset_index(drop=True)


def value_columns():
    """The value columns of the final forecast, always the same and in this order: TOTAL first, then each category of
    warehouse.py (CATEGORY_OF_TYPE and OTHER), each for windows 1, 2, 3 (days 1-30, 31-60, 61-90 after the cut-off
    day) and P10, EV, P90: TOTAL_w1_p10, TOTAL_w1_ev, TOTAL_w1_p90, TOTAL_w2_p10, ..., UTILITIES_w3_p90."""
    from warehouse import CATEGORY_OF_TYPE, OTHER
    categories = [TOTAL] + sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER})
    return [f"{c}_w{h}_{v}" for c in categories for h in HORIZONS for v in VALUES]


def one_row_per_client(table, cutoff, no_postings=()):
    """The final forecast: ONE row per client. table: forecast_table (one row per client, category and window).
    Columns: client_no, cutoff_date (the last day of data), w1_start, w1_end, w2_start, w2_end, w3_start, w3_end (the
    days of each window), status, history_days, then value_columns() (empty for a category the client does not have,
    and for a client without enough history). no_postings: active clients with no posting at all, added with status
    NO_POSTINGS and no values."""
    columns = value_columns()
    keys = ["client_no", "category", "window"]
    values = pd.DataFrame(index=pd.Index([], name="client_no"))
    if len(table):
        values = table.set_index(keys)[list(VALUES)].unstack(["category", "window"])
        values.columns = [f"{c}_w{h}_{v}" for v, c, h in values.columns]
    unknown = sorted(set(values.columns) - set(columns))
    if unknown:
        raise ValueError(f"Forecast for categories not in warehouse.py: {', '.join(sorted({u.split('_w')[0] for u in unknown}))}")
    info = table.groupby("client_no")[["status", "history_days"]].first()
    missing = pd.DataFrame({"status": "NO_POSTINGS", "history_days": 0},
                           index=pd.Index(sorted(no_postings), name="client_no"))
    info = pd.concat([info, missing])
    info["history_days"] = pd.to_numeric(info["history_days"]).fillna(0).astype(int)
    days = {"cutoff_date": cutoff}
    for h in HORIZONS:
        days[f"w{h}_start"], days[f"w{h}_end"] = window(cutoff, h)
    return pd.concat([pd.DataFrame(days, index=info.index), info,
                      values.reindex(index=info.index, columns=columns).astype(float)], axis=1).reset_index()
