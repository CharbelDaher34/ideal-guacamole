"""Shared settings and helpers for the cash-flow forecast recipes (project library: python/cashflow.py)."""
import numpy as np
import pandas as pd

# category code: (name, direction, predicted by the models?)
# Financing instalments and deposit maturities are fully scheduled: no model, their obligations are added as they are.
CATEGORIES = {
    "PAYROLL": ("Payroll", "OUT", True),
    "RENT": ("Rent", "OUT", True),
    "FINANCING": ("Financing instalments", "OUT", False),
    "SADAD_GOV": ("SADAD and government bills", "OUT", True),
    "SUPPLIERS": ("Supplier payments", "OUT", True),
    "CORP_CARD": ("Corporate card", "OUT", True),
    "OTHER_OUT": ("Other out", "OUT", True),
    "RECEIPTS": ("Customer receipts", "IN", True),
    "CARD_SALES": ("Card sales settlements", "IN", True),
    "CHEQUES_IN": ("Cheques in", "IN", True),
    "DEPOSIT_MAT": ("Deposit maturities", "IN", False),
    "OTHER_IN": ("Other in", "IN", True),
    "NET": ("Net cash flow", "NET", True),      # client total: money in minus money out
}
MODELLED = [c for c, (_, _, m) in CATEGORIES.items() if m]
MIN_HISTORY_MONTHS = 12     # a client needs this many months before the cut-off to get a forecast
HORIZONS = (1, 2, 3)        # months ahead
HOLDOUT_MONTHS = 6          # the last months kept aside to test the models
QUANTILES = (0.1, 0.9)      # P10 and P90
CATBOOST = {"iterations": 600, "depth": 6, "learning_rate": 0.08, "random_seed": 0, "verbose": 0}
FX_TO_SAR = {"SAR": 1.0, "USD": 3.75, "EUR": 4.05}

# Category mapping: (origin type, transaction type tag, direction) -> category, one row per combination the
# warehouse sends (NONE = no origin or tag). Anything not in the mapping is other in / other out. The setup copies
# it into the editable dataset category_mapping: edit it there (Flow > category_mapping > Edit), then run the
# scenario Nightly batch. Postings and scheduled payments use the same mapping (visual Join recipes).
DEFAULT_CATEGORY_MAPPING = [
    ("NONE", "COLLECTION", "IN", "RECEIPTS"),
    ("NONE", "SUPPLIER", "OUT", "SUPPLIERS"),
    ("NONE", "RENT", "OUT", "RENT"),
    ("NONE", "OTHER", "IN", "OTHER_IN"),
    ("NONE", "OTHER", "OUT", "OTHER_OUT"),
    ("NONE", "FEE", "OUT", "OTHER_OUT"),
    ("INVOICE", "COLLECTION", "IN", "RECEIPTS"),
    ("CARD_SETTLEMENT", "CARD_SETTLEMENT", "IN", "CARD_SALES"),
    ("CHEQUE", "CHEQUE", "IN", "CHEQUES_IN"),
    ("CORP_CARD", "CARD", "OUT", "CORP_CARD"),
    ("DEPOSIT", "OWN", "IN", "DEPOSIT_MAT"),
    ("DEPOSIT", "DEPOSIT", "IN", "DEPOSIT_MAT"),
    ("LC", "SUPPLIER", "OUT", "SUPPLIERS"),
    ("LOAN", "FINANCING", "OUT", "FINANCING"),
    ("PAYROLL", "SALARY", "OUT", "PAYROLL"),
    ("SADAD", "UTILITY", "OUT", "SADAD_GOV"),
    ("SADAD", "GOV", "OUT", "SADAD_GOV"),
    ("STANDING_ORDER", "RENT", "OUT", "RENT"),
    ("STANDING_ORDER", "SUPPLIER", "OUT", "SUPPLIERS"),
    ("STANDING_ORDER", "OTHER", "OUT", "OTHER_OUT"),
]
# formula of the join's category column: the mapped category, else other in / other out
CATEGORY_FORMULA = 'if(isBlank(mapped_category), if(direction == "IN", "OTHER_IN", "OTHER_OUT"), mapped_category)'

CAT_FEATURES = ["category", "direction", "segment", "sector", "region"]
SERIES_FEATURES = ["last_amount", "avg_3m", "same_month_last_year", "free_last", "free_avg_3m", "oblig_target_month"]
CLIENT_FEATURES = ["client_in_avg_3m", "client_out_avg_3m", "balance_end", "oblig_in_next_3m", "oblig_out_next_3m"]
OTHER_FEATURES = ["months_ahead", "target_month_of_year", "history_months", "log_scale", "series_share",
                  "log_annual_sales", "employees"]
FEATURES = CAT_FEATURES + SERIES_FEATURES + CLIENT_FEATURES + OTHER_FEATURES


def read_dataset(name):
    """A dataset as a DataFrame, with dates as plain (time zone free) timestamps."""
    import dataiku
    df = dataiku.Dataset(name).get_dataframe()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert(None)
    return df


def month_start(dates):
    """First day of the month, as a plain (time zone free) timestamp."""
    d = pd.to_datetime(dates)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert(None)
    return d.dt.to_period("M").dt.to_timestamp()


def months_between(start, end):
    """Whole months from start to end (both month timestamps)."""
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def add_months(month, n):
    return month + pd.offsets.DateOffset(months=n)


def cutoff_month():
    """The forecast's cut-off: the last complete month of the latest checked warehouse load (project variable
    cutoff_month, set by the load check that starts each scenario)."""
    import dataiku
    return pd.Timestamp(dataiku.get_custom_variables()["cutoff_month"] + "-01")


def model_matrix(rows):
    rows = rows.assign(series_share=rows["series_scale"] / rows["scale"])
    X = rows[FEATURES].copy()
    for c in SERIES_FEATURES:
        X[c] = rows[c] / rows["series_scale"]
    for c in CLIENT_FEATURES:
        X[c] = rows[c] / rows["scale"]
    for c in CAT_FEATURES:
        X[c] = X[c].fillna("NA").astype(str)
    return X


def fit_models(rows):
    """EV model (expected value) and P10/P90 model (both quantiles in one model), on amounts / scale."""
    from catboost import CatBoostRegressor
    X, y = model_matrix(rows), rows["target"] / rows["series_scale"]
    ev = CatBoostRegressor(loss_function="RMSE", **CATBOOST)
    ev.fit(X, y, cat_features=CAT_FEATURES)
    alphas = ",".join(str(q) for q in QUANTILES)
    pq = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", **CATBOOST)
    pq.fit(X, y, cat_features=CAT_FEATURES)
    return {"ev": ev, "p10_p90": pq}


def finish(ev, p10, p90, category, index=None):
    """P10 <= EV <= P90, and no negative amounts except for the net cash flow (in SAR)."""
    ev, p10, p90 = (np.asarray(v, dtype=float) for v in (ev, p10, p90))
    p10, p90 = np.minimum(p10, ev), np.maximum(p90, ev)
    floor = np.where(np.asarray(category) == "NET", -np.inf, 0.0)
    return pd.DataFrame({"p10": np.maximum(p10, floor), "ev": np.maximum(ev, floor),
                         "p90": np.maximum(p90, floor)}, index=index).round(2)


def predict(models, rows):
    """P10, EV and P90 in SAR from the code models."""
    X = model_matrix(rows)
    scale = rows["series_scale"].values
    q = models["p10_p90"].predict(X)
    return finish(models["ev"].predict(X) * scale, q[:, 0] * scale, q[:, 1] * scale, rows["category"], rows.index)


def scores(d):
    """Test scores of predictions (columns target, p10, ev, p90, baseline): EV error (WAPE) of the model and of the
    simple guess, and the share of actual values below P10, between P10 and P90, and above P90."""
    size = d["target"].abs().sum()
    wape = lambda column: round((d["target"] - d[column]).abs().sum() / size, 4) if size else np.nan
    return pd.Series({"rows": len(d), "actual_total": round(d["target"].sum(), 0), "wape_model": wape("ev"),
                      "wape_baseline": wape("baseline") if "baseline" in d else np.nan,
                      "coverage_p10_p90": round(((d["target"] >= d["p10"]) & (d["target"] <= d["p90"])).mean(), 4),
                      "below_p10": round((d["target"] < d["p10"]).mean(), 4),
                      "above_p90": round((d["target"] > d["p90"]).mean(), 4)})


def holdout(rows):
    """The test rows: predicted month in the last 6 months before the cut-off, with the simple guess."""
    rows = rows.assign(target_month=month_start(rows["target_month"]))
    test = rows[rows["target_month"] >= add_months(cutoff_month(), -(HOLDOUT_MONTHS - 1))].copy()
    test["baseline"] = np.where(test["category"] == "NET", test["free_avg_3m"], test["free_avg_3m"].clip(lower=0))
    return test
