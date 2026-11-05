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


def model_rows(flows, obligations, clients, balances, last_month):
    """One row per client, category, cut-off month and months ahead (1, 2, 3): the model inputs, and the answer
    (target) where it is known. Inputs only use what was known at the end of the cut-off month; the answer is the
    amount in the predicted month minus the obligations already known at the cut-off.
    flows: client_no, month, category, amount, known_1, known_2, known_3 (monthly_flows)
    obligations: client_no, category, direction, due_month, known_month, amount (monthly_obligations)
    clients: client_no, status, segment, sector, region, last_annual_sales, number_of_employees
    balances: client_no, month, balance_end"""
    flows = flows.assign(month=month_start(flows["month"]))
    obl = obligations.assign(due_month=month_start(obligations["due_month"]),
                             known_month=month_start(obligations["known_month"]))
    bal = balances.assign(month=month_start(balances["month"]))[["client_no", "month", "balance_end"]]

    # ---- every client's months, every category, zeros where nothing moved ----
    span = flows.groupby("client_no")["month"].agg(first_month="min", last_seen="max").reset_index()
    span = span.merge(clients[["client_no", "status"]], on="client_no")
    span["end_month"] = np.where(span["status"] == "ACTIVE", last_month, span["last_seen"])
    real = [c for c in CATEGORIES if c != "NET"]
    grid = pd.DataFrame([(r.client_no, m, c) for r in span.itertuples()
                         for m in pd.date_range(r.first_month, r.end_month, freq="MS") for c in real],
                        columns=["client_no", "month", "category"])
    grid["direction"] = grid["category"].map({c: v[1] for c, v in CATEGORIES.items()})
    values = ["amount", "known_1", "known_2", "known_3"]
    grid = grid.merge(flows[["client_no", "month", "category"] + values], on=["client_no", "month", "category"],
                      how="left")
    grid[values] = grid[values].fillna(0.0)

    # client totals per month, and the net cash flow as one more series
    sign = np.where(grid["direction"] == "IN", 1.0, -1.0)
    totals = (grid.assign(money_in=grid["amount"].where(sign > 0, 0.0), money_out=grid["amount"].where(sign < 0, 0.0))
              .groupby(["client_no", "month"], as_index=False)[["money_in", "money_out"]].sum())
    net = (grid[values].mul(sign, axis=0).assign(client_no=grid["client_no"], month=grid["month"])
           .groupby(["client_no", "month"], as_index=False)[values].sum().assign(category="NET", direction="NET"))
    series = pd.concat([grid, net], ignore_index=True).sort_values(["client_no", "category", "month"])

    # ---- inputs at each cut-off (the month itself is the cut-off) ----
    g = series.groupby(["client_no", "category"])
    roll = lambda col, n: g[col].transform(lambda s: s.rolling(n, min_periods=1).mean())
    series["free"] = series["amount"] - series["known_1"]
    series["last_amount"] = series["amount"]
    series["avg_3m"], series["avg_12m"] = roll("amount", 3), roll("amount", 12)
    series["free_last"] = series["free"]
    series["free_avg_3m"] = series.groupby(["client_no", "category"])["free"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    for k in (9, 10, 11):
        series[f"lag_{k}"] = g["amount"].shift(k)

    totals = totals.sort_values(["client_no", "month"])
    t = totals.groupby("client_no")
    for col, short in (("money_in", "in"), ("money_out", "out")):
        totals[f"client_{short}_avg_3m"] = t[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        totals[f"client_{short}_avg_12m"] = t[col].transform(lambda s: s.rolling(12, min_periods=1).mean())
    totals["scale"] = ((totals["client_in_avg_12m"] + totals["client_out_avg_12m"]) / 2).clip(lower=1000.0)

    base = (series.merge(totals[["client_no", "month", "client_in_avg_3m", "client_out_avg_3m", "scale"]],
                         on=["client_no", "month"], how="left")
            .merge(bal, on=["client_no", "month"], how="left")
            .merge(span[["client_no", "first_month", "end_month"]], on="client_no"))
    base["balance_end"] = base["balance_end"].fillna(0.0)
    base["history_months"] = months_between(base["first_month"], base["month"]) + 1

    # ---- obligations known at the cut-off ----
    expanded = []
    for h in HORIZONS:
        o = obl.assign(cutoff=obl["due_month"].apply(lambda m: add_months(m, -h)), months_ahead=h)
        expanded.append(o[o["known_month"] <= o["cutoff"]])
    expanded = pd.concat(expanded, ignore_index=True)
    expanded["cutoff"] = pd.to_datetime(expanded["cutoff"])
    ob_cat = expanded.groupby(["client_no", "category", "cutoff", "months_ahead"], as_index=False)["amount"].sum()
    signed = expanded.assign(amount=np.where(expanded["direction"] == "IN", 1.0, -1.0) * expanded["amount"])
    ob_net = signed.groupby(["client_no", "cutoff", "months_ahead"], as_index=False)["amount"].sum().assign(
        category="NET")
    ob_target = pd.concat([ob_cat, ob_net], ignore_index=True).rename(columns={"amount": "oblig_target_month"})
    next_3m = ["oblig_in_next_3m", "oblig_out_next_3m"]
    ob_next = (expanded.assign(oblig_in_next_3m=expanded["amount"].where(expanded["direction"] == "IN", 0.0),
                               oblig_out_next_3m=expanded["amount"].where(expanded["direction"] == "OUT", 0.0))
               .groupby(["client_no", "cutoff"], as_index=False)[next_3m].sum())

    # ---- one row per months ahead, with its answer ----
    answers = series[["client_no", "category", "month", "amount", "known_1", "known_2", "known_3"]].rename(
        columns={"month": "target_month", "amount": "actual"})
    profile = clients[["client_no", "segment", "sector", "region", "last_annual_sales", "number_of_employees"]]
    rows = []
    for h in HORIZONS:
        r = base.assign(months_ahead=h, target_month=base["month"].apply(lambda m: add_months(m, h)))
        r["same_month_last_year"] = r[f"lag_{12 - h}"]
        r = r.merge(answers, on=["client_no", "category", "target_month"], how="left")
        r["obligations_known"] = r[f"known_{h}_y"]
        r["target"] = r["actual"] - r["obligations_known"]
        rows.append(r)
    rows = pd.concat(rows, ignore_index=True).rename(columns={"month": "cutoff_month"})
    rows = rows[rows["category"].isin(MODELLED)]
    rows = (rows.merge(ob_target.rename(columns={"cutoff": "cutoff_month"}),
                       on=["client_no", "category", "cutoff_month", "months_ahead"], how="left")
            .merge(ob_next.rename(columns={"cutoff": "cutoff_month"}), on=["client_no", "cutoff_month"], how="left")
            .merge(profile, on="client_no", how="left"))
    fill = ["oblig_target_month", "oblig_in_next_3m", "oblig_out_next_3m", "same_month_last_year"]
    rows[fill] = rows[fill].fillna(0.0)
    rows["target_month_of_year"] = rows["target_month"].dt.month
    rows["log_scale"] = np.log10(rows["scale"])
    rows["log_annual_sales"] = np.log10(rows["last_annual_sales"].astype(float).clip(lower=1.0))
    rows["employees"] = rows["number_of_employees"]
    rows["enough_history"] = rows["history_months"] >= MIN_HISTORY_MONTHS
    # a category the client did not use in the last 12 months is forecast as 0 (plus its obligations), not modelled
    rows["active"] = (rows["category"] == "NET") | (rows["avg_12m"] > 0)
    rows["series_scale"] = np.where(rows["category"] == "NET", rows["scale"],
                                    np.maximum(rows["avg_12m"], 0.01 * rows["scale"]))

    keep = ["client_no", "category", "direction", "cutoff_month", "months_ahead", "target_month", "history_months",
            "enough_history", "active", "scale", "series_scale", "segment", "sector", "region", "last_amount",
            "avg_3m", "avg_12m", "same_month_last_year", "free_last", "free_avg_3m", "client_in_avg_3m",
            "client_out_avg_3m", "balance_end", "oblig_target_month", "oblig_in_next_3m", "oblig_out_next_3m",
            "target_month_of_year", "log_scale", "log_annual_sales", "employees", "actual", "obligations_known",
            "target"]
    return rows[keep].reset_index(drop=True)


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
