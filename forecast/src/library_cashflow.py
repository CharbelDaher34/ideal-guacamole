"""Shared by every zone (project library: python/cashflow.py): the categories and their default mapping, dates, the
business rule on amounts, and the two interfaces between the zones:
    1. the model rows (model_rows): data prep -> training and inference. Training rows also have the ANSWER columns.
       Add a column freely (a model that does not read it ignores it); rename or remove one only when training and
       inference both agree.
    3. the saved models (SAVED_MODELS): training adds versions, inference reads the active one.
Interface 2, the forecaster (lab -> training), is in models.py."""
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
SAVED_MODELS = {"ev": "forecast_ev", "p10": "forecast_p10", "p90": "forecast_p90"}   # forecast column: saved model

# Category mapping: (T24 module SYS_ID, transaction type TRNS_TYP, direction) -> category, one row per combination the
# warehouse sends (NONE = no type). Anything not in the mapping is other in / other out. The setup copies it into the
# editable dataset category_mapping: edit it there with the bank's codes (Flow > category_mapping), then run the
# scenario Nightly batch. Statement entries and scheduled payments use the same mapping (visual Join recipes).
# These defaults are the sample data's codes: the bank's real SYS_ID and TRNS_TYP values replace them.
DEFAULT_CATEGORY_MAPPING = [
    ("FT", "COLLECTION", "IN", "RECEIPTS"),
    ("FT", "SUPPLIER", "OUT", "SUPPLIERS"),
    ("FT", "RENT", "OUT", "RENT"),
    ("FT", "OTHER", "IN", "OTHER_IN"),
    ("FT", "OTHER", "OUT", "OTHER_OUT"),
    ("FT", "FEE", "OUT", "OTHER_OUT"),
    ("FT", "CARD_SETTLEMENT", "IN", "CARD_SALES"),
    ("FT", "CARD", "OUT", "CORP_CARD"),
    ("FT", "SALARY", "OUT", "PAYROLL"),
    ("FT", "UTILITY", "OUT", "SADAD_GOV"),
    ("FT", "GOV", "OUT", "SADAD_GOV"),
    ("CQ", "CHEQUE", "IN", "CHEQUES_IN"),
    ("AA", "FINANCING", "OUT", "FINANCING"),
    ("AA", "OWN", "IN", "DEPOSIT_MAT"),
    ("AA", "DEPOSIT", "IN", "DEPOSIT_MAT"),
    ("LC", "SUPPLIER", "OUT", "SUPPLIERS"),
    ("STO", "RENT", "OUT", "RENT"),
    ("STO", "SUPPLIER", "OUT", "SUPPLIERS"),
    ("STO", "OTHER", "OUT", "OTHER_OUT"),
]
# formula of the join's category column: the mapped category, else other in / other out
CATEGORY_FORMULA = 'if(isBlank(mapped_category), if(direction == "IN", "OTHER_IN", "OTHER_OUT"), mapped_category)'

# Interface 1, the model rows: one row per client, category, cut-off month and months ahead, with these columns.
# Training rows also have ANSWER (target: what the models predict); scoring rows do not.
MODEL_ROW_COLUMNS = [
    "client_no", "category", "direction", "cutoff_month", "months_ahead", "target_month", "history_months",
    "enough_history", "active", "scale", "series_scale", "segment", "sector", "city", "sole_proprietor",
    "last_amount", "avg_3m",
    "avg_12m", "same_month_last_year", "free_last", "free_avg_3m", "client_in_avg_3m", "client_out_avg_3m",
    "oblig_target_month", "oblig_in_next_3m", "oblig_out_next_3m", "target_month_of_year",
    "log_scale", "log_annual_sales", "employees"]
ANSWER = ["actual", "obligations_known", "target"]


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


def finish(ev, p10, p90, category, index=None, floor=True):
    """P10 <= EV <= P90 and, with floor, no negative amounts except for the net cash flow (in SAR). The models'
    part is not floored (it can be below zero when scheduled payments are not all made); the final forecast is."""
    ev, p10, p90 = (np.asarray(v, dtype=float) for v in (ev, p10, p90))
    p10, p90 = np.minimum(p10, ev), np.maximum(p90, ev)
    floor = np.where(np.asarray(category) == "NET", -np.inf, 0.0) if floor else np.full(len(ev), -np.inf)
    return pd.DataFrame({"p10": np.maximum(p10, floor), "ev": np.maximum(ev, floor),
                         "p90": np.maximum(p90, floor)}, index=index).round(2)


def model_rows(flows, obligations, clients, last_month):
    """Interface 1, the model rows: one row per client, category, cut-off month and months ahead (1, 2, 3), with the
    columns MODEL_ROW_COLUMNS and the answer (ANSWER) where it is known. Training rows and scoring rows both come from
    here, so a model sees the same inputs in training and in the forecast. Inputs only use what was known at the end
    of the cut-off month. The answer (target) is the amount of the predicted month minus the scheduled payments due
    that month and already known at the cut-off: the same scheduled payments the forecast adds back.
    flows: client_no, month, category, amount (monthly_flows)
    obligations: client_no, category, direction, due_month, known_month, amount (monthly_obligations)
    clients: client_no, status, segment, sector, city, sole_proprietor, last_annual_sales, number_of_employees"""
    flows = flows.assign(month=month_start(flows["month"]))
    obl = obligations.assign(due_month=month_start(obligations["due_month"]),
                             known_month=month_start(obligations["known_month"]))

    # ---- every client's months, every category, zeros where nothing moved ----
    span = flows.groupby("client_no")["month"].agg(first_month="min", last_seen="max").reset_index()
    span = span.merge(clients[["client_no", "status"]], on="client_no")
    span["end_month"] = np.where(span["status"] == "ACTIVE", last_month, span["last_seen"])
    real = [c for c in CATEGORIES if c != "NET"]
    grid = pd.DataFrame([(r.client_no, m, c) for r in span.itertuples()
                         for m in pd.date_range(r.first_month, r.end_month, freq="MS") for c in real],
                        columns=["client_no", "month", "category"])
    grid["direction"] = grid["category"].map({c: v[1] for c, v in CATEGORIES.items()})
    grid = grid.merge(flows[["client_no", "month", "category", "amount"]], on=["client_no", "month", "category"],
                      how="left")
    grid["amount"] = grid["amount"].fillna(0.0)

    # client totals per month, and the net cash flow as one more series
    sign = np.where(grid["direction"] == "IN", 1.0, -1.0)
    totals = (grid.assign(money_in=grid["amount"].where(sign > 0, 0.0), money_out=grid["amount"].where(sign < 0, 0.0))
              .groupby(["client_no", "month"], as_index=False)[["money_in", "money_out"]].sum())
    net = (grid.assign(amount=grid["amount"] * sign).groupby(["client_no", "month"], as_index=False)["amount"].sum()
           .assign(category="NET", direction="NET"))
    series = pd.concat([grid, net], ignore_index=True).sort_values(["client_no", "category", "month"])

    # ---- scheduled payments known at each cut-off, for the month 1, 2 or 3 months ahead ----
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
    # each month's part already known a month before (the cut-off before it): the rest is the "free" part
    known_before = ob_target[ob_target["months_ahead"] == 1].assign(
        month=lambda d: d["cutoff"].apply(lambda m: add_months(m, 1)))
    series = series.merge(known_before[["client_no", "category", "month", "oblig_target_month"]].rename(
        columns={"oblig_target_month": "known_before"}), on=["client_no", "category", "month"], how="left")
    series["known_before"] = series["known_before"].fillna(0.0)

    # ---- inputs at each cut-off (the month itself is the cut-off) ----
    series = series.sort_values(["client_no", "category", "month"]).reset_index(drop=True)
    series["free"] = series["amount"] - series["known_before"]
    g = series.groupby(["client_no", "category"])
    roll = lambda col, n: g[col].transform(lambda s: s.rolling(n, min_periods=1).mean())
    series["last_amount"] = series["amount"]
    series["avg_3m"], series["avg_12m"] = roll("amount", 3), roll("amount", 12)
    series["free_last"] = series["free"]
    series["free_avg_3m"] = roll("free", 3)
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
            .merge(span[["client_no", "first_month", "end_month"]], on="client_no"))
    base["history_months"] = months_between(base["first_month"], base["month"]) + 1

    # ---- one row per months ahead, with its answer ----
    answers = series[["client_no", "category", "month", "amount"]].rename(
        columns={"month": "target_month", "amount": "actual"})
    profile = clients[["client_no", "segment", "sector", "city", "sole_proprietor", "last_annual_sales",
                       "number_of_employees"]]
    rows = []
    for h in HORIZONS:
        r = base.assign(months_ahead=h, target_month=base["month"].apply(lambda m: add_months(m, h)))
        r["same_month_last_year"] = r[f"lag_{12 - h}"]
        rows.append(r.merge(answers, on=["client_no", "category", "target_month"], how="left"))
    rows = pd.concat(rows, ignore_index=True).rename(columns={"month": "cutoff_month"})
    rows = rows[rows["category"].isin(MODELLED)]
    rows = (rows.merge(ob_target.rename(columns={"cutoff": "cutoff_month"}),
                       on=["client_no", "category", "cutoff_month", "months_ahead"], how="left")
            .merge(ob_next.rename(columns={"cutoff": "cutoff_month"}), on=["client_no", "cutoff_month"], how="left")
            .merge(profile, on="client_no", how="left"))
    fill = ["oblig_target_month", "oblig_in_next_3m", "oblig_out_next_3m", "same_month_last_year"]
    rows[fill] = rows[fill].fillna(0.0)
    rows["obligations_known"] = rows["oblig_target_month"]
    rows["target"] = rows["actual"] - rows["obligations_known"]
    rows["target_month_of_year"] = rows["target_month"].dt.month
    rows["log_scale"] = np.log10(rows["scale"])
    rows["log_annual_sales"] = np.log10(rows["last_annual_sales"].astype(float).clip(lower=1.0))
    rows["employees"] = rows["number_of_employees"].astype(float)
    rows["enough_history"] = rows["history_months"] >= MIN_HISTORY_MONTHS
    # a category the client did not use in the last 12 months is forecast as 0 (plus its obligations), not modelled
    rows["active"] = (rows["category"] == "NET") | (rows["avg_12m"] > 0)
    rows["series_scale"] = np.where(rows["category"] == "NET", rows["scale"],
                                    np.maximum(rows["avg_12m"], 0.01 * rows["scale"]))
    return rows[MODEL_ROW_COLUMNS + ANSWER].reset_index(drop=True)
