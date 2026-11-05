"""Shared by every zone (project library: python/cashflow.py): dates, the business rule on amounts, and the two
interfaces between the zones:
    1. the model rows (model_rows): staging -> training and inference. Training rows also have the ANSWER column.
       Add a column freely (a model that does not read it ignores it); rename or remove one only when training and
       inference both agree.
    3. the saved models (SAVED_MODELS): training adds versions, inference reads the active one.
Interface 2, the forecaster (lab -> training), is in models.py.
The forecast is one value per client and month: the net cash flow (money in minus money out), 1, 2 and 3 months
ahead, as P10, EV and P90. There are no categories."""
import numpy as np
import pandas as pd

MIN_HISTORY_MONTHS = 12     # a client needs this many months before the cut-off to get a forecast
HORIZONS = (1, 2, 3)        # months ahead
SAVED_MODELS = {"ev": "forecast_ev", "p10": "forecast_p10", "p90": "forecast_p90"}   # forecast column: saved model

# Interface 1, the model rows: one row per client, cut-off month and months ahead, with these columns. Training rows
# also have ANSWER (target: the net cash flow of the predicted month); scoring rows do not.
MODEL_ROW_COLUMNS = [
    "client_no", "cutoff_month", "months_ahead", "target_month", "history_months", "enough_history", "scale",
    "segment", "sector", "city", "sole_proprietor", "last_amount", "avg_3m", "avg_12m", "same_month_last_year",
    "client_in_avg_3m", "client_out_avg_3m", "target_month_of_year", "log_scale", "log_annual_sales", "employees"]
ANSWER = ["target"]


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
    """The forecast's cut-off: the last complete month (project variable
    cutoff_month, set by the first step of each scenario: warehouse.set_cutoff)."""
    import dataiku
    return pd.Timestamp(dataiku.get_custom_variables()["cutoff_month"] + "-01")


def finish(ev, p10, p90, index=None):
    """P10 <= EV <= P90 (SAR). The net cash flow can be below zero."""
    ev, p10, p90 = (np.asarray(v, dtype=float) for v in (ev, p10, p90))
    return pd.DataFrame({"p10": np.minimum(p10, ev), "ev": ev, "p90": np.maximum(p90, ev)}, index=index).round(2)


def model_rows(flows, clients, last_month):
    """Interface 1, the model rows: one row per client, cut-off month and months ahead (1, 2, 3), with the columns
    MODEL_ROW_COLUMNS and the answer (target: the net cash flow of the predicted month) where it is known. Training
    rows and scoring rows both come from here, so a model sees the same inputs in training and in the forecast. Inputs
    only use what was known at the end of the cut-off month.
    flows: client_no, month, direction (IN / OUT), amount (monthly_flows)
    clients: client_no, status, segment, sector, city, sole_proprietor, last_annual_sales, number_of_employees"""
    flows = flows.assign(month=month_start(flows["month"]))

    # ---- every client's months, zeros where nothing moved: money in, money out, net ----
    span = flows.groupby("client_no")["month"].agg(first_month="min", last_seen="max").reset_index()
    span = span.merge(clients[["client_no", "status"]], on="client_no")
    span["end_month"] = np.where(span["status"] == "ACTIVE", last_month, span["last_seen"])
    grid = pd.DataFrame([(r.client_no, m) for r in span.itertuples()
                         for m in pd.date_range(r.first_month, r.end_month, freq="MS")], columns=["client_no", "month"])
    sums = (flows.pivot_table(index=["client_no", "month"], columns="direction", values="amount", aggfunc="sum")
            .reindex(columns=["IN", "OUT"]).fillna(0.0).reset_index())
    series = grid.merge(sums, on=["client_no", "month"], how="left").rename(columns={"IN": "money_in", "OUT": "money_out"})
    series[["money_in", "money_out"]] = series[["money_in", "money_out"]].fillna(0.0)
    series["amount"] = series["money_in"] - series["money_out"]
    series = series.sort_values(["client_no", "month"]).reset_index(drop=True)

    # ---- inputs at each cut-off (the month itself is the cut-off) ----
    g = series.groupby("client_no")
    roll = lambda col, n: g[col].transform(lambda s: s.rolling(n, min_periods=1).mean())
    series["last_amount"] = series["amount"]
    series["avg_3m"], series["avg_12m"] = roll("amount", 3), roll("amount", 12)
    series["client_in_avg_3m"], series["client_out_avg_3m"] = roll("money_in", 3), roll("money_out", 3)
    series["scale"] = ((roll("money_in", 12) + roll("money_out", 12)) / 2).clip(lower=1000.0)
    for k in (9, 10, 11):
        series[f"lag_{k}"] = g["amount"].shift(k)
    base = series.merge(span[["client_no", "first_month"]], on="client_no")
    base["history_months"] = months_between(base["first_month"], base["month"]) + 1

    # ---- one row per months ahead, with its answer ----
    answers = series[["client_no", "month", "amount"]].rename(columns={"month": "target_month", "amount": "target"})
    profile = clients[["client_no", "segment", "sector", "city", "sole_proprietor", "last_annual_sales",
                       "number_of_employees"]]
    rows = []
    for h in HORIZONS:
        r = base.assign(months_ahead=h, target_month=base["month"].apply(lambda m: add_months(m, h)))
        r["same_month_last_year"] = r[f"lag_{12 - h}"]
        rows.append(r.merge(answers, on=["client_no", "target_month"], how="left"))
    rows = pd.concat(rows, ignore_index=True).rename(columns={"month": "cutoff_month"})
    rows = rows.merge(profile, on="client_no", how="left")
    rows["same_month_last_year"] = rows["same_month_last_year"].fillna(0.0)
    rows["target_month_of_year"] = rows["target_month"].dt.month
    rows["log_scale"] = np.log10(rows["scale"])
    rows["log_annual_sales"] = np.log10(rows["last_annual_sales"].astype(float).clip(lower=1.0))
    rows["employees"] = rows["number_of_employees"].astype(float)
    rows["enough_history"] = rows["history_months"] >= MIN_HISTORY_MONTHS
    return rows[MODEL_ROW_COLUMNS + ANSWER].reset_index(drop=True)
