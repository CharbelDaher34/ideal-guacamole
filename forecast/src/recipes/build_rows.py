# Build rows: one row per client, category, cut-off month and months ahead (1, 2, 3).
# Inputs only use what was known at the end of the cut-off month; the answer (target) is the amount in the
# predicted month minus the obligations already known at the cut-off.
# training_rows: every past cut-off with an answer. scoring_rows: the latest cut-off, to forecast.
import dataiku
import numpy as np
import pandas as pd

from cashflow import (CATEGORIES, HORIZONS, MIN_HISTORY_MONTHS, MODELLED, add_months, cutoff_month, month_start,
                      months_between, read_dataset)

last_month = cutoff_month()
flows = read_dataset("monthly_flows")
flows["month"] = month_start(flows["month"])
obl = read_dataset("monthly_obligations")
obl["due_month"], obl["known_month"] = month_start(obl["due_month"]), month_start(obl["known_month"])
clients = read_dataset("clients")

# ---- every client's months, every category, zeros where nothing moved ----------------------------
span = flows.groupby("client_no")["month"].agg(first_month="min", last_seen="max").reset_index()
span = span.merge(clients[["client_no", "status"]], on="client_no")
span["end_month"] = np.where(span["status"] == "ACTIVE", last_month, span["last_seen"])
real = [c for c in CATEGORIES if c != "NET"]
grid = pd.DataFrame([(r.client_no, m, c) for r in span.itertuples()
                     for m in pd.date_range(r.first_month, r.end_month, freq="MS") for c in real],
                    columns=["client_no", "month", "category"])
grid["direction"] = grid["category"].map({c: v[1] for c, v in CATEGORIES.items()})
values = ["amount", "known_1", "known_2", "known_3"]
grid = grid.merge(flows[["client_no", "month", "category"] + values], on=["client_no", "month", "category"], how="left")
grid[values] = grid[values].fillna(0.0)

# client totals per month, and the net cash flow as one more series
sign = np.where(grid["direction"] == "IN", 1.0, -1.0)
totals = (grid.assign(money_in=grid["amount"].where(sign > 0, 0.0), money_out=grid["amount"].where(sign < 0, 0.0))
          .groupby(["client_no", "month"], as_index=False)[["money_in", "money_out"]].sum())
net = (grid[values].mul(sign, axis=0).assign(client_no=grid["client_no"], month=grid["month"])
       .groupby(["client_no", "month"], as_index=False)[values].sum().assign(category="NET", direction="NET"))
series = pd.concat([grid, net], ignore_index=True).sort_values(["client_no", "category", "month"])

# ---- inputs at each cut-off (the month itself is the cut-off) -----------------------------------
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

bal = read_dataset("monthly_balances")
bal["month"] = month_start(bal["month"])

base = (series.merge(totals[["client_no", "month", "client_in_avg_3m", "client_out_avg_3m", "scale"]],
                     on=["client_no", "month"], how="left")
        .merge(bal, on=["client_no", "month"], how="left")
        .merge(span[["client_no", "first_month", "end_month"]], on="client_no"))
base["balance_end"] = base["balance_end"].fillna(0.0)
base["history_months"] = months_between(base["first_month"], base["month"]) + 1

# ---- obligations known at the cut-off ----------------------------------------------------------
expanded = []
for h in HORIZONS:
    o = obl.assign(cutoff=obl["due_month"].apply(lambda m: add_months(m, -h)), months_ahead=h)
    expanded.append(o[o["known_month"] <= o["cutoff"]])
expanded = pd.concat(expanded, ignore_index=True)
ob_cat = expanded.groupby(["client_no", "category", "cutoff", "months_ahead"], as_index=False)["amount"].sum()
signed = expanded.assign(amount=np.where(expanded["direction"] == "IN", 1.0, -1.0) * expanded["amount"])
ob_net = signed.groupby(["client_no", "cutoff", "months_ahead"], as_index=False)["amount"].sum().assign(category="NET")
ob_target = pd.concat([ob_cat, ob_net], ignore_index=True).rename(columns={"amount": "oblig_target_month"})
ob_next = (expanded.pivot_table(index=["client_no", "cutoff"], columns="direction", values="amount", aggfunc="sum",
                                fill_value=0.0)
           .rename(columns={"IN": "oblig_in_next_3m", "OUT": "oblig_out_next_3m"}).reset_index())
for c in ("oblig_in_next_3m", "oblig_out_next_3m"):
    if c not in ob_next:
        ob_next[c] = 0.0

# ---- one row per months ahead, with its answer -------------------------------------------------
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
rows["log_annual_sales"] = np.log10(rows["last_annual_sales"].clip(lower=1.0))
rows["employees"] = rows["number_of_employees"]
rows["enough_history"] = rows["history_months"] >= MIN_HISTORY_MONTHS
# a category the client did not use in the last 12 months is forecast as 0 (plus its obligations), not modelled
rows["active"] = (rows["category"] == "NET") | (rows["avg_12m"] > 0)
rows["series_scale"] = np.where(rows["category"] == "NET", rows["scale"],
                                np.maximum(rows["avg_12m"], 0.01 * rows["scale"]))

keep = ["client_no", "category", "direction", "cutoff_month", "months_ahead", "target_month", "history_months",
        "enough_history", "active", "scale", "series_scale", "segment", "sector", "region", "last_amount", "avg_3m",
        "avg_12m", "same_month_last_year", "free_last", "free_avg_3m", "client_in_avg_3m", "client_out_avg_3m",
        "balance_end", "oblig_target_month", "oblig_in_next_3m", "oblig_out_next_3m", "target_month_of_year",
        "log_scale", "log_annual_sales", "employees", "actual", "obligations_known", "target"]
rows = rows[keep]
training = rows[rows["enough_history"] & rows["active"] & rows["target"].notna()]
scoring = rows[(rows["cutoff_month"] == last_month)].drop(columns=["actual", "obligations_known", "target"])
dataiku.Dataset("training_rows").write_with_schema(training.round(4).reset_index(drop=True))
dataiku.Dataset("scoring_rows").write_with_schema(scoring.round(4).reset_index(drop=True))
