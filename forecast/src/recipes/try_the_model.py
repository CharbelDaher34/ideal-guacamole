# Try the model: forecasts the clients in the CSV files of the folder try_model_uploads with the production model,
# the same way as the nightly batch, but on data you upload (made-up clients, or a few clients' history). Nothing
# else in the project changes. One row per client, month and category, amounts in SAR (positive):
#   client_no, month (2026-10 or 2026-10-31), category (a category code: PAYROLL, RENT, SUPPLIERS, RECEIPTS, ...),
#   amount
# Optional columns (one value per client; balance_end per client and month): segment, sector, region,
# last_annual_sales, number_of_employees, balance_end. The cut-off is the latest month in the files; a client needs
# 12 months up to it. Scheduled payments are not uploaded: every amount counts as not scheduled.
# Also draws forecast_chart.png in the folder try_model_charts (shown on the dashboard Try the model): per client,
# the uploaded net cash flow and the forecast (EV, P10-P90 band).
import io

import dataiku
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, forecast_table, model_rows, month_start
from tracking import production_models

REQUIRED = ["client_no", "month", "category", "amount"]
MAX_CHARTS = 12                     # clients drawn in the chart (all are in try_model_forecast)
PROFILE = ["segment", "sector", "region", "last_annual_sales", "number_of_employees"]

folder = dataiku.Folder("try_model_uploads")
files = [p for p in folder.list_paths_in_partition() if p.lower().endswith(".csv")]
if not files:
    raise ValueError("No CSV file in the folder try_model_uploads: upload one (columns " + ", ".join(REQUIRED) + ")")
parts = []
for path in files:
    with folder.get_download_stream(path) as stream:
        part = pd.read_csv(stream, dtype={"client_no": str})
    part.columns = [c.strip().lower() for c in part.columns]
    missing = [c for c in REQUIRED if c not in part.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)} (needed: {', '.join(REQUIRED)})")
    parts.append(part.assign(source_file=path.lstrip("/")))
upload = pd.concat(parts, ignore_index=True)
upload["category"] = upload["category"].astype(str).str.strip().str.upper()
unknown = sorted(set(upload["category"]) - (set(CATEGORIES) - {"NET"}))
if unknown:
    raise ValueError(f"Unknown categories {unknown}: use {', '.join(c for c in CATEGORIES if c != 'NET')}")
upload["month"] = month_start(upload["month"].astype(str).str.slice(0, 7) + "-01")
upload["amount"] = pd.to_numeric(upload["amount"]).abs()
last_month = upload["month"].max()

flows = (upload.groupby(["client_no", "month", "category"], as_index=False)["amount"].sum()
         .assign(known_1=0.0, known_2=0.0, known_3=0.0))
clients = upload.groupby("client_no", as_index=False).agg(source_file=("source_file", "first"))
for c in PROFILE:
    clients[c] = upload.groupby("client_no")[c].first().values if c in upload else np.nan
money_in = upload[upload["category"].map(lambda c: CATEGORIES[c][1]) == "IN"]
recent_in = money_in[money_in["month"] > last_month - pd.offsets.DateOffset(months=12)].groupby("client_no")["amount"].sum()
clients["last_annual_sales"] = clients["last_annual_sales"].fillna(clients["client_no"].map(recent_in)).fillna(0.0)
clients["status"] = "ACTIVE"
balances = (upload.groupby(["client_no", "month"], as_index=False)["balance_end"].last() if "balance_end" in upload
            else pd.DataFrame(columns=["client_no", "month", "balance_end"]))
no_obligations = pd.DataFrame({"client_no": pd.Series(dtype=str), "category": pd.Series(dtype=str),
                               "direction": pd.Series(dtype=str), "due_month": pd.Series(dtype="datetime64[ns]"),
                               "known_month": pd.Series(dtype="datetime64[ns]"), "amount": pd.Series(dtype=float)})

rows = model_rows(flows, no_obligations, clients, balances, last_month)
rows = rows[rows["cutoff_month"] == last_month].drop(columns=["actual", "obligations_known", "target"])
model_id, models = production_models()
for name, model in models.items():
    rows[name] = np.asarray(model.predict(rows), dtype=float)
frame = forecast_table(rows.reset_index(drop=True), no_obligations, last_month)
frame["model_id"] = model_id
frame["source_file"] = frame["client_no"].map(clients.set_index("client_no")["source_file"])
frame["run_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
cols = ["source_file", "client_no", "cutoff_month", "months_ahead", "target_month", "category", "category_name",
        "direction", "p10", "ev", "p90", "history_months", "status", "model_id", "run_at"]
dataiku.Dataset("try_model_forecast").write_with_schema(frame[cols].round(2))

# ---- the chart: uploaded net cash flow (money in minus money out) and the forecast, one panel per client ----
charts = dataiku.Folder("try_model_charts")
charts.clear()
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                 # comes with catboost; without it, the forecast is still written
    plt = None
    print("matplotlib not in the code env: no chart")
if plt is not None:
    sign = upload["category"].map(lambda c: 1.0 if CATEGORIES[c][1] == "IN" else -1.0)
    history = (upload.assign(net=sign * upload["amount"]).groupby(["client_no", "month"])["net"].sum()
               .reset_index())
    history = history[history["month"] > last_month - pd.offsets.DateOffset(months=24)]
    net = frame[frame["category"] == "NET"]
    shown = clients["client_no"].tolist()[:MAX_CHARTS]
    ncols = min(3, len(shown))
    nrows = -(-len(shown) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat[len(shown):]:
        ax.axis("off")
    for ax, client in zip(axes.flat, shown):
        h = history[history["client_no"] == client].sort_values("month")
        f = net[net["client_no"] == client].sort_values("target_month")
        ax.plot(h["month"], h["net"] / 1000, color="0.45", marker=".", label="uploaded")
        if f["ev"].notna().any():
            months = [last_month] + list(pd.to_datetime(f["target_month"]))
            start = h["net"].iloc[-1] / 1000 if len(h) else np.nan
            ax.fill_between(months[1:], f["p10"] / 1000, f["p90"] / 1000, color="tab:blue", alpha=0.2,
                            label="P10 to P90")
            ax.plot(months, [start] + list(f["ev"] / 1000), color="tab:blue", marker="o", label="forecast (EV)")
            ax.set_title(f"{client}", fontsize=10)
        else:
            ax.set_title(f"{client}: not enough history (needs 12 months)", fontsize=10)
        ax.axhline(0, color="0.8", linewidth=0.8)
        ax.set_ylabel("net cash flow, thousand SAR", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Try the model: net cash flow, cut-off {last_month:%Y-%m}, model {model_id}", fontsize=11)
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=110)
    charts.upload_data("forecast_chart.png", image.getvalue())
print(f"{len(clients)} clients from {len(files)} files, cut-off {last_month:%Y-%m}, model {model_id}")
