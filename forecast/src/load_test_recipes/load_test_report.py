# Load test, the report of the last batch (the last run of the scenario Run the load test), from load_test_log:
#   the folder load_test_report: report_<batch>.html (one per batch, kept) and latest_report.html, each one page with
#   everything: what was found in plain words, one row per test, the time and the memory as charts
#   the dashboard "Load test" (Dashboards > Load test): the same, made again each time (do not edit it by hand:
#   copy it first)
# A test's time: the forecast from start to end (the wall clock) plus the SQL load; not generating the fake data nor
# fitting the model on 2 clients (production does neither at night). The steps (reading, preparing, model,
# formatting, writing) are shown as shares of the workers' time.
import base64
import html
import io

import cashflow  # noqa: F401  (quieter job log: hides Dataiku's writer chatter, as in every recipe)
import dataiku
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dataiku import insights  # noqa: E402

DASHBOARD = "Load test"
STEPS = {"seconds_reading": "reading", "seconds_preparing": "preparing", "seconds_model": "model",
         "seconds_formatting": "formatting", "seconds_writing": "writing"}
COLORS = ["#2a6f97", "#e07a5f"]
NUMBERS = list(STEPS) + ["clients", "months", "postings_per_month", "clients_per_group", "groups", "workers",
                         "seconds_forecast", "seconds_sql_load",
                         "clients_per_second", "peak_memory_mb", "minutes_at_bank_size", "bank_clients", "test_no",
                         "daily_rows", "clients_ok", "clients_not_enough_history",
                         "clients_no_postings"]

log = dataiku.Dataset("load_test_log").get_dataframe()
if log.empty:
    raise ValueError("load_test_log is empty: run the scenario Run the load test")
for column in NUMBERS:
    log[column] = pd.to_numeric(log.get(column), errors="coerce")
batch_id = log["batch_id"].max()
batch = log[log["batch_id"] == batch_id].sort_values("test_no").reset_index(drop=True)
ok = batch[batch["status"] == "SUCCESS"]
bank = int(batch["bank_clients"].dropna().iloc[0]) if batch["bank_clients"].notna().any() else 420000
folder = dataiku.Folder("load_test_report")
print(f"PROGRESS load test report: batch {batch_id}, {len(batch)} tests, {len(ok)} successful", flush=True)


def gb(mb):
    return f"{mb / 1024:,.1f} GB"


def duration(seconds):
    """3.2 s, 45 s, 2.5 min, 1.2 h."""
    if pd.isna(seconds):
        return ""
    return (f"{seconds:,.1f} s" if seconds < 10 else f"{seconds:,.0f} s" if seconds < 90
            else f"{seconds / 60:,.1f} min" if seconds < 5400 else f"{seconds / 3600:,.1f} h")


# 1. what was found, in plain words
found = []
if len(ok):
    best = ok.sort_values("clients_per_second", ascending=False).iloc[0]
    found.append(f"Fastest: <b>{html.escape(best['test'])}</b> ({int(best['workers'])} workers), "
                 f"{best['clients_per_second']:,.0f} clients a second, at most {gb(best['peak_memory_mb'])} of memory.")
    found.append(f"At the bank's size ({bank:,} clients), the nightly forecast and its SQL load would take about "
                 f"<b>{best['minutes_at_bank_size']:,.0f} minutes</b> with that setup (its time per client, "
                 "times the bank's clients).")
    lowest = ok.sort_values("peak_memory_mb").iloc[0]
    if lowest["test"] != best["test"]:
        found.append(f"Least memory: <b>{html.escape(lowest['test'])}</b>, {gb(lowest['peak_memory_mb'])}, about "
                     f"{lowest['minutes_at_bank_size']:,.0f} minutes at the bank's size.")
    steps = ok[list(STEPS)].sum()
    main = steps.idxmax()
    found.append(f"Most of the workers' time goes to <b>{STEPS[main]}</b> ({steps[main] / steps.sum():.0%}).")
for _, row in batch[batch["status"] != "SUCCESS"].iterrows():
    found.append(f"<span style='color:#b00020'>Test <b>{html.escape(str(row['test']))}</b> failed: "
                 f"{html.escape(str(row['error']))}</span>")

# 2. one row per test
table = pd.DataFrame({
    "test": batch["test"], "status": batch["status"],
    "workers": batch["workers"].map("{:,.0f}".format),
    "forecast": batch["seconds_forecast"].map(duration),
    "SQL load": batch["seconds_sql_load"].map(duration),
    **{f"{label} %": (batch[c] / batch[list(STEPS)].sum(axis=1) * 100).map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
       for c, label in STEPS.items()},
    "clients / second": batch["clients_per_second"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else ""),
    "memory (GB)": batch["peak_memory_mb"].map(lambda v: f"{v / 1024:,.1f}" if pd.notna(v) else ""),
    f"min at {bank:,} clients": batch["minutes_at_bank_size"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else ""),
})
table_html = table.to_html(index=False, border=0, escape=True).replace(
    "<table", "<table style='border-collapse:collapse;font-size:12px' cellpadding='5'")


# 3. charts: the time (forecast, SQL load), the memory
def figure(fig, name):
    fig.tight_layout()
    insights.save_figure(f"load_test_{name}", fig, label=f"Load test: {name}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    with folder.get_writer(f"{name}.png") as writer:
        writer.write(buffer.getvalue())
    return base64.b64encode(buffer.getvalue()).decode()


charts = {}
height = 1.6 + 0.45 * max(len(ok), 1)
labels = ok["test"].tolist()[::-1]
fig, ax = plt.subplots(figsize=(10, height))
if len(ok):
    forecast, load = ok["seconds_forecast"].fillna(0)[::-1], ok["seconds_sql_load"].fillna(0)[::-1]
    ax.barh(labels, forecast, color=COLORS[0], label="forecast")
    ax.barh(labels, load, left=forecast, color=COLORS[1], label="SQL load")
    ax.set_xlabel("seconds")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
else:
    ax.text(0.5, 0.5, "No successful test", ha="center", va="center")
    ax.axis("off")
ax.set_title("Time per test: the forecast, then the SQL load", loc="left", fontsize=11, pad=24)
charts["times"] = figure(fig, "times")
fig, ax = plt.subplots(figsize=(10, height))
if len(ok):
    ax.barh(labels, (ok["peak_memory_mb"] / 1024)[::-1], color=COLORS[0])
    for y, value in enumerate((ok["peak_memory_mb"] / 1024)[::-1]):
        ax.text(value, y, f" {value:,.1f}", va="center", fontsize=8)
    ax.set_xlabel("GB")
    ax.spines[["top", "right"]].set_visible(False)
else:
    ax.text(0.5, 0.5, "No successful test", ha="center", va="center")
    ax.axis("off")
ax.set_title("The most memory used, per test (all the workers together)", loc="left", fontsize=11)
charts["memory"] = figure(fig, "memory")

# 4. the page, and the dashboard
first = batch.iloc[0]
summary = (f"<h2 style='margin:0 0 4px'>Load test, {html.escape(batch_id)}</h2>"
           f"<p style='margin:0 0 10px;color:#555'>{len(batch)} tests, {len(ok)} successful, each on the same fake "
           f"data: <b>{int(first['clients']):,} clients</b>, {int(first['months'])} months, "
           f"{int(first['postings_per_month'])} postings a month, {int(first['clients_per_group']):,} clients per group "
           f"({int(first['daily_rows']):,} daily rows, up to "
           f"{html.escape(str(first['cutoff_date']))}). Model: "
           f"{html.escape(', '.join(sorted(set(ok['model'].dropna())) or ['-']))}, the last "
           f"{', '.join(sorted({f'{v:.0f}' for v in pd.to_numeric(ok['input_days'], errors='coerce').dropna()}) or ['365'])}"
           " days of each client, fitted on 2 clients (its forecasts mean nothing, its time is real). Time: the "
           "forecast from start to end, then the SQL load (not making the fake data). The step columns: shares of the "
           "workers' time.</p>"
           "<ul style='margin:0 0 12px'>" + "".join(f"<li>{line}</li>" for line in found) + "</ul>")
page = ("<!doctype html><html><head><meta charset='utf-8'><title>Load test report</title></head>"
        "<body style='font-family:sans-serif;max-width:1100px;margin:20px auto;padding:0 16px'>"
        + summary + table_html
        + "".join(f"<p><img style='max-width:100%' src='data:image/png;base64,{png}'></p>" for png in charts.values())
        + "<p style='color:#777;font-size:12px'>Every test of every batch: the dataset load_test_log.</p>"
          "</body></html>")
name = f"report_{batch_id.replace(':', '').replace(' ', '_')}.html"
for path in (name, "latest_report.html"):
    with folder.get_writer(path) as writer:
        writer.write(page.encode())
insights.save_data("load_test_summary", ("<div style='font-family:sans-serif'>" + summary + table_html
                                        + "</div>").encode(), "text/html", label="Load test: summary")

project = dataiku.api_client().get_default_project()
existing = [d for d in project.list_dashboards(as_type="listitems") if d.name == DASHBOARD]
dashboard = existing[0].to_dashboard() if existing else project.create_dashboard(DASHBOARD)


def tile(insight_id, top, height):
    return {"tileType": "INSIGHT", "insightId": insight_id, "insightType": "static_file",
            "tileParams": {"numDisplayedComments": 5}, "displayMode": "INSIGHT", "clickAction": "DO_NOTHING",
            "box": {"left": 0, "top": top, "width": 36, "height": height},
            "titleOptions": {"showTitle": "NO"}, "resizeImageMode": "FIT_SIZE", "autoLoad": True}


rows = 8 + 2 * len(batch)
settings = dashboard.get_settings()
settings.get_raw()["pages"] = [{"id": None, "title": "Last batch", "show": True, "grid": {"tiles": [
    tile("load_test_summary", 0, rows), tile("load_test_times", rows, 10), tile("load_test_memory", rows + 10, 10)]}}]
settings.save()
print(f"PROGRESS load test report: done: {name} and latest_report.html in the folder load_test_report, and the "
      f"dashboard {DASHBOARD!r}", flush=True)
