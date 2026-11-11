# Load test, the report of the last run of the scenario Run the load test: its tests side by side (one row per test
# in load_test_log, all with the same batch), then the earlier tests below:
#   1. this run: one line per test (its data, its workers, its times, its memory), the fastest marked, in a table
#   2. charts of this run's tests: the total time (forecast + SQL load), where the forecast's time goes (reading,
#      preparing, model, formatting, writing), the memory
#   3. the earlier tests, as a table
# In the folder load_test_report: report_<batch>.html (one per run, kept) and latest_report.html, one page each; and
# the dashboard "Load test" (Dashboards > Load test), made again each time (do not edit it by hand: copy it first).
# A test's time: the forecast from start to end (the wall clock) plus the SQL load; not making the fake data nor
# fitting the model on 2 clients (production does neither at night).
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
EARLIER = 20                    # the earlier tests shown
STEPS = {"seconds_reading": "reading", "seconds_preparing": "preparing", "seconds_model": "model",
         "seconds_formatting": "formatting", "seconds_writing": "writing"}
STEP_COLORS = ["#2a6f97", "#61a5c2", "#e07a5f", "#f2cc8f", "#81b29a"]
COLORS = {"forecast": "#2a6f97", "SQL load": "#e07a5f", "memory": "#81b29a"}
NUMBERS = ["test", "workers", "clients", "months", "postings_per_month", "clients_per_group", "seconds_forecast",
           "seconds_sql_load", "clients_per_second", "peak_memory_mb", *STEPS]
CELL = "padding:5px 9px;border-bottom:1px solid #ddd;text-align:left"

log = dataiku.Dataset("load_test_log").get_dataframe()
if log.empty:
    raise ValueError("load_test_log is empty: run the scenario Run the load test")
for column in NUMBERS:
    log[column] = pd.to_numeric(log[column], errors="coerce") if column in log else float("nan")
log["batch"] = log["batch"].fillna("").astype(str) if "batch" in log else ""
log["batch"] = log["batch"].where(log["batch"] != "", log["run_at"])      # tests from before batches: one run each
log["test"] = log["test"].fillna(1)
log["seconds_total"] = log["seconds_forecast"].fillna(0) + log["seconds_sql_load"].fillna(0)
log = log.sort_values(["run_at", "test"]).reset_index(drop=True)
batch_id = log.iloc[-1]["batch"]
run = log[log["batch"] == batch_id].sort_values("test")             # this run's tests
earlier = log[log["batch"] != batch_id].tail(EARLIER).iloc[::-1]     # the newest first
ok = run[run["status"] == "SUCCESS"]
fastest = ok.loc[ok["seconds_total"].idxmin()] if len(ok) else None
folder = dataiku.Folder("load_test_report")


def duration(seconds):
    """3.2 s, 45 s, 2.5 min, 1.2 h."""
    if pd.isna(seconds):
        return ""
    return (f"{seconds:,.1f} s" if seconds < 10 else f"{seconds:,.0f} s" if seconds < 90
            else f"{seconds / 60:,.1f} min" if seconds < 5400 else f"{seconds / 3600:,.1f} h")


def data(t):
    """The test's fake data, short: 50,000 clients · 500 per group · 36 months · 40 postings/month."""
    return (f"{int(t['clients']):,} clients · {int(t['clients_per_group']):,} per group · {int(t['months'])} months · "
            f"{int(t['postings_per_month'])} postings/month")


def label(t):
    return (f"Test {int(t['test'])}: {int(t['clients']):,} clients, {int(t['clients_per_group']):,}/group, "
            f"{int(t['workers'])} workers")


def main_step(t):
    steps = t[list(STEPS)].astype(float)
    return f"{STEPS[steps.idxmax()]} ({steps.max() / steps.sum():.0%})" if steps.sum() > 0 else ""


def table(headers, rows, styles=None):
    head = "".join(f"<th style='{CELL};background:#f3f5f7'>{h}</th>" for h in headers)
    body = "".join(f"<tr style='{(styles or {}).get(i, '')}'>" + "".join(f"<td style='{CELL}'>{v}</td>" for v in row)
                   + "</tr>" for i, row in enumerate(rows))
    return f"<table style='border-collapse:collapse;font-size:13px;margin:6px 0 14px'>{head}{body}</table>"


# 1. this run: one line per test
lines, styles = [], {}
for i, (_, t) in enumerate(run.iterrows()):
    if t["status"] != "SUCCESS":
        lines.append([int(t["test"]), html.escape(data(t)), int(t["workers"]),
                      f"<span style='color:#b00020'>FAILED: {html.escape(str(t['error'])[:200])}</span>",
                      "", "", "", "", "", ""])
        continue
    is_fastest = t.name == fastest.name
    if is_fastest:
        styles[i] = "background:#e8f5e9;font-weight:bold"
    lines.append([int(t["test"]), html.escape(data(t)), int(t["workers"]), duration(t["seconds_forecast"]),
                  duration(t["seconds_sql_load"]), f"<b>{duration(t['seconds_total'])}</b>",
                  f"{t['clients_per_second']:,.0f}", f"{t['peak_memory_mb'] / 1024:,.1f} GB", main_step(t),
                  "the fastest" if is_fastest else
                  f"{t['seconds_total'] / fastest['seconds_total'] - 1:.0%} slower"])
this_run = table(["Test", "Fake data", "Workers", "Forecast", "SQL load", "Total", "Clients / second", "Memory",
                  "Most of the forecast's time", "Compared with the fastest"], lines, styles)
words = (f"<p>The fastest: <b>{html.escape(label(fastest))}</b>, {duration(fastest['seconds_total'])} in all "
         f"({fastest['clients_per_second']:,.0f} clients a second).</p>" if fastest is not None else
         "<p style='color:#b00020'>No test of this run succeeded.</p>")
if len(run) > len(ok):
    words += f"<p style='color:#b00020'>{len(run) - len(ok)} of {len(run)} tests failed: see the table.</p>"


# 3. the earlier tests
def earlier_row(t):
    done = t["status"] == "SUCCESS"
    return [html.escape(str(t["batch"])), int(t["test"]), html.escape(data(t)) if pd.notna(t["clients"]) else "",
            int(t["workers"]) if pd.notna(t["workers"]) else "", duration(t["seconds_forecast"]),
            duration(t["seconds_sql_load"]), duration(t["seconds_total"]) if done else "",
            f"{t['clients_per_second']:,.0f}" if pd.notna(t["clients_per_second"]) else "",
            f"{t['peak_memory_mb'] / 1024:,.1f} GB" if pd.notna(t["peak_memory_mb"]) else "",
            html.escape(str(t["status"]))]


history = (table(["Run", "Test", "Fake data", "Workers", "Forecast", "SQL load", "Total", "Clients / second", "Memory",
                  "Status"], [earlier_row(t) for _, t in earlier.iterrows()]) if len(earlier) else "<p>None yet.</p>")


# 2. charts of this run's successful tests, test 1 at the top
def figure(fig, name):
    fig.tight_layout()
    insights.save_figure(f"load_test_{name}", fig, label=f"Load test: {name}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    with folder.get_writer(f"{name}.png") as writer:
        writer.write(buffer.getvalue())
    return base64.b64encode(buffer.getvalue()).decode()


shown = ok.iloc[::-1]
names = [label(t) for _, t in shown.iterrows()]
height = 1.8 + 0.5 * max(len(shown), 1)
charts = {}
for name, title in (("times", "Total time per test: the forecast, then the SQL load (shorter is better)"),
                    ("steps", "Where the forecast's time goes, per test (share of the workers' time)"),
                    ("memory", "The most memory used, per test")):
    fig, ax = plt.subplots(figsize=(11, height))
    if not len(shown):
        ax.text(0.5, 0.5, "No successful test in this run", ha="center", va="center")
        ax.axis("off")
    elif name == "times":
        forecast, load = shown["seconds_forecast"].fillna(0), shown["seconds_sql_load"].fillna(0)
        ax.barh(names, forecast, color=COLORS["forecast"], label="forecast")
        ax.barh(names, load, left=forecast, color=COLORS["SQL load"], label="SQL load")
        for y, total in enumerate(forecast + load):
            ax.text(total, y, f"  {duration(total)}", va="center", fontsize=9)
        ax.set_xlabel("seconds")
    elif name == "steps":
        shares = shown[list(STEPS)].div(shown[list(STEPS)].sum(axis=1), axis=0) * 100
        left = pd.Series(0.0, index=shares.index)
        for (column, step), color in zip(STEPS.items(), STEP_COLORS):
            ax.barh(names, shares[column], left=left, color=color, label=step)
            for y, (value, start) in enumerate(zip(shares[column], left)):
                if value >= 6:
                    ax.text(start + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=8,
                            color="white" if color in ("#2a6f97", "#e07a5f") else "black")
            left += shares[column]
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of the forecast's work")
    else:
        memory = shown["peak_memory_mb"] / 1024
        ax.barh(names, memory, color=COLORS["memory"])
        for y, value in enumerate(memory):
            ax.text(value, y, f"  {value:,.1f} GB", va="center", fontsize=9)
        ax.set_xlabel("GB")
    if len(shown) and name != "memory":
        ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, pad=24 if name != "memory" else 6)
    charts[name] = figure(fig, name)

# 4. the page, and the dashboard
model = html.escape(str(run.iloc[-1].get("model") or "daily_catboost"))
summary = (f"<h2 style='margin:0 0 4px'>Load test, {html.escape(str(batch_id))}: {len(run)} "
           f"test{'s' if len(run) > 1 else ''}</h2>"
           f"<p style='margin:0 0 10px;color:#555'>Each test: the nightly forecast of every fake client with the "
           f"model {model} (fitted on 2 clients only: its forecasts mean nothing, its time is real), then its load "
           "into SQL. Total = forecast + SQL load; making the fake data is not timed. The tests ran one after the "
           "other.</p>" + words + this_run)
earlier_html = (f"<h3 style='margin:18px 0 4px'>Earlier tests (the last {len(earlier)}, the newest first)</h3>"
                + history)
page = ("<!doctype html><html><head><meta charset='utf-8'><title>Load test report</title></head>"
        "<body style='font-family:sans-serif;max-width:1200px;margin:20px auto;padding:0 16px'>"
        + summary
        + "".join(f"<p><img style='max-width:100%' src='data:image/png;base64,{png}'></p>" for png in charts.values())
        + earlier_html
        + "<p style='color:#777;font-size:12px'>Every test: the dataset load_test_log.</p></body></html>")
name = f"report_{str(batch_id).replace(':', '').replace(' ', '_')}.html"
for path in (name, "latest_report.html"):
    with folder.get_writer(path) as writer:
        writer.write(page.encode())
for insight, content, title in (("load_test_summary", summary, "this run"),
                                ("load_test_earlier", earlier_html, "earlier tests")):
    insights.save_data(insight, f"<div style='font-family:sans-serif'>{content}</div>".encode(), "text/html",
                       label=f"Load test: {title}")

project = dataiku.api_client().get_default_project()
existing = [d for d in project.list_dashboards(as_type="listitems") if d.name == DASHBOARD]
dashboard = existing[0].to_dashboard() if existing else project.create_dashboard(DASHBOARD)


def tile(insight_id, top, height):
    return {"tileType": "INSIGHT", "insightId": insight_id, "insightType": "static_file",
            "tileParams": {"numDisplayedComments": 5}, "displayMode": "INSIGHT", "clickAction": "DO_NOTHING",
            "box": {"left": 0, "top": top, "width": 36, "height": height},
            "titleOptions": {"showTitle": "NO"}, "resizeImageMode": "FIT_SIZE", "autoLoad": True}


tiles, top = [], 0
for insight_id, height in (("load_test_summary", 8 + 2 * len(run)), ("load_test_times", 5 + 2 * len(shown)),
                           ("load_test_steps", 5 + 2 * len(shown)), ("load_test_memory", 5 + 2 * len(shown)),
                           ("load_test_earlier", 4 + 2 * len(earlier))):
    tiles.append(tile(insight_id, top, height))
    top += height
settings = dashboard.get_settings()
settings.get_raw()["pages"] = [{"id": None, "title": "Last run", "show": True, "grid": {"tiles": tiles}}]
settings.save()
print(f"PROGRESS load test report: done: {name} and latest_report.html in the folder load_test_report, and the "
      f"dashboard {DASHBOARD!r}", flush=True)
