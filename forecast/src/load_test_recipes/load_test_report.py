# Load test, the report of the last run of the scenario Run the load test (its tests: the rows of load_test_log with
# the same batch). The number of clients is not a setting (it is how many clients the bank has), so each client count
# is shown on its own, and only the settings (workers, clients per group) are compared, at the same client count:
#   1. the run: what a test times, the model, the fake data, the machine (CPUs, RAM)
#   2. one section per client count: a table (one line per setting: the forecast's 5 parts in seconds, the forecast,
#      the final load, the total, memory, compared with the best setting at this client count) and a chart of the
#      same times. The parts are measured added up over the workers (they run at the same time), so each part is
#      shown as its share of that work times the forecast's own time: the 5 add up to the forecast.
#   3. where the forecast's time goes, per setting, at the largest client count
#   4. each setting's time per 100,000 clients, at each client count
#   5. the earlier tests, as a table
# In the folder load_test_report: report_<batch>.html (one per run, kept) and latest_report.html, one page each; and
# the dashboard "Load test" (Dashboards > Load test), made again each time (do not edit it by hand: copy it first).
# A test's time: the forecast from start to end (the wall clock) plus the final load (a CSV file or the SQL table
# load_test_final, as set by FINAL_LOAD in the setup notebook); not making the fake data nor
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
PER = 100_000                   # section 4: the time per this many clients
STEPS = {"seconds_reading": "reading", "seconds_preparing": "preparing", "seconds_model": "model",
         "seconds_formatting": "formatting", "seconds_writing": "writing"}
STEP_COLORS = ["#2a6f97", "#61a5c2", "#e07a5f", "#f2cc8f", "#81b29a"]
FINAL_COLOR = "#7d6b9e"         # the final load, after the forecast's 5 parts (STEP_COLORS)
NUMBERS = ["test", "workers", "clients", "months", "postings_per_month", "clients_per_group", "seconds_forecast",
           "seconds_sql_load", "peak_memory_mb", "cpus", "ram_gb", "cpu_limit", "ram_limit_gb", "ram_free_gb", "disk_gb",
           "disk_free_gb", *STEPS]
CELL = "padding:5px 9px;border-bottom:1px solid #ddd;text-align:left"
BEST = "background:#e8f5e9;font-weight:bold"
NOTE = "margin:2px 0 8px;color:#555;font-size:13px"

log = dataiku.Dataset("load_test_log").get_dataframe()
if log.empty:
    raise ValueError("load_test_log is empty: run the scenario Run the load test")
for column in NUMBERS:
    log[column] = pd.to_numeric(log[column], errors="coerce") if column in log else float("nan")
log["batch"] = log["batch"].fillna("").astype(str) if "batch" in log else ""
log["batch"] = log["batch"].where(log["batch"] != "", log["run_at"])      # tests from before batches: one run each
log["test"] = log["test"].fillna(1)
log["seconds_total"] = log["seconds_forecast"] + log["seconds_sql_load"]
log = log.sort_values(["run_at", "test"]).reset_index(drop=True)
batch_id = log.iloc[-1]["batch"]
run = log[log["batch"] == batch_id].sort_values("test")             # this run's tests, in their order
earlier = log[log["batch"] != batch_id].tail(EARLIER).iloc[::-1]     # the newest first
ok = run[run["status"] == "SUCCESS"]
data_varies = len(run[["months", "postings_per_month"]].drop_duplicates()) > 1
folder = dataiku.Folder("load_test_report")
dashboard_error = []        # why the dashboard could not be made (the HTML report is made all the same)


def to_dashboard(call, *args, **kwargs):
    """A call that makes the dashboard (an insight): when Dataiku refuses it (a licence limit, a right), the report
    goes on without the dashboard."""
    if dashboard_error:
        return
    try:
        call(*args, **kwargs)
    except Exception as e:
        dashboard_error.append(f"{type(e).__name__}: {e}")
        print(f"PROGRESS the dashboard is not made ({dashboard_error[0][:300]}): the HTML report still is", flush=True)


def duration(seconds):
    """3.2 s, 45 s, 2.5 min, 1.2 h."""
    if pd.isna(seconds):
        return ""
    return (f"{seconds:,.1f} s" if seconds < 10 else f"{seconds:,.0f} s" if seconds < 90
            else f"{seconds / 60:,.1f} min" if seconds < 5400 else f"{seconds / 3600:,.1f} h")


def gb(mb):
    return "" if pd.isna(mb) else f"{mb / 1024:,.1f} GB"


def setting(t, full=data_varies):
    """A test's setting: 4 workers · 5,000 per group (and its fake data's months and postings when they vary)."""
    workers = int(t["workers"])
    text = f"{workers} worker{'s' if workers != 1 else ''} · {int(t['clients_per_group']):,} per group"
    if full and pd.notna(t["months"]):
        text += f" · {int(t['months'])} months · {int(t['postings_per_month'])} postings/month"
    return text


def clock_parts(tests):
    """The forecast's 5 parts in seconds of the clock, one row per test: each part's share of the work (added up over
    the workers) times the forecast's own time, so they add up to the forecast."""
    work = tests[list(STEPS)].astype(float)
    return work.div(work.sum(axis=1).where(lambda total: total > 0), axis=0).mul(tests["seconds_forecast"], axis=0)


def table(headers, rows, styles=None):
    head = "".join(f"<th style='{CELL};background:#f3f5f7'>{h}</th>" for h in headers)
    body = "".join(f"<tr style='{(styles or {}).get(i, '')}'>" + "".join(f"<td style='{CELL}'>{v}</td>" for v in row)
                   + "</tr>" for i, row in enumerate(rows))
    return f"<table style='border-collapse:collapse;font-size:13px;margin:6px 0 14px'>{head}{body}</table>"


def failed(t):
    return f"<span style='color:#b00020'>FAILED: {html.escape(str(t['error'])[:200])}</span>"


def figure(fig, name):
    """The chart saved as an insight (for the dashboard) and a PNG in the folder; returns the PNG, base64."""
    fig.tight_layout()
    to_dashboard(insights.save_figure, f"load_test_{name}", fig, label=f"Load test: {name}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    with folder.get_writer(f"{name}.png") as writer:
        writer.write(buffer.getvalue())
    return base64.b64encode(buffer.getvalue()).decode()


def image(png):
    return f"<p><img style='max-width:100%' src='data:image/png;base64,{png}'></p>"


def bar_chart(title, shown, xlabel):
    """A horizontal bar per test of `shown` (its setting), the first at the top. Returns the figure and its axes."""
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.45 * max(len(shown), 1)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, pad=24)
    ax.set_xlabel(xlabel)
    return fig, ax, [setting(t) for _, t in shown.iloc[::-1].iterrows()], shown.iloc[::-1]


# 1. the run
first = run.iloc[0]
model = html.escape(str(run.iloc[-1].get("model") or "daily_catboost"))


def gb(v):
    return f"{v / 1024:,.1f} TB" if v >= 1024 else f"{v:,.0f} GB" if v >= 10 else f"{v:,.1f} GB"


def recorded(column):
    """The column's value at the run's first test (the machine as the run started), None when not recorded (tests from
    before it was measured: then the report says so)."""
    return first[column] if pd.notna(first[column]) else None


cpus, cpu_cap = recorded("cpus"), recorded("cpu_limit")
ram, ram_cap, ram_free = recorded("ram_gb"), recorded("ram_limit_gb"), recorded("ram_free_gb")
disk, disk_free = recorded("disk_gb"), recorded("disk_free_gb")
measured = run[["ram_free_gb", "disk_gb"]].notna().any().any()     # this run's code measured the limits too
cpu_text = ("not recorded" if cpus is None else f"{int(cpus)} CPUs it may run on" + (
    f", but the job may use only {cpu_cap:g} CPUs' worth of time (its limit)" if cpu_cap is not None and cpu_cap < cpus
    else ", no CPU limit on the job" if measured else ""))
ram_text = ("not recorded" if ram is None else f"{gb(ram)} in the server" + (
    f", the job is limited to {gb(ram_cap)}" if ram_cap is not None
    else ", no memory limit on the job" if measured else "")
    + (f"; {gb(ram_free)} free when the run started" if ram_free is not None else ""))
disk_text = (f"{gb(disk_free)} free of {gb(disk)}, on the disk of the folder load_test_forecast"
             if disk is not None and disk_free is not None else "not recorded")
machine = (f"<ul style='margin:2px 0 0'><li>CPUs: {cpu_text}</li><li>RAM: {ram_text}</li><li>Disk: {disk_text}</li>"
           "</ul><span style='color:#555'>Read from Linux as each test starts (/proc, the job's cgroup: the limits set "
           "on its container or service, the folder's disk), the values of the first test. The server's other jobs share them.</span>")
final_to = sorted({str(v) for v in run.get("final_load_to", pd.Series(dtype=str)).dropna() if str(v)})
final_to = html.escape(" / ".join(final_to) or "the SQL table")
data = ("as in each setting" if data_varies else
        f"{int(first['months'])} months of daily flows per client, about {int(first['postings_per_month'])} postings "
        "a month per client")
header = (
    f"<h2 style='margin:0 0 4px'>Load test, {html.escape(str(batch_id))}: {len(run)} test{'s' if len(run) > 1 else ''}"
    "</h2>"
    "<p style='margin:0 0 6px;color:#555'>Each test: the nightly forecast of every fake client, then its final load "
    f"into {final_to}. <b>Total = forecast + final load</b> (making the fake data is not timed). The tests ran one "
    "after the other on the same machine.</p>"
    "<p style='margin:0 0 6px'><b>How to read it:</b> the number of clients is not a setting, it is how many clients "
    "the bank has. So each client count has its own table, and only the settings (workers, clients per group) are "
    "compared there.</p>"
    f"<ul style='margin:0 0 10px;font-size:13px'><li>Machine: {machine}</li><li>Fake data: {data}</li>"
    f"<li>Model: {model}, fitted on 2 clients only (its forecasts mean nothing, its time is real)</li></ul>")
if len(run) > len(ok):
    header += f"<p style='color:#b00020'>{len(run) - len(ok)} of {len(run)} tests failed: see the tables.</p>"
if not len(ok):
    header += "<p style='color:#b00020'>No test of this run succeeded.</p>"

# 2. one section per client count
sections = []          # (client count, its table's HTML, its chart's PNG or None, its number of tests)
for clients, tests in run.groupby("clients", sort=True):
    good = tests[tests["status"] == "SUCCESS"]
    best = good["seconds_total"].min() if len(good) else None
    parts = clock_parts(tests)
    lines, styles = [], {}
    for i, (index, t) in enumerate(tests.iterrows()):
        if t["status"] != "SUCCESS":
            lines.append([html.escape(setting(t)), failed(t)] + [""] * (len(STEPS) + 5))
            continue
        is_best = t["seconds_total"] == best
        if is_best:
            styles[i] = BEST
        lines.append([html.escape(setting(t)), *[duration(parts.loc[index, c]) for c in STEPS],
                      f"<b>{duration(t['seconds_forecast'])}</b>", duration(t["seconds_sql_load"]),
                      f"<b>{duration(t['seconds_total'])}</b>", gb(t["peak_memory_mb"]),
                      "the best here" if is_best else f"{t['seconds_total'] / best - 1:.0%} longer"])
    section = (f"<h3 style='margin:22px 0 2px'>{int(clients):,} clients</h3>"
               f"<p style='{NOTE}'>Each setting with {int(clients):,} fake clients. Compared only with the other "
               "settings at this client count. The forecast = its 5 parts: reading the daily flows, preparing the "
               "model's input rows, the model's prediction, formatting one row per client, writing the Parquet files "
               "(each part: its share of the workers' work × the forecast's time, so they add up to the forecast). "
               "Then the final load.</p>"
               + table(["Setting", *[s.capitalize() for s in STEPS.values()], "Forecast", "Final load", "Total",
                        "Memory (most used)", "Compared with the best here"], lines, styles))
    png = None
    if len(good):
        fig, ax, names, shown = bar_chart(f"{int(clients):,} clients: the total time per setting, the forecast's 5 "
                                          "parts then the final load (shorter is better)", good, "seconds")
        left = pd.Series(0.0, index=shown.index)
        for (column, step), color in zip(STEPS.items(), STEP_COLORS):
            ax.barh(names, parts.loc[shown.index, column].fillna(0), left=left, color=color, label=step)
            left += parts.loc[shown.index, column].fillna(0)
        ax.barh(names, shown["seconds_sql_load"], left=shown["seconds_forecast"], color=FINAL_COLOR,
                label="final load")
        for y, total in enumerate(shown["seconds_total"]):
            ax.text(total, y, f"  {duration(total)}", va="center", fontsize=9)
        ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False)
        png = figure(fig, f"{int(clients)}_clients")
    sections.append((int(clients), section, png, len(tests)))

# 3. where the forecast's time goes, at the largest client count
steps_png, steps_html, steps_rows = None, "", 0
if len(ok):
    largest = ok["clients"].max()
    shown_tests = ok[ok["clients"] == largest]
    fig, ax, names, shown = bar_chart(f"Where the forecast's time goes, per setting, at {int(largest):,} clients "
                                      "(share of the workers' time)", shown_tests, "% of the forecast's work")
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
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False)
    steps_png, steps_rows = figure(fig, "steps"), len(shown_tests)
    steps_html = ("<h3 style='margin:22px 0 2px'>Where the forecast's time goes</h3>"
                  f"<p style='{NOTE}'>reading: the daily flows; preparing: the model's input rows; model: its "
                  "prediction; formatting: one row per client; writing: the forecast's files. The final load is not in "
                  "it.</p>")

# 4. each setting's time per 100,000 clients, at each client count
counts = sorted(int(c) for c in ok["clients"].unique())
rate_lines = []
for name, tests in ok.assign(setting=ok.apply(setting, axis=1)).groupby("setting", sort=False):
    per_count = tests.groupby("clients")["seconds_total"].mean()
    lost = sorted(int(t["clients"]) for _, t in run.iterrows() if t["status"] != "SUCCESS" and setting(t) == name)
    rate_lines.append([html.escape(name) + "".join(f"<br><span style='color:#b00020'>failed at {c:,} clients</span>"
                                                   for c in lost),
                       *[duration(per_count[c] / c * PER) if c in per_count.index else "" for c in counts],
                       gb(tests["peak_memory_mb"].max())])
rate_html = (
    f"<h3 style='margin:22px 0 2px'>Time per {PER:,} clients</h3>"
    f"<p style='{NOTE}'>A test's total time (forecast + final load) ÷ its clients × {PER:,}, for each setting at each "
    "client count. When the values in a row are close, the time grows in step with the clients.</p>"
    + (table(["Setting", *[f"At {c:,} clients" for c in counts], "Memory (most used)"], rate_lines)
       if rate_lines else "<p>No successful test.</p>"))


# 5. the earlier tests
def earlier_row(t):
    done = t["status"] == "SUCCESS"
    return [html.escape(str(t["batch"])), int(t["test"]),
            f"{int(t['clients']):,}" if pd.notna(t["clients"]) else "",
            html.escape(setting(t, full=True)) if pd.notna(t["workers"]) else "",
            duration(t["seconds_forecast"]), duration(t["seconds_sql_load"]),
            duration(t["seconds_total"]) if done else "", gb(t["peak_memory_mb"]), html.escape(str(t["status"]))]


earlier_html = (f"<h3 style='margin:22px 0 4px'>Earlier tests (the last {len(earlier)}, the newest first)</h3>"
                + (table(["Run", "Test", "Clients", "Setting", "Forecast", "Final load", "Total", "Memory", "Status"],
                         [earlier_row(t) for _, t in earlier.iterrows()]) if len(earlier) else "<p>None yet.</p>"))

# the page
page = ("<!doctype html><html><head><meta charset='utf-8'><title>Load test report</title></head>"
        "<body style='font-family:sans-serif;max-width:1200px;margin:20px auto;padding:0 16px'>"
        + header
        + "".join(section + (image(png) if png else "") for _, section, png, _ in sections)
        + (steps_html + image(steps_png) if steps_png else "")
        + rate_html + earlier_html
        + "<p style='color:#777;font-size:12px'>Every test: the dataset load_test_log.</p></body></html>")
name = f"report_{str(batch_id).replace(':', '').replace(' ', '_')}.html"
for path in (name, "latest_report.html"):
    with folder.get_writer(path) as writer:
        writer.write(page.encode())

# the dashboard: the same parts, one tile each, in the same order
tiles_wanted = []                   # (insight ID, height)


def html_insight(insight_id, content, title, height):
    to_dashboard(insights.save_data, insight_id, f"<div style='font-family:sans-serif'>{content}</div>".encode(),
                 "text/html", label=f"Load test: {title}")
    tiles_wanted.append((insight_id, height))


html_insight("load_test_summary", header, "this run", 10)
for clients, section, png, n in sections:
    html_insight(f"load_test_{clients}_table", section, f"{clients:,} clients", 6 + 2 * n)
    if png:
        tiles_wanted.append((f"load_test_{clients}_clients", 5 + 2 * n))
if steps_png:
    html_insight("load_test_steps_text", steps_html, "where the time goes", 4)
    tiles_wanted.append(("load_test_steps", 5 + 2 * steps_rows))
html_insight("load_test_per_100k", rate_html, f"time per {PER:,} clients", 6 + 2 * len(rate_lines))
html_insight("load_test_earlier", earlier_html, "earlier tests", 4 + 2 * len(earlier))

def tile(insight_id, top, height):
    return {"tileType": "INSIGHT", "insightId": insight_id, "insightType": "static_file",
            "tileParams": {"numDisplayedComments": 5}, "displayMode": "INSIGHT", "clickAction": "DO_NOTHING",
            "box": {"left": 0, "top": top, "width": 36, "height": height},
            "titleOptions": {"showTitle": "NO"}, "resizeImageMode": "FIT_SIZE", "autoLoad": True}


def make_dashboard():
    project = dataiku.api_client().get_default_project()
    existing = [d for d in project.list_dashboards(as_type="listitems") if d.name == DASHBOARD]
    dashboard = existing[0].to_dashboard() if existing else project.create_dashboard(DASHBOARD)
    tiles, top = [], 0
    for insight_id, height in tiles_wanted:
        tiles.append(tile(insight_id, top, height))
        top += height
    settings = dashboard.get_settings()
    settings.get_raw()["pages"] = [{"id": None, "title": "Last run", "show": True, "grid": {"tiles": tiles}}]
    settings.save()


to_dashboard(make_dashboard)
print(f"PROGRESS load test report: done: {name} and latest_report.html in the folder load_test_report"
      + (f", not the dashboard ({dashboard_error[0][:200]})" if dashboard_error else f", and the dashboard {DASHBOARD!r}"),
      flush=True)
