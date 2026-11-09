# Load test, charts: the runs of load_test_log drawn on the dashboard "Load test" (Dashboards > Load test), after
# every run of the scenario Load test. Only reads load_test_log; the charts are also saved as files in the folder
# load_test_charts. The dashboard is made again each time (do not edit it by hand: copy it first).
#   1. seconds per step of each run (the last 15)
#   2. speed: clients forecast per second, by number of clients (each engine and model its own colour and mark)
#   3. memory: the most memory used, by number of clients
#   4. CSV or Parquet: seconds to write and to read back the forecast, each format (the last 15 runs)
#   5. the last 20 runs, as a table
import io

import dataiku
import matplotlib
import matplotlib.ticker
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dataiku import insights  # noqa: E402

DASHBOARD = "Load test"
STEPS = {"seconds_reading": "reading the daily flows", "seconds_preparing": "preparing (30-day sums, inputs)",
         "seconds_model": "model (predictions)", "seconds_formatting": "formatting (one row per client)",
         "seconds_writing_csv": "writing the forecast (CSV)", "seconds_writing_parquet": "writing the forecast (Parquet)"}
FORMATS = {"seconds_writing_csv": "write CSV", "seconds_writing_parquet": "write Parquet",
           "seconds_reading_back_csv": "read back CSV", "seconds_reading_back_parquet": "read back Parquet"}

log = dataiku.Dataset("load_test_log").get_dataframe()
for column in set(STEPS) | set(FORMATS) | {"clients", "clients_per_second", "peak_memory_mb"}:
    log[column] = pd.to_numeric(log.get(column), errors="coerce")
log["engine"] = log.get("engine", pd.Series(index=log.index, dtype=object)).fillna("pandas").replace("", "pandas")
log = log.sort_values("run_at")
done = log[log["status"] == "SUCCESS"]
last = done.tail(15)
label = lambda r: f"{r.run_at[5:16]} | {int(r.clients):,} clients | {r.model} | {r.engine}"
folder = dataiku.Folder("load_test_charts")
print(f"PROGRESS load test charts: {len(log)} runs in load_test_log, {len(done)} successful", flush=True)


def publish(insight_id, title, fig):
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    insights.save_figure(insight_id, fig, label=title)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    with folder.get_writer(f"{insight_id}.png") as writer:
        writer.write(buffer.getvalue())
    plt.close(fig)


def empty(ax, text="No successful run yet: run the scenario Load test"):
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")


# 1. seconds per step
fig, ax = plt.subplots(figsize=(11, 6))
if len(last):
    times = last.set_index(last.apply(label, axis=1))[list(STEPS)].fillna(0).rename(columns=STEPS)
    times.plot.barh(stacked=True, ax=ax, width=0.7)
    ax.set_xlabel("seconds")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=8)
else:
    empty(ax)
publish("load_test_times", "Seconds per step, each run (the last 15; generating the fake data not included)", fig)

# 2. speed and 3. memory, by number of clients
for insight_id, column, title, unit in (
        ("load_test_speed", "clients_per_second", "Speed: clients forecast per second (reading included)",
         "clients per second"),
        ("load_test_memory", "peak_memory_mb", "Memory: the most used by the forecast step", "MB")):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(done):
        for (engine, model), runs in done.groupby(["engine", "model"]):
            ax.plot(runs["clients"], runs[column], marker="o" if engine == "pandas" else "s", linestyle="",
                    markersize=8, label=f"{model}, {engine}")
        ax.set_xlabel("clients")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    else:
        empty(ax)
    publish(insight_id, title, fig)

# 4. CSV or Parquet
fig, ax = plt.subplots(figsize=(11, 6))
compared = last.dropna(subset=["seconds_writing_parquet"])
if len(compared):
    compared.set_index(compared.apply(label, axis=1))[list(FORMATS)].rename(columns=FORMATS).plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("seconds")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=8)
else:
    empty(ax, "No run with both formats yet: run the scenario Load test")
publish("load_test_formats", "CSV or Parquet: seconds to write and read back the forecast", fig)

# 5. the last runs, as a table
shown = ["run_at", "status", "model", "engine", "clients", "months", "postings_per_month", "daily_rows",
         "clients_per_chunk", "seconds_reading", "seconds_preparing", "seconds_model", "seconds_formatting",
         "seconds_writing_csv",
         "seconds_writing_parquet", "seconds_total", "clients_per_second", "peak_memory_mb", "error"]
table = log.reindex(columns=shown).tail(20).iloc[::-1]
html = ("<div style='font-family:sans-serif;font-size:12px'><b>The last 20 runs of the scenario Load test</b> "
        "(newest first; all of them in the dataset load_test_log)"
        + table.to_html(index=False, na_rep="", float_format=lambda v: f"{v:,.1f}", border=0) + "</div>")
html = html.replace("<table", "<table style='border-collapse:collapse' cellpadding='4'")
insights.save_data("load_test_runs", html.encode(), "text/html", label="The last 20 runs")
with folder.get_writer("load_test_runs.html") as writer:
    writer.write(html.encode())

# the dashboard: made again each time, with the 5 insights
project = dataiku.api_client().get_default_project()
found = [d for d in project.list_dashboards(as_type="listitems") if d.name == DASHBOARD]
dashboard = found[0].to_dashboard() if found else project.create_dashboard(DASHBOARD)


def tile(insight_id, left, top, width, height):
    return {"tileType": "INSIGHT", "insightId": insight_id, "insightType": "static_file",
            "tileParams": {"numDisplayedComments": 5}, "displayMode": "INSIGHT", "clickAction": "DO_NOTHING",
            "box": {"left": left, "top": top, "width": width, "height": height},
            "titleOptions": {"showTitle": "NO"}, "resizeImageMode": "FIT_SIZE", "autoLoad": True}


settings = dashboard.get_settings()
settings.get_raw()["pages"] = [{"id": None, "title": "Runs", "show": True, "grid": {"tiles": [
    tile("load_test_times", 0, 0, 18, 12), tile("load_test_formats", 18, 0, 18, 12),
    tile("load_test_speed", 0, 12, 18, 11), tile("load_test_memory", 18, 12, 18, 11),
    tile("load_test_runs", 0, 23, 36, 14)]}}]
settings.save()
print(f"PROGRESS load test charts: done, dashboard {DASHBOARD!r} updated (Dashboards > {DASHBOARD})", flush=True)
