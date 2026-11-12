# Load test, zone Load test: every test set in step 1 of the scenario Run the load test, one after the other, each
# on the fake data of its size in load_test_flows (made by the step before, not timed), each in its own process
# (forked from this one: its memory is measured on its own, and a crash such as running out of memory is logged as
# FAILED and the next test still runs). Adds one row per test to load_test_log, all with the same batch (this run),
# written together at the end.
# A test (loadtest.run_test): the placeholder model fitted on 2 clients, then the nightly forecast of every fake
# client with the forecast project's own code (forecast.forecast_all: `workers` groups of clients at a time, each
# read, prepared, forecast and written on its own), then the final load (a CSV file, or the table FINAL_SCHEMA.
# FINAL_TABLE on FINAL_CONNECTION, as set by FINAL_LOAD and FINAL_* in the setup notebook: loadtest.run_test; on Oracle
# with oracledb, else through the dataset load_test_final, an output of this recipe then). Its seconds:
#   reading, preparing, model, formatting, writing   added up over the groups (so over the workers too)
#   forecast                                         from start to end of the forecast (the wall clock)
#   sql_load                                         the final load (the CSV file, or the table)
import pandas as pd

from cashflow import append_row
from forecast import STEPS
from loadtest import LOG_COLUMNS, data_info, in_own_process, log_to_sql, machine, run_test, size_path, step1_tests

LOG = LOG_COLUMNS        # load_test_log's columns (loadtest.py), also those of its SQL copy (log_to_sql)
assert all(f"seconds_{k}" in LOG for k in STEPS), "loadtest.LOG_COLUMNS lacks a step of forecast.STEPS"
tests = step1_tests()
batch = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
rows, failed = [], []
for n, test in enumerate(tests, 1):
    row = {"batch": batch, "test": n, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), **test, **machine()}
    print(f"PROGRESS test {n}/{len(tests)}: machine at the start: " + ", ".join(
        f"{k} {row[k]}" for k in ("cpus", "cpu_limit", "ram_gb", "ram_limit_gb", "ram_free_gb", "disk_gb",
                                  "disk_free_gb")), flush=True)
    print(f"PROGRESS test {n}/{len(tests)}: {test['workers']} workers, {test['clients']:,} clients, {test['months']} "
          f"months, {test['postings_per_month']} postings a month, {test['clients_per_group']:,} per group", flush=True)
    info = data_info(test)
    if info is None:
        result = ("error", f"no fake data {size_path(test)[1:]} in load_test_flows: run the scenario Run the load "
                           "test (its step 2 makes it), not the recipe alone", "")
    else:
        row.update({k: info.get(k) for k in ("daily_rows", "data_mb")}, cutoff_date=info["cutoff"])
        result = in_own_process(run_test, test, info)
    if result[0] == "ok":
        row.update(result[1], status="SUCCESS")
        print(f"PROGRESS test {n}/{len(tests)}: forecast {row['seconds_forecast']:.0f} s + final load "
              f"{row['seconds_sql_load']:.0f} s, {row['clients_per_second']:,.0f} clients a second, "
              f"{row['peak_memory_mb']:,} MB at most", flush=True)
    else:
        row.update(status="FAILED", error=result[1][:1000])
        failed.append(f"test {n}: {result[1][:300]}")
        print(f"PROGRESS test {n}/{len(tests)}: FAILED: {result[1]}\n{result[2]}", flush=True)
    rows.append(row)
append_row("load_test_log", LOG, rows)
try:                                # the SQL copy of the log, when the setup notebook set one (LOG_SQL_*)
    done = log_to_sql(rows, LOG)
    print(f"PROGRESS the SQL log: {done or 'none set (LOG_SQL_CONNECTION in the setup notebook)'}", flush=True)
except Exception as e:              # the rows are in load_test_log all the same; the job fails below to show it
    failed.append(f"the SQL log could not be written ({type(e).__name__}: {str(e)[:500]})")
    print(f"PROGRESS the SQL log: FAILED: {e}", flush=True)
if failed:
    raise RuntimeError(f"{len(failed)} problem(s) (the report still runs): " + "; ".join(failed))
