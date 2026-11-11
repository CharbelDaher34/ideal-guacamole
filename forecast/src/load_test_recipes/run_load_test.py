# Load test, zone Load test: every test listed in step 1 of the scenario Run the load test, one after the other, each
# on the fake data of its size in load_test_flows (made by the step before, not timed), each in its own process
# (forked from this one: its memory is measured on its own, and a crash such as running out of memory is logged as
# FAILED and the next test still runs). Adds one row per test to load_test_log, all with the same batch (this run),
# written together at the end.
# A test (loadtest.run_test): the placeholder model fitted on 2 clients, then the nightly forecast of every fake
# client with the forecast project's own code (forecast.forecast_all: `workers` groups of clients at a time, each
# read, prepared, forecast and written on its own), then the SQL load (forecast.load_to_sql). Its seconds:
#   reading, preparing, model, formatting, writing   added up over the groups (so over the workers too)
#   forecast                                         from start to end of the forecast (the wall clock)
#   sql_load                                         the forecast loaded into the SQL table load_test_final
import pandas as pd

from cashflow import append_row
from forecast import STEPS
from loadtest import data_info, in_own_process, run_test, size_path, step1_tests

LOG = {"batch": "string", "test": "bigint", "run_at": "string", "status": "string", "error": "string",
       "workers": "bigint", "clients": "bigint", "months": "bigint", "postings_per_month": "bigint",
       "clients_per_group": "bigint", "groups": "bigint", "model": "string", "input_days": "bigint",
       "cutoff_date": "string", "daily_rows": "bigint", "data_mb": "double", "seconds_fitting_2_clients": "double",
       **{f"seconds_{k}": "double" for k in STEPS}, "seconds_forecast": "double", "seconds_sql_load": "double",
       "clients_per_second": "double", "peak_memory_mb": "bigint",
       "daily_rows_read": "bigint", "clients_ok": "bigint", "clients_not_enough_history": "bigint",
       "clients_no_postings": "bigint"}
tests = step1_tests()
batch = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
rows, failed = [], []
for n, test in enumerate(tests, 1):
    row = {"batch": batch, "test": n, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), **test}
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
        print(f"PROGRESS test {n}/{len(tests)}: forecast {row['seconds_forecast']:.0f} s + SQL load "
              f"{row['seconds_sql_load']:.0f} s, {row['clients_per_second']:,.0f} clients a second, "
              f"{row['peak_memory_mb']:,} MB at most", flush=True)
    else:
        row.update(status="FAILED", error=result[1][:1000])
        failed.append(f"test {n}: {result[1][:300]}")
        print(f"PROGRESS test {n}/{len(tests)}: FAILED: {result[1]}\n{result[2]}", flush=True)
    rows.append(row)
append_row("load_test_log", LOG, rows)
if failed:
    raise RuntimeError(f"{len(failed)} of {len(tests)} tests failed (the report still runs): " + "; ".join(failed))
