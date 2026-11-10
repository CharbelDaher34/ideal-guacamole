# Load test, zone Load test (scenario "Run the load test"): runs every test of the test plan (load_test_plan, set by
# step 1 of the scenario), one after the other, on the fake data in load_test_flows (made by the scenario "Create the fake
# data": the number of clients is always the data's), and adds one row per test to load_test_log, all with the same
# batch_id (this run). The recipe load_test_report then makes the report of the batch.
# Each test runs in its own process (forked from this one): its memory is measured on its own and freed at its end,
# and a test that crashes (out of memory, for example) fails alone, with status FAILED, and the next tests still run.
# Each test (loadtest.run_test): the placeholder model fitted on 2 clients, then the nightly forecast of every fake
# client with the forecast project's own code (forecast.forecast_all: `workers` groups of clients at a time, each
# read, prepared, forecast and written on its own), then the SQL load (forecast.load_to_sql). Its seconds:
#   reading, preparing, model, formatting, writing   added up over the groups (so over the workers too)
#   forecast                                         from start to end of the forecast (the wall clock)
#   sql_load                                         the forecast loaded into the SQL table load_test_final
import time

import pandas as pd

from cashflow import append_row
from forecast import STEPS
from loadtest import RUN, data_info, in_own_process, read_plan, run_test, scenario_numbers

LOG = {"batch_id": "string", "test_no": "bigint", "test": "string", "run_at": "string", "status": "string",
       "error": "string", "workers": "bigint", "clients": "bigint", "months": "bigint", "postings_per_month": "bigint",
       "clients_per_group": "bigint", "groups": "bigint", "model": "string", "input_days": "bigint",
       "cutoff_date": "string", "daily_rows": "bigint", "data_mb": "double", "seconds_fitting_2_clients": "double",
       **{f"seconds_{k}": "double" for k in STEPS}, "seconds_forecast": "double", "seconds_sql_load": "double",
       "clients_per_second": "double", "peak_memory_mb": "bigint", "bank_clients": "bigint",
       "minutes_at_bank_size": "double", "daily_rows_read": "bigint", "clients_ok": "bigint",
       "clients_not_enough_history": "bigint", "clients_no_postings": "bigint"}
numbers = scenario_numbers(RUN, ["load_test_plan", "bank_clients"])
plan = read_plan(numbers["load_test_plan"])
bank_clients = int(float(numbers["bank_clients"] or 420000))
batch_id = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
info = data_info()
if info is None:
    raise ValueError("No fake data in load_test_flows: run the scenario Create the fake data first")
data = {k: info.get(k) for k in ("clients", "months", "postings_per_month", "clients_per_group", "daily_rows",
                                 "data_mb")}
started = time.time()
print(f"PROGRESS load test batch {batch_id}: {len(plan)} tests on the fake data: {info['clients']:,} clients, "
      f"{info['months']} months, {info['postings_per_month']} postings a month, up to {info['cutoff']}", flush=True)
failed = 0
for i, test in enumerate(plan):
    print(f"PROGRESS test {i + 1}/{len(plan)}: {test['test']} ({test['workers']} workers; "
          f"{(time.time() - started) / 60:.1f} min so far)", flush=True)
    row = {"batch_id": batch_id, "test_no": i + 1, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
           "bank_clients": bank_clients, "cutoff_date": info["cutoff"], **data, **test}
    result = in_own_process(run_test, test, info)
    if result[0] == "ok":
        row.update(result[1], status="SUCCESS")
        seconds = row["seconds_forecast"] + row["seconds_sql_load"]
        row["minutes_at_bank_size"] = round(seconds / info["clients"] * bank_clients / 60, 1)
        print(f"PROGRESS test {test['test']}: forecast {row['seconds_forecast']:.0f} s + SQL load "
              f"{row['seconds_sql_load']:.0f} s, {row['clients_per_second']:,.0f} "
              f"clients a second, {row['peak_memory_mb']:,} MB at most; at the bank's size ({bank_clients:,} "
              f"clients): about {row['minutes_at_bank_size']:.0f} min", flush=True)
    else:
        failed += 1
        row.update(status="FAILED", error=result[1][:1000])
        print(f"PROGRESS test {test['test']}: FAILED: {result[1]}\n{result[2]}", flush=True)
    append_row("load_test_log", LOG, row)
print(f"PROGRESS load test batch {batch_id}: done in {(time.time() - started) / 60:.1f} min, "
      f"{len(plan) - failed} of {len(plan)} tests successful (load_test_log, then the report)", flush=True)
