"""Run the scenario Load test with several configurations, one after the other, on the sandbox, then print a short
report. Test machine only. Each configuration sets step 1 of the scenario (its variables), runs it and waits; the
fake data of a size is only generated once (load_test_data keeps it while the size does not change). All the
numbers are also in load_test_log and on the dashboard Load test. The scenario's own numbers are put back at the end.
    python3 sandbox/benchmark.py            # run the configurations below, then the report
    python3 sandbox/benchmark.py report     # the report only, from load_test_log
"""
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dss  # noqa: E402

PROJECT, SCENARIO = "DCM_FORECAST", "Load_test"
SIZE = {"loadtest_months": 24, "loadtest_postings_per_month": 40, "loadtest_model": "catboost"}
# one size is enough: the time grows in a straight line with the clients. 400,000: about the bank's ~420,000
RUNS = [(400_000, k) for k in (2_000, 5_000, 20_000)]  # clients, clients per chunk
READ_LOG = r"""
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("*.csv*")], ignore_index=True)
print(d.to_csv(index=False))
"""


def scenario():
    return dss.api("GET", "scenarios/get", {"projectKey": PROJECT, "scenarioId": SCENARIO})


def set_variables(data, variables):
    data["params"]["steps"][0]["params"]["variables"] = variables
    dss.api("POST", "scenarios/save", {"projectKey": PROJECT, "data": json.dumps(data), "saveInfo": "{}"})


def wait_idle():
    """Wait while a run of the scenario is still going (one started by hand, or by an earlier benchmark)."""
    while True:
        last = dss.api("GET", "scenarios/get-last-scenario-runs", {"projectKey": PROJECT, "smartScenarioId": SCENARIO,
                                                                  "withFullScenario": "false", "limit": "1",
                                                                  "onlyFinishedRuns": "false"})
        if not last or last[0].get("result"):
            return
        print("waiting: a run of the scenario is still going", flush=True)
        time.sleep(30)


def run_all():
    dss.login()
    wait_idle()
    before = dict(scenario()["params"]["steps"][0]["params"]["variables"])
    started = time.time()
    try:
        for i, (clients, chunk) in enumerate(RUNS, 1):
            dss.login()                                       # long runs: a fresh session each time
            set_variables(scenario(), {**SIZE, "loadtest_clients": clients,
                                       "loadtest_clients_per_chunk": chunk})
            print(f"\n[{i}/{len(RUNS)}] {clients:,} clients, {chunk:,} per chunk "
                  f"({(time.time() - started) / 60:.0f} min so far)", flush=True)
            dss.run_scenario(PROJECT, SCENARIO)
    finally:
        dss.login()
        set_variables(scenario(), before)
        print(f"\nDone in {(time.time() - started) / 60:.0f} min; step 1 put back to {before}")
    return started


def report(since=None):
    """The runs (CatBoost, successful) as one simple table per size, fastest first."""
    import pandas as pd
    text = subprocess.run(["docker", "exec", "-w", "/home/dataiku/dss/managed_datasets/DCM_FORECAST/load_test_log",
                           "dataiku-dss", "/home/dataiku/dss/code-envs/python/cashflow_forecast/bin/python", "-c",
                           READ_LOG], check=True, capture_output=True, text=True).stdout
    d = pd.read_csv(io.StringIO(text))
    d = d[(d["status"] == "SUCCESS") & (d["model"] == SIZE["loadtest_model"])]
    if since:
        d = d[d["run_at"] >= since]
    d = d.sort_values("run_at").drop_duplicates(["clients", "clients_per_chunk"], keep="last")
    d["minutes"] = ((d["seconds_reading"] + d["seconds_preparing"] + d["seconds_model"] + d["seconds_formatting"]
                     + d["seconds_writing_parquet"]) / 60).round(1)
    d["memory_gb"] = (d["peak_memory_mb"] / 1024).round(1)
    columns = {"clients_per_chunk": "per chunk", "minutes": "minutes", "memory_gb": "memory GB",
               "seconds_reading": "reading s", "seconds_preparing": "preparing s", "seconds_model": "model s",
               "seconds_formatting": "formatting s", "seconds_writing_csv": "write CSV s",
               "seconds_writing_parquet": "write Parquet s"}
    for clients, runs in d.groupby("clients"):
        print(f"\n{clients:,} clients ({runs['daily_rows'].iloc[0] / 1e6:,.0f}M daily rows), fastest first; "
              "minutes = reading + preparing + model + formatting + writing Parquet")
        print(runs.sort_values("minutes")[list(columns)].rename(columns=columns).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        run_all()
        report(since=start)
