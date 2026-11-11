"""Load test (project library of the load test project: python/loadtest.py). Times the nightly forecast of the
forecast project on fake data of the same shape, with the same code: cashflow.py, models.py and forecast.py are copies
of the forecast project's, made by the same setup files. Nothing here reads or writes the forecast project.

One scenario, "Run the load test". Its step 1 sets the tests, run one after the other in one go (step1_tests):
    {"tests": {"clients": [100000, 400000, 800000], "months": 24, "postings_per_month": 60,
               "clients_per_group": [2000, 5000], "workers": [2, 4, 8]}}
tests is a grid (each number, or a list of numbers: every combination is a test, the clients first, the workers last)
or a list of tests, each with every number, [{"clients": ..., "months": ..., ..., "workers": ...}, ...]. The numbers:
    clients              how many fake clients
    months               months of daily flows per client up to the day they are made (at least MIN_MONTHS)
    postings_per_month   the average number of postings per client and month
    clients_per_group    clients per group (production: cashflow.CLIENTS_PER_GROUP)
    workers              groups of clients forecast at a time (production: the project variable inference_workers)
Zone Data prep (recipe make_fake_data, not timed): the fake daily flows of each size listed (the first 4 numbers) in
load_test_flows, one subfolder per size (size_path), each laid out like daily_flows (one file per group of clients and
month), made only when not there yet; nothing is deleted by code (remove unused sizes by hand).
Zone Load test (recipe run_load_test, then load_test_report): each test on its size's data: the placeholder model
(models.DailyCatBoost) fitted on 2 clients only (its forecasts mean nothing, its prediction time is real), the nightly
forecast of every fake client (forecast.forecast_all) into the folder load_test_forecast, then its final load, timed
(run_test), in its own process (in_own_process). The final load is set by FINAL_LOAD in the setup notebook (the project
variable final_load): "csv", one CSV file in the folder load_test_forecast (load_to_file: no table, no dataset), or
"sql", the SQL table load_test_final (forecast.load_to_sql, as production loads final_forecast)."""
import itertools
import json
import multiprocessing
import os
import time
import traceback

import numpy as np
import pandas as pd

from cashflow import INPUT_DAYS, PERIOD_DAYS, Progress, client_groups, local_path, whole_number, write_daily
from warehouse import CATEGORY_OF_TYPE, COLUMNS, OTHER

SCENARIO = "Run the load test"
FLOWS = "load_test_flows"         # folder: the fake daily flows, laid out like daily_flows
OUTPUT = "load_test_forecast"     # folder: the forecast, like the folder forecast
SQL = "load_test_final"           # final load "sql": the SQL table, like final_forecast
FINAL_FILE = "/final/load_test_final.csv"    # final load "csv": this file of the folder OUTPUT
MIN_MONTHS = -(-(INPUT_DAYS + PERIOD_DAYS * 3) // PERIOD_DAYS)    # a year of input and 3 windows: 16 months
SIZE = ("clients", "months", "postings_per_month", "clients_per_group")     # the fake data's size
TEST = SIZE + ("workers",)                                                  # the numbers of one test


def check(test, n=1):
    """One test of step 1 (TEST: whole numbers, months at least MIN_MONTHS), checked. n: its place in the list."""
    if not isinstance(test, dict):
        raise ValueError(f"test {n} must be an object {{\"clients\": ..., ...}}: set it in step 1 of the scenario "
                         f"{SCENARIO}")
    checked = {}
    for name in TEST:
        checked[name] = whole_number(name, test.get(name))
        if checked[name] is None:
            raise ValueError(f"test {n}: {name} must be a whole number (1 or more): set it in step 1 of the scenario "
                             f"{SCENARIO}")
    if checked["months"] < MIN_MONTHS:
        raise ValueError(f"test {n}: months is {checked['months']}: it must be at least {MIN_MONTHS} (a year of input, "
                         f"{INPUT_DAYS} days, and the 3 windows after it)")
    return checked


def grid_tests(grid):
    """A grid {"clients": [100000, 400000], "workers": [2, 4], "months": 24, ...} (each a number or a list): every
    combination, as a list of tests, in the order of TEST (the clients first, the workers last)."""
    lists = {name: value if isinstance(value, list) else [value] for name, value in grid.items()}
    return [dict(zip(TEST, numbers)) for numbers in itertools.product(*(lists.get(name, [None]) for name in TEST))]


def check_tests(tests):
    """The tests of step 1 (a grid, or a list of tests), as a list, each checked."""
    if isinstance(tests, dict):
        tests = grid_tests(tests)
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"tests must be a grid {{\"clients\": [...], \"workers\": [...], ...}} or a list of tests "
                         f"[{{\"clients\": ..., ...}}, ...]: set it in step 1 of the scenario {SCENARIO}")
    return [check(test, n) for n, test in enumerate(tests, 1)]


def step1_variables():
    """The variables set in step 1 of the scenario, read from its settings, so they are the same whether the scenario
    runs, a recipe is built from the Flow or the setup notebook runs it."""
    import dataiku
    project = dataiku.api_client().get_default_project()
    ids = {s["name"]: s["id"] for s in project.list_scenarios()}
    steps = (project.get_scenario(ids[SCENARIO]).get_settings().get_raw()["params"].get("steps", [])
             if SCENARIO in ids else [])
    return steps[0]["params"].get("variables", {}) if steps and steps[0].get("type") == "define_vars" else {}


def step1_tests():
    """The tests set in step 1 of the scenario, as a list, each checked."""
    return check_tests(step1_variables().get("tests"))


def machine():
    """The machine the tests run on: its CPUs (the ones this process may use) and its RAM in GB (the container's
    limit when it is lower)."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    ram = None
    try:
        with open("/proc/meminfo") as f:
            ram = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemTotal:"))
        for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            if os.path.exists(limit):
                with open(limit) as f:
                    value = f.read().strip()
                if value.isdigit():
                    ram = min(ram, int(value))
                break
    except (OSError, StopIteration):
        pass
    return {"cpus": cpus, "ram_gb": round(ram / 1024 ** 3, 1) if ram else None}


def size_path(test):
    """The subfolder of load_test_flows holding the fake data of a test's size."""
    return (f"/{test['clients']}_clients_{test['months']}_months_{test['postings_per_month']}_postings_"
            f"{test['clients_per_group']}_per_group")


def data_info(test):
    """What fake data of this test's size is in load_test_flows (its data.json: size, cut-off day, rows, MB), or
    None."""
    import dataiku
    path = local_path(dataiku.Folder(FLOWS), f"{size_path(test)}/data.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def in_own_process(function, *args):
    """function(*args) in a forked process: its memory is its own (measured on its own, freed at its end), and a
    crash (out of memory, for example) stops it alone. Returns ("ok", its result) or ("error", the error, details)."""
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def target():
        try:
            send.send(("ok", function(*args)))
        except BaseException as e:
            send.send(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))
        finally:
            send.close()

    process = context.Process(target=target)
    process.start()
    send.close()                        # only the child sends: when it ends without a word, recv stops
    try:
        result = receive.recv()
    except EOFError:
        result = None
    process.join()
    if result is None:
        code = process.exitcode
        why = " (killed: most likely out of memory)" if code in (-9, 137) else ""
        result = ("error", f"the process stopped with code {code}{why}", "")
    return result


def fake_clients(n):
    """The fake clients: client_no LT00000000, LT00000001, ..., all ACTIVE (the columns of the clients dataset)."""
    clients = pd.DataFrame({column: "LOAD TEST" for column in COLUMNS["clients"]}, index=range(n))
    clients["client_no"] = np.array([f"LT{i:08d}" for i in range(n)])
    clients["status"] = "ACTIVE"
    return clients


def fake_data(wanted, cutoff):
    """Zone Data prep: the fake daily flows of the size `wanted` (SIZE) in their subfolder of load_test_flows
    (size_path): kept when they are already there (whatever their cut-off day), else made up to `cutoff`. Returns what
    is there (data.json). Each client gets its own rate around the average (a few busy clients, many quiet ones), and 10% of
    them a shorter history, so some get NOT_ENOUGH_HISTORY, as in production."""
    import dataiku
    started = time.time()
    wanted, prefix = {k: wanted[k] for k in SIZE}, size_path(wanted)
    info = data_info(wanted)
    if info:
        print(f"PROGRESS fake data {prefix[1:]}: already there (up to {info['cutoff']}): kept", flush=True)
        return info
    folder = dataiku.Folder(FLOWS)
    clients, months, postings = wanted["clients"], wanted["months"], wanted["postings_per_month"]
    groups = max(1, -(-clients // wanted["clients_per_group"]))
    days = months * PERIOD_DAYS
    categories = np.array(sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER}))
    numbers = fake_clients(clients)["client_no"].to_numpy()
    group_of = client_groups(numbers, groups).to_numpy()
    rng = np.random.default_rng(7)
    totals = {"postings": 0, "rows": 0}
    print(f"PROGRESS fake data {prefix[1:]}: {clients:,} clients ({groups:,} groups) x {months} months x about {postings} postings "
          f"a month = about {clients * months * postings:,} postings, up to {cutoff:%Y-%m-%d}", flush=True)

    def parts():
        """The fake daily flows, one group of clients at a time."""
        progress = Progress("fake data: writing the daily flows (groups of clients)", groups)
        for g in range(groups):
            who = numbers[group_of == g]
            n = len(who)
            rate = postings * rng.gamma(2.0, 0.5, n)                       # postings a month, average `postings`
            span = np.where(rng.random(n) < 0.1, rng.integers(30, days + 1, n), days)   # days of history
            client = np.repeat(np.arange(n), rng.poisson(rate * span / PERIOD_DAYS))
            amount = rng.lognormal(8, 1.2, n)[client] * rng.lognormal(0, 0.8, len(client))     # SAR
            money_in = rng.random(len(client)) < 0.55
            one = pd.DataFrame({
                "client": client,
                "back": (rng.random(len(client)) * span[client]).astype(int),     # days before the cut-off day
                "category": rng.integers(0, len(categories), len(client)),
                "money_in": np.where(money_in, amount, 0.0), "money_out": np.where(money_in, 0.0, amount)})
            daily = (one.groupby(["client", "back", "category"])
                     .agg(money_in=("money_in", "sum"), money_out=("money_out", "sum"), postings=("money_in", "size"))
                     .reset_index())
            totals["postings"] += len(one)
            totals["rows"] += len(daily)
            progress.add()
            yield pd.DataFrame({"client_no": who[daily["client"].to_numpy()],
                                "day": (cutoff - pd.to_timedelta(daily["back"].to_numpy(), unit="D")).values,
                                "category": categories[daily["category"].to_numpy()],
                                "money_in": daily["money_in"].round(2).to_numpy(),
                                "money_out": daily["money_out"].round(2).to_numpy(),
                                "postings": daily["postings"].to_numpy()})
        progress.finish()

    manifest = write_daily(parts(), groups, FLOWS, prefix=prefix)
    info = {**wanted, "groups": groups, "cutoff": f"{cutoff:%Y-%m-%d}", "postings": totals["postings"],
            "daily_rows": manifest["rows"], "files": sum(len(m) for m in manifest["files"].values()),
            "data_mb": round(sum(folder.get_path_details(p).get("size", 0)
                                 for p in folder.list_paths_in_partition() if p.startswith(prefix + "/")) / 1e6, 1),
            "seconds_generating": round(time.time() - started, 1)}
    with folder.get_writer(f"{prefix}/data.json") as writer:
        writer.write(json.dumps(info).encode())
    print(f"PROGRESS fake data: done in {info['seconds_generating']:.0f} s: {info['postings']:,} postings, "
          f"{info['daily_rows']:,} daily rows, {info['files']:,} files ({info['data_mb']} MB)", flush=True)
    return info


def prepare_data(tests, cutoff):
    """Zone Data prep: the fake data of every size the tests use (fake_data), made only when not there yet.
    Nothing is deleted: remove the sizes no longer needed by hand (the folder load_test_flows).
    Returns {size_path: data.json}."""
    sizes = {size_path(t): t for t in tests}
    return {prefix: fake_data(test, cutoff) for prefix, test in sizes.items()}


def final_load():
    """The final load of the tests, "csv" or "sql": the project variable final_load (set by the setup notebook)."""
    import dataiku
    return dataiku.get_custom_variables().get("final_load", "sql")


def load_to_file(output=OUTPUT):
    """The final load "csv": the current forecast of the folder `output` (one Parquet file per group) into one CSV
    file of the same folder (FINAL_FILE), replacing the previous one, one group's file at a time."""
    from cashflow import read_parquet
    from forecast import current_output
    folder, version, paths = current_output(output)
    with folder.get_writer(FINAL_FILE) as writer:
        for n, path in enumerate(paths):
            writer.write(read_parquet(folder, path).to_pandas().to_csv(header=n == 0, index=False).encode())


def run_test(test, info):
    """Zone Load test: one test, on the fake data of its size in load_test_flows (info: its data.json): fit option A
    on 2 clients, forecast every fake client as the nightly forecast does with `workers`, then load the forecast into
    the CSV file or the SQL table (final_load). Returns the values for load_test_log."""
    from cashflow import read_group, read_manifest
    from forecast import STEPS, forecast_all, load_to_sql
    from models import DailyCatBoost
    cutoff = pd.Timestamp(info["cutoff"])
    print("PROGRESS fitting the model on 2 clients only (no real training: only the forecast is timed)", flush=True)
    ticked = time.time()
    prefix = size_path(test)
    folder, manifest = read_manifest(FLOWS, prefix)
    daily = read_group(folder, manifest, next(iter(manifest["files"]))).to_pandas()          # one group of clients
    two = daily.groupby("client_no")["day"].min().nsmallest(2).index                       # its 2 oldest clients
    model = DailyCatBoost().fit(daily[daily["client_no"].isin(two)], None, cutoff)
    log = {"model": model.name, "input_days": model.params["input_days"],
           "seconds_fitting_2_clients": round(time.time() - ticked, 1)}
    del daily

    done = forecast_all(model, "load_test", fake_clients(info["clients"]), cutoff, test["workers"], FLOWS, OUTPUT,
                        {"forecaster": model.name, "load_id": "load_test"}, prefix)
    to_csv = final_load() == "csv"
    ticked = time.time()
    load_to_file(OUTPUT) if to_csv else load_to_sql(SQL, OUTPUT)
    seconds_sql = time.time() - ticked
    log["final_load_to"] = f"a CSV file ({OUTPUT}{FINAL_FILE})" if to_csv else f"the SQL table {SQL}"
    log.update({f"seconds_{k}": round(done[k], 1) for k in STEPS}, groups=done["groups"],
               daily_rows_read=done["daily_rows"], seconds_forecast=round(done["seconds"], 1),
               seconds_sql_load=round(seconds_sql, 1), clients_ok=done.get("OK", 0),
               clients_not_enough_history=done.get("NOT_ENOUGH_HISTORY", 0), clients_no_postings=done.get("NO_POSTINGS", 0),
               clients_per_second=round(info["clients"] / max(done["seconds"], 0.1), 1),
               peak_memory_mb=done["peak_memory_mb"])
    return log
