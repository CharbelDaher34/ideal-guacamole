"""Load test (project library of the load test project: python/loadtest.py). Times the nightly forecast of the
forecast project on fake data of the same shape, with the same code: cashflow.py, models.py and forecast.py are copies
of the forecast project's, made by the same setup files. Nothing here reads or writes the forecast project.

Zone Data prep, scenario "Create the fake data" (recipe make_fake_data, not timed): one set of fake daily flows in
the folder load_test_flows, laid out like daily_flows (one file per group of clients and month), of the size its
step 1 sets (data_size):
    fake_clients                 how many fake clients
    fake_months                  months of daily flows per client up to the day they are made (at least MIN_MONTHS)
    fake_postings_per_month      the average number of postings per client and month
    fake_clients_per_group       clients per group (production: cashflow.CLIENTS_PER_GROUP)
Zone Load test, scenario "Run the load test" (recipe run_test_plan): every test of the test plan (load_test_plan, set
by its step 1) on that data, so the number of clients is always the data's. Each test:
    test                 a name, for the report
    workers              groups of clients forecast at a time (production: the project variable inference_workers)
fits the placeholder model (models.DailyCatBoost, option A) on 2 clients only (its forecasts mean nothing, its
prediction time is real), runs the nightly forecast of every fake client (forecast.forecast_all) into the folder
load_test_forecast, then loads it into the SQL table load_test_final (forecast.load_to_sql), timed step by step
(run_test), in its own process (in_own_process)."""
import json
import multiprocessing
import time
import traceback

import numpy as np
import pandas as pd

from cashflow import INPUT_DAYS, PERIOD_DAYS, Progress, client_groups, whole_number, write_daily
from warehouse import CATEGORY_OF_TYPE, COLUMNS, OTHER

FLOWS = "load_test_flows"         # folder: the fake daily flows, laid out like daily_flows
OUTPUT = "load_test_forecast"     # folder: the forecast, like the folder forecast
SQL = "load_test_final"           # SQL table: the forecast loaded, like final_forecast
MIN_MONTHS = -(-(INPUT_DAYS + PERIOD_DAYS * 3) // PERIOD_DAYS)    # a year of input and 3 windows: 16 months
SIZE = {"clients": "fake_clients", "months": "fake_months", "postings_per_month": "fake_postings_per_month",
        "clients_per_group": "fake_clients_per_group"}
CREATE, RUN = "Create the fake data", "Run the load test"         # the 2 scenarios


def scenario_numbers(scenario, names):
    """The numbers set in step 1 of the scenario (a define-variables step), by name: the scenario's variables when it
    runs; otherwise (a build from the Flow, or the setup notebook) read from its step 1, so they are always the same."""
    import dataiku
    variables = dataiku.get_custom_variables()
    if all(n in variables for n in names):
        return {n: variables[n] for n in names}
    project = dataiku.api_client().get_default_project()
    ids = {s["name"]: s["id"] for s in project.list_scenarios()}
    steps = (project.get_scenario(ids[scenario]).get_settings().get_raw()["params"].get("steps", [])
             if scenario in ids else [])
    found = steps[0]["params"].get("variables", {}) if steps and steps[0].get("type") == "define_vars" else {}
    return {n: found.get(n, variables.get(n)) for n in names}


def data_size(variables):
    """The size of the fake data to make (step 1 of the scenario Create the fake data), checked."""
    size = {}
    for key, name in SIZE.items():
        size[key] = whole_number(name, variables.get(name))
        if size[key] is None:
            raise ValueError(f"{name} must be a whole number (1 or more): set it in step 1 of the scenario Create the "
                             "fake data, and run the scenario (not the recipe alone)")
    if size["months"] < MIN_MONTHS:
        raise ValueError(f"fake_months is {size['months']}: it must be at least {MIN_MONTHS} (a year of input, "
                         f"{INPUT_DAYS} days, and the 3 windows after it)")
    return size


def read_plan(value):
    """The test plan (load_test_plan, step 1 of the scenario Run the load test: a list, or its JSON text), checked."""
    try:
        plan = json.loads(value) if isinstance(value, str) else value
    except ValueError as e:
        raise ValueError(f"load_test_plan is not valid JSON ({e})") from None
    if not isinstance(plan, list) or not plan or not all(isinstance(t, dict) for t in plan):
        raise ValueError("load_test_plan must be a list of tests, each {\"test\": \"a name\", \"workers\": 4}: set it "
                         "in step 1 of the scenario Run the load test, and run the scenario (not the recipe alone)")
    tests = [{"test": str(t.get("test", "")).strip() or f"test {n}",
              "workers": whole_number(f"workers of test {n}", t.get("workers"))} for n, t in enumerate(plan, 1)]
    if any(t["workers"] is None for t in tests):
        raise ValueError("Each test of load_test_plan needs workers: a whole number (1 or more)")
    if len({t["test"] for t in tests}) != len(tests):
        raise ValueError(f"Give each test of load_test_plan its own name: {[t['test'] for t in tests]}")
    return tests


def data_info():
    """What fake data is in load_test_flows (its data.json: size, cut-off day, rows, MB), or None if none yet."""
    import dataiku
    folder = dataiku.Folder(FLOWS)
    if "/data.json" not in folder.list_paths_in_partition():
        return None
    with folder.get_download_stream("data.json") as stream:
        return json.loads(stream.read())


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
    """Zone Data prep: makes sure load_test_flows holds fake daily flows of the size `wanted` (data_size): kept when
    they are already there (whatever their cut-off day), else made again up to `cutoff`, replacing the earlier ones.
    Returns what is there (data.json). Each client gets its own rate around the average (a few busy clients, many
    quiet ones), and 10% of them a shorter history, so some get NOT_ENOUGH_HISTORY, as in production."""
    import dataiku
    started = time.time()
    info = data_info()
    if info and {k: info.get(k) for k in wanted} == wanted:
        print(f"PROGRESS fake data: already there ({wanted}, up to {info['cutoff']}): kept. Change one of these numbers "
              "to make it again", flush=True)
        return info
    clients, months, postings = wanted["clients"], wanted["months"], wanted["postings_per_month"]
    groups = max(1, -(-clients // wanted["clients_per_group"]))
    days = months * PERIOD_DAYS
    categories = np.array(sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER}))
    numbers = fake_clients(clients)["client_no"].to_numpy()
    group_of = client_groups(numbers, groups).to_numpy()
    rng = np.random.default_rng(7)
    totals = {"postings": 0, "rows": 0}
    print(f"PROGRESS fake data: {clients:,} clients ({groups:,} groups) x {months} months x about {postings} postings "
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

    manifest = write_daily(parts(), groups, FLOWS)
    folder = dataiku.Folder(FLOWS)
    info = {**wanted, "groups": groups, "cutoff": f"{cutoff:%Y-%m-%d}", "postings": totals["postings"],
            "daily_rows": manifest["rows"], "files": sum(len(m) for m in manifest["files"].values()),
            "data_mb": round(sum(folder.get_path_details(p).get("size", 0)
                                 for p in folder.list_paths_in_partition()) / 1e6, 1),
            "seconds_generating": round(time.time() - started, 1)}
    with folder.get_writer("data.json") as writer:
        writer.write(json.dumps(info).encode())
    print(f"PROGRESS fake data: done in {info['seconds_generating']:.0f} s: {info['postings']:,} postings, "
          f"{info['daily_rows']:,} daily rows, {info['files']:,} files ({info['data_mb']} MB)", flush=True)
    return info


def run_test(test, info):
    """Zone Load test: one test on the fake data in load_test_flows (info: its data.json): fit option A on 2 clients,
    forecast every fake client as the nightly forecast does, then load the forecast into SQL. Returns the values for
    load_test_log."""
    from cashflow import read_group, read_manifest
    from forecast import STEPS, forecast_all, load_to_sql
    from models import DailyCatBoost
    cutoff = pd.Timestamp(info["cutoff"])
    print("PROGRESS fitting the model on 2 clients only (no real training: only the forecast is timed)", flush=True)
    ticked = time.time()
    folder, manifest = read_manifest(FLOWS)
    daily = read_group(folder, manifest, next(iter(manifest["files"]))).to_pandas()          # one group of clients
    two = daily.groupby("client_no")["day"].min().nsmallest(2).index                       # its 2 oldest clients
    model = DailyCatBoost().fit(daily[daily["client_no"].isin(two)], None, cutoff)
    log = {"model": model.name, "input_days": model.params["input_days"],
           "seconds_fitting_2_clients": round(time.time() - ticked, 1)}
    del daily

    done = forecast_all(model, "load_test", fake_clients(info["clients"]), cutoff, test["workers"], FLOWS, OUTPUT,
                        {"forecaster": model.name, "load_id": test["test"]})
    ticked = time.time()
    load_to_sql(SQL, OUTPUT)
    seconds_sql = time.time() - ticked
    log.update({f"seconds_{k}": round(done[k], 1) for k in STEPS}, workers=done["workers"], groups=done["groups"],
               daily_rows_read=done["daily_rows"], seconds_forecast=round(done["seconds"], 1),
               seconds_sql_load=round(seconds_sql, 1), clients_ok=done.get("OK", 0),
               clients_not_enough_history=done.get("NOT_ENOUGH_HISTORY", 0), clients_no_postings=done.get("NO_POSTINGS", 0),
               clients_per_second=round(info["clients"] / max(done["seconds"], 0.1), 1),
               peak_memory_mb=done["peak_memory_mb"])
    return log
