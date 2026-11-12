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
"sql", the table FINAL_SCHEMA.FINAL_TABLE on FINAL_CONNECTION (the project variables final_*: final_table), made by
the setup notebook with SQL and loaded by forecast.load_to_table: on Oracle with oracledb (no Dataiku dataset), on
another database through the dataset load_test_final linked to it."""
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
SQL = "load_test_final"           # final load "sql", not on Oracle: the dataset linked to the table (final_table)
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


# load_test_log: its columns and their Dataiku types (string, bigint, double). run_load_test writes them into the
# dataset load_test_log, and into the SQL table set by the project variables log_sql_* when there is one (log_to_sql).
LOG_COLUMNS = {
    "batch": "string", "test": "bigint", "run_at": "string", "status": "string", "error": "string",
    "workers": "bigint", "clients": "bigint", "months": "bigint", "postings_per_month": "bigint",
    "clients_per_group": "bigint", "groups": "bigint", "model": "string", "input_days": "bigint",
    "cutoff_date": "string", "daily_rows": "bigint", "data_mb": "double", "seconds_fitting_2_clients": "double",
    **{f"seconds_{k}": "double" for k in ("reading", "preparing", "model", "formatting", "writing")},  # forecast.STEPS
    "seconds_forecast": "double", "seconds_sql_load": "double", "clients_per_second": "double",
    "peak_memory_mb": "bigint", "daily_rows_read": "bigint", "clients_ok": "bigint",
    "clients_not_enough_history": "bigint", "clients_no_postings": "bigint", "cpus": "bigint", "ram_gb": "double",
    "final_load_to": "string",
    "cpu_limit": "double", "ram_limit_gb": "double", "ram_free_gb": "double", "disk_gb": "double",
    "disk_free_gb": "double"}       # the machine: machine(), read from Linux as each test starts

# The SQL copy of load_test_log (log_to_sql): a table the project's user may create and insert into, set by the
# setup notebook (LOG_SQL_CONNECTION, LOG_SQL_SCHEMA, LOG_SQL_TABLE) as the project variables log_sql_*. Made and
# written with SQL statements on the connection (forecast.ensure_table, forecast.run_sql: dataiku.SQLExecutor2), not
# through a Dataiku dataset: no dataset is created. The tables' helpers are forecast.py's (shared with production).


def sql_target(prefix):
    """(connection, "SCHEMA.TABLE") set by the project variables <prefix>_connection, _schema, _table, or None when
    the connection or the table is empty."""
    import dataiku
    from forecast import table_name
    variables = dataiku.get_custom_variables()
    connection, schema, table = (str(variables.get(f"{prefix}_{k}") or "").strip()
                                 for k in ("connection", "schema", "table"))
    return (connection, table_name(schema, table)) if connection and table else None


def log_table():
    """(connection, "SCHEMA.TABLE") of the SQL copy of load_test_log, from the project variables, or None (not set)."""
    return sql_target("log_sql")


def ensure_log_table(connection, table, columns=None):
    """Creates the table when it is not there; when it is, adds the columns it lacks (a newer version of the code).
    Returns what was done."""
    from forecast import ensure_table
    return ensure_table(connection, table, columns or LOG_COLUMNS)


def sql_value(value, kind):
    """A value as an SQL literal: NULL when missing, a quoted string (quotes doubled, at most 1,000 characters), or a
    number."""
    try:
        if value is None or pd.isna(value):
            return "NULL"
    except (TypeError, ValueError):
        pass
    if kind == "string":
        text = str(value)[:1000].replace("'", "''")
        return f"'{text}'" if text else "NULL"
    number = float(value)
    if not np.isfinite(number):
        return "NULL"
    return str(int(round(number))) if kind == "bigint" else repr(number)


def log_to_sql(rows, columns=None):
    """Adds the rows (this run's tests) to the SQL copy of load_test_log, made first if needed (ensure_log_table).
    Returns what was done, or None when no table is set (the project variables log_sql_*)."""
    from forecast import run_sql
    target = log_table()
    if target is None:
        return None
    columns = columns or LOG_COLUMNS
    connection, table = target
    done = ensure_log_table(connection, table, columns)
    names = ", ".join(columns)
    run_sql(connection, [f"INSERT INTO {table} ({names}) VALUES "
                         f"({', '.join(sql_value(row.get(c), kind) for c, kind in columns.items())})"
                         for row in rows])
    return f"{done}; {len(rows)} rows added"


GB = 1024 ** 3


def _read(path):
    """A small file's text, stripped, or None when it can't be read."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cgroup_dirs(v1_controller):
    """The cgroup folders that limit this process for one controller (cgroup v2, or v1's v1_controller), its own first
    then each parent up to the root (a limit set on a parent holds too). Read from /proc/self/cgroup."""
    dirs = []
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":                                               # cgroup v2: one tree
            root = "/sys/fs/cgroup"
        elif v1_controller in controllers.split(","):                       # cgroup v1: one tree per controller
            root = next((r for r in (f"/sys/fs/cgroup/{controllers}", f"/sys/fs/cgroup/{v1_controller}")
                         if os.path.isdir(r)), None)
            if root is None:
                continue
        else:
            continue
        here = os.path.join(root, path.lstrip("/"))
        if not os.path.isdir(here):          # in a container the path is the host's: the container's cgroup is the root
            here = root
        while True:
            dirs.append(here)
            if os.path.normpath(here) == os.path.normpath(root):
                break
            here = os.path.dirname(here)
    return dirs


def cpu_limit():
    """The CPU limit of this process's cgroup (container), in CPUs: its CPU time quota over the period (cgroup v2
    cpu.max, v1 cpu.cfs_quota_us / cpu.cfs_period_us), the lowest of its own and its parents'. None: no limit."""
    limits = []
    for d in _cgroup_dirs("cpu"):
        v2 = (_read(os.path.join(d, "cpu.max")) or "").split()
        if len(v2) == 2 and v2[0].isdigit() and v2[1].isdigit() and int(v2[1]):
            limits.append(int(v2[0]) / int(v2[1]))
        quota, period = _read(os.path.join(d, "cpu.cfs_quota_us")), _read(os.path.join(d, "cpu.cfs_period_us"))
        if quota and period and quota.isdigit() and period.isdigit() and int(period):    # -1 (no limit) is not digits
            limits.append(int(quota) / int(period))
    return round(min(limits), 2) if limits else None


def _meminfo():
    """/proc/meminfo in bytes: {"MemTotal": ..., "MemAvailable": ..., ...}."""
    info = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        name, _, value = line.partition(":")
        if value.split() and value.split()[0].isdigit():
            info[name] = int(value.split()[0]) * 1024
    return info


def memory_limit():
    """The memory limit of this process's cgroup (container) and the memory free under it, in bytes: (limit, free),
    the cgroup with the lowest limit (v2 memory.max, v1 memory.limit_in_bytes; a v1 'limit' above the server's RAM
    means none). Free = the limit - what the cgroup uses, not counting the file cache it can drop (inactive_file, as
    docker stats counts it). (None, None): no limit."""
    total = _meminfo().get("MemTotal")
    best = (None, None)
    for d in _cgroup_dirs("memory"):
        limit, used = _read(os.path.join(d, "memory.max")), _read(os.path.join(d, "memory.current"))
        stat_name = "inactive_file"
        if limit is None:
            limit, used, stat_name = (_read(os.path.join(d, "memory.limit_in_bytes")),
                                      _read(os.path.join(d, "memory.usage_in_bytes")), "total_inactive_file")
        if not (limit and limit.isdigit()) or (total and int(limit) >= total):
            continue
        stat = dict(line.split() for line in (_read(os.path.join(d, "memory.stat")) or "").splitlines()
                    if len(line.split()) == 2)
        cache = int(stat.get(stat_name, 0)) if str(stat.get(stat_name, 0)).isdigit() else 0
        free = int(limit) - (int(used) - cache) if used and used.isdigit() else None
        if best[0] is None or int(limit) < best[0]:
            best = (int(limit), free)
    return best


def machine(folder=OUTPUT):
    """The machine, read from Linux when a test starts (nothing is set by hand):
        cpus           CPUs this process may run on (os.sched_getaffinity, else os.cpu_count)
        cpu_limit      its container's CPU limit, in CPUs (cpu_limit), empty when there is none
        ram_gb         the server's RAM (/proc/meminfo MemTotal)
        ram_limit_gb   its container's memory limit (memory_limit), empty when there is none
        ram_free_gb    the RAM free for it: /proc/meminfo MemAvailable, or less when the container's limit leaves less
        disk_gb, disk_free_gb   the size and free space of the disk of the folder `folder` (shutil.disk_usage)
    Each is empty when it can't be read."""
    import shutil
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    info = _meminfo()
    limit, limit_free = memory_limit()
    free = [v for v in (info.get("MemAvailable"), limit_free) if v is not None]
    disk = (None, None)
    try:
        import dataiku
        usage = shutil.disk_usage(local_path(dataiku.Folder(folder)))
        disk = (usage.total, usage.free)
    except Exception as e:
        print(f"machine: the disk of the folder {folder} could not be read: {e}", flush=True)
    gb = lambda v: round(v / GB, 1) if v is not None else None  # noqa: E731
    return {"cpus": cpus, "cpu_limit": cpu_limit(), "ram_gb": gb(info.get("MemTotal")), "ram_limit_gb": gb(limit),
            "ram_free_gb": gb(min(free)) if free else None, "disk_gb": gb(disk[0]), "disk_free_gb": gb(disk[1])}


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
    return dataiku.get_custom_variables().get("final_load", "csv")


def final_table():
    """The final load "sql": (connection, "SCHEMA.TABLE", the Dataiku dataset linked to it or None, how it is
    written), from the project variables final_* (FINAL_CONNECTION, FINAL_SCHEMA, FINAL_TABLE of the setup notebook).
    Oracle: oracledb, no dataset; another database: Dataiku's writer into the dataset load_test_final (SQL)."""
    from forecast import sql_dialect
    target = sql_target("final")
    if target is None:
        raise ValueError("The final load is \"sql\" but FINAL_CONNECTION or FINAL_TABLE is empty: set them in the "
                         "setup notebook and run it again")
    connection, table = target
    oracle = sql_dialect(connection) == "oracle"
    return connection, table, None if oracle else SQL, "oracledb in bulk" if oracle else "Dataiku's writer"


def load_to_file(output=OUTPUT):
    """The final load "csv": the current forecast of the folder `output` (one Parquet file per group) into one CSV
    file of the same folder (FINAL_FILE), replacing the previous one, one group's file at a time. Polars writes the
    CSV (in Rust, on several threads): pandas' to_csv took most of the final load turning the numbers into text."""
    from cashflow import read_parquet
    from forecast import current_output
    folder, version, paths = current_output(output)
    with folder.get_writer(FINAL_FILE) as writer:
        for n, path in enumerate(paths):
            writer.write(csv_bytes(read_parquet(folder, path), header=n == 0))


def csv_bytes(frame, header=True):
    """A Polars DataFrame as CSV bytes (dates as 2026-11-10, empty for a missing value, like pandas wrote them)."""
    try:
        return frame.write_csv(include_header=header).encode()
    except TypeError:                         # Polars before 0.19.17 names it has_header
        return frame.write_csv(has_header=header).encode()


def run_test(test, info):
    """Zone Load test: one test, on the fake data of its size in load_test_flows (info: its data.json): fit option A
    on 2 clients, forecast every fake client as the nightly forecast does with `workers`, then load the forecast into
    the CSV file or the SQL table (final_load, final_table). Returns the values for load_test_log."""
    from cashflow import read_group, read_manifest
    from forecast import STEPS, check_table_writer, forecast_all, load_to_table
    from models import DailyCatBoost
    to_csv = final_load() == "csv"
    if not to_csv:                  # before the forecast: a table that can't be written fails the test at once
        connection, table, dataset, how = final_table()
        check_table_writer(connection)
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
    ticked = time.time()
    if to_csv:
        load_to_file(OUTPUT)
        log["final_load_to"] = f"a CSV file ({OUTPUT}{FINAL_FILE})"
    else:
        load_to_table(connection, table, dataset, OUTPUT)
        log["final_load_to"] = f"the table {table} on {connection} ({how})"
    seconds_sql = time.time() - ticked
    log.update({f"seconds_{k}": round(done[k], 1) for k in STEPS}, groups=done["groups"],
               daily_rows_read=done["daily_rows"], seconds_forecast=round(done["seconds"], 1),
               seconds_sql_load=round(seconds_sql, 1), clients_ok=done.get("OK", 0),
               clients_not_enough_history=done.get("NOT_ENOUGH_HISTORY", 0), clients_no_postings=done.get("NO_POSTINGS", 0),
               clients_per_second=round(info["clients"] / max(done["seconds"], 0.1), 1),
               peak_memory_mb=done["peak_memory_mb"])
    return log
