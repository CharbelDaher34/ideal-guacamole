# Load test, step 1 of 2: fake data, the same shape as the real data (interface 1), in the Load test zone only.
# It never reads or writes the real data: it writes
#   load_test_clients: one row per fake client (the columns of clients, all ACTIVE)
#   load_test_flows: fake daily flows, Parquet files exactly like daily_flows (one row per client, day and category:
#                    money_in, money_out, postings; each client's rows in one file), and benchmark.json (what was
#                    generated and how long it took, for load_test_run's log)
# Its settings are variables, set in the first step of the scenario Load test (Scenarios > Load test > Steps):
#   loadtest_clients             how many fake clients
#   loadtest_months              months of history up to the cut-off day (yesterday)
#   loadtest_postings_per_month  the average number of postings (transactions) per client and month
# The fake data is kept when these numbers did not change since the last run (not generated again).
# Each client gets its own rate around that average (a few busy clients, many quiet ones), and 10% of the clients a
# shorter history, so some get NOT_ENOUGH_HISTORY, as in production.
import json
import time

import dataiku
import numpy as np
import pandas as pd

from cashflow import Progress, whole_number, write_daily
from warehouse import CATEGORY_OF_TYPE, COLUMNS, OTHER

started = time.time()
variables = dataiku.get_custom_variables()


def setting(name, default):
    value = whole_number(name, variables.get(name, default))
    if value is None:
        raise ValueError(f"The variable {name} must be a number (1 or more): set it in the scenario Load test, "
                         "step 1")
    return value


CLIENTS = setting("loadtest_clients", 1000)
MONTHS = setting("loadtest_months", 24)
POSTINGS = setting("loadtest_postings_per_month", 40)
PER_FILE = 2000                                   # fake clients per Parquet file
cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

# the same numbers and cut-off day as the fake data already there: kept as it is (so each configuration of the
# forecast is timed on the same data, without generating it again)
folder = dataiku.Folder("load_test_flows")
reuse = False
if "/benchmark.json" in folder.list_paths_in_partition():
    with folder.get_download_stream("benchmark.json") as stream:
        before = json.loads(stream.read())
    if (before.get("clients"), before.get("months"), before.get("postings_per_month"), before.get("cutoff")) == \
            (CLIENTS, MONTHS, POSTINGS, f"{cutoff:%Y-%m-%d}"):
        print(f"PROGRESS load test 1/2, fake data: the same numbers as the data already there ({CLIENTS:,} clients, "
              f"{MONTHS} months, {POSTINGS} postings a month): kept, not generated again", flush=True)
        reuse = True


def generate():
    """The fake clients and daily flows, and benchmark.json."""
    days = MONTHS * 30
    categories = np.array(sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER}))
    rng = np.random.default_rng(7)
    print(f"PROGRESS load test 1/2, fake data: {CLIENTS:,} clients x {MONTHS} months x about {POSTINGS} postings a "
          f"month = about {CLIENTS * MONTHS * POSTINGS:,} postings, up to {cutoff:%Y-%m-%d}", flush=True)

    numbers = np.array([f"LT{i:08d}" for i in range(CLIENTS)])
    clients = pd.DataFrame({column: "LOAD TEST" for column in COLUMNS["clients"]}, index=range(CLIENTS))
    clients["client_no"], clients["status"], clients["sole_proprietor"] = numbers, "ACTIVE", False
    dataiku.Dataset("load_test_clients").write_with_schema(clients)

    totals = {"postings": 0, "rows": 0}

    def files():
        """The fake daily flows, PER_FILE clients at a time (each client's rows together, as staging writes them)."""
        progress = Progress("load test 1/2, writing the fake daily flows (clients)", CLIENTS)
        for start in range(0, CLIENTS, PER_FILE):
            who = numbers[start:start + PER_FILE]
            n = len(who)
            rate = POSTINGS * rng.gamma(2.0, 0.5, n)                       # postings a month, average POSTINGS
            span = np.where(rng.random(n) < 0.1, rng.integers(30, days + 1, n), days)   # days of history
            count = rng.poisson(rate * span / 30)
            client = np.repeat(np.arange(n), count)
            size = rng.lognormal(8, 1.2, n)[client]                        # the client's typical amount (SAR)
            amount = size * rng.lognormal(0, 0.8, len(client))
            money_in = rng.random(len(client)) < 0.55
            postings = pd.DataFrame({
                "client": client,
                "back": (rng.random(len(client)) * span[client]).astype(int),     # days before the cut-off day
                "category": rng.integers(0, len(categories), len(client)),
                "money_in": np.where(money_in, amount, 0.0), "money_out": np.where(money_in, 0.0, amount)})
            daily = (postings.groupby(["client", "back", "category"], sort=True)
                     .agg(money_in=("money_in", "sum"), money_out=("money_out", "sum"), postings=("money_in", "size"))
                     .reset_index())
            daily = daily.sort_values(["client", "back"], ascending=[True, False])
            part = pd.DataFrame({"client_no": who[daily["client"].to_numpy()],
                                 "day": (cutoff - pd.to_timedelta(daily["back"].to_numpy(), unit="D")).values,
                                 "category": categories[daily["category"].to_numpy()],
                                 "money_in": daily["money_in"].round(2).to_numpy(),
                                 "money_out": daily["money_out"].round(2).to_numpy(),
                                 "postings": daily["postings"].astype("int64").to_numpy()})
            totals["postings"] += len(postings)
            totals["rows"] += len(part)
            progress.add(n)
            yield part
        progress.finish()


    write_daily(files(), "load_test_flows")
    paths = [p for p in folder.list_paths_in_partition() if p.endswith(".parquet")]
    size = sum(folder.get_path_details(p).get("size", 0) for p in paths)
    info = {"cutoff": f"{cutoff:%Y-%m-%d}", "clients": CLIENTS, "months": MONTHS, "postings_per_month": POSTINGS,
            "postings": totals["postings"], "daily_rows": totals["rows"], "parquet_files": len(paths),
            "parquet_mb": round(size / 1e6, 1), "seconds_generating": round(time.time() - started, 1)}
    with folder.get_writer("benchmark.json") as writer:
        writer.write(json.dumps(info).encode())
    print(f"PROGRESS load test 1/2, fake data: done in {info['seconds_generating']:.0f} s: {info['postings']:,} postings, "
          f"{info['daily_rows']:,} daily rows, {info['parquet_files']} Parquet files ({info['parquet_mb']} MB)", flush=True)


if not reuse:
    generate()
