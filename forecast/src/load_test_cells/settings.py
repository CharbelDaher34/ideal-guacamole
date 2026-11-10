PROJECT_NAME = "Digital Cash Management Forecast - Load test"

# ── 1. PYTHON AND OUTPUT ────────────────────────────────────────────────────────────────────────────────────────────
CODE_ENV = "YOUR_CODE_ENV"               # the forecast project's code env (polars, catboost), e.g. "cashflow_forecast"
OUTPUT_SQL_CONNECTION = "YOUR_SQL_CONNECTION"   # the forecast project's output database: the table load_test_final
OUTPUT_CONNECTION = "filesystem_managed"         # datasets: load_test_log
OUTPUT_FOLDER_CONNECTION = "filesystem_folders"  # folders: the fake daily flows, the forecast, the reports

# ── 2. THE NUMBERS THE 2 SCENARIOS START WITH ──────────────────────────────────────────────────────────────────────
# Each scenario sets its numbers in its step 1 (Scenarios > the scenario > Steps > 1). These are only its first values,
# when the scenario is made: after that, change them there. Running this notebook again keeps them.

# Scenario "Create the fake data" (not timed), step 1 "Set the fake data's size": one set of fake daily flows, the
# same shape as the real ones. The load test runs on it: its number of clients is the load test's.
FAKE_CLIENTS = 420_000          # fake_clients: how many fake clients (420,000: the bank's size)
FAKE_MONTHS = 24                # fake_months: months of daily flows per client, up to yesterday (at least 16)
FAKE_POSTINGS_PER_MONTH = 40    # fake_postings_per_month: average postings (transactions) per client and month
FAKE_CLIENTS_PER_GROUP = 5_000  # fake_clients_per_group: clients per group (a file per group and month)

# Scenario "Run the load test", step 1 "Set the test plan": one line per test, all on the fake data above
#   test      a name, for the report
#   workers   groups of clients forecast at a time (in production: the project variable inference_workers)
TEST_PLAN = [                   # load_test_plan
    {"test": "1 worker", "workers": 1},
    {"test": "4 workers", "workers": 4},
    {"test": "8 workers", "workers": 8},
]
BANK_CLIENTS = 420_000          # bank_clients: the report gives the time (forecast and SQL load) for this many clients

RUN_NOW = True                  # at the end of this notebook: run what the 2 scenarios run, with their step 1
