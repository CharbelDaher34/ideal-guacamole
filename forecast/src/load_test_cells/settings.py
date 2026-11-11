PROJECT_NAME = "Digital Cash Management Forecast - Load test"

# ── 1. PYTHON AND OUTPUT ────────────────────────────────────────────────────────────────────────────────────────────
CODE_ENV = "YOUR_CODE_ENV"               # the forecast project's code env (polars, catboost), e.g. "cashflow_forecast"
FINAL_LOAD = "csv"              # where each test's forecast is loaded at the end (timed):
                                #   "csv": a CSV file in the folder load_test_forecast (no table, no new dataset)
                                #   "sql": the table load_test_final on OUTPUT_SQL_CONNECTION, as production does
OUTPUT_SQL_CONNECTION = "YOUR_SQL_CONNECTION"   # only for FINAL_LOAD = "sql": the forecast project's output database
OUTPUT_CONNECTION = "filesystem_managed"         # datasets: load_test_log
OUTPUT_FOLDER_CONNECTION = "filesystem_folders"  # folders: the fake daily flows, the forecast, the reports

# ── 2. THE TESTS THE SCENARIO STARTS WITH ──────────────────────────────────────────────────────────────────────────
# The scenario "Run the load test" sets its tests in its step 1 (Scenarios > Run the load test > Steps > 1), run one
# after the other in one go. These are only its first tests, when the scenario is made: after that, change them
# there. Running this notebook again keeps them. TESTS is a grid: each number, or a list of numbers, and every
# combination is a test (here 3 x 2 x 3 = 18 tests). A list of tests, each with every number, works too.
#   clients              how many fake clients
#   months               months of daily flows per client, up to yesterday (at least 16)
#   postings_per_month   average postings (transactions) per client and month
#   clients_per_group    clients per group (a file per group and month)
#   workers              groups of clients forecast at a time (production: inference_workers)
# Tests with the same first 4 numbers share the same fake data (made once, not timed).
TESTS = {"clients": [100_000, 400_000, 800_000], "months": 24, "postings_per_month": 60,
         "clients_per_group": [2_000, 5_000], "workers": [2, 4, 8]}

RUN_NOW = True                  # at the end of this notebook: run what the scenario runs, with its step 1
