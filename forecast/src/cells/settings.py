PROJECT_NAME = "Digital Cash Management Forecast"

# ── 1. INPUT: the warehouse, read-only through SQL (nothing is ever written there) ──────────────────────────────────
# The project reads 2 tables of this schema: DIM_CUSTOMER and FCT_STATEMENT_ENTRY.
EDW_CONNECTION = "YOUR_EDW_CONNECTION"   # the Dataiku connection to the warehouse (Administration > Connections).
                                         # At the bank, for example "EDW_AI"; on the test machine "EDW_LOCAL"
EDW_SCHEMA = "EDW"                       # the schema that holds the 2 tables
HISTORY_YEARS = 2                        # years of postings read the first time (and by the scenario Full rebuild),
                                         # up to yesterday; at least 2. None: the whole history. Each night then adds
                                         # the new days. To change it later: run the setup again, then Full rebuild

# ── 2. OUTPUT: where the project writes its own results ─────────────────────────────────────────────────────────────
OUTPUT_CONNECTION = "filesystem_managed"         # datasets (tables): clients, final_forecast, test_results, logs
OUTPUT_FOLDER_CONNECTION = "filesystem_folders"  # folders (files): daily_flows (Parquet), models, forecast archive

# ── 3. PYTHON ───────────────────────────────────────────────────────────────────────────────────────────────────────
CODE_ENV = "YOUR_CODE_ENV"               # an existing Python code env with scikit-learn and mlflow==2.22.0 (an admin
                                         # creates it once), for example "cashflow_forecast"
