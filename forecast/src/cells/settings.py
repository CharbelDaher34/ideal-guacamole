PROJECT_NAME = "Digital Cash Management Forecast"

# ── 1. INPUT: the warehouse, read-only through SQL (nothing is ever written there) ──────────────────────────────────
# The project reads 2 tables, both through the same connection (staging joins them in one query).
EDW_CONNECTION = "YOUR_EDW_CONNECTION"   # the Dataiku connection to the warehouse (Administration > Connections).
                                         # At the bank, for example "EDW_AI"; on the test machine "EDW_LOCAL"
CLIENTS_SCHEMA = "EDW"                   # the clients: schema and table (the name as the database shows it)
CLIENTS_TABLE = "DIM_CUSTOMER"
POSTINGS_SCHEMA = "EDW"                  # the daily postings: schema and table
POSTINGS_TABLE = "FCT_STATEMENT_ENTRY"
HISTORY_MONTHS = 24                      # months of postings read the first time (and by the scenario Full rebuild),
                                         # up to yesterday. -1: the whole history. Each night then adds the new days
CLIENTS_LIMIT = -1                       # how many business clients to read (the lowest customer numbers): -1 all;
                                         # for example 1000 to test fast. Both: to change them later, run the setup
                                         # again, then the scenario Full rebuild

# ── 2. OUTPUT: where the project writes its own results ─────────────────────────────────────────────────────────────
FORECAST_CONNECTION = "cm_test_env"      # the forecast, one row per active client: the database connection it goes
                                         # to (one you may write into, never the warehouse), its schema and table.
FORECAST_SCHEMA = "EDW_AI"               # The setup creates the table if it is not there, with this name (letters,
FORECAST_TABLE = "FINAL_FORECAST"        # digits and _ only). In the Flow it is the dataset final_forecast. Oracle:
                                         # the code env needs oracledb, and the connection's "Details readable by"
                                         # must include your group
OUTPUT_CONNECTION = "filesystem_managed"         # datasets (CSV files): clients, test_results, logs
OUTPUT_FOLDER_CONNECTION = "filesystem_folders"  # folders: daily_flows (Parquet), models, forecast (Parquet, by night)

# ── 3. PYTHON ───────────────────────────────────────────────────────────────────────────────────────────────────────
CODE_ENV = "YOUR_CODE_ENV"               # an existing Python code env with polars, catboost, scikit-learn and
                                         # mlflow==2.22.0 (an admin
                                         # creates it once), for example "cashflow_forecast"
