PROJECT_NAME = "Digital Cash Management Forecast"
WAREHOUSE_SQL_CONNECTION = ""         # production: the read-only database connection to the warehouse (Oracle), to
WAREHOUSE_SQL_SCHEMA = "EDW"          # read DIM_CUSTOMER and FCT_STATEMENT_ENTRY of this schema as they are. Empty:
LANDING_CONNECTION = "EDW_LANDING"    # read them as CSV files from the landing folder (read-only connection). Neither
                                      # is used for tables the warehouse simulator has written here (sandbox: CSV).
CONNECTION = "filesystem_managed"     # where the project writes its own datasets (CSV files)
FOLDER_CONNECTION = "filesystem_folders"   # where the tracked models and the forecast archive are kept
HISTORY_YEARS = 2                     # years of postings read the first time (and by the scenario Full rebuild), up
                                      # to yesterday; at least 2. None: the whole history. Then each night adds the
                                      # new days. Changing it later: run the setup again, then the scenario Full rebuild
CODE_ENV = "cashflow_forecast"        # Python code env for the recipes: an existing one (an admin creates it once)
