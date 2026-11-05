PROJECT_NAME = "Digital Cash Management Forecast"
WAREHOUSE_SQL_CONNECTION = ""         # production: the read-only database connection to the warehouse (Oracle), to
WAREHOUSE_SQL_SCHEMA = "EDW"          # read DIM_CUSTOMER and FCT_STATEMENT_ENTRY of this schema as they are. Empty:
LANDING_CONNECTION = "EDW_LANDING"    # read them as CSV files from the landing folder (read-only connection). Neither
                                      # is used for tables the warehouse simulator has written here (sandbox: CSV).
CONNECTION = "filesystem_managed"     # where the project writes its own datasets (CSV files)
FOLDER_CONNECTION = "filesystem_folders"   # where the tracked models and the forecast archive are kept
CODE_ENV = "cashflow_forecast"        # Python code env for the recipes: an existing one (an admin creates it once)
