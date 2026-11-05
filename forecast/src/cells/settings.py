PROJECT_NAME = "Digital Cash Management Forecast"
WAREHOUSE_SQL_CONNECTION = ""         # production: a database connection to the warehouse, to read the 3 tables and
WAREHOUSE_SQL_SCHEMA = "EDW"          # the load record as tables of this schema. Empty: read them as CSV files from
LANDING_CONNECTION = "EDW_LANDING"    # the landing folder (read-only connection). Neither is used for tables the
                                      # warehouse simulator has already written here (sandbox: CSV only).
CONNECTION = "filesystem_managed"     # where the project writes its own datasets (CSV files)
FOLDER_CONNECTION = "filesystem_folders"   # where the tracked models and the forecast archive are kept
CODE_ENV = "cashflow_forecast"        # Python code env for the recipes: an existing one (an admin creates it once)
