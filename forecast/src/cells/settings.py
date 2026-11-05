PROJECT_NAME = "Digital Cash Management Forecast"
LANDING_CONNECTION = "EDW_LANDING"    # production: read-only connection to the warehouse landing folder (CSV files).
                                      # Not used when the warehouse simulator has already written the tables here.
CONNECTION = "filesystem_managed"     # where the project writes its own datasets (CSV files)
FOLDER_CONNECTION = "filesystem_folders"   # where the tracked models and the forecast archive are kept
CODE_ENV = "cashflow_forecast"        # Python code env for the recipes: an existing one (an admin creates it once)
