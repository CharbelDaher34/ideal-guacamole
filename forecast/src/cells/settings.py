EDW_CONNECTION = "EDW_AI"             # read-only connection to the warehouse tables built by the warehouse team
EDW_SCHEMA = "edw_ai"                 # schema of those tables
CONNECTION = "postgres_local"         # where the project writes its own datasets (a SQL connection)
FOLDER_CONNECTION = "filesystem_folders"   # where the code models are saved
CODE_ENV = "cashflow_forecast"        # Python code env for the recipes (created below with CatBoost if missing)
MODEL_SOURCE = "code"                 # which models feed final_forecast: "code" (CatBoost recipes) or "visual"
