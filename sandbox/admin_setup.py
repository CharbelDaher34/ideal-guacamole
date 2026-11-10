# Run once inside the fresh sandbox DSS (by sandbox/rebuild.py, in a temporary project it deletes afterwards): what
# an admin does by hand at the bank. Test machine only. __PG_PASSWORD__ and __PACKAGES__ are filled in by rebuild.py.
import dataiku

client = dataiku.api_client()

# 1. the connection to the local test database that plays the warehouse (schema EDW)
if "EDW_LOCAL" not in client.list_connections():
    client.create_connection("EDW_LOCAL", "PostgreSQL", {
        "host": "postgres", "port": 5432, "db": "dataiku", "user": "dataiku", "password": "__PG_PASSWORD__"})
print("connection EDW_LOCAL ready")
# the database the forecast is written into (at the bank: a connection the project may write into, never the
# warehouse). Here the same local database, its own connection
if "FORECAST_OUT_LOCAL" not in client.list_connections():
    client.create_connection("FORECAST_OUT_LOCAL", "PostgreSQL", {
        "host": "postgres", "port": 5432, "db": "dataiku", "user": "dataiku", "password": "__PG_PASSWORD__"})
print("connection FORECAST_OUT_LOCAL ready")

# 2. the code env of the recipes and the lab, with its Jupyter kernel
if "cashflow_forecast" not in [e["envName"] for e in client.list_code_envs()]:
    client.create_code_env("PYTHON", "cashflow_forecast", "DESIGN_MANAGED",
                           {"pythonInterpreter": "PYTHON39", "installCorePackages": True,
                            "installJupyterSupport": True})
env = client.get_code_env("PYTHON", "cashflow_forecast")
settings = env.get_settings()
settings.set_required_packages(*__PACKAGES__)
settings.save()
result = env.update_packages()
print("code env cashflow_forecast:", result.get("messages", {}).get("messages", [{}])[0].get("title", result))
env.set_jupyter_support(True)
print("code env packages:", env.get_settings().get_required_packages(as_list=True))

# 3. the empty forecast project
if "DCM_FORECAST" not in client.list_project_keys():
    client.create_project("DCM_FORECAST", "Digital Cash Management Forecast", "admin")
print("project DCM_FORECAST ready")
