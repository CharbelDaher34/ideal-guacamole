"""Set up a fresh sandbox DSS from nothing, after `make start` on empty data folders (`make reset` does both).
Test machine only. Steps:
    1. the Free Edition licence back (sandbox/keep/license.json, saved by `make wipe`), then a restart
    2. the schema EDW in the local test database (the fake warehouse)
    3. as an admin would: the connection EDW_LOCAL, the code env cashflow_forecast (with its Jupyter kernel), the
       empty project DCM_FORECAST (sandbox/admin_setup.py, run in a temporary project, deleted afterwards)
    4. the forecast: warehouse_simulator.ipynb (fake tables), then build_forecast_project.ipynb with the local
       settings (the flow, the first run), and forecast_lab.ipynb and daily_flows_viewer.ipynb imported
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FORECAST = os.path.join(ROOT, "forecast")
sys.path.insert(0, HERE)
import dss  # noqa: E402

PACKAGES = ["scikit-learn", "mlflow==2.22.0", "catboost==1.2.7"]   # the code env cashflow_forecast
LOCAL_SETTINGS = {"YOUR_EDW_CONNECTION": "EDW_LOCAL", "YOUR_CODE_ENV": "cashflow_forecast"}


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()


def env_file():
    values = {}
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                values[k] = v.strip().strip("'\"")
    return values


def wait_ready():
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"{dss.DSS}/dip/api/get-configuration", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    sys.exit("DSS is not ready after 10 min: docker compose logs dss")


def step(text):
    print(f"\n##### {text}", flush=True)


step("1. licence")
licence = os.path.join(HERE, "keep", "license.json")
if os.path.exists(licence):
    sh("docker", "cp", licence, "dataiku-dss:/tmp/license.json")
    sh("docker", "exec", "dataiku-dss", "/home/dataiku/dss/bin/dsscli", "set-license", "/tmp/license.json")
    sh("docker", "restart", "dataiku-dss")
    wait_ready()
    sh("docker", "exec", "dataiku-dss", "/home/dataiku/dss/bin/dsscli", "user-edit", "admin", "--password", "admin")
    print("licence restored (Free Edition)")
else:
    print("no saved licence (sandbox/keep/license.json): register the Free Edition in the browser first, then run "
          "this again")
    sys.exit(1)

step("2. schema EDW in the local test database")
pg = env_file()
sh("docker", "exec", "dataiku-postgres", "psql", "-U", pg["POSTGRES_USER"], "-d", pg["POSTGRES_DB"], "-qc",
   'CREATE SCHEMA IF NOT EXISTS "EDW"')
print("schema EDW ready")

step("3. connection, code env (a few minutes), project")
dss.login()
dss.create_project("SANDBOX_ADMIN", "Sandbox admin (temporary)")
with open(os.path.join(HERE, "admin_setup.py")) as f:
    code = f.read().replace("__PG_PASSWORD__", pg["POSTGRES_PASSWORD"]).replace("__PACKAGES__", json.dumps(PACKAGES))
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "admin_setup.ipynb")
    with open(path, "w") as f:
        json.dump({"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                              "source": code}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f)
    ok = dss.run_notebook("SANDBOX_ADMIN", path)
sh("docker", "exec", "dataiku-dss", "/home/dataiku/dss/bin/dsscli", "project-delete", "SANDBOX_ADMIN", check=False)
if not ok:
    sys.exit("admin setup failed")

step("4. the forecast: fake warehouse, then the setup notebook")
sh(sys.executable, os.path.join(FORECAST, "make_notebook.py"))
with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(FORECAST, "build_forecast_project.ipynb")) as f:
        text = f.read()
    for placeholder, value in LOCAL_SETTINGS.items():
        text = text.replace(placeholder, value)
    build = os.path.join(tmp, "build_forecast_project.ipynb")
    with open(build, "w") as f:
        f.write(text)
    if not dss.run_notebook("DCM_FORECAST", os.path.join(FORECAST, "warehouse_simulator.ipynb"), quiet=True):
        sys.exit("warehouse_simulator failed")
    print("fake warehouse tables written")
    if not dss.run_notebook("DCM_FORECAST", build):
        sys.exit("build_forecast_project failed")
for name in ("forecast_lab.ipynb", "daily_flows_viewer.ipynb"):
    dss.import_notebook("DCM_FORECAST", os.path.join(FORECAST, name))
print("\nSandbox ready: http://127.0.0.1:10000 (admin / admin), project DCM_FORECAST")
