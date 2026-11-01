"""Cash-flow outlook outside Dataiku: a small web app that reads the results of CORP_CASHFLOW and
CORP_CASHFLOW_CODE through the DSS public REST API and runs their scenarios.

    uvicorn app:app --host 127.0.0.1 --port 8060      # then open http://127.0.0.1:8060

The browser never talks to DSS: it calls this app, which holds the API key (env DSS_API_KEY, or the
git-ignored file ../.dss_api_key) and only exposes what the page needs. DSS_URL is also the address the page's
"Run in Dataiku" links open, so it must work from the browser (here: through the SSH tunnel).
"""
import csv
import io
import json
import os
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
DSS_URL = os.environ.get("DSS_URL", "http://127.0.0.1:10000")
# The Free Edition refuses to start scenarios through the API ("Automated triggering of scenarios is not
# available in your license"): the page then opens the scenario in DSS, where you click Run, and follows the
# run. With a license that allows it, DSS_API_RUN=1 makes the Run buttons start scenarios from here.
API_RUN = os.environ.get("DSS_API_RUN") == "1"
KEY_FILE = HERE.parent / ".dss_api_key"
PROJECTS = {"CORP_CASHFLOW": "Visual", "CORP_CASHFLOW_CODE": "Code"}
SCENARIOS = {"DAILY_FORECAST": "Daily forecast", "TRAIN_MODEL": "Train model", "PREPARE_DATA": "Prepare data"}
RUNNABLE = {"DAILY_FORECAST", "TRAIN_MODEL"}
MODEL_MAX_AGE_DAYS = 45          # same limits as the model check of the Daily forecast scenario
MODEL_MAX_MASE = 1.0

app = FastAPI(title="Cash-flow outlook")


def api_key():
    key = os.environ.get("DSS_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        raise HTTPException(500, f"No DSS API key: set DSS_API_KEY or write it to {KEY_FILE}")
    return key


def dss(method, path, **kwargs):
    """Call the DSS public API: /public/api/<path>, authenticated with the API key as the user name."""
    try:
        r = requests.request(method, f"{DSS_URL}/public/api/{path}", auth=(api_key(), ""), timeout=60, **kwargs)
    except requests.ConnectionError:
        raise HTTPException(502, f"DSS is not reachable at {DSS_URL}")
    if not r.ok:
        try:
            message = r.json().get("message", r.text)
        except ValueError:
            message = r.text
        raise HTTPException(502, f"DSS {path}: {message[:300]}")
    return r


def rows(project, dataset):
    """All rows of a dataset, as a list of dicts (numbers converted)."""
    text = dss("GET", f"projects/{project}/datasets/{dataset}/data/", params={"format": "tsv-excel-header"}).text
    out = []
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        for k, v in row.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


def check_project(project):
    if project not in PROJECTS:
        raise HTTPException(404, f"Unknown project {project}")


def model_status(project):
    """When the model in use was trained and its test error, from the saved model (visual project) or from
    model_info.json in the forecast_model folder (code project)."""
    if project == "CORP_CASHFLOW":
        models = dss("GET", f"projects/{project}/savedmodels/").json()
        if not models:
            return {"ok": False, "message": "No saved model: run Train model"}
        versions = dss("GET", f"projects/{project}/savedmodels/{models[0]['id']}/versions").json()
        active = next((v for v in versions if v.get("active")), None)
        if not active:
            return {"ok": False, "message": "No active model version: run Train model"}
        details = dss("GET", f"projects/{project}/savedmodels/{models[0]['id']}/versions/{active['id']}/details").json()
        trained = active["trainDate"] / 1000
        mase = details.get("perf", {}).get("aggregatedMetrics", {}).get("mase")
        algorithm = details.get("modeling", {}).get("algorithm", "").split("_")[-1].capitalize()   # GLUONTS_TORCH_DEEPAR
    else:
        folders = dss("GET", f"projects/{project}/managedfolders/").json()
        folder = next((f for f in folders if f["name"] == "forecast_model"), None)
        r = folder and requests.get(f"{DSS_URL}/public/api/projects/{project}/managedfolders/{folder['id']}"
                                    "/contents/model_info.json", auth=(api_key(), ""), timeout=60)
        if not r or not r.ok:
            return {"ok": False, "message": "No trained model in forecast_model: run Train model"}
        info = r.json()
        trained, mase, algorithm = info["trained_at"], info["mase"], "Gradient boosting"
    age = (time.time() - trained) / 86400
    problems = ([f"trained {age:.0f} days ago (limit {MODEL_MAX_AGE_DAYS})"] if age > MODEL_MAX_AGE_DAYS else []) \
        + ([f"MASE {mase:.2f} is not below {MODEL_MAX_MASE}"] if mase is not None and mase >= MODEL_MAX_MASE else [])
    return {"ok": not problems, "message": "; ".join(problems) or "OK", "trained_at": trained,
            "age_days": age, "mase": mase, "algorithm": algorithm}


def last_runs(project, scenario, limit=5):
    """The latest runs, newest first: running ones have no outcome yet."""
    runs = dss("GET", f"projects/{project}/scenarios/{scenario}/get-last-runs/",
               params={"limit": limit, "onlyFinishedRuns": "false"}).json()
    out = []
    for run in runs:
        result = run.get("result") or {}
        out.append({"run_id": run.get("runId"), "start": run.get("start", 0) / 1000,
                    "end": (run.get("end") or 0) / 1000, "outcome": result.get("outcome") or "RUNNING",
                    "by": (run.get("trigger") or {}).get("trigger", {}).get("type", "")})
    return out


def failure(project, scenario, run_id):
    """Name and message of the step that failed in a run."""
    details = dss("GET", f"projects/{project}/scenarios/{scenario}/get-run-details",
                  params={"runId": run_id}).json()
    for step in details.get("stepRuns", []):
        result = step.get("result") or {}
        if result.get("outcome") == "FAILED":
            thrown = result.get("thrown") or {}
            return {"step": (step.get("step") or {}).get("name", ""), "message": thrown.get("message", "")}
    return None


@app.get("/")
def page():
    return FileResponse(HERE / "index.html")


@app.get("/api/projects")
def projects():
    return {"projects": [{"key": k, "label": v} for k, v in PROJECTS.items()], "dss_url": DSS_URL,
            "api_run": API_RUN}


@app.get("/api/{project}/outlook")
def outlook(project: str):
    """One row per account (account_outlook), and the balance per day (daily_balance_path)."""
    check_project(project)
    return {"accounts": rows(project, "account_outlook"), "balance_path": rows(project, "daily_balance_path")}


@app.get("/api/{project}/model")
def model(project: str):
    check_project(project)
    return model_status(project)


@app.get("/api/{project}/scenarios")
def scenarios(project: str):
    """Last runs of each scenario; for a failed latest run, the step that failed and why."""
    check_project(project)
    out = []
    for sid, name in SCENARIOS.items():
        runs = last_runs(project, sid)
        latest = runs[0] if runs else None
        out.append({"id": sid, "name": name, "runnable": sid in RUNNABLE, "runs": runs,
                    "failure": failure(project, sid, latest["run_id"])
                    if latest and latest["outcome"] == "FAILED" else None})
    return out


@app.post("/api/{project}/scenarios/{scenario}/run")
def run(project: str, scenario: str):
    """Start a scenario (Daily forecast or Train model) and return at once: the page then follows its runs."""
    check_project(project)
    if not API_RUN:
        raise HTTPException(403, "Starting scenarios through the API is off (DSS_API_RUN): run them in DSS")
    if scenario not in RUNNABLE:
        raise HTTPException(403, f"{scenario} cannot be started from here")
    if any(r["outcome"] == "RUNNING" for r in last_runs(project, scenario, 1)):
        raise HTTPException(409, f"{SCENARIOS[scenario]} is already running")
    dss("POST", f"projects/{project}/scenarios/{scenario}/run", data=json.dumps({}),
        headers={"Content-Type": "application/json"})
    return {"started": scenario}
