# Cash-flow outlook, outside Dataiku

A small web app that shows the results of `CORP_CASHFLOW` and `CORP_CASHFLOW_CODE` and gets their
scenarios run, through the **DSS public REST API**. The browser only talks to this app (`app.py`, FastAPI);
the app holds the API key and calls DSS.

```
browser ──► app.py (127.0.0.1:8060) ──► DSS public API (127.0.0.1:10000/public/api/…)
```

## Set up once: an API key

The app reads the key from `dataiku/.dss_api_key` (git-ignored) or from `DSS_API_KEY`. The current one is the
global admin key labelled `outlook-dashboard` (Administration → Security → Global API keys). To make a new one:

```bash
docker exec dataiku-dss /home/dataiku/dss/bin/dsscli api-key-create --label outlook-dashboard --admin true
```

then paste its `key` into `dataiku/.dss_api_key`.

## Run

```bash
make dashboard        # = cd outlook && uvicorn app:app --host 127.0.0.1 --port 8060
```

Open http://127.0.0.1:8060 (from your laptop: tunnel port 8060 as you do for 10000).

## Running scenarios: Free Edition

The Free Edition refuses to start a scenario through the API or `dsscli` (*"Automated triggering of
scenarios is not available in your license"*). So **Run in Dataiku ↗** opens the scenario in DSS; click
**Run** there. The dashboard checks the runs every 5 s (for up to 20 min), shows *Running*, and reloads the
numbers when the run ends.

With a license that allows it, start the app with `DSS_API_RUN=1`: the buttons become **Run** and start the
scenario from the dashboard (`POST /projects/{p}/scenarios/{id}/run`), one run at a time.

## What it shows

| Part | Comes from (DSS public API) |
|---|---|
| HIGH / MEDIUM accounts, lowest balance, accounts table | `GET /projects/{p}/datasets/account_outlook/data/` |
| Balance per day of the selected account | `GET /projects/{p}/datasets/daily_balance_path/data/` |
| Model: trained when, MASE (limits: 45 days, MASE < 1) | visual: `GET /projects/{p}/savedmodels/{id}/versions` and `…/versions/{v}/details`; code: `GET /projects/{p}/managedfolders/{id}/contents/model_info.json` |
| Runs: last outcome, failed step and message, last 5 runs | `GET /projects/{p}/scenarios/{id}/get-last-runs/`, `…/get-run-details` |

Only `DAILY_FORECAST` and `TRAIN_MODEL` get a run button; `PREPARE_DATA` is shown for its checks.

## The app's own API

| Call | Returns |
|---|---|
| `GET /api/projects` | the two projects, the DSS address, whether API runs are on |
| `GET /api/{project}/outlook` | `accounts` (account_outlook rows), `balance_path` (daily_balance_path rows) |
| `GET /api/{project}/model` | `ok`, `message`, `trained_at`, `age_days`, `mase`, `algorithm` |
| `GET /api/{project}/scenarios` | per scenario: last runs, and the failed step of the latest run |
| `POST /api/{project}/scenarios/{id}/run` | starts it with `DSS_API_RUN=1` (409 if already running), else 403 |

Interactive docs: http://127.0.0.1:8060/docs.
