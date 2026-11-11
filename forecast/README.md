# Digital Cash Management Forecast: run it at the bank

Forecasts, for every active business client, the net cash flow of the next 30, 60 and 90 days (from the client's
daily flows of the last 365 days up to yesterday), per category and in TOTAL,
as P10 / EV / P90. How it works, step by step: [`docs/how_it_works.md`](../docs/how_it_works.md).

## Before you start (ask once)

1. **A Python code env** named `cashflow_forecast` (Python 3.9 or later) with `polars`, `catboost`, `scikit-learn` and `mlflow==2.22.0`.
   Only a Dataiku admin can create it. The setup notebook checks it and, if something is missing, prints the request
   to send the admin.
2. **A read-only database connection** to the warehouse (Oracle), usable by your group, for example `EDW`. The
   project only reads `EDW.DIM_CUSTOMER` and `EDW.FCT_STATEMENT_ENTRY`, with a `SELECT` of only the columns used. It never
   writes there.
3. **An empty Dataiku project** where you are at least a designer. Don't reuse a project from an older version.

## Set it up (once)

You only need `build_forecast_project.ipynb`: it holds all the code and creates the whole project.

1. **Import** `build_forecast_project.ipynb` into a new, empty project (Notebooks → New notebook → Import).
2. **Put the bank's codes in its cell `WAREHOUSE_CODE = r'''`** (section 2, Code):
   - `CATEGORY_OF_TYPE`: the real transaction types (`TRNS_TYP`) and their category. Today it's a mock list.
   - `CLIENTS_BUSINESS` and `CLIENTS_CURRENT_ROW`: which rows of `DIM_CUSTOMER` are read, as SQL conditions. Already
     set to the bank's query: business clients = `PRIME_RM LIKE '12%' OR SCTR_DSC = 'Sole Proprietorships'`, current
     row = `END_DT = DATE '2199-12-31'` (about 420k rows, one per customer). If staging warns that `DIM_CUSTOMER` sent
     several rows per client, check them.
   - `POSTINGS_LOAD_DATE` / `POSTINGS_LOAD_MARGIN_DAYS`: `FCT_STATEMENT_ENTRY` is partitioned by month on its load
     date `AS_OF_DT`, so staging also filters on it and Oracle reads only the partitions needed (each night: the
     latest one or two instead of the whole table). Margin: 31 days; a posting loaded more than that many days
     *before* its value date would be missed.
   - Check `ACTIVE_STATUS` and `SOLE_PROPRIETOR` against `DIM_CUSTOMER`.
   - `check_warehouse_loaded`: today a mock that always passes. Replace it with the bank's check that the
     warehouse's load of yesterday is finished (for example a query on its load-control table).

   A new version of the notebook starts again from the mock codes: send the real ones to whoever maintains the code,
   so the next versions already have them.
3. **Fill its first cell (settings):**

   | Setting | What it is | At the bank |
   |---|---|---|
   | `EDW_CONNECTION` | **input**: the Dataiku connection to the warehouse (read-only) | for example `"EDW_AI"` |
   | `EDW_SCHEMA` | **input**: the schema of `DIM_CUSTOMER` and `FCT_STATEMENT_ENTRY` | `"EDW"` |
   | `HISTORY_MONTHS` | **input**: months of postings read the first time, up to yesterday; `-1` reads everything. Under 18 months, the setup only stages the data (training needs 18) | for example `24`, or `3` to test fast |
   | `CLIENTS_LIMIT` | **input**: how many business clients to read (the lowest customer numbers); `-1` all | `-1`, or for example `1000` to test fast |
   | `OUTPUT_SQL_CONNECTION` | **output**: the database the forecast goes to, the table `final_forecast` (a connection you may write into, never the warehouse) | your output database |
   | `OUTPUT_CONNECTION` | **output**: where the project writes its other datasets (CSV: clients, logs) | `"filesystem_managed"` |
   | `OUTPUT_FOLDER_CONNECTION` | **output**: where it writes its folders (Parquet: `daily_flows`, `forecast`; models) | `"filesystem_folders"` |
   | `CODE_ENV` | the Python code env | for example `"cashflow_forecast"` |

   The project reads the warehouse through SQL only: there is no CSV input.

4. **Run All.** It first checks it can read both warehouse tables (it stops with a clear message if not). Then it
   builds the flow, reads the history once (the last `HISTORY_MONTHS` months), trains and promotes a first model, and
   runs the first forecast. This first read is the long part. Check it printed `DIM_CUSTOMER: linked to EDW.DIM_CUSTOMER
   of <your connection>` (and the same for `FCT_STATEMENT_ENTRY`).
5. **Check the last cell:** the production model, its test scores, `final_forecast` (one row per active client)
   and the logs.
6. **Switch on the scenario triggers** (Scenarios → each scenario → Settings): they are created switched off.

The test-machine notebook `warehouse_simulator.ipynb` is not part of the bank's package: it uses fake data.

## Every day and every month

| Scenario | When | What it does |
|---|---|---|
| **Nightly batch** | daily, 06:00 | reads the new days, forecasts every active client with the production model |
| **Retrain, test and promote** | monthly, on the 5th | trains a new model and activates it only if it passes the promotion rules |
| **Full rebuild of the daily flows** | by hand | reads the history again (the last `HISTORY_MONTHS` months) |

**To follow a long run:** the setup notebook prints progress lines while it runs (rows read of the total, %, about how long is left). For a scenario, open its job (Jobs) → Log and search `PROGRESS`.

**After each run, check:**
- `inference_log`: `status` is `SUCCESS`; `active_clients` = `clients_ok` + `clients_not_enough_history` +
  `clients_no_postings`.
- `staging_log`: what was read from the warehouse.
- `promotion_log`: whether the new model was promoted, and why.

## When something goes wrong

| You see | Do |
|---|---|
| `staging_log` FAILED: "the warehouse load of <day> looks late" | wait for the warehouse load, then run the Nightly batch again |
| "run the scenario Full rebuild" | run **Full rebuild of the daily flows**, then the Nightly batch |
| Staging looks stuck | Jobs → the job → Log, search `PROGRESS`. `no row yet, the warehouse is still running the query` = Oracle is still working on it: the line above shows the exact SQL, to give the DBA. The row count going up slowly = rows are arriving, it's just long |
| `inference_log` FAILED | read its `error`; `final_forecast` and the folder `forecast` still hold the previous forecast |
| The Nightly batch's last step (SQL load) FAILED | the forecast is complete in the folder `forecast`: fix the database, then build `final_forecast` alone |
| "No production model yet" | run **Retrain, test and promote** |
| The setup stops on the code env | send the admin the request it printed |
| The setup says "Cannot read EDW.<table>" | check `EDW_CONNECTION`, `EDW_SCHEMA`, the table's columns, and that the connection's user may read it |
| The setup says "already has a dataset ... that is not the table" | fake or old data in the project: delete that dataset, or use a new empty project |
| Opening a warehouse dataset says "Invalid number of columns" | the dataset's Settings → **Test table & get schema** → Save (the recipes don't need it) |
| A wrong forecast after a model change | activate the previous version of the saved model `cash_forecast`, run the Nightly batch |

## Settings you can change without code (project variables)

| Variable | Default | What it does |
|---|---|---|
| `forecaster` | `daily_catboost` | the model Retrain trains |
| `promotion_min_coverage`, `promotion_max_coverage` | 0.70, 0.90 | promotion rule: P10–P90 coverage range |
| `promotion_beat_simple_guess` | true | promotion rule: must beat "same as the last 30 days" |
| `promotion_min_gain` | 0.0 | promotion rule: how much better than production |
| `inference_workers` | 4 | groups of clients forecast side by side: more is faster (up to the server's cores) but needs more memory |
| `history_months`, `clients_limit` | from `HISTORY_MONTHS`, `CLIENTS_LIMIT` | set by the setup; to change them, run the setup again, then **Full rebuild** |
| `cutoff_override` | empty | forecast from another day's data (YYYY-MM-DD) instead of yesterday |


## Load test (a project of its own)

Times the nightly forecast on fake data, in a separate project, so it never touches this one. It runs the same code
(`cashflow.py`, `models.py`, `forecast.py`, copied by its setup notebook) with the placeholder model fitted on 2
clients only: its forecasts mean nothing, its prediction time is real.

1. **Set it up:** in a new empty project, import `build_load_test_project.ipynb`, fill in `CODE_ENV` and
   `OUTPUT_SQL_CONNECTION`, and run all cells. It sets up the project, runs the load test once and shows the result.
2. **One scenario, Run the load test.** Its **step 1** (Scenarios → Run the load test → Steps → 1) sets `tests`, a
   list of tests run one after the other in one go; change it there, Save, then Run. Running the setup notebook again
   keeps it. Each test has every number:

   | Number | What it is |
   |---|---|
   | `clients` | fake clients |
   | `months` | months of history per client (at least 16) |
   | `postings_per_month` | average postings (transactions) per client and month |
   | `clients_per_group` | clients per group: one Parquet file per group and month (production: 5000) |
   | `workers` | groups of clients forecast at a time (production: `inference_workers`) |

   For example, 4 then 8 workers on the same data:

   ```json
   {"tests": [
     {"clients": 420000, "months": 24, "postings_per_month": 40, "clients_per_group": 5000, "workers": 4},
     {"clients": 420000, "months": 24, "postings_per_month": 40, "clients_per_group": 5000, "workers": 8}
   ]}
   ```

   It makes the fake data of each size if it is not there yet (zone Data prep, not timed, one subfolder of
   `load_test_flows` per size, kept for the next runs; tests of the same size share it), runs each test (the
   forecast, then its SQL load; zone Load test), then the report. Each test runs in its own process: its memory is
   measured on its own, and running out of memory is logged as a failed test; the next test still runs.
3. **Results:** the dashboard **Load test**, and `latest_report.html` in the folder `load_test_report` (one page: the
   tests of this run side by side and next to the earlier ones, in plain words and as a table, the time and the
   memory). Every test: one row of `load_test_log`, its `batch` the run it was part of.

After the forecast code changes, run `build_load_test_project.ipynb` again so the load test times the new code.

## Files

The bank's package is `build_forecast_project.ipynb`, `build_load_test_project.ipynb`, `forecast_lab.ipynb`, `daily_flows_viewer.ipynb` and this README. The rest stays in the code repository:

| File | What it is |
|---|---|
| `src/` | the code the notebooks are built from; edit it, then run `python make_notebook.py` |
| `forecast_lab.ipynb` | the data scientist's lab (outside the flow): EDA and model comparison; then set `forecaster` and run Retrain |
| `warehouse_simulator.ipynb` | test machine only: fake warehouse tables in a local test database |
| `build_load_test_project.ipynb` | sets up the load test, a project of its own (see above) |
| `daily_flows_viewer.ipynb` | reads and shows the Parquet files of `daily_flows` (or, in the load test project, one size of `load_test_flows`); only reads |
| `alerts_plan.md` | planned email / Teams alerts (not built) |
