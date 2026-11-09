# Digital Cash Management Forecast: run it at the bank

Forecasts, for every active business client, the net cash flow of the next 30, 60 and 90 days (from all the data up
to yesterday), per category and in TOTAL,
as P10 / EV / P90. How it works, step by step: [`docs/how_it_works.md`](../docs/how_it_works.md).

## Before you start (ask once)

1. **A Python code env** named `cashflow_forecast` (Python 3.9 or later) with `scikit-learn` and `mlflow==2.22.0`.
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
   | `OUTPUT_CONNECTION` | **output**: where the project writes its datasets (tables) | `"filesystem_managed"` |
   | `OUTPUT_FOLDER_CONNECTION` | **output**: where it writes its folders (files: `daily_flows`, models, archive) | `"filesystem_folders"` |
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
| **Load test** | by hand | times the flow on fake data (zone Load test only, never the real data); see below |

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
| `inference_log` FAILED | read its `error`; `final_forecast` still holds the previous forecast |
| "No production model yet" | run **Retrain, test and promote** |
| The setup stops on the code env | send the admin the request it printed |
| The setup says "Cannot read EDW.<table>" | check `EDW_CONNECTION`, `EDW_SCHEMA`, the table's columns, and that the connection's user may read it |
| The setup says "already has a dataset ... that is not the table" | fake or old data in the project: delete that dataset, or use a new empty project |
| Opening a warehouse dataset says "Invalid number of columns" | the dataset's Settings → **Test table & get schema** → Save (the recipes don't need it) |
| A wrong forecast after a model change | activate the previous version of the saved model `cash_forecast`, run the Nightly batch |

## Settings you can change without code (project variables)

| Variable | Default | What it does |
|---|---|---|
| `forecaster` | `quantile_boosting` | the model Retrain trains |
| `promotion_min_coverage`, `promotion_max_coverage` | 0.70, 0.90 | promotion rule: P10–P90 coverage range |
| `promotion_beat_simple_guess` | true | promotion rule: must beat "same as the last 30 days" |
| `promotion_min_gain` | 0.0 | promotion rule: how much better than production |
| `inference_clients_per_chunk` | 5000 | clients forecast at a time: bigger is faster but needs more memory |
| `history_months`, `clients_limit` | from `HISTORY_MONTHS`, `CLIENTS_LIMIT` | set by the setup; to change them, run the setup again, then **Full rebuild** |
| `cutoff_override` | empty | forecast from another day's data (YYYY-MM-DD) instead of yesterday |


## Load test

Times the flow on fake data in its own zone, **Load test**. It never reads or writes the real data, models or logs.

1. **Scenarios → Load test → Steps → 1. Set the load test numbers**, change the values, Save:

   | Variable | Default | What it is |
   |---|---|---|
   | `loadtest_clients` | 1000 | fake clients (for example 420000 for the bank's size) |
   | `loadtest_months` | 24 | months of history per client |
   | `loadtest_postings_per_month` | 40 | average postings (transactions) per client and month |
   | `loadtest_clients_per_chunk` | 5000 | clients forecast at a time |
   | `loadtest_model` | dummy | `dummy`, or a real forecaster's name (`quantile_boosting`) to time it |

2. **Run** the scenario. Progress: the job's log (Jobs → the last job → the recipe → lines starting with `PROGRESS`).
3. **Results:** `load_test_log`, one row per run: the data size (clients, postings, daily rows, Parquet MB), the
   seconds of each step (generating, reading, training and testing, forecasting, writing, total), clients per second
   and the most memory used (`peak_memory_mb`). Compare runs with different numbers.

Running the setup again keeps the numbers set in step 1.

## Files

The bank's package is `build_forecast_project.ipynb`, `forecast_lab.ipynb`, `daily_flows_viewer.ipynb` and this README. The rest stays in the code repository:

| File | What it is |
|---|---|
| `src/` | the code the notebooks are built from; edit it, then run `python make_notebook.py` |
| `forecast_lab.ipynb` | the data scientist's lab (outside the flow): EDA and model comparison; then set `forecaster` and run Retrain |
| `warehouse_simulator.ipynb` | test machine only: fake warehouse tables in a local test database |
| `daily_flows_viewer.ipynb` | reads and shows the Parquet files of `daily_flows` (or `load_test_flows`); only reads |
| `alerts_plan.md` | planned email / Teams alerts (not built) |
