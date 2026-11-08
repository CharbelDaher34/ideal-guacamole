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

1. **Put the bank's codes in `src/warehouse.py`:**
   - `CATEGORY_OF_TYPE`: the real transaction types (`TRNS_TYP`) and their category. Today it's a mock list.
   - Check `ACTIVE_STATUS`, `RETAIL`, `BUSINESS_ID`, `SOLE_PROPRIETOR` against `DIM_CUSTOMER`.
   - `check_warehouse_loaded`: today a mock that always passes. Replace it with the bank's check that the
     warehouse's load of yesterday is finished (for example a query on its load-control table).

   Then rebuild the notebooks: `python make_notebook.py`.
2. **Import** `build_forecast_project.ipynb` into the project (Notebooks → New notebook → Import).
3. **Fill its first cell:**

   | Setting | Set it to |
   |---|---|
   | `WAREHOUSE_SQL_CONNECTION` | the connection's name, for example `"EDW"` |
   | `WAREHOUSE_SQL_SCHEMA` | the tables' schema, for example `"EDW"` |
   | `CONNECTION`, `FOLDER_CONNECTION` | where the project may write its datasets and folders |
   | `CODE_ENV` | `"cashflow_forecast"` |

4. **Run All.** It builds the flow, reads the whole history once, trains and promotes a first model, and runs the
   first forecast. The first read of the whole history is the long part.
5. **Check the last cell:** the production model, its test scores, `final_forecast` (one row per active client)
   and the logs.
6. **Switch on the scenario triggers** (Scenarios → each scenario → Settings): they are created switched off.

Don't import or run `warehouse_simulator.ipynb` or `inference_load_test.ipynb` in this project: they replace the
data with fake data.

## Every day and every month

| Scenario | When | What it does |
|---|---|---|
| **Nightly batch** | daily, 06:00 | reads the new days, forecasts every active client with the production model |
| **Retrain, test and promote** | monthly, on the 5th | trains a new model and activates it only if it passes the promotion rules |
| **Full rebuild of the daily flows** | by hand | reads the whole history again |

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
| `inference_log` FAILED | read its `error`; `final_forecast` still holds the previous forecast |
| "No production model yet" | run **Retrain, test and promote** |
| The setup stops on the code env | send the admin the request it printed |
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
| `cutoff_override` | empty | forecast from another day's data (YYYY-MM-DD) instead of yesterday |

## Optional: benchmark the bank's server

In a separate sandbox project with the forecast set up, run `inference_load_test.ipynb`. It times the nightly
forecast for 420,000 fake clients and shows the memory used, which helps choose `inference_clients_per_chunk`.

## Files

| File | What it is |
|---|---|
| `build_forecast_project.ipynb` | sets up the project |
| `forecast_lab.ipynb` | the data scientist's lab: EDA and model comparison |
| `warehouse_simulator.ipynb` | sandbox only: fake warehouse tables |
| `inference_load_test.ipynb` | sandbox only: the 420k-client benchmark |
| `alerts_plan.md` | planned email / Teams alerts (not built) |
| `src/` | the code the notebooks are built from; edit it, then run `python make_notebook.py` |
