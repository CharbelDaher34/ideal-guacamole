"""Assemble the notebooks from the files in src/ (edit those, then run: python make_notebook.py):
build_forecast_project.ipynb (sets up the forecast project), build_load_test_project.ipynb (sets up the load test, a
project of its own, with the same forecast code), forecast_lab.ipynb (the data scientist's lab),
daily_flows_viewer.ipynb (reads the daily flows), warehouse_simulator.ipynb (test machine only). A notebook whose files are not in src/
is skipped: the bank's package has only what build_forecast_project.ipynb needs."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
RECIPES = ["stage_tables", "train_and_test", "champion_challenger", "nightly_forecast", "load_forecast_sql"]
LOAD_TEST_RECIPES = ["make_fake_data", "run_load_test", "load_test_report"]


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def embedded(name, path):
    text = path.read_text().strip("\n")
    assert "'''" not in text, f"{path} must not contain triple single quotes"
    return f"{name} = r'''\n{text}\n'''"


def read(path):
    return (SRC / path).read_text()


def write_if_present(name, folder, cells, kernel=None):
    """Writes the notebook only if src/<folder> is there (cells: a function returning its cells)."""
    if not (SRC / folder).is_dir():
        print(f"skipped {name} (no src/{folder})")
        return
    write(name, cells(), kernel)


def write(name, cells, kernel=None):
    """kernel: the Jupyter kernel the notebook opens with (default: Dataiku's built-in Python)."""
    kernelspec = kernel or {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook = {"cells": cells, "metadata": {"kernelspec": kernelspec, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {name} ({len(cells)} cells)")


write("build_forecast_project.ipynb", [
    md("""
# Digital Cash Management Forecast: set up the project

This notebook sets up the Digital Cash Management Forecast in this Dataiku project (it also gives the project
that name), the way it runs in production. The daily flows and the forecast are Parquet files in folders (for speed),
the forecast is then loaded into a SQL table, `final_forecast` (`OUTPUT_SQL_CONNECTION`); the logs and small tables
are CSV files.
It reads 2 warehouse tables, read-only, and creates or writes nothing in the warehouse: the clients
(`EDW.DIM_CUSTOMER`) and the daily postings (`EDW.FCT_STATEMENT_ENTRY`), which the staging recipe sums per client,
day and category in Dataiku. The first run reads each client's whole history; after that, each run reads only the
new days (through the database connection, a `SELECT` with a date filter) and adds them to `daily_flows`. It always
reads them through SQL, through the read-only database connection `EDW_CONNECTION`: at the bank, the warehouse;
on the test machine, a local test database with fake data (`warehouse_simulator.ipynb`). The same code, only the
connection differs. Names, codes and the categories (a mock list until the bank's transaction types are known):
`python/warehouse.py` in the project library.

It forecasts, per client, per category and in TOTAL (all categories together), the net cash flow (money in minus
money out) of each of the next 30, 60 and 90 days after the cut-off day (yesterday: days 1-30, 31-60, 61-90), as
P10, EV and P90, from each client's daily flows of the last 365 days. It builds 4 flow zones, each with an owner:

- **Raw data** and **Staging** (shared): the 2 warehouse tables, read-only, then `clients` (the business clients)
  and `daily_flows` (money in and out per client, day and category): what every model gets. `staging_log` records
  each read.
- **Training** (data scientist): each training run is one model with its own **model ID**, trained with the
  **forecaster** the project variable `forecaster` names, tested on the last 6 periods of 30 days, tracked in Dataiku's
  **Experiment Tracking** and deployed as a new **version** of the saved model `cash_forecast`. A champion /
  challenger test activates the new version if it passes the **promotion rules** (project variables `promotion_*`).
- **Inference** (inference owner): the production model's forecast for every active client, one row per client
  (every row says which model and which load produced it), Polars from start to end: the groups of clients of
  `daily_flows` run side by side (project variable `inference_workers`), each read (its last months only),
  prepared, forecast and written on its own into the folder `forecast` (one subfolder per night, kept). Then it is
  loaded into the SQL table `final_forecast`. The inference log has one row per run, with timings and memory.
The data scientist tries new forecasters outside the flow, in the notebook `forecast_lab.ipynb` (it never touches
production), then sets the project variable `forecaster` and runs **Retrain, test and promote**.

The load test is a project of its own (`build_load_test_project.ipynb`, the same code): nothing of it is here.

The zones only meet through three interfaces: **1. the data** (`daily_flows` and `clients`: staging → training, lab
and inference), **2. the forecaster** (`models.py`: lab → training), **3. the active version of the saved model**
(training → inference). Until the data scientist's models are ready, the forecaster is a placeholder,
`daily_catboost`: ONE CatBoost model that gives P10, EV and P90 together, for every category and the 3 windows. Its
inputs: the client's net cash flow of each of the last 365 days (one column per day), the category and the window (1,
2, 3), the rows the nightly forecast builds (`cashflow.model_rows`); the model only predicts (`predict_rows`). Two
independent scenarios run it:
**Nightly batch** (inference) and **Retrain, test and promote** (training). Each starts by setting the cut-off day
(`set_cutoff`: yesterday, or the project variable `cutoff_override`) and checking the warehouse load of that day is
finished (`check_warehouse_loaded`: a mock for now, to replace with the bank's real check); the staging recipe stops
if the postings do not reach the cut-off day.

It then trains, tests and promotes a first model, and runs the first nightly batch.

**How to use it (at the bank):** in a new empty project, import this notebook only (Notebooks > New notebook >
Import), fill in the settings below (`EDW_CONNECTION`, `CODE_ENV`, ...) and run all cells (Cell > Run All). It first
checks it can read the 2 warehouse tables, and stops with a clear message if not. It can be run again in the same
project: what exists is kept (data, saved model versions, zones, scenario history), and the recipes' code is updated.

On the test machine only: run `warehouse_simulator.ipynb` first (it writes fake tables into the local test
database), then this notebook with `EDW_CONNECTION` set to that database's connection.
"""),
    code(read("cells/settings.py")),
    md("""
## 1. Code environment

The recipes need CatBoost (the placeholder model), scikit-learn, and MLflow for Dataiku's Experiment Tracking, in an existing code env (`CODE_ENV`
above). This notebook does not create it: an admin creates it once. This checks it is there with those packages,
and otherwise stops with the request to send the admin.
"""),
    code(read("cells/check_code_env.py") + "\n\n" + read("cells/code_env.py")),
    md("""
## 2. Code

The project library gets these files, by owner:

- shared: `warehouse.py` (the warehouse tables and columns read, codes, categories, daily sums, cut-off day, warehouse check) and
  `cashflow.py` (dates, the daily flows' files, the model's input rows, the saved model name)
- training and lab: `models.py` (the model interface, the placeholder models, the test and its scores) and
  `tracking.py` (model IDs, Experiment Tracking, saved model versions, promotion rules, champion / challenger)
- inference: `forecast.py` (the production model, the nightly forecast in parallel, its SQL load)
- `forecasters.py`: the data scientist's own forecasters. Written only when it is created: running this notebook
  again never overwrites it.

Each recipe below becomes a Python recipe in the flow.
"""),
    code(embedded("WAREHOUSE_CODE", SRC / "warehouse.py")),
    code(embedded("LIBRARY_CODE", SRC / "library_cashflow.py")),
    code(embedded("MODELS_CODE", SRC / "models.py")),
    code(embedded("TRACKING_CODE", SRC / "tracking.py")),
    code(embedded("FORECAST_CODE", SRC / "forecast.py")),
    code(embedded("FORECASTERS_CODE", SRC / "forecasters.py")),
    code("RECIPES = {}"),
    *[code(embedded(f'RECIPES["{r}"]', SRC / "recipes" / f"{r}.py")) for r in RECIPES],
    md("""
## 3. Build the flow

Uses the simulated warehouse tables, or links the warehouse's (read-only), and creates the Python recipes and their
datasets (CSV files, and the SQL table `final_forecast`), the folders `daily_flows` (Parquet files), `experiments`
(tracked models) and `forecast` (Parquet, one subfolder per night), the saved model `cash_forecast`, and the project
variables `forecaster` (`daily_catboost`), `inference_workers` (4) and the promotion rules (`promotion_*`) if they are
not set yet, in 4 zones.
Training (folder `experiments`) and the promotion log are only rebuilt when asked by name, so the nightly batch never
retrains by accident.
"""),
    code(read("cells/create_flow.py")),
    md("""
## 4. First run

Builds the daily flows (the whole history, the first time), trains, tests and deploys a model, runs champion / challenger (with no production model yet,
the first model is promoted, with a warning in promotion_log if its checks fail), then runs the nightly batch:
forecast, log, SQL load.
"""),
    code(read("cells/build_job.py") + "\n\n" + read("cells/first_run.py")),
    md("""
## 5. Scenarios

- **Nightly batch**: set the cut-off, add the new days to the daily flows, forecast with the production model and
  log the run, then load the forecast into the SQL table `final_forecast`. It never trains.
- **Retrain, test and promote**: set the cut-off, add the new days to the daily flows, train, test and deploy a new model with
  the forecaster the project variable `forecaster` names, champion / challenger with the promotion rules, activate
  it if it passes. It does not forecast: the next Nightly batch uses the new active version.

- **Full rebuild of the daily flows** (run by hand, no trigger): empty `daily_flows` and `staging_log`, then read the
  whole history again. For example after a reversal cancels a posting from a day already staged.

Their triggers (daily at 06:00 and monthly on the 5th, Riyadh time) are created switched off; switch them on
in production.
"""),
    code(read("cells/scenarios.py")),
    md("""
## 6. Results

To use another model (for example to roll back), activate its version in the saved model `cash_forecast` and run
the Nightly batch. To change when a model is promoted, change the project variables `promotion_*` (no code change).
"""),
    code(read("cells/results.py")),
])

write_if_present("build_load_test_project.ipynb", "load_test_cells", lambda: [
    md("""
# Digital Cash Management Forecast: load test

This notebook sets up the load test in this Dataiku project, a project of its own: it never reads or writes the
forecast project. It times the nightly forecast on fake data of the same shape as the real daily flows, with the
forecast project's own code (`cashflow.py`, `models.py`, `forecast.py`: the same files, copied by this notebook).

One scenario, **Run the load test**, run by hand. Its **step 1** (Scenarios > Run the load test > Steps > 1) sets
`tests`: a list of tests run one after the other in one go, each with every number: the fake data's size (`clients`,
`months`, `postings_per_month`, `clients_per_group`) and `workers`. For example:

```json
{"tests": [
  {"clients": 420000, "months": 24, "postings_per_month": 40, "clients_per_group": 5000, "workers": 4},
  {"clients": 420000, "months": 24, "postings_per_month": 40, "clients_per_group": 5000, "workers": 8}
]}
```

Change them there, then Run. It runs, in 2 zones:

- **Data prep** (not timed): the fake daily flows of each size listed in `load_test_flows`, one subfolder per size,
  Parquet files laid out exactly like `daily_flows`; a size already there is kept, a size no longer listed removed.
  Tests with the same size share its data.
- **Load test**: the placeholder model (`daily_catboost`: one CatBoost model, P10, EV and P90 together, the last 365
  days as inputs) fitted on 2 clients only (its forecasts mean nothing, its prediction time is real), then the nightly
  forecast of every fake client, `workers` groups of clients at a time, as production runs it, then its load into the
  SQL table `load_test_final`, timed (the forecast, the SQL load, and the share of each step: reading, preparing,
  model, formatting, writing). Each test runs in its own process: its memory is measured on its own, and running out
  of memory is logged as a failed test (the next test still runs).
- **The report** (even when a test failed): the dashboard **Load test** and `latest_report.html` in the folder
  `load_test_report`: the tests of this run side by side, next to the earlier ones. Every test: one row of the
  dataset `load_test_log` (its `batch`: the run it was part of).

**How to use it:** in a new empty project, import this notebook (Notebooks > New notebook > Import), fill in
`CODE_ENV` and `OUTPUT_SQL_CONNECTION` (and the scenario's first tests if you like) and run all cells (Cell > Run
All). It sets up the project, then (`RUN_NOW`) runs what the scenario runs and shows the result. Later: change the
tests in step 1 of the scenario, then Run. Run this notebook again after the forecast code changes, so the load test
times the new code: it keeps the tests set on the scenario.
"""),
    code(read("load_test_cells/settings.py")),
    md("## 1. Code environment, code, flow\n\nChecks the code env has CatBoost, writes the project library, the 3 "
       "recipes in 2 zones (Data prep: `make_fake_data`; Load test: `run_load_test`, `load_test_report`) and the "
       "scenario, its tests in step 1."),
    code(read("cells/check_code_env.py")),
    code(embedded("WAREHOUSE_CODE", SRC / "warehouse.py")),
    code(embedded("LIBRARY_CODE", SRC / "library_cashflow.py")),
    code(embedded("MODELS_CODE", SRC / "models.py")),
    code(embedded("FORECAST_CODE", SRC / "forecast.py")),
    code(embedded("LOADTEST_CODE", SRC / "loadtest.py")),
    code("RECIPES = {}"),
    *[code(embedded(f'RECIPES["{r}"]', SRC / "load_test_recipes" / f"{r}.py")) for r in LOAD_TEST_RECIPES],
    code(read("load_test_cells/setup.py")),
    md("## 2. Run it\n\nWhat the scenario **Run the load test** runs, with the tests of its step 1 (with many "
       "clients it takes long: the progress shows here)."),
    code(read("cells/build_job.py") + "\n\n" + read("load_test_cells/run.py")),
])

write_if_present("forecast_lab.ipynb", "lab_cells", lambda: [
    md("""
# Digital Cash Management Forecast: lab (data scientist)

Explore the data and try models here without touching production: nothing in this notebook changes the flow, the
saved models or the production model.

1. **EDA**: the daily flows per category, totals over time, seasonality, and one client day by day.
2. **Models to try**: several forecasters trained and tested side by side on the same test as champion / challenger
   (trained without the last 6 periods of 30 days before the cut-off, tested on them), next to the production model
   and the simple guess (the same as the 30 days before), with the decision champion / challenger would take for each.
   Each try is recorded in Experiment Tracking, experiment `cashflow_lab` (same folder as the production models,
   `experiments`, but its own experiment); lab tries are never deployed.
3. **Put one in production.**

**Kernel:** the code env `cashflow_forecast`. If it is not in the kernel list, ask whoever manages it to enable its
Jupyter support. Meanwhile a forecaster can be tried safely through **Retrain, test and promote**: it goes live
only if it beats the production model.

In the sandbox the data is simulated: there, the scores show that the code works, not which model is better.
"""),
    code(read("lab_cells/setup.py")),
    md("""
## 1. EDA

The data every model gets (interface 1): `daily_flows`, one row per client, day and category with money in and
money out, and `clients`. The models predict, per client, per category and in TOTAL (all categories together), the
net cash flow (money in minus money out) of each of the next 30, 60 and 90 days. First, per category and per segment
(**negative_months**: the share of months where more money went out than came in).
"""),
    code(read("lab_cells/eda_overview.py")),
    md("""
Net cash flow per category over time, and seasonality. Ramadan and Eid move about 11 days earlier each year, so a
calendar-month profile blurs them.
"""),
    code(read("lab_cells/eda_charts.py")),
    md("""
One client day by day (what the models see) and its monthly totals.
"""),
    code(read("lab_cells/eda_target.py")),
    md("""
## 2. Models to try

A model is a class with `name`, `params` (its settings), `fit(daily, clients, cutoff)` and `predict_rows(rows)`:
`rows` is a Polars table, one row per client, category (and TOTAL) and window (1, 2, 3: days 1-30, 31-60, 61-90
after the cut-off day), with `day_365 ... day_1` (the net cash flow of each of the last 365 days); it returns P10, EV
and P90 for each row (details in `models.py`, project library). Built on `DayModel`, it also gets `predict(daily,
clients, cutoff)` for the test.
Compared here:

- `daily_catboost`: the placeholder in production. ONE CatBoost model (MultiQuantile loss) gives P10, EV and P90
  together; its inputs, one row per client, category (and TOTAL) and window: the net cash flow of each of the last 365
  days, the category and the window (1, 2, 3).
- `daily_catboost_deeper`: the same with deeper, more numerous trees, defined below: an example to change.
- `recent_average`: no machine learning (the average of the last 3 periods of 30 days), and `recent_average_6m` (over
  6 periods, an example in `forecasters.py`)

Change them or add your own, then run the comparison. CatBoost, scikit-learn, pandas and matplotlib are in the code env; other
packages need the admin.
"""),
    code(read("lab_cells/forecasters_to_try.py")),
    code(read("lab_cells/compare.py")),
    code(read("lab_cells/compare_chart.py")),
    md("""
## 3. Put one in production

1. Copy its class into `forecasters.py` (project library) with the imports it needs (for example
   `import pandas as pd`, `from models import DailyCatBoost`) and add it to
   `FORECASTERS`. The setup notebook never overwrites that file.
2. Set the project variable `forecaster` to its name.
3. Run the scenario **Retrain, test and promote**. Champion / challenger promotes it only if it passes the
   promotion rules (project variables `promotion_*`) against the production model on the same 6 periods; the next
   Nightly batch forecasts with it.
4. To stop training it, set `forecaster` back to `daily_catboost`. To use an earlier model now, activate its
   version in the saved model `cash_forecast`.
"""),
], kernel={"display_name": "Python (env cashflow_forecast)", "language": "python",    # the code env's own kernel
           "name": "py-dku-venv-cashflow_forecast"})

write_if_present("warehouse_simulator.ipynb", "sim_cells", lambda: [
    md("""
# Digital Cash Management Forecast: warehouse simulator (sandbox only)

**Test machine only, never at the bank.** Plays the warehouse with fake data. Run it in the same project as the
forecast, before `build_forecast_project.ipynb`.

It generates sample data up to `SIM_END` and writes, as the warehouse would have them at the end of `AS_OF`, the 2
tables the forecast reads, shaped like the bank's (`warehouse.py`): `DIM_CUSTOMER` (the business clients, and a few
retail customers the forecast must leave out) and `FCT_STATEMENT_ENTRY` (the daily postings). It writes them into a
local test database (`EDW_CONNECTION`, schema `EDW_SCHEMA`), and the forecast reads them through SQL exactly as it
reads the bank's warehouse. It cannot write into the bank's warehouse: that connection is read-only. It also sets
the project variable `cutoff_override` to `AS_OF`.

To simulate the next month, move `AS_OF` to the next month's end, run all cells again, then run the scenario
**Nightly batch**.
"""),
    code(read("sim_cells/settings.py")),
    code(read("sim_cells/setup.py")),
    md("""
## Sample data, warehouse names, publishing

The sample data generator, the warehouse table and column names, the publishing as of a date, and the 2 tables.
"""),
    code(read("generate_data.py")),
    code(read("warehouse.py")),
    code(read("simulator.py")),
    md("""
## Load

Generates the data (about 2 minutes), publishes it as of `AS_OF` and sets the project variable `cutoff_override`.
"""),
    code(read("sim_cells/load.py")),
])

write_if_present("daily_flows_viewer.ipynb", "viewer_cells", lambda: [
    md("""
# Digital Cash Management Forecast: daily flows viewer

Reads the Parquet files of `daily_flows` (what staging writes and every model gets: one row per client, day and
category with `money_in`, `money_out` and `postings`) and shows them. It only reads: nothing here changes the
project. In the load test project (import it there), set `FOLDER` to `load_test_flows` to look at its fake data.

**Kernel:** the project's code env (the setup's `CODE_ENV`, which has polars), like `forecast_lab.ipynb`.
"""),
    code(read("viewer_cells/settings.py")),
    code(read("viewer_cells/setup.py")),
    md("## 1. Files\n\nOne Parquet file per group of clients and month: rows per month, and the first files."),
    code(read("viewer_cells/files.py")),
    md("## 2. Summary\n\nRows, clients, postings and money per category, over the first `SUMMARY_GROUPS` groups of clients."),
    code(read("viewer_cells/summary.py")),
    md("## 3. Sample rows\n\nThe columns and the first rows of the first file."),
    code(read("viewer_cells/sample.py")),
    md("## 4. One client\n\n`CLIENT_NO` day by day, and per month."),
    code(read("viewer_cells/client.py")),
], kernel={"display_name": "Python (env cashflow_forecast)", "language": "python",
           "name": "py-dku-venv-cashflow_forecast"})
