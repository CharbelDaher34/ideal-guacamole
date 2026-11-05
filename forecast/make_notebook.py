"""Assemble the notebooks from the files in src/ (edit those, then run: python make_notebook.py):
build_forecast_project.ipynb (sets up the forecast project), forecast_lab.ipynb (the data scientist's lab) and
warehouse_simulator.ipynb (sandbox only)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
RECIPES = ["stage_tables", "obligations", "build_training_rows", "train_and_test", "champion_challenger",
           "build_scoring_rows", "final_forecast", "archive_forecast", "try_forecaster"]


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
that name), the way it runs in production. Everything is stored as CSV files. It reads the 3 tables the warehouse
team builds from the bank's main data (`warehouse_tables.sql`): clients (`EDW.DIM_CUSTOMER`), money in and out per
month (`EDW.FCT_STATEMENT_ENTRY`) and the scheduled payments, and the load record (in production: tables through a
database connection, `WAREHOUSE_SQL_CONNECTION`, or CSV files in the landing folder): in the
sandbox, the datasets the notebook `warehouse_simulator.ipynb` wrote in this project; in production, the warehouse
team's CSV files in the landing folder, read-only through `EDW_LANDING` (names: `python/warehouse.py` in the
project library). It builds 6 flow zones, each with an owner:

- **Raw data** and **Staging** (shared): the 3 tables and the load record, read-only, then the same tables with the
  standard column names and the project's values (direction, client status, sole proprietor).
- **Data prep** (shared by training and inference): the category mapping (a CSV dataset) applied by visual Join
  and Group recipes, the obligations.
- **Training** (data scientist): the training rows, then each training run is one model with its own **model ID**,
  trained with the **forecaster** the project variable `forecaster` names, tracked in Dataiku's **Experiment
  Tracking** and deployed as a new **version** of the saved models `forecast_ev`, `forecast_p10`, `forecast_p90`.
  A champion / challenger test activates the new version if it beats the production model.
- **Inference** (inference owner): the scoring rows (latest month), visual Score recipes with the active version, the 3-month
  forecast (every row says which model and which load produced it), the forecast archive, and the inference log
  (one row per run: when, which model, which load, how many clients).
- **Lab** (data scientist): `try_forecaster` tries the forecaster the project variable `lab_forecaster` names
  against the production model and adds the result to `lab_results`, without touching production. The notebook
  `forecast_lab.ipynb` does the same interactively.

The zones only meet through three interfaces: **1. the model rows** (`model_rows` in `cashflow.py`: data prep →
training and inference), **2. the forecaster** (`models.py`: lab → training), **3. the active version of the saved
models** (training → inference). Two independent scenarios run it: **Nightly batch** (inference) and **Retrain,
test and promote** (training). Each starts with the load check (`check_latest_load`): it stops the scenario if the
latest load is incomplete, and sets `cutoff_month`.

It then trains, tests and promotes a first model, and runs the first nightly batch.

**How to use it:** in the project, import this notebook and `warehouse_simulator.ipynb` (Notebooks > New notebook
> Import). In the sandbox, run the simulator first. Then check the settings below and run all cells (Cell > Run All):
a few minutes with the small sample data. It can be run again in the same project: what exists is kept (data, the
category mapping, saved model versions, zones, scenario history), and the recipes' code is updated.
"""),
    code(read("cells/settings.py")),
    md("""
## 1. Code environment

The recipes need CatBoost, and MLflow for Dataiku's Experiment Tracking, in an existing code env (`CODE_ENV`
above). This notebook does not create it: an admin creates it once. This checks it is there with those packages,
and otherwise stops with the request to send the admin.
"""),
    code(read("cells/code_env.py")),
    md("""
## 2. Code

The project library gets these files, by owner:

- shared: `warehouse.py` (warehouse table and column names, the load check) and `cashflow.py` (categories and the
  default category mapping, the model rows, the saved model names)
- training and lab: `models.py` (the forecaster interface, the reference CatBoost forecaster, the test and its
  scores) and `tracking.py` (model IDs, Experiment Tracking, saved model versions, champion / challenger, promotion)
- inference: `forecast.py` (the production model's ID, the final forecast table)
- `forecasters.py`: the data scientist's own forecasters. Written only when it is created: running this notebook
  again never overwrites it.

Each recipe below becomes a Python recipe in the flow; the visual recipes are created in the next step.
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

Uses the simulated warehouse tables, or links the landing files (read-only), and creates the category
mapping, the recipes (Python, Join, Group, Score) and their CSV datasets, the folders `experiments` (tracked
models), `forecast_archive` and `lab_experiments` (lab tries), the three saved models, and the project variables
`forecaster` (`catboost`) and `lab_forecaster` (`catboost_deeper`) if they are not set yet, in 6 zones. Training
(folder `experiments`), the promotion log, the saved models and the lab are only rebuilt when asked by name, so the
nightly batch never retrains by accident.
"""),
    code(read("cells/create_flow.py")),
    md("""
## 4. First run

Checks the load, trains, tests and deploys a model, runs champion / challenger (with no production model yet, the
first model is promoted, with a warning in promotion_log if its checks fail), then runs the nightly batch: score,
forecast, archive, log.
"""),
    code(read("cells/first_run.py")),
    md("""
## 5. Scenarios

- **Nightly batch**: check the load, build the scoring rows, forecast with the production model, archive it and
  log the run. It does not touch the training data.
- **Retrain, test and promote**: check the load, prepare the training data, train, test and deploy a new model
  with the forecaster the project variable `forecaster` names, champion / challenger, activate it if better. It
  does not forecast: the next Nightly batch uses the new active version.

Their triggers (daily at 06:00 and monthly on the 5th, Riyadh time) are created switched off; switch them on
in production.
"""),
    code(read("cells/scenarios.py")),
    md("""
## 6. Results

To use another model (for example to roll back), activate its version in the three saved models
(`forecast_ev`, `forecast_p10`, `forecast_p90`) and run the Nightly batch.
"""),
    code(read("cells/results.py")),
])

write("forecast_lab.ipynb", [
    md("""
# Digital Cash Management Forecast: lab (data scientist)

Explore the data and try models here without touching production: nothing in this notebook changes the flow, the
saved models or the production model.

1. **EDA**: the training rows by category, totals over time, seasonality, and what the models learn.
2. **Models to try**: several forecasters trained and tested side by side on the same test as champion / challenger
   (trained without the last 6 months before the cut-off, tested on them), next to the production model and the
   simple guess (the average of the last 3 months), with the decision champion / challenger would take for each.
   Each try is recorded in Experiment Tracking, experiment `cashflow_lab` (Lab → Experiment tracking); lab tries
   are never deployed.
3. **Put one in production.**

**Kernel:** the code env `cashflow_forecast`. If it is not in the kernel list, ask whoever manages it to enable its
Jupyter support; meanwhile, the recipe `try_forecaster` tries one forecaster without a notebook (set the project
variable `lab_forecaster` and build `lab_results`, Lab zone).

In the sandbox the data is simulated: there, the scores show that the code works, not which model is better.
"""),
    code(read("lab_cells/setup.py")),
    md("""
## 1. EDA

The training rows: one row per client, category, cut-off month and months ahead (1 to 3). The models predict
`target`: the month's amount (`actual`) minus the part already scheduled at the cut-off (`obligations_known`), which
the forecast adds back.

- **zero_months**: the share of months with no movement. Lumpy categories are hard to forecast.
- **known_at_cutoff**: the share of the amount already scheduled one month ahead. That part is not predicted.
- **volatility**: how much a client's month varies (median coefficient of variation of the monthly amounts).
"""),
    code(read("lab_cells/eda_overview.py")),
    md("""
Totals over time and seasonality. Ramadan and Eid move about 11 days earlier each year, so a calendar-month profile
blurs them.
"""),
    code(read("lab_cells/eda_charts.py")),
    md("""
What the models learn: `target` divided by the series scale (the client's usual amount in that category), its spread
per category, and how much each input moves with it.
"""),
    code(read("lab_cells/eda_target.py")),
    md("""
## 2. Models to try

A forecaster is a class with `name`, `params` (its settings), `inputs` (the model-row columns it reads), `fit(rows)`
and `predict(rows)`, which returns `p10`, `ev` and `p90` in SAR (details in `models.py`, project library). Three
examples, next to the reference CatBoost forecaster (`catboost`, the production recipe):

- `simple_guess`: no model, the average of the last 3 months, with a band from its own past errors
- `catboost_depth8`: the reference CatBoost forecaster with deeper trees
- `sklearn_boosting`: scikit-learn gradient boosting, one model per output (EV, P10, P90)

Change them or add your own, then run the comparison. CatBoost, scikit-learn, pandas and matplotlib are in the code
env; other packages need the admin.
"""),
    code(read("lab_cells/forecasters_to_try.py")),
    code(read("lab_cells/compare.py")),
    code(read("lab_cells/compare_chart.py")),
    md("""
## 3. Put one in production

1. Copy its class into `forecasters.py` (project library) with the imports it needs (for example
   `import numpy as np`, `import pandas as pd`, `from models import CAT_FEATURES, CatBoostForecaster, model_matrix`)
   and add it to `FORECASTERS`. The setup notebook never overwrites that file.
2. Set the project variable `forecaster` to its name.
3. Run the scenario **Retrain, test and promote**. Champion / challenger promotes it only if it beats the
   production model on the same 6 months; the next Nightly batch forecasts with it.
4. To stop training it, set `forecaster` back to `catboost`. To use an earlier model now, activate its version in
   the three saved models.
"""),
], kernel={"display_name": "Python (env cashflow_forecast)", "language": "python",    # the code env's own kernel
           "name": "py-dku-venv-cashflow_forecast"})

write("warehouse_simulator.ipynb", [
    md("""
# Digital Cash Management Forecast: warehouse simulator (sandbox only)

Plays the warehouse team (Informatica) until the real files exist. Run it in the same project as the forecast,
before `build_forecast_project.ipynb`.

It generates sample data up to `SIM_END` (accounts, postings, loans, deposits, standing orders and so on) and
publishes, as the warehouse would have them at the end of `AS_OF` (what had happened or was scheduled by then), the
3 tables the forecast reads, built the way the warehouse team builds them from EDW (`warehouse_tables.sql`): the
clients (`EDW.DIM_CUSTOMER`), the statement entries summed per month (`EDW.FCT_STATEMENT_ENTRY`) and the scheduled
payments. They are written as CSV datasets of this project, in their EDW column names (`warehouse.py`). Each load
adds one row per table to `CFF_LOAD_RECORD`, which the forecast checks before it runs.

To simulate the next load, move `AS_OF` to the next month's end, run all cells again, then run the scenario
**Nightly batch**.
"""),
    code(read("sim_cells/settings.py")),
    code(read("sim_cells/setup.py")),
    md("""
## Sample data, warehouse names, publishing

The sample data generator, the warehouse table and column names, the publishing as of a date, and the 3 tables.
"""),
    code(read("generate_data.py")),
    code(read("warehouse.py")),
    code(read("simulator.py")),
    md("""
## Load

Generates the data (about 2 minutes), publishes it as of `AS_OF` and records the load.
"""),
    code(read("sim_cells/load.py")),
])
