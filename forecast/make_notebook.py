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
that name), the way it runs in production. Everything is stored as CSV files. It reads the 4 curated warehouse
tables (client, monthly flows by origin, scheduled payments, monthly balances) and the load record: in the
sandbox, the datasets the notebook `warehouse_simulator.ipynb` wrote in this project; in production, the warehouse
team's CSV files in the landing folder, read-only through `EDW_LANDING` (names: `python/warehouse.py` in the
project library). It builds 6 flow zones, each with an owner:

- **Raw data** and **Staging** (shared): the 4 curated tables and the load record, read-only, then the same tables
  with the standard column names.
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

Try other models here without touching production: nothing in this notebook changes the flow, the saved models or
the production model.

- **Same test as production.** Your forecaster is trained without the last 6 months before the cut-off and tested
  on them, the way champion / challenger tests a new model. Your scores sit next to the production model's and the
  simple guess (the average of the last 3 months), with the decision champion / challenger would take.
- **Each try is recorded** in Experiment Tracking, experiment `cashflow_lab`: its settings, its test scores and the
  trained forecaster. Compare tries there (Lab → Experiment tracking). Lab tries are never deployed.
- **The same try without a notebook:** set the project variable `lab_forecaster` to a forecaster's name and build
  `lab_results` (Lab zone). Each try adds its rows to `lab_results`.

**Kernel:** the code env `cashflow_forecast` (Kernel → Change kernel). If it is not in the list, the env has no
Jupyter support yet: ask whoever manages it to enable it, and meanwhile use `lab_results`.

**A forecaster** is a class with `name`, `params` (its settings), `inputs` (the model-row columns it reads),
`fit(rows)` and `predict(rows)`, which returns `p10`, `ev` and `p90` in SAR (details in `models.py`, project
library). The amount to predict is the column `target`: the month's amount minus the obligations already known at
the cut-off, which the forecast adds back.
"""),
    code(read("lab_cells/setup.py")),
    code(read("lab_cells/try_model.py")),
    md("""
## Put it in production

1. Copy the class into `forecasters.py` (project library) and add it to `FORECASTERS`. The setup notebook never
   overwrites that file.
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

It generates sample data for the 17 detailed tables of the data requirements, up to `SIM_END`, and publishes them
as the warehouse would have them at the end of `AS_OF` (what had happened or was scheduled by then), then builds
the 4 curated tables from them, as the warehouse team would: client, monthly flows by origin, scheduled payments,
monthly balances. They are written as CSV datasets of this project, in the warehouse's table and column names
(`warehouse.py`). Each load adds one row per table to `CFF_LOAD_RECORD`, which the forecast checks before it runs.
Set `WRITE_DETAILED_TABLES = True` to also write the 17 detailed tables (the forecast does not read them).

To simulate the next load, move `AS_OF` to the next month's end, run all cells again, then run the scenario
**Nightly batch**.
"""),
    code(read("sim_cells/settings.py")),
    code(read("sim_cells/setup.py")),
    md("""
## Sample data, warehouse names, publishing

The sample data generator, the warehouse table and column names, the publishing as of a date, and the curated
tables.
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
