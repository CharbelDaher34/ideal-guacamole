"""Assemble the two notebooks from the files in src/ (edit those, then run: python make_notebook.py):
build_forecast_project.ipynb (sets up the forecast project) and warehouse_simulator.ipynb (sandbox only)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
RECIPES = ["stage_tables", "obligations", "build_training_rows", "train_and_test", "champion_challenger",
           "build_scoring_rows", "final_forecast", "archive_forecast", "try_the_model"]


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


def write(name, cells):
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
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
project library). It builds:

- **Raw data**: the 4 curated tables and the load record, read-only.
- **Staging**: the same tables with the standard column names.
- **Data prep** (shared by training and inference): the category mapping (a CSV dataset) applied by visual Join
  and Group recipes, the obligations.
- **Training**: the training rows, then each training run is one model with its own **model ID**, tracked in Dataiku's
  **Experiment Tracking** and deployed as a new **version** of the saved models `forecast_ev`, `forecast_p10`,
  `forecast_p90`. A champion / challenger test activates the new version if it beats the production model.
- **Inference**: the scoring rows (latest month), visual Score recipes with the active version, the 3-month
  forecast (every row says which model
  and which load produced it), the forecast archive, and the inference log (one row per run: when, which model,
  which load, how many clients).
- **Try the model**: upload CSV files of clients (real or made up) to the folder `try_model_uploads` and get the
  production model's forecast for them in `try_model_forecast`, with a chart in `try_model_charts`, all on the
  dashboard **Try the model**. Changes nothing else.
- Three scenarios: **Nightly batch** and **Retrain, test and promote** each start with the load check
  (`check_latest_load`): it stops the scenario if the latest load is incomplete, and sets `cutoff_month`.
  **Try the model** is run by hand after uploading files.

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

The project library gets three files: `warehouse.py` (warehouse table and column names, the load check),
`cashflow.py` (categories and the default category mapping, the inputs the models see, training and test scores)
and `tracking.py` (model IDs, Experiment Tracking, saved model versions, promotion). Each recipe
below becomes a Python recipe in the flow; the visual recipes are created in the next step.
"""),
    code(embedded("WAREHOUSE_CODE", SRC / "warehouse.py")),
    code(embedded("LIBRARY_CODE", SRC / "library_cashflow.py")),
    code(embedded("TRACKING_CODE", SRC / "tracking.py")),
    code("RECIPES = {}"),
    *[code(embedded(f'RECIPES["{r}"]', SRC / "recipes" / f"{r}.py")) for r in RECIPES],
    md("""
## 3. Build the flow

Uses the simulated warehouse tables, or links the landing files (read-only), and creates the category
mapping, the recipes (Python, Join, Group, Score) and their CSV datasets, the folders `experiments` (tracked
models), `forecast_archive`, `try_model_uploads` and `try_model_charts`, and the three saved models, in 6 zones. Training (folder
`experiments`), the promotion log and the saved models are only rebuilt when asked by name, so the nightly batch
never retrains by accident.
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
## 5. Try the model

Upload CSV files to the folder `try_model_uploads` (Flow > Try the model > `try_model_uploads` > Upload), then run
the scenario **Try the model** (the dashboard **Try the model** has a button for it, the chart and the table). One row per client, month and category, amounts in
SAR (positive):

| client_no | month | category | amount |
|---|---|---|---|
| TEST_1 | 2025-11 | RECEIPTS | 250000 |
| TEST_1 | 2025-11 | PAYROLL | 90000 |

Categories: PAYROLL, RENT, FINANCING, SADAD_GOV, SUPPLIERS, CORP_CARD, OTHER_OUT, RECEIPTS, CARD_SALES, CHEQUES_IN,
DEPOSIT_MAT, OTHER_IN. Optional columns: segment, sector, region, last_annual_sales, number_of_employees,
balance_end. A client needs 12 months up to the latest month in the files (the cut-off). Scheduled payments are not
uploaded: every amount counts as not scheduled. The first time, this writes `example_clients.csv` (two clients of
the data, renamed) to show the format; delete it when you upload your own.
"""),
    code(read("cells/try_model.py")),
    md("""
## 6. Scenarios

- **Nightly batch**: check the load, build the scoring rows, forecast with the production model, archive it and
  log the run. It does not touch the training data.
- **Retrain, test and promote**: check the load, prepare the training data, train, test and deploy a new model,
  champion / challenger, activate it if better, then forecast with the production model.
- **Try the model**: forecast the uploaded clients (no trigger, run by hand).

Their triggers (daily at 06:00 and monthly on the 5th, Riyadh time) are created switched off; switch them on
in production.
"""),
    code(read("cells/scenarios.py")),
    md("""
## 7. Dashboard Try the model

A dashboard (Dashboards menu) to test the model on your own clients: how to upload, the uploaded files, a button
that runs the scenario **Try the model**, the chart (each client's net cash flow and its forecast) and the forecast
table. Created once; when the setup runs again it is kept as it is.
"""),
    code(read("cells/dashboard.py")),
    md("""
## 8. Results

To use another model (for example to roll back), activate its version in the three saved models
(`forecast_ev`, `forecast_p10`, `forecast_p90`) and run the Nightly batch.
"""),
    code(read("cells/results.py")),
])

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
