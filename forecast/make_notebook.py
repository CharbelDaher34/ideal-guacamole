"""Assemble the two notebooks from the files in src/ (edit those, then run: python make_notebook.py):
build_forecast_project.ipynb (sets up the forecast project) and warehouse_simulator.ipynb (sandbox only)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
RECIPES = ["load_check", "categorise", "obligations", "build_rows", "train_models", "test_models", "ml_rows", "score",
           "visual_predictions", "compare_models", "final_forecast"]
SQL_RECIPES = []          # SQL recipes other than staging (none since the warehouse curates the data)


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
# Cash-flow forecast: set up the project

This notebook sets up the cash-flow forecasting project in this (empty) Dataiku project, the way it runs in
production. It reads the 4 curated tables the warehouse team builds from the 17 detailed ones (client, monthly
flows by origin, scheduled payments, monthly balances), through the read-only `EDW_AI` connection (table and
column names: `python/warehouse.py` in the project library), and builds:

- **Raw data**: the 4 curated warehouse tables and the load record, read-only.
- **Staging**: one SQL recipe per table that gives back the standard column names, and a load check
  that stops everything if the latest load is incomplete. The load also sets the forecast's cut-off month.
- **Data prep**: the category rules (an editable dataset), monthly flows by category, obligations, and the model
  rows.
- **Training (code)** and **Training (visual)**: the same rows trained two ways (Python recipes with
  CatBoost; visual ML, tracked by Dataiku), each tested on the last 6 months.
- **Inference**: the 3-month forecast from either (project variable `model_source`), plus the obligations.
- Two scenarios: **Monthly forecast** (after each load) and **Retrain and test the models**.

It then trains and tests the models once and builds the first forecast.

**How to use it:** create an empty project, import this notebook (Notebooks > New notebook > Import), check the
settings below, then run all cells (Cell > Run All): about 15 minutes the first time (creating the code env and
training). It only sets up an empty project: after that, work in the flow and run the scenarios.

In the sandbox, the warehouse tables come from the notebook `warehouse_simulator.ipynb` (run it first).
"""),
    code(read("cells/settings.py")),
    md("""
## 1. Code environment

The Python recipes need CatBoost. This creates a code env with it, unless it already exists. If you are not
allowed to create code envs, ask an admin, or set `CODE_ENV` above to an existing env that has CatBoost. The
visual models use Dataiku's built-in Python.
"""),
    code(read("cells/code_env.py")),
    md("""
## 2. Code

The project library gets two files: `warehouse.py` (the warehouse tables and their column names) and
`cashflow.py` (categories and their default rules, the inputs the models see, how the code models are trained).
Each recipe below becomes a Python recipe in the flow.
"""),
    code(embedded("WAREHOUSE_CODE", SRC / "warehouse.py")),
    code(embedded("LIBRARY_CODE", SRC / "library_cashflow.py")),
    code("RECIPES, SQL_RECIPES = {}, {}"),
    *[code(embedded(f'RECIPES["{r}"]', SRC / "recipes" / f"{r}.py")) for r in RECIPES],
    *[code(embedded(f'SQL_RECIPES["{r}"]', SRC / "recipes" / f"{r}.sql")) for r in SQL_RECIPES],
    md("""
## 3. Build the flow

Links the curated warehouse tables (read-only), creates the editable category rules, the staging recipes, the
load check, the data prep and the code training. The code models' folder is only rebuilt when asked by name, like saved models, so building the
forecast never retrains by accident.
"""),
    code(read("cells/create_flow.py")),
    md("""
## 4. Check the load and prepare the data

Runs the load check, the staging and the data prep. The visual models need their data to exist before they can
be trained.
"""),
    code(read("cells/run_prep.py")),
    md("""
## 5. Visual models and scenarios

Three visual ML tasks in the Lab (Lab > Visual analyses), trained on `ml_train` and tested on `ml_test`:

- **EV (visual)**: LightGBM and XGBoost, the better one (lowest RMSE) is deployed.
- **P10 (visual)** and **P90 (visual)**: LightGBM set to predict the 10th and 90th percentile (a custom
  algorithm in the Lab), scored with the pinball loss and the share of actual values below the prediction.

Each is deployed to the flow as a saved model, with an Evaluate recipe on the last 6 months (results kept in a
model evaluation store, so every retraining is tracked) and a Score recipe on the latest cut-off.

Then the two scenarios, with their monthly triggers switched off (switch them on in production):

- **Monthly forecast**: check the latest load, then stage, prepare and forecast with the current models.
- **Retrain and test the models**: retrain both ways, test them on the last 6 months, compare, forecast.
"""),
    code(read("cells/visual_ml.py")),
    md("""
## 6. First training and forecast

Trains and tests the code models, tests the visual ones, and builds the comparison and the first forecast.
"""),
    code(read("cells/run_flow.py")),
    md("""
## 7. Results

`model_comparison`: the code and visual models on the same last 6 months (both trained without them).
`wape` is the EV error as a share of the actual amounts (lower is better); `coverage_p10_p90` should be close
to 0.8. `test_results` has more detail on the code models (per month ahead, and a simple guess to beat).

`final_forecast`: per client, category and month ahead, P10, EV and P90 in SAR, with the obligations
included. `NET` is the client's net cash flow. To switch models, set the project variable `model_source` to
`code` or `visual` and run the scenario Monthly forecast.
"""),
    code(read("cells/results.py")),
])

write("warehouse_simulator.ipynb", [
    md("""
# Warehouse simulator (sandbox only)

Plays the warehouse team (Informatica) until the real tables exist. Run it in its own project
(for example `EDW_SIMULATOR`), not in the forecast project.

It generates sample data for the 17 detailed tables of the data requirements, up to `SIM_END`, and publishes them
as the warehouse would have them at the end of `AS_OF` (what had happened or was scheduled by then), then builds
the 4 curated tables from them, as the warehouse team would: client, monthly flows by origin, scheduled payments,
monthly balances. All in the warehouse's table and column names (`warehouse.py`), into the schema `edw_ai`.
Each load adds one row per table to `CFF_LOAD_RECORD`, which the forecast project checks before it runs. The
forecast project reads only the curated tables.

To simulate the next monthly load, move `AS_OF` to the next month's end, run all cells again, then run the
forecast project's scenario **Monthly forecast**.

The first time, it also creates the schema `edw_ai` and two connections on the local PostgreSQL: `EDW_AI_LOADER`
(the warehouse team's, read and write) and `EDW_AI` (the forecast project's, read-only). In the bank, these
already exist.
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
