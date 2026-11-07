"""Assemble the notebooks from the files in src/ (edit those, then run: python make_notebook.py):
build_forecast_project.ipynb (sets up the forecast project), forecast_lab.ipynb (the data scientist's lab) and
warehouse_simulator.ipynb (sandbox only)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
RECIPES = ["stage_tables", "train_and_test", "champion_challenger", "final_forecast", "archive_forecast",
           "try_forecaster"]


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
that name), the way it runs in production. Everything the project writes is a CSV file. It reads 2 warehouse tables
as they are, read-only, and creates nothing in the warehouse: the clients (`EDW.DIM_CUSTOMER`) and the daily
postings (`EDW.FCT_STATEMENT_ENTRY`), which the staging recipe sums per client, day and category in Dataiku, over
each client's whole history. In the sandbox, they are the datasets the notebook `warehouse_simulator.ipynb` wrote in
this project; in production, the warehouse's tables through the read-only database connection
`WAREHOUSE_SQL_CONNECTION` (or CSV files in the landing folder). Names, codes and the categories (a mock list until
the bank's transaction types are known): `python/warehouse.py` in the project library.

It forecasts, per client, per category and in TOTAL (all categories together), the net cash flow (money in minus
money out) of each of the next 3 months, as P10, EV and P90. It builds 5 flow zones, each with an owner:

- **Raw data** and **Staging** (shared): the 2 warehouse tables, read-only, then `clients` (the business clients)
  and `daily_flows` (money in and out per client, day and category): what every model gets.
- **Training** (data scientist): each training run is one model with its own **model ID**, trained with the
  **forecaster** the project variable `forecaster` names, tested on the last 6 months, tracked in Dataiku's
  **Experiment Tracking** and deployed as a new **version** of the saved model `cash_forecast`. A champion /
  challenger test activates the new version if it passes the **promotion rules** (project variables `promotion_*`).
- **Inference** (inference owner): `final_forecast`, the production model's forecast for every active client (every
  row says which model and which load produced it), the forecast archive, and the inference log (one row per run).
- **Lab** (data scientist): `try_forecaster` tries the forecaster the project variable `lab_forecaster` names
  against the production model and adds the result to `lab_results`, without touching production. The notebook
  `forecast_lab.ipynb` does the same interactively.

The zones only meet through three interfaces: **1. the data** (`daily_flows` and `clients`: staging → training, lab
and inference), **2. the forecaster** (`models.py`: lab → training), **3. the active version of the saved model**
(training → inference). Until the data scientist's models are ready, the forecaster is a placeholder,
`recent_average` (the average of the last 3 months, no machine learning). Two independent scenarios run it:
**Nightly batch** (inference) and **Retrain, test and promote** (training). Each starts by setting the cut-off month
(`set_cutoff`: the month before today, or the project variable `cutoff_override`); the staging recipe stops if the
postings do not reach it.

It then trains, tests and promotes a first model, and runs the first nightly batch.

**How to use it:** in the project, import this notebook and `warehouse_simulator.ipynb` (Notebooks > New notebook
> Import). In the sandbox, run the simulator first. Then check the settings below and run all cells (Cell > Run All):
a few minutes with the small sample data. It can be run again in the same project: what exists is kept (data,
saved model versions, zones, scenario history), and the recipes' code is updated.
"""),
    code(read("cells/settings.py")),
    md("""
## 1. Code environment

The recipes need scikit-learn, and MLflow for Dataiku's Experiment Tracking, in an existing code env (`CODE_ENV`
above). This notebook does not create it: an admin creates it once. This checks it is there with those packages,
and otherwise stops with the request to send the admin.
"""),
    code(read("cells/code_env.py")),
    md("""
## 2. Code

The project library gets these files, by owner:

- shared: `warehouse.py` (the warehouse tables and columns read, codes, categories, daily sums, cut-off month) and
  `cashflow.py` (dates, the data interface, days to months, the saved model name)
- training and lab: `models.py` (the forecaster interface, the placeholder forecaster, the test and its scores) and
  `tracking.py` (model IDs, Experiment Tracking, saved model versions, promotion rules, champion / challenger)
- inference: `forecast.py` (the production model, the final forecast table)
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
CSV datasets, the folders `experiments` (tracked models), `forecast_archive` and `lab_experiments` (lab tries), the
saved model `cash_forecast`, and the project variables `forecaster` (`recent_average`), `lab_forecaster`
(`recent_average_6m`) and the promotion rules (`promotion_*`) if they are not set yet, in 5 zones. Training (folder
`experiments`), the promotion log and the lab are only rebuilt when asked by name, so the nightly batch never
retrains by accident.
"""),
    code(read("cells/create_flow.py")),
    md("""
## 4. First run

Builds the daily flows, trains, tests and deploys a model, runs champion / challenger (with no production model yet,
the first model is promoted, with a warning in promotion_log if its checks fail), then runs the nightly batch:
forecast, archive, log.
"""),
    code(read("cells/first_run.py")),
    md("""
## 5. Scenarios

- **Nightly batch**: set the cut-off, build the daily flows, forecast with the production model, archive it and
  log the run. It never trains.
- **Retrain, test and promote**: set the cut-off, build the daily flows, train, test and deploy a new model with
  the forecaster the project variable `forecaster` names, champion / challenger with the promotion rules, activate
  it if it passes. It does not forecast: the next Nightly batch uses the new active version.

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

write("forecast_lab.ipynb", [
    md("""
# Digital Cash Management Forecast: lab (data scientist)

Explore the data and try models here without touching production: nothing in this notebook changes the flow, the
saved models or the production model.

1. **EDA**: the daily flows per category, totals over time, seasonality, and one client day by day.
2. **Models to try**: several forecasters trained and tested side by side on the same test as champion / challenger
   (trained without the last 6 months before the cut-off, tested on them), next to the production model and the
   simple guess (the same as the month before), with the decision champion / challenger would take for each.
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

The data every model gets (interface 1): `daily_flows`, one row per client, day and category with money in and
money out, and `clients`. The models predict, per client, per category and in TOTAL (all categories together), the
net cash flow (money in minus money out) of each of the next 3 months. First, per category and per segment
(**negative_months**: the share of months where more money went out than came in).
"""),
    code(read("lab_cells/eda_overview.py")),
    md("""
Net cash flow per category over time, and seasonality. Ramadan and Eid move about 11 days earlier each year, so a
calendar-month profile blurs them.
"""),
    code(read("lab_cells/eda_charts.py")),
    md("""
One client day by day (what the models see) and its monthly totals (what they predict).
"""),
    code(read("lab_cells/eda_target.py")),
    md("""
## 2. Models to try

A forecaster is a class with `name`, `params` (its settings), `fit(daily, clients, cutoff)` and
`predict(daily, clients, cutoff)`, which returns one row per client, category (and TOTAL) and months ahead with `p10`,
`ev` and `p90` in SAR (details in `models.py`, project library). `cashflow.monthly_matrix(daily, cutoff)` turns the
days into months. Compared here:

- `recent_average`: the placeholder in production (the average of the last 3 months, no machine learning)
- `recent_average_6m`: the same over 6 months (an example in `forecasters.py`)
- `lag_boosting`: an example of a small model that learns: scikit-learn gradient boosting on the last 12 months

Change them or add your own, then run the comparison. scikit-learn, pandas and matplotlib are in the code env; other
packages need the admin.
"""),
    code(read("lab_cells/forecasters_to_try.py")),
    code(read("lab_cells/compare.py")),
    code(read("lab_cells/compare_chart.py")),
    md("""
## 3. Put one in production

1. Copy its class into `forecasters.py` (project library) with the imports it needs (for example
   `import pandas as pd`, `from cashflow import FORECAST_COLUMNS, HORIZONS, TOTAL, monthly_matrix`) and add it to
   `FORECASTERS`. The setup notebook never overwrites that file.
2. Set the project variable `forecaster` to its name.
3. Run the scenario **Retrain, test and promote**. Champion / challenger promotes it only if it passes the
   promotion rules (project variables `promotion_*`) against the production model on the same 6 months; the next
   Nightly batch forecasts with it.
4. To stop training it, set `forecaster` back to `recent_average`. To use an earlier model now, activate its
   version in the saved model `cash_forecast`.
"""),
], kernel={"display_name": "Python (env cashflow_forecast)", "language": "python",    # the code env's own kernel
           "name": "py-dku-venv-cashflow_forecast"})

write("warehouse_simulator.ipynb", [
    md("""
# Digital Cash Management Forecast: warehouse simulator (sandbox only)

Plays the warehouse until the project is connected to the real one. Run it in the same project as the forecast,
before `build_forecast_project.ipynb`.

It generates sample data up to `SIM_END` and writes, as the warehouse would have them at the end of `AS_OF`, the 2
tables the forecast reads, shaped like the bank's (`warehouse.py`): `DIM_CUSTOMER` (the business clients, and a few
retail customers the forecast must leave out) and `FCT_STATEMENT_ENTRY` (the daily postings). They are CSV datasets
of this project. It also sets the project variable `cutoff_override` to the month of `AS_OF`.

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
