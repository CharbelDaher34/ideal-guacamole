# Cash-flow forecast

A Dataiku project that forecasts each business client's money in and out for the next 3 months, by category
(P10, EV, P90), set up the way it runs in production. How it works, in plain words:
[`../docs/cashflow_forecast_design.html`](../docs/cashflow_forecast_design.html).

Two notebooks:

| Notebook | Project | Role |
|---|---|---|
| [`warehouse_simulator.ipynb`](warehouse_simulator.ipynb) | `EDW_SIMULATOR` | **Sandbox only.** Plays the warehouse team: writes the 17 detailed tables (sample data), the 4 curated tables built from them, and the load record into the schema `edw_ai`. Replaced by Informatica in the bank. |
| [`build_forecast_project.ipynb`](build_forecast_project.ipynb) | `CASHFLOW_FORECAST` | Sets up the forecast project once, in an empty project: flow, models, scenarios, first forecast. |

## Run it

1. **Warehouse (sandbox):** create a project `EDW_SIMULATOR`, import `warehouse_simulator.ipynb`
   (Notebooks → New notebook → Import), **Run All**. The first time it also creates the schema `edw_ai` and two
   connections: `EDW_AI_LOADER` (the warehouse team's, read and write) and `EDW_AI` (read-only).
2. **Forecast project:** create an empty project `CASHFLOW_FORECAST`, import `build_forecast_project.ipynb`, check
   the settings in its first cell, **Run All** (about 15 minutes the first time). It refuses to run in a project
   that already has a flow.
3. **Every month after a load:** Scenarios → **Monthly forecast** → Run. To retrain: **Retrain and test the models**.

To simulate the next monthly load: in `EDW_SIMULATOR`, set `AS_OF` to the next month's end (for example
`2026-11-30`), Run All, then run the scenario **Monthly forecast** in `CASHFLOW_FORECAST`: the cut-off moves to
the new month.

In the bank: no simulator. The warehouse team builds the curated tables (and keeps the detailed ones) with the
names in [`src/warehouse.py`](src/warehouse.py) (our proposal until they confirm; if they change, only that
file changes) and the load record; set `EDW_CONNECTION` and `EDW_SCHEMA` in the first cell.

## What the project builds

| Zone | Recipes | Outputs |
|---|---|---|
| Raw data | (none: read-only, `EDW_AI`) | the 4 curated warehouse tables: `CFF_CLIENT`, `CFF_CLIENT_MNTHLY_FLOW` (client × month × direction × origin type × tag), `CFF_SCHEDULED_PYMT` (every scheduled payment and when it became known), `CFF_CLIENT_MNTHLY_BAL`; and `CFF_LOAD_RECORD` |
| Staging | `stage_*` (SQL, one per table) | the same tables with the standard column names (`clients`, `monthly_flows_by_origin`, `scheduled_payments`, `monthly_balances`); flags Y / N → true / false |
| Data prep | (editable dataset) | `category_rules`: origin type + transaction type tag + direction → category, first match wins. Edit it in the flow, then run Monthly forecast |
| Data prep | `categorise` | `monthly_flows`: money in / out per client, month and category |
| Data prep | `obligations` | `monthly_obligations`: scheduled payments per client, category, month due and month known |
| Data prep | `build_rows` | `training_rows` (every past cut-off with its answer), `scoring_rows` (latest cut-off) |
| Training (code) | `train_models`, `test_models` | `models` folder (CatBoost EV and P10/P90; rebuilt only when built by name), `test_predictions`, `test_results` |
| Training (visual) | `ml_rows`, Lab analyses, `train_*_visual`, `evaluate_*_visual` | `ml_train`, `ml_test`, `ml_scoring`; saved models `ev_visual`, `p10_visual`, `p90_visual`; an evaluation store per model |
| Inference | `score`, `score_*_visual`, `visual_predictions`, `compare_models`, `final_forecast` | `predictions`, `predictions_visual`, `model_comparison`, `final_forecast` |

Project variable `model_source` (`code` or `visual`) picks which models feed `final_forecast`. The cut-off comes
from the load record, not from today's date: each scenario starts with the **load check**
(`check_latest_load` in the library file `warehouse.py`). For the latest load in `CFF_LOAD_RECORD`, every
curated table must be loaded with status SUCCESS and hold the number of rows the record says; otherwise the
scenario stops before anything is built. On success it sets the project variables `load_id`, `as_of_date` and
`cutoff_month` (the last complete month), which the recipes read. Building datasets by hand uses the cut-off of
the last successful check.

## Scenarios

| Scenario | Steps | Trigger (off in the sandbox) |
|---|---|---|
| Monthly forecast | check the load (Python step) → stage, prepare, forecast with the current models | monthly, 2nd at 07:00 Riyadh |
| Retrain and test the models | check the load (Python step) → prepare training data → retrain and test the code models → retrain and test the visual models → compare → forecast | every 3 months, 5th at 07:00 Riyadh |

The code models' folder, the visual saved models and the evaluation stores are only rebuilt when built by name,
so the monthly forecast never retrains by accident. The Free Edition only runs scenarios from the browser.

## Change the code

The notebooks are assembled from `src/`. Edit those, then run `python make_notebook.py`:

- `src/warehouse.py`: warehouse table and column names (curated and detailed), and the staging SQL built from them
- `src/library_cashflow.py`: the project library `cashflow.py` (categories and default category rules, model
  inputs, training)
- `src/recipes/*.py`: one file per recipe
- `src/cells/*.py`: the setup notebook's own cells
- `src/generate_data.py`, `src/simulator.py`, `src/sim_cells/*.py`: the warehouse simulator (sample data,
  publishing as of a date, curated tables)
