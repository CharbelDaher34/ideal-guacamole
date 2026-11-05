# Digital Cash Management Forecast

A Dataiku project that forecasts each business client's money in and out for the next 3 months, by category
(P10, EV, P90), set up the way it runs in production, in a single project. All data is stored as CSV files. How it
works, in plain words: [`../docs/cashflow_forecast_design.html`](../docs/cashflow_forecast_design.html).

Two notebooks, both in the same project:

| Notebook | Role |
|---|---|
| [`warehouse_simulator.ipynb`](warehouse_simulator.ipynb) | **Sandbox only.** Plays the warehouse team: writes the 4 curated warehouse tables (sample data) and the load record as CSV datasets of the project. Replaced by Informatica in the bank. |
| [`build_forecast_project.ipynb`](build_forecast_project.ipynb) | Sets up the project once (and names it Digital Cash Management Forecast): flow, first model, scenarios, first nightly batch. |

## Run it

1. In the project, import both notebooks (Notebooks → New notebook → Import).
2. **Sandbox:** run `warehouse_simulator.ipynb` (Run All, about 1 minute). It writes `CFF_CLIENT`,
   `CFF_CLIENT_MNTHLY_FLOW`, `CFF_SCHEDULED_PYMT`, `CFF_CLIENT_MNTHLY_BAL` and `CFF_LOAD_RECORD` as of `AS_OF`,
   for `N_CLIENTS` sample clients (40: a quick run; scores need more, for example 300).
3. Run `build_forecast_project.ipynb` (check the settings in its first cell, Run All, a few minutes).
   It needs an existing Python code env (`CODE_ENV`) with `catboost==1.2.7`, `scikit-learn` and `mlflow`: 2.22.0
   (Python 3.9 or later) or 3.10.1 (Python 3.10 or later), the versions Dataiku 14 and 15 tested for MLflow models.
   Tested on DSS 14.2.0 with Python 3.9 and mlflow 2.22.0:
   it does not create one (that takes admin rights). It checks the env and, if something is missing, stops with
   the request to send a Dataiku admin.
   It uses the simulated tables if they are there, otherwise it links the warehouse team's landing files
   (`LANDING_CONNECTION`, read-only). It needs no admin right on the project (without one, it cannot rename the
   project and says so). It can be run again in the same project: what exists is kept (data, the category
   mapping, saved model versions, zones, scenario history) and the recipes' code is updated.
4. **Every night after a load:** Scenarios → **Nightly batch** → Run. To retrain: **Retrain, test and promote**.

To simulate the next load: set `AS_OF` in the simulator to the next month's end (for example `2026-11-30`), Run
All, then run **Nightly batch**: the cut-off moves to the new month.

## What the project builds

| Zone | Recipes | Outputs (CSV files unless noted) |
|---|---|---|
| Raw data (Dataiku's Default zone, renamed) | (simulator, or read-only landing files) | `CFF_CLIENT`, `CFF_CLIENT_MNTHLY_FLOW`, `CFF_SCHEDULED_PYMT`, `CFF_CLIENT_MNTHLY_BAL`, `CFF_LOAD_RECORD` |
| Staging | `stage_tables` | `clients`, `monthly_flows_by_origin`, `scheduled_payments`, `monthly_balances`: standard column names, flags Y / N → true / false |
| Data prep | (CSV dataset) | `category_mapping`: origin type + transaction type tag + direction → category, one row per combination |
| Data prep (shared) | `join_flows_category` (Join), `group_monthly_flows` (Group), `join_scheduled_category` (Join), `obligations` | `flows_with_category`, `monthly_flows`, `scheduled_with_category`, `monthly_obligations` |
| Training | `build_training_rows`, `train_and_test`, `champion_challenger` | `training_rows`, folder `experiments` (Experiment Tracking), `test_results`, `model_reference` (a sample of the model inputs, gives each version its input columns), `promotion_log` (appended) |
| Inference | `build_scoring_rows`, `score_ev`, `score_p10`, `score_p90` (Score), `final_forecast`, `archive_forecast` | `scoring_rows` (the latest month), saved models `forecast_ev`, `forecast_p10`, `forecast_p90` (one version per model ID, next to the Score recipes that use them), `scored_*`, `final_forecast` (with `model_id`, `model_promoted_at`, `load_id`), folder `forecast_archive`, `inference_log` (appended) |
| Try the model | `try_the_model` | folder `try_model_uploads` (your CSV files), `try_model_forecast`, folder `try_model_charts` (the chart); dashboard **Try the model** |

The zones run left to right in one row: Raw data → Staging → Data prep → Training → Inference → Try the model.
Training and inference share the data prep, then split: `build_training_rows` and `build_scoring_rows` use the
same code (`model_rows` in `cashflow.py`), so the model sees the same inputs in training and in the forecast. The
only link between the two paths is the saved models: training adds versions, inference reads the active one.

Categories: anything not in `category_mapping` becomes other in / other out (a formula column in the Join). To
change a category, change `category_mapping` from a notebook (read it, change rows, write it back) and run Nightly
batch. Running the setup again keeps it.

## Models: tracking, versions, promotion

- **Tracking:** every run of `train_and_test` is one model with a model ID like `CFF_202610_20261104T1329`
  (cut-off month, training time), recorded in **Experiment Tracking** (Lab → Experiment tracking): settings,
  cut-off, load, row counts, test scores on the last 6 months (EV error overall and per category, P10–P90
  coverage), the backtest models, and three MLflow models (EV, P10, P90). The experiment is `cashflow_forecast`;
  if that one points to a deleted `experiments` folder, training uses `cashflow_forecast_<folder id>` instead.
- **Versions:** those three are deployed as a new version, named by the model ID, of the saved models
  `forecast_ev`, `forecast_p10` and `forecast_p90` (not activated yet).
- **Promotion (automatic if better):** `champion_challenger` tests the new model and the production model on the
  same last 6 months (each with its version trained without those months). With no production model to compare
  with (none yet, or its tracked files were deleted), the new model is promoted (the forecast needs one), with a
  warning in `promotion_log` if it fails the checks below.
  Otherwise it is promoted if its EV error is lower (a tie keeps production), its P10–P90 coverage is between 70%
  and 90%, and it beats the simple guess. The step "Promote if better" activates its version in the three saved
  models.
- **Nightly batch:** the Score recipes always use the **active version**; `final_forecast` checks the three saved
  models have the same one and writes it on every row. To roll back, activate an earlier version in the three
  saved models.

## Inference log

`inference_log` gets one row per forecast run: `run_at`, `load_id`, `as_of_date`, `cutoff_month`, `model_id`,
`model_promoted_at`, `clients`, `clients_not_enough_history`, `forecast_rows`, `archive_file` (the run's file in
`forecast_archive`). Run history and failures are in the scenarios' Last runs; in production, add a reporter
(email, Slack) to the scenarios.

## Try the model

Everything is on the dashboard **Try the model** (Dashboards menu):

1. Upload CSV files to the folder `try_model_uploads` (Flow → zone Try the model → `try_model_uploads` → Upload).
2. Click **Run** on the dashboard (it runs the scenario **Try the model**, about 30 seconds), then refresh the page.
3. Read the chart (each client's uploaded net cash flow, then the forecast EV with its P10–P90 band) and the table
   `try_model_forecast` (P10, EV, P90 per client, category and month ahead).

The forecast is made the same way as the nightly forecast, with the production model; nothing else changes.

One row per client, month and category, amounts in SAR (positive): `client_no`, `month` (`2026-10` or
`2026-10-31`), `category` (PAYROLL, RENT, FINANCING, SADAD_GOV, SUPPLIERS, CORP_CARD, OTHER_OUT, RECEIPTS,
CARD_SALES, CHEQUES_IN, DEPOSIT_MAT, OTHER_IN), `amount`. Optional: `segment`, `sector`, `region`,
`last_annual_sales` (else the last 12 months of money in), `number_of_employees`, `balance_end`. A client needs 12
months up to the latest month in the files (the cut-off). Scheduled payments are not uploaded: every amount counts
as not scheduled. The setup writes `example_clients.csv` (two clients of the data, renamed) to show the format;
delete it when you upload your own.

## Scenarios

| Scenario | Steps | Trigger (off in the sandbox) |
|---|---|---|
| Nightly batch | check the load → build the scoring rows → forecast with the production model, archive it and log the run (never touches the training data) | daily, 06:00 Riyadh |
| Retrain, test and promote | check the load → prepare the training data → train, test and deploy a new model → champion / challenger → promote if better → forecast with the production model, archive and log | monthly, 5th at 07:00 Riyadh |
| Try the model | forecast the clients uploaded to `try_model_uploads` | none: run by hand |

The first two start with the load check (`check_latest_load` in `warehouse.py`): for the latest load in
`CFF_LOAD_RECORD`, every curated table must be loaded with status SUCCESS and hold the number of rows the record
says; otherwise the scenario stops. On success it sets `load_id`, `as_of_date` and `cutoff_month`. Training
(folder `experiments`), the promotion log and the saved models are only rebuilt when built by name, so the nightly
batch never retrains by accident. The Free Edition only runs scenarios from the browser and has no scheduled
triggers.

## Change the code

The notebooks are assembled from `src/`. Edit those, then run `python make_notebook.py`:

- `src/warehouse.py`: warehouse table and column names, staging, the load check
- `src/library_cashflow.py`: the project library `cashflow.py` (categories and the default mapping, the model rows
  shared by training, inference and Try the model, the forecast table, training and test scores)
- `src/tracking.py`: model IDs, Experiment Tracking, saved model versions, promotion
- `src/recipes/*.py`: one file per Python recipe; the visual recipes are created in `src/cells/create_flow.py`
- `src/cells/*.py`: the setup notebook's own cells
- `src/generate_data.py`, `src/simulator.py`, `src/sim_cells/*.py`: the warehouse simulator
