# Digital Cash Management Forecast

A Dataiku project that forecasts each business client's money in and out for the next 3 months, by category
(P10, EV, P90), set up the way it runs in production, in a single project. All data is stored as CSV files. How it
works, in plain words: [`../docs/cashflow_forecast_design.html`](../docs/cashflow_forecast_design.html).

Three notebooks, all in the same project:

| Notebook | Role |
|---|---|
| [`warehouse_simulator.ipynb`](warehouse_simulator.ipynb) | **Sandbox only.** Plays the warehouse team: writes the 4 curated warehouse tables (sample data) and the load record as CSV datasets of the project. Replaced by Informatica in the bank. |
| [`build_forecast_project.ipynb`](build_forecast_project.ipynb) | Sets up the project once (and names it Digital Cash Management Forecast): flow, first model, scenarios, first nightly batch. |
| [`forecast_lab.ipynb`](forecast_lab.ipynb) | **The data scientist's lab.** EDA of the training rows, then several forecasters side by side against the production model on the same test; each try is recorded in Experiment Tracking; never touches production. Opens in the code env's kernel. |

## Run it

1. In the project, import the notebooks (Notebooks → New notebook → Import).
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
| Training (data scientist) | `build_training_rows`, `train_and_test`, `champion_challenger` | `training_rows`, folder `experiments` (Experiment Tracking), `test_results`, `model_reference` (a sample of the forecaster's inputs, gives each version its input columns), `promotion_log` (appended, with each model's forecaster) |
| Inference (inference owner) | `build_scoring_rows`, `score_ev`, `score_p10`, `score_p90` (Score), `final_forecast`, `archive_forecast` | `scoring_rows` (the latest month), saved models `forecast_ev`, `forecast_p10`, `forecast_p90` (one version per model ID, next to the Score recipes that use them), `scored_*`, `final_forecast` (with `model_id`, `load_id`), folder `forecast_archive`, `inference_log` (appended) |
| Lab (data scientist) | `try_forecaster` | folder `lab_experiments` (Experiment Tracking, experiment `cashflow_lab`), `lab_results` (appended: one block of rows per try) |

Each zone has an owner, and the zones only meet through three interfaces:

1. **The model rows** (data prep → training and inference): `build_training_rows` and `build_scoring_rows` both
   use `model_rows` in `cashflow.py`, so a model sees the same inputs in training and in the forecast. The columns
   are `MODEL_ROW_COLUMNS` (training rows also have `ANSWER`). Add a column freely; rename or remove one only when
   training and inference both agree.
2. **The forecaster** (lab → training): a class with `name`, `params`, `inputs`, `fit(rows)` and `predict(rows)`
   (P10, EV, P90 in SAR), described in `models.py`. The project variable `forecaster` names the one Retrain trains.
3. **The active version of the saved models** (training → inference): the Score recipes use it and
   `final_forecast` writes its ID on every row. Inference reads nothing else from training.

Categories: anything not in `category_mapping` becomes other in / other out (a formula column in the Join). To
change a category, change `category_mapping` from a notebook (read it, change rows, write it back) and run Nightly
batch. Running the setup again keeps it.

## Models: tracking, versions, promotion

- **Forecaster:** `train_and_test` trains the forecaster the project variable `forecaster` names: `catboost` (the
  reference, in `models.py`: one CatBoost model for the EV, one for P10 and P90) or one of the data scientist's in
  `forecasters.py`.
- **Tracking:** every run of `train_and_test` is one model with a model ID like `CFF_202610_20261104T1329`
  (cut-off month, training time), recorded in **Experiment Tracking** (Lab → Experiment tracking): the forecaster
  and its settings, cut-off, load, row counts, test scores on the last 6 months (EV error overall and per
  category, P10–P90 coverage), the backtest forecaster, and three MLflow models (EV, P10, P90). The experiment is `cashflow_forecast`;
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

## Trying other models (lab)

The data scientist tries forecasters without touching production, on the same test as champion / challenger (the
last 6 months, each forecaster trained without them), next to the production model and the simple guess:

- **Notebook `forecast_lab.ipynb`** (it opens in the kernel of the code env `cashflow_forecast`, which needs
  Jupyter support: an option of the code env, set by whoever manages it):
  1. **EDA** of the training rows: per category the share of months with no movement, the share already scheduled
     at the cut-off and the volatility; monthly totals in and out; seasonality without the trend; the spread of what
     the models learn and how much each input moves with it.
  2. **Models to try**: several forecasters side by side with `lab_compare` (examples: the simple guess with a
     band, CatBoost with deeper trees, scikit-learn gradient boosting), next to the production model: EV error and
     P10–P90 coverage overall, per category and per months ahead, and the decision champion / challenger would
     take for each. `catboost`, the production recipe trained again, matches the production model's scores: a
     check that the test is the same.
- **Without a notebook:** set the project variable `lab_forecaster` to a forecaster's name and build `lab_results`
  (Lab zone, only built when asked by name). Each try adds its rows to `lab_results`.

Every try is recorded in Experiment Tracking, experiment `cashflow_lab` (folder `lab_experiments`); lab tries are
never deployed. To put a forecaster in production: add its class to `forecasters.py` and to `FORECASTERS` (the
setup notebook creates that file once and never overwrites it), set the project variable `forecaster` to its name,
and run **Retrain, test and promote**.

## Inference log

`inference_log` gets one row per forecast run: `run_at`, `load_id`, `as_of_date`, `cutoff_month`, `model_id`
(when it was promoted is in `promotion_log`), `clients`, `clients_not_enough_history`, `forecast_rows`,
`archive_file` (the run's file in `forecast_archive`). Run history and failures are in the scenarios' Last runs; in production, add a reporter
(email, Slack) to the scenarios.

## Scenarios

| Scenario | Steps | Trigger (off in the sandbox) |
|---|---|---|
| Nightly batch | check the load → build the scoring rows → forecast with the production model, archive it and log the run (never touches the training data) | daily, 06:00 Riyadh |
| Retrain, test and promote | check the load → prepare the training data → train, test and deploy a new model (the forecaster `forecaster` names) → champion / challenger → promote if better. It does not forecast: the next Nightly batch uses the new active version | monthly, 5th at 07:00 Riyadh |

The two scenarios are independent. Each starts with the load check (`check_latest_load` in `warehouse.py`): for the latest load in
`CFF_LOAD_RECORD`, every curated table must be loaded with status SUCCESS and hold the number of rows the record
says; otherwise the scenario stops. On success it sets `load_id`, `as_of_date` and `cutoff_month`. Training
(folder `experiments`), the promotion log and the saved models are only rebuilt when built by name, so the nightly
batch never retrains by accident. The Free Edition only runs scenarios from the browser and has no scheduled
triggers.

## Change the code

The notebooks are assembled from `src/`. Edit those, then run `python make_notebook.py`:

- `src/warehouse.py`: warehouse table and column names, staging, the load check
- `src/library_cashflow.py`: the project library `cashflow.py`, shared (categories and the default mapping, the
  model rows, the saved model names, P10 ≤ EV ≤ P90)
- `src/models.py`: training and lab (the forecaster interface, the reference CatBoost forecaster, the test and its
  scores, `lab_try` and `lab_compare`)
- `src/tracking.py`: training and lab (model IDs, Experiment Tracking, saved model versions, champion / challenger,
  promotion)
- `src/forecast.py`: inference (the production model's ID, the final forecast table)
- `src/forecasters.py`: the first version of the data scientist's `forecasters.py` (written once in the project)
- `src/recipes/*.py`: one file per Python recipe; the visual recipes are created in `src/cells/create_flow.py`
- `src/cells/*.py`: the setup notebook's own cells; `src/lab_cells/*.py`: the lab notebook's
- `src/generate_data.py`, `src/simulator.py`, `src/sim_cells/*.py`: the warehouse simulator
