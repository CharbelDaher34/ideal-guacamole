# Digital Cash Management Forecast

A Dataiku project that forecasts, for each business client, the net cash flow (money in minus money out) of each of
the next 3 months, **per category and in TOTAL**, as one value with a range (P10, EV, P90). It is set up the way it
runs in production, in a single project, and stores all its data as CSV files. For a plain-words explanation of how
it works, see [`../docs/cashflow_forecast_design.html`](../docs/cashflow_forecast_design.html).

**Who owns what:**
- **This project** covers everything around the model: getting the data into a format the data scientist can work
  with, the inference pipeline, and the promotion pipeline with its rules.
- **The model is a placeholder**, `recent_average` (no machine learning), until the data scientist's models replace
  it through the forecaster interface.

There are three notebooks, all in the same project:

| Notebook | Role |
|---|---|
| [`warehouse_simulator.ipynb`](warehouse_simulator.ipynb) | **Sandbox only.** Plays the warehouse: writes sample `DIM_CUSTOMER` and `FCT_STATEMENT_ENTRY` tables as CSV datasets of the project. |
| [`build_forecast_project.ipynb`](build_forecast_project.ipynb) | Sets up the project (and names it Digital Cash Management Forecast): the flow, the first model, the scenarios and the first nightly batch. |
| [`forecast_lab.ipynb`](forecast_lab.ipynb) | **The data scientist's lab.** An EDA of the daily flows, then several forecasters compared side by side with the production model on the same test. Each try is recorded in Experiment Tracking, and nothing here touches production. It opens in the code env's kernel. |

## Run it

1. In the project, import the notebooks (Notebooks → New notebook → Import).
2. **Sandbox only:** run `warehouse_simulator.ipynb` (Run All, about 1 minute).
   - It writes sample `DIM_CUSTOMER` and `FCT_STATEMENT_ENTRY` tables as of `AS_OF`, shaped like the bank's.
   - They cover `N_CLIENTS` sample business clients (40 by default), plus a few retail customers that the forecast
     leaves out.
3. Run `build_forecast_project.ipynb`: check the settings in its first cell, then Run All (a few minutes).
   - **Code env:** it needs an existing Python code env (`CODE_ENV`) with `scikit-learn` and `mlflow`, either
     2.22.0 (Python 3.9 or later) or 3.10.1 (Python 3.10 or later). It was tested on DSS 14.2.0 with Python 3.9 and
     mlflow 2.22.0. The notebook doesn't create the env, because that needs admin rights. If something is missing,
     it stops and shows the request to send a Dataiku admin.
   - **Data:** it uses the simulated tables if they're there; otherwise it reads the warehouse (see below).
   - **Rights:** it needs no admin right on the project. Without one, it can't rename the project and says so.
   - **Running it again in the same project** keeps what already exists (data, saved model versions, zones,
     scenario history) and updates the recipes' code.
4. **Every night after a load:** Scenarios → **Nightly batch** → Run. To retrain: **Retrain, test and promote**.

### In production: connect to the warehouse

Nothing is created or run in the warehouse. The project only reads `EDW.DIM_CUSTOMER` and `EDW.FCT_STATEMENT_ENTRY`
as they are.

1. **Ask a Dataiku admin once** for a read-only database connection to the warehouse (Oracle), usable by your
   group, for example named `EDW`.
2. **In the setup notebook's first cell**, set:
   - `WAREHOUSE_SQL_CONNECTION = "EDW"` (the connection's name);
   - `WAREHOUSE_SQL_SCHEMA = "EDW"` (the tables' schema).

   Don't run the simulator in that project.
3. **Replace the mock categories** in `src/warehouse.py` (`CATEGORY_OF_TYPE`) with the bank's real transaction type
   codes (`TRNS_TYP`), then run `python make_notebook.py`.
4. Run the setup notebook.

To simulate the next month in the sandbox, set `AS_OF` in the simulator to the next month's end, Run All, then run
**Nightly batch**.

## Input data (interface 1: what every model gets)

The project reads 2 tables of the warehouse, read-only and as they are. The names, codes and categories it expects
are all in `src/warehouse.py`, the only file to change if they differ at the bank.

| Table | What the project reads | What it becomes (in Dataiku) |
|---|---|---|
| `EDW.DIM_CUSTOMER` | the clients: CIF, customer type, line of business, legal entity, ID type, segment, sector, industry, city, status. Personal fields are never read. | **`clients`**: the business clients (every line of business except retail, plus retail clients with a commercial registration, i.e. sole proprietors), one row per CIF |
| `EDW.FCT_STATEMENT_ENTRY` | the daily postings: CIF, account, value date, credit / debit, amount in SAR, transaction type (`TRNS_TYP`), reference (`TRNS_RFR`), reversal marker | **`daily_flows`**: one row per client, day and category, with `money_in`, `money_out` and `postings`, over each client's whole history up to the cut-off month |

**Categories**
- Each posting gets its category from its transaction type, through `CATEGORY_OF_TYPE`. A type that isn't listed
  goes to `OTHER`.
- **The list is a mock** built from the sample data's types: customer receipts, salaries, rent, suppliers,
  financing, government, utilities, cards, fees, other. At the bank, replace it with the real codes; nothing else
  changes.

**Daily postings, summed in Dataiku**
- The staging recipe reads the postings in chunks, keeps the business clients' postings up to the end of the
  cut-off month, and sums them per client, day and category.
- Reversed postings (and what they reverse) are left out, and so are transfers between the client's own accounts.

**Cut-off month**
- It's the month before today, unless the project variable `cutoff_override` (YYYY-MM) is set; the simulator sets
  it.
- If the postings don't reach the cut-off month, or there are far fewer than the month before, the run stops,
  because the warehouse load looks late.

**Days to months:** `cashflow.monthly_matrix(daily, cutoff)` gives the net cash flow per client and category (and
TOTAL) and month, for the models that work by month.

## What the project builds

| Zone | Recipes | Outputs (CSV files unless noted) |
|---|---|---|
| Raw data | (the simulator, or the read-only warehouse) | `DIM_CUSTOMER`, `FCT_STATEMENT_ENTRY` |
| Staging | `stage_tables` | `clients`, `daily_flows` |
| Training (data scientist) | `train_and_test`, `champion_challenger` | folder `experiments` (Experiment Tracking), `test_results`, `promotion_log` (appended) |
| Inference (inference owner) | `final_forecast`, `archive_forecast` | saved model `cash_forecast` (one version per model ID), `final_forecast`, folder `forecast_archive`, `inference_log` (appended) |
| Lab (data scientist) | `try_forecaster` | folder `lab_experiments` (Experiment Tracking, experiment `cashflow_lab`), `lab_results` (appended) |

The zones only meet through three interfaces:

1. **The data** (staging → training, lab and inference): `daily_flows` and `clients`.
2. **The forecaster** (lab → training): a class described in `models.py`, with:
   - `name` and `params`;
   - `fit(daily, clients, cutoff)`;
   - `predict(daily, clients, cutoff)`, which returns one row per client, category (and `TOTAL`) and months ahead
     (1, 2, 3), with `p10`, `ev` and `p90` in SAR.

   The project variable `forecaster` names the one that Retrain trains.
3. **The active version of the saved model `cash_forecast`** (training → inference): `final_forecast` loads it and
   writes its ID on every row. Inference reads nothing else from training.

## The forecast (`final_forecast`)

- **One row per** active client, category (and `TOTAL`) and months ahead, with these columns:
  - `client_no`, `category`, `cutoff_month`, `months_ahead`, `target_month`;
  - `p10`, `ev`, `p90`;
  - `history_months`, `status`, `model_id`, `load_id`.
- **Not enough history:** a client with less than 12 months of history (`MIN_HISTORY_MONTHS`) gets no values, and
  its status is `NOT_ENOUGH_HISTORY`.
- **For now, all the rows are computed in one go.** That's fine for the sample data. Making inference handle the
  bank's ~420k clients is the next step.

## Models: tracking, versions, promotion

**Placeholder model**
- `recent_average`, in `models.py`:
  - EV is the average of the last 3 months.
  - P10 and P90 are EV ∓ width × the series' spread over the last 12 months.
  - `fit` sets the width so that 80% of past months fell inside.
- It's the same for the 3 months ahead.

**Tracking**
- Every run of `train_and_test` is one model, with a model ID like `CFF_202610_20261107T1000` (cut-off month,
  training time).
- It's recorded in **Experiment Tracking**, experiment `cashflow_forecast`:
  - the forecaster and its settings;
  - the test scores on the last 6 months, per level (TOTAL, categories) and per months ahead;
  - the backtest forecaster;
  - one MLflow model.

**Versions:** that MLflow model is deployed as a new version of the saved model `cash_forecast`, named by the model
ID. It isn't activated yet.

**Promotion**
- `champion_challenger` tests the new model and the production model on the same last 6 months, each with its
  version trained without those months.
- The rules are **project variables**, so you can change them without touching the code:

| Variable | Default | The candidate is promoted only if |
|---|---|---|
| `promotion_min_coverage` | 0.70 | its P10–P90 coverage is at least this |
| `promotion_max_coverage` | 0.90 | … and at most this (80% is the aim) |
| `promotion_beat_simple_guess` | true | its EV error is lower than the simple guess (the same as the month before) |
| `promotion_min_gain` | 0.0 | its EV error is lower than the production model's by more than this share (0.02 = 2%). A tie keeps production. |

- With no production model to compare with, the new model is promoted (the forecast needs one), with a warning if
  it fails the checks.
- Each decision, its reason and the rules used are added to `promotion_log`.
- "Promote if better" activates the version. To roll back, activate an earlier version of `cash_forecast`.

## Trying other models (lab)

The data scientist tries forecasters without touching production. They're tested the same way champion /
challenger tests a model, next to the production model and the simple guess.

**Notebook `forecast_lab.ipynb`**
- **EDA:**
  - the daily flows per category and per segment;
  - the net cash flow per category over time;
  - seasonality;
  - one client day by day.
- **Models to try** (with `lab_compare`):
  - `recent_average`, the production placeholder. Trained again here, it matches the production model's scores,
    which checks that the test is the same.
  - `recent_average_6m`;
  - `lag_boosting`: an example of a small scikit-learn model that learns from the last 12 months.

**Without a notebook:** set the project variable `lab_forecaster` and build `lab_results`.

**To put a forecaster in production:**
1. Add its class to `forecasters.py` and to `FORECASTERS`.
2. Set the project variable `forecaster` to its name.
3. Run **Retrain, test and promote**.

## Scenarios

| Scenario | Steps | Trigger (off in the sandbox) |
|---|---|---|
| Nightly batch | set the cut-off month → build the daily flows → forecast with the production model, archive it and log the run. It never trains. | daily, 06:00 Riyadh |
| Retrain, test and promote | set the cut-off month → build the daily flows → train, test and deploy a new model → champion / challenger → promote if better. It doesn't forecast. | monthly, on the 5th at 07:00 Riyadh |

## Change the code

The notebooks are assembled from `src/`. Edit those files, then run `python make_notebook.py`:

- `src/warehouse.py`: the warehouse tables and columns, the codes, **the categories**, the daily sums and the
  cut-off month
- `src/library_cashflow.py`: the project library `cashflow.py`, shared by all zones (the data interface, days to
  months, the saved model name)
- `src/models.py`: the forecaster interface, the placeholder, the test and its scores, `lab_try` and `lab_compare`
- `src/tracking.py`: model IDs, Experiment Tracking, saved model versions, **the promotion rules**, champion /
  challenger
- `src/forecast.py`: inference (the production model and the final forecast table)
- `src/forecasters.py`: the first version of the data scientist's `forecasters.py`
- `src/recipes/*.py`: one file per Python recipe
- `src/cells/*.py` (setup notebook) and `src/lab_cells/*.py` (lab notebook)
- `src/generate_data.py`, `src/simulator.py`, `src/sim_cells/*.py`: the warehouse simulator
