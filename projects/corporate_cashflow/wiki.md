## What this project does

For each of 10 corporate accounts: **forecast the money in and the money out of every day of the next 30
days** with Dataiku's **time series forecasting**, then follow the balance day by day from yesterday's
balance to find its **lowest point** and when it happens. A month can end positive and still go below the
overdraft limit on payroll day: this is what the daily view catches.

Data is synthetic: 10 companies, 3 years of transactions up to yesterday (D-1 = 30 Sep 2026), each with a
signed amount (+ in, - out) and a bank description. Forecast: **1 - 30 Oct 2026**. No code: visual recipes
and Visual ML.

## The flow: three zones

| Zone | What | Run by |
|---|---|---|
| **Data prep** | From the raw transactions to `daily_flows_scoring`: every series and day, history + the 30 days to forecast | both scenarios |
| **Training** | `daily_flows` (history only) -> the Visual ML model -> saved model `cashflow_forecast_model` | **Train model** |
| **Inference** | The saved model scores `daily_flows_scoring` -> forecast -> balance path -> `account_outlook` | **Daily forecast** |

The saved model is the link between the two: training writes a new version of it, inference always
scores with the **active** version. Its rebuild behavior is **explicit** (saved model > Settings), so
building `account_outlook` never retrains it. Datasets used by another zone are *shared* into it (they show
in both).

| Zone (tag) | Recipe (type) | Output | What it does |
|---|---|---|---|
| Data prep (1-input) | uploaded CSVs | `accounts`, `transactions`, `calendar` | The data; `calendar` = one row per day with calendar features |
| Data prep (2-clean) | `prepare_transactions` (Prepare) | `transactions_clean` | Description rules -> `category`; sign -> `flow` (IN/OUT); `excluded` = internal transfer or one-off |
| Data prep (3-daily) | `group_daily_totals` (Group) | `daily_totals` | Money in / out per account and day, without the excluded rows |
| Data prep (3-daily) | `join_account_days` (Join, cross) | `account_days` | Every calendar day for every account |
| Data prep (3-daily) | `join_daily_totals` (Join) | `account_days_totals` | Add the day's totals (empty = nothing happened) |
| Data prep (3-daily) | `prepare_daily_flows` (Prepare) | `daily_flows_scoring` | 0 on quiet days, up to yesterday + 30 future days, one row per flow |
| Training (4-train) | `prepare_training` (Prepare) | `daily_flows` | History only: **the training data** |
| Training (4-train) | `train_cashflow_forecast_model` (**Visual ML**) | `cashflow_forecast_model` | The best of 9 algorithms, see below |
| Inference (5-forecast) | `score_flows` (Score) | `scored_flows` | Forecast and range on the 30 future days |
| Inference (5-forecast) | `prepare_forecast` (Prepare) | `forecast` | Future days only |
| Inference (6-balance) | `group_daily_forecast` (Group) | `daily_forecast` | IN and OUT side by side per day, net and its uncertainty |
| Inference (6-balance) | `window_running_totals` (Window) | `daily_running` | Running totals since today |
| Inference (6-balance) | `join_opening_balance` (Join) | `daily_running_accounts` | Add yesterday's balance |
| Inference (6-balance) | `prepare_balance_path` (Prepare) | `daily_balance_path` | **Balance per future day** |
| Inference (7-outlook) | `group_outlook` (Group) | `account_lowest` | Totals, lowest balance, lowest pessimistic balance |
| Inference (7-outlook) | `join_accounts` (Join) | `account_lowest_details` | The day of the lowest balance (join on account + balance = lowest), company details |
| Inference (7-outlook) | `prepare_outlook` (Prepare) | `account_outlook` | Closing balance and **risk level** |

## Cleaning: what the description is used for

`prepare_transactions` gives each transaction a category with simple rules: the first text found in the
description wins (`TRF FROM` -> customer_receipt, `SALARY` -> payroll, `INTERNAL TRF` -> internal_transfer,
`SALE OF` -> one_off, ...). Internal transfers and one-off deals are not repeatable cash flow, so they are
removed before the daily totals. The categories are only used for this cleaning: the model forecasts IN
and OUT.

## Time series forecasting: what you give it

`daily_flows`: one row per **series** and **day**.

| Setting | Here |
|---|---|
| Target | `amount` (total of the day, 0 if nothing happened) |
| Time variable | `date` (time step: 1 day) |
| Time series identifiers | `account_id` + `flow`: 20 series (10 accounts x IN / OUT) |
| Forecast horizon | 30 days |
| External features | `day_of_month`, `day_of_week`, `is_weekend`, `is_month_start`, `is_month_end` |

External features are columns **known in advance**: the calendar of the next 30 days is known, so the
models that support them (Prophet, LightGBM, the neural networks) can learn "payroll leaves around the
25th" or "nothing happens on weekends". That is why the dataset to score, `daily_flows_scoring`, also holds
the 30 future days, with their calendar but an empty amount.

## How it is tested

The last 30 days of each series (1 - 30 Sep 2026) are hidden, each algorithm is trained on what is before
and forecasts them, and the forecast is compared with what really happened.

The main metric is **MASE**: the error divided by the error of a naive "same as the day before" guess on
the training data. Below 1 = better than that. Over 30 days ahead, simple methods often score above 1.

## The algorithms compared (Lab > cashflow_forecast_model)

| Algorithm | Idea | MASE |
|---|---|---|
| Naive (trivial identity) | every future day = the last known day | 1.14 |
| Seasonal naive | every future day = same weekday of the last known week | 1.34 |
| ETS (Holt-Winters) | smoothed level + trend + weekly pattern | 0.72 |
| Seasonal LOESS (STL) | split into trend + weekly pattern, forecast each | 1.14 |
| ARIMA (2,0,0)(0,1,0)7 | weekly difference + the last 2 days | 1.44 |
| Prophet | trend + weekly / yearly seasonality + calendar features | 0.73 |
| LightGBM | gradient boosting on past values (30+ days old) + calendar features | 0.77 |
| Simple feed-forward | small neural network, all series together, + calendar features | 0.70 |
| **DeepAR** | recurrent neural network, all series together, + calendar features | **0.57** (deployed) |

Open the analysis to see, per model: the metrics, the **forecast chart** per series (test days vs actual),
the metrics **per time series**, and the settings it used.

## Balance path and risk level

`balance (day) = yesterday's balance + running total of (money in - money out)`.

The **pessimistic balance** is the 1-in-10 bad case of that running total. Each day's 10%-90% range says how
uncertain the day is (the range is about 2.56 standard deviations); over several days the uncertainties add
up as variances (`variance_running`), and the pessimistic balance is 1.28 standard deviations below the
expected one. Simply adding up every day's worst case would assume all 30 days go wrong at once, which is
far too pessimistic.

- **HIGH**: the lowest balance goes past the overdraft limit
- **MEDIUM**: below 0, expected or in the pessimistic case
- **LOW**: otherwise

`account_outlook` gives, per account: opening and closing balance, totals in and out, the **lowest balance
and its date**, the pessimistic lowest balance and the risk level.

## Scenarios: training and inference (Scenarios tab > Run)

| Scenario | When | Steps |
|---|---|---|
| **Daily forecast** | every day, once yesterday's data is in | Rebuild `account_outlook` and everything upstream: data prep, score with the active model, balance path, outlook. The model is not retrained. |
| **Train model** | monthly, or when the forecasts get worse | 1. rebuild `daily_flows`; 2. retrain the saved model (the deployed algorithm, on the latest history): a new version, made active. |

Retraining reuses the deployed algorithm and its settings; to compare algorithms again, train in the Lab
and redeploy. Older versions stay in the saved model: open it to compare them or roll back. The Free
Edition has no time-based triggers; a cron on the server can run
`dsscli scenario-run CORP_CASHFLOW DAILY_FORECAST`.

## Things to try (in the Lab analysis)

- **Design > Forecasting parameters**: change the horizon, retrain, compare.
- **Design > Train / Test set**: switch to **K-fold cross-test** (several test windows) for steadier metrics.
- **Design > External features**: remove one and see which models get worse.
- **Design > Algorithms**: ARIMA orders, LightGBM trees, epochs of the neural networks.
- Chart `daily_balance_path`: X = `date`, Y = `balance` and `pessimistic_balance`, subgroup by `account_id`.
