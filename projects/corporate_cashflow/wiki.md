## What this project does

For each of 10 corporate accounts: **forecast the net cash flow of the next 3 months** (Oct - Dec 2026)
with Dataiku's **time series forecasting**, check whether the balance will go below zero or past the
overdraft limit, and have an **LLM write a note** for the relationship manager.

Data is synthetic: 10 companies, 36 months of transactions (Oct 2023 - Sep 2026). No code: visual recipes
and Visual ML.

## The flow, left to right

| Tag | Recipe (type) | Output | What it does |
|---|---|---|---|
| 1-input | uploaded CSVs | `accounts`, `transactions` | The data. Open them, try the Explore tab and column statistics |
| 2-monthly | `prepare_transactions` (Prepare) | `transactions_prepared` | Month of each transaction, signed amount `net` (+ in, - out) |
| 2-monthly | `group_by_month` (Group) | `monthly_net` | **The time series**: one row per account and month |
| 3-forecast | `train_cashflow_forecast_model` (**Visual ML**) | `cashflow_forecast_model` | The best of 9 algorithms, see below |
| 3-forecast | `forecast_next_months` (Score) | `forecast` | History + Oct - Dec 2026 with `forecast` and quantiles |
| 4-outlook | `group_forecast_3m` (Group) | `forecast_3m` | Per account: 3-month total, worst month, pessimistic total |
| 4-outlook | `join_accounts` (Join) | `forecast_accounts` | Add company, balance, overdraft limit |
| 4-outlook | `prepare_outlook` (Prepare) | `account_outlook` | Projected balance and risk level |
| 5-genai | `prompt_account_notes` (Prompt) | `account_notes` | The LLM's note per account (`llm_output`) |

## Time series forecasting: what you give it

A dataset with one row per **series** and **time step**: here `monthly_net`.

| Setting | Here |
|---|---|
| Target | `net` |
| Time variable | `month` (time step: 1 month) |
| Time series identifiers | `account_id`: 10 series, one per company |
| Forecast horizon | 3 months |

No feature engineering: the algorithms find trend and seasonality themselves.

## How it is tested

The last 3 months of each series (Jul - Sep 2026) are hidden, each algorithm is trained on what is before
and forecasts them, and the forecast is compared with what really happened.

The main metric is **MASE**: the error divided by the error of a naive "same as last month" guess on the
training data. Below 1 = better than naive. Also shown: MAPE, sMAPE, MAE, RMSE, MSIS (quality of the range).

## The algorithms compared (Lab > cashflow_forecast_model)

| Algorithm | Idea | MASE |
|---|---|---|
| Naive (trivial identity) | next months = last month | 0.90 |
| Seasonal naive | next months = same months last year | 0.98 |
| ETS (Holt-Winters) | smoothed level + trend + yearly season | 0.70 |
| Seasonal LOESS (STL) | split into trend + season, forecast each | 0.79 |
| **ARIMA** (1,0,0)(0,1,0)12 | yearly difference + last month's value | **0.68** (deployed) |
| AutoARIMA | ARIMA that searches its own orders | 0.78 |
| Prophet | trend with changepoints + seasonality | 0.89 |
| Simple feed-forward | small neural network, all series together | 0.86 |
| DeepAR | recurrent neural network, all series together | 0.91 |

Open the analysis to see, per model: the metrics, the **forecast chart** per account (test months vs
actual), the metrics **per time series**, and the settings it used.

## Risk level

`projected balance = balance today + 3-month forecast` (end of December). The pessimistic case uses the
10% quantile instead of the forecast.

- **HIGH**: projected balance past the overdraft limit
- **MEDIUM**: below 0, expected or in the pessimistic case
- **LOW**: otherwise

## Things to try (in the Lab analysis)

- **Design > Forecasting parameters**: change the horizon (e.g. 6 months), retrain, compare.
- **Design > Train / Test set**: switch to **K-fold cross-test** (several test windows instead of one) for
  steadier metrics.
- **Design > Algorithms**: change ARIMA's orders, ETS trend / seasonality (additive, multiplicative,
  damped), or the epochs of the neural networks.
- **Design > External features**: add a column known in advance (e.g. a holiday flag) to help the models.
- Deploy another model: from its result page, **Deploy** > update `cashflow_forecast_model`, then rerun
  the flow.
- Chart `forecast`: X = `month`, Y = `net` and `forecast`, subgroup by `account_id`.
