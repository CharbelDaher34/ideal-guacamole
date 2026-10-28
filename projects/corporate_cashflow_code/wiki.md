## What this project does

The **scripted** version of `CORP_CASHFLOW`: same data, close goal (forecast each account's net cash flow
for next month, flag liquidity risk, have an LLM write a note), built with **Python and SQL recipes**.

It also uses another way to forecast: a **regression**. The features are built by hand (this month, 3-month
average, same month last year) and a model learns next month's value from them. `CORP_CASHFLOW` uses
Dataiku's **time series forecasting** instead, where the algorithms find trend and seasonality themselves.

Data is synthetic: 10 companies, 36 months of transactions (Oct 2023 - Sep 2026). Forecast: **Oct 2026**.

## The flow, left to right

| Tag | Recipe | Output | What it does |
|---|---|---|---|
| 1-input | uploaded CSVs | `accounts`, `transactions` | The data |
| 2-monthly | Python `compute_monthly_net` | `monthly_net` | Net cash flow per account and month |
| 3-ml | SQL `compute_features` | `features` | 3-month average, same month last year, next month, row role — *inside* PostgreSQL |
| 3-ml | Python `compute_forecast` | `account_outlook`, `model_evaluation` | Linear regression: train, test, forecast Oct 2026, risk level |
| 4-genai | Python `compute_account_notes` | `account_notes` | The LLM's note per account |

## The model

- Learns **next month's net** from this month's net, the 3-month average and the same month last year.
- Trained on the `train` rows, checked on the `test` rows (Jun - Aug 2026). `model_evaluation` compares
  its average error with two simple baselines.

## Risk level

`projected balance = balance today + forecast`: **HIGH** past the overdraft limit, **MEDIUM** below 0,
**LOW** otherwise.

## The GenAI note

`compute_account_notes` calls the LLM in the project variable `llm_id` (**... > Variables**) through the
**LLM Mesh** connection `openai`, one call per account. No fallback: if a call fails, the recipe fails.

## Things to try

- Open a recipe, edit the code, **Run**; use the notebook button to explore it interactively.
- Add a feature in `features.sql` and use it in `compute_forecast`.
