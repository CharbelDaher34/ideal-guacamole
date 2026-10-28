## What this project does

The **scripted** version of `CORP_CASHFLOW`: same data, same goal (forecast each corporate account's
next 3 months of cash flow, flag liquidity risk, have an LLM write a report), built with **Python and
SQL recipes** instead of visual recipes and Visual ML. Open both projects side by side to compare.

All data is synthetic: 30 corporate accounts, 36 months of transactions (Oct 2023 - Sep 2026).
The forecast covers **Oct - Dec 2026**.

## The flow, left to right

| Tag | Recipe | Output | What it does |
|---|---|---|---|
| 1-ingest | uploaded CSVs | `accounts_raw`, `transactions_raw` | Raw data |
| 2-prepare | Python `compute_transactions_prepared` | `transactions_prepared`, `accounts` | Types, de-duplication, signed amount, month |
| 3-aggregate | SQL `compute_monthly_cashflow` | `monthly_cashflow` | Sums per account and month, *inside* PostgreSQL |
| 4-ml | Python `compute_training_set` | `training_set` | One row per account and month: features and targets |
| 4-ml | Python `compute_cashflow_forecast` | `cashflow_forecast`, `model_evaluation`, `feature_importance`, `backtest_predictions` | 6 LightGBM models, backtest vs baselines, forecast |
| 5-insights | SQL `compute_account_outlook` | `account_outlook` | Balances and risk level per account |
| 6-genai | Python `compute_cashflow_report` | `cashflow_report` | The report, written by the LLM |

## How the "training" works

- `training_set`: for each account and month *t*: what was known at *t* (this month, last 2 months,
  3- and 12-month averages, the same month last year) and what happened at *t+1, t+2, t+3* (the targets).
- `compute_cashflow_forecast` trains 6 LightGBM models (inflow and outflow x 1/2/3 months ahead), shared
  by all accounts. Amounts are divided by each account's average monthly inflow.
- **Backtest**: the recipe replays the last 6 month-ends, training only on data available at that time,
  and compares with what really happened. `model_evaluation` shows the error (WAPE) of the model and of
  two simple baselines; `backtest_predictions` lets you chart predicted vs actual.
- The spread of backtest errors gives the 10%-90% range (`net_lower` / `net_upper`).
- The final models are retrained on all known history before forecasting.

## The GenAI report

- `compute_cashflow_report` calls the LLM in the project variable `llm_id` (**... > Variables**), through
  the **LLM Mesh** connection `openai`. One call per account + one for the portfolio summary.
- No fallback: if the variable is empty or a call fails, the recipe fails.

## Things to try

- Open a recipe, edit the code, **Run**; use the notebook button to explore it interactively.
- Compare `model_evaluation` here with the one in `CORP_CASHFLOW` (Visual ML).
