## What this project does

Forecasts the **next 3 months of cash flow for every corporate account** from its transaction history,
flags accounts heading for liquidity trouble, and has an LLM write a **report**.

All data is synthetic: 30 corporate accounts, 36 months of transactions (Oct 2023 - Sep 2026).
The forecast covers **Oct - Dec 2026**.

The flow has **no code**: Dataiku visual recipes and **Visual ML** (open any recipe or the model to see
its settings).

## The flow, left to right

| Tag | Recipe (type) | Output | What it does |
|---|---|---|---|
| 1-ingest | uploaded CSVs | `accounts_raw`, `transactions_raw` | Raw data. Explore tab, column statistics |
| 2-prepare | `prepare_accounts` (Prepare) | `accounts` | Parse the opening date |
| 2-prepare | `prepare_transactions` (Prepare) | `transactions_prepared` | Drop incomplete rows, parse the date, month, signed amount, amount per direction / category (formulas) |
| 3-aggregate | `group_monthly_cashflow` (Group) | `monthly_cashflow` | Sum per account and month |
| 3-aggregate | `group_account_balance` (Group) | `account_balance` | Total net cash flow (=> current balance) and latest month |
| 4-features | `window_monthly_history` (Window) | `monthly_history` | Per account, by month: previous months (lag), same month last year, 12-month average, next 3 months (lead) |
| 4-features | `join_account_segment` (Join) | `monthly_history_accounts` | Add sector, segment, latest month |
| 4-features | `prepare_training_set` (Prepare) | `training_set` | 3-month averages, month of year, keep months with a year of history |
| 5-ml | `prepare_ml_dataset` (Prepare) | `ml_dataset` | One row per account x month x flow x horizon, amounts divided by the account's size, target, role |
| 5-ml | `split_ml_dataset` (Split) | `ml_train`, `ml_test`, `ml_forecast` | By role |
| 5-ml | `train_cashflow_model` (**Visual ML** training) | saved model `cashflow_model` | LightGBM, tested on `ml_test`, 10%-90% prediction intervals |
| 5-ml | `evaluate_cashflow_model` (Evaluate) | `test_predictions`, `model_metrics` | Model metrics on the held-out months |
| 5-ml | `group_test_errors` (Group) + `prepare_model_evaluation` (Prepare) | `model_evaluation` | Model vs 2 baselines (WAPE) |
| 5-ml | `score_forecast` (Score) | `forecast_scored` | The forecast from the latest month |
| 5-ml | `prepare_forecast_amounts` (Prepare), `group_forecast_by_month` (Group), `prepare_cashflow_forecast` (Prepare) | `cashflow_forecast` | Back to USD, one row per account and month |
| 6-insights | `window_forecast_path` (Window) | `forecast_path` | Running total of the forecast net cash flow |
| 6-insights | `group_forecast_summary` (Group) | `forecast_summary` | 3-month totals, lowest point, monthly detail as text |
| 6-insights | `join_outlook_inputs` (Join) | `outlook_inputs` | Account + balance + last 3 months + forecast |
| 6-insights | `prepare_account_outlook` (Prepare) | `account_outlook` | Balances and risk level (formulas) |
| 7-genai | `prompt_account_reports` (Prompt) | `account_reports` | One LLM call per account |
| 7-genai | `group_portfolio_facts` (Group) | `portfolio_facts` | Whole portfolio in one row |
| 7-genai | `group_model_accuracy` (Group) | `model_accuracy` | Model and baseline errors in one row |
| 7-genai | `join_portfolio_inputs` (Join, cross) | `portfolio_inputs` | Both one-row datasets side by side |
| 7-genai | `prompt_portfolio_report` (Prompt) | `portfolio_report` | One LLM call for the executive summary |
| 7-genai | `stack_report` (Stack) | `report_rows` | Portfolio row + account rows |
| 7-genai | `prepare_cashflow_report` (Prepare) | `cashflow_report` | **The report**: `report_markdown` per row |

Use the **Tags** filter in the flow to highlight one step at a time. Recipes reading and writing
PostgreSQL run *inside* the database (Dataiku translates them to SQL: see the recipe's engine).

## How the machine learning works

- `training_set`: for each account and month *t*: what was known at *t* (this month, last 2 months,
  3- and 12-month averages, the same month last year) and what happened at *t+1, t+2, t+3*.
- `ml_dataset` unfolds it to one row per **flow** (inflow, outflow, net) and **horizon** (1-3 months ahead),
  so **one model** serves every account, flow and horizon. Amounts are divided by the account's average
  monthly inflow (`scale`) so small and large companies look alike to the model.
- **Roles**: rows whose target falls in the last 6 months of data are `TEST` (held out), older ones `TRAIN`,
  the latest month is `FORECAST`.
- **The model**: open the saved model `cashflow_model` (diamond in the flow), or the Lab analysis it comes
  from (**Lab** menu), to see its settings, test metrics, feature importance and explanations. To try
  other algorithms: Lab > the analysis > Design > Algorithms, train, then *Deploy* the best one.
- **Evaluation**: `model_evaluation` compares the model's error (WAPE) on the held-out months with two
  simple baselines ("same month last year", "average of last 3 months").
- **Prediction intervals**: the model gives a 10%-90% range for each prediction (`net_lower` / `net_upper`
  in `cashflow_forecast`), used for the pessimistic case in the risk level.
- The model used for the forecast is trained on `ml_train` only (data up to 6 months ago).

## The GenAI report

- The two **Prompt** recipes use the **LLM Mesh** connection `openai` (Administration > Connections),
  model `gpt-6-luna`. Open one to see the prompt: `{{column}}` is replaced by that column's value.
- Change the model or the prompt in the recipe, then use **Test** on a few rows before running it.
- The `generator` column of `cashflow_report` shows which model wrote each row (or the LLM error).
- Try prompts interactively in **GenAI > Prompt Studios** on top of `account_outlook`.

## Things to try

- Charts on `monthly_cashflow` (net cash flow by month, one line per account).
- A dashboard with `account_outlook` (risk level counts) and the portfolio report.
- Edit a recipe, then **Build** with "upstream" or "downstream" to see how Dataiku recomputes the flow.
