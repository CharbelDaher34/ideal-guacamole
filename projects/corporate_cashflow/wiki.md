## What this project does

For each of 10 corporate accounts: **forecast next month's net cash flow** (money in minus money out),
check whether the balance will go below zero or past the overdraft limit, and have an **LLM write a note**
for the relationship manager.

Data is synthetic: 10 companies, 24 months of transactions (Oct 2024 - Sep 2026). Forecast: **Oct 2026**.
No code: visual recipes and **Visual ML** (open any recipe or the model to see its settings).

## The flow, left to right

| Tag | Recipe (type) | Output | What it does |
|---|---|---|---|
| 1-input | uploaded CSVs | `accounts`, `transactions` | The data. Open them, try the Explore tab and column statistics |
| 2-monthly | `prepare_transactions` (Prepare) | `transactions_prepared` | Month of each transaction, signed amount `net` (+ in, - out) |
| 2-monthly | `group_by_month` (Group) | `monthly_net` | Sum of `net` per account and month |
| 3-ml | `window_features` (Window) | `features` | Per account, by month: 3-month average, same month last year, next month (= what to predict) |
| 3-ml | `prepare_roles` (Prepare) | `features_roles` | Label rows: train, test (Jun - Aug 2026), forecast (Sep 2026), warm_up (first year) |
| 3-ml | `split_by_role` (Split) | `ml_train`, `ml_test`, `ml_forecast` | One dataset per role |
| 3-ml | `train_next_month_model` (**Visual ML**) | `next_month_model` | Linear regression, tested on `ml_test` |
| 3-ml | `score_next_month` (Score) | `forecast` | The model's `prediction` for Oct 2026 |
| 4-outlook | `join_accounts` (Join) | `forecast_accounts` | Add company name, balance, overdraft limit |
| 4-outlook | `prepare_outlook` (Prepare) | `account_outlook` | Projected balance and risk level |
| 5-genai | `prompt_account_notes` (Prompt) | `account_notes` | The LLM's note per account (`llm_output`) |

## The model

- One row = one account in one month. The model learns **next month's net** (`net_next_month`) from
  three numbers: this month's net, the average of the last 3 months, and the same month last year.
- Open `next_month_model`: **Performance** shows the error on the test months, **Coefficients** how much
  each input counts.
- Test months error: $35.5k on average, against $45.9k for "same month last year" and $52.8k for
  "average of the last 3 months".

## Risk level

`projected balance = balance today + forecast`

- **HIGH**: projected balance past the overdraft limit
- **MEDIUM**: projected balance below 0, within the overdraft
- **LOW**: projected balance positive

## Things to try

- In **Lab**, open the `next_month_model` analysis, enable another algorithm (e.g. Random Forest), train
  and compare.
- Change the prompt in `prompt_account_notes` and run it again.
