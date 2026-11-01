## What this project does

For 10 corporate accounts: forecast the **money in** and the **normal expenses** of every day from tomorrow
to the end of the third month after this one, add the **loan installments** from the schedule, and follow the
balance day by day and month by month. A month can end positive and still go below the overdraft limit on
payroll day: the daily balance shows when, the monthly table says which month.

Data is synthetic: transactions up to yesterday (30 Sep 2026). Outlook: 1 Oct 2026 - 31 Jan 2027. No code.

## Three amounts per day

| Column | What | From |
|---|---|---|
| **inflows** | money in | the model |
| **expenses** | payroll, suppliers, rent, utilities, tax, fees | the model |
| **obligations** | loan installments on their due dates | `loan_schedule`, exact |

`outflows = expenses + obligations`, `net = inflows - outflows`, `balance = yesterday's balance + net`.

## The flow, zone by zone

**Data prep** (built by both scenarios)

- `prepare_transactions`: the description gives the `category`; a row with a `loan_id` is an **OBLIGATION**,
  otherwise the sign gives **IN** or **EXPENSE**; internal transfers and one-off deals are `excluded`.
- `group_daily_totals`, `group_series`: amount per series and day; the list of series (account × flow ×
  category, installments and excluded rows left out).
- `join_account_days`, `prepare_account_calendar`: every day of every account, numbered from yesterday
  (`days_ahead`) and by month (`month_index`: 0 = this month, 1-3 the next ones, < 0 the past).
- `join_series_days`, `join_daily_totals`, `prepare_daily_series`: `daily_series`, every series on every day,
  0 on quiet days, plus 123 future days with an empty amount and their calendar columns.
- `join_obligations`: the installments due from tomorrow to the end of month 3.

**Training**: `training_series` (the history part of `daily_series`) → Visual ML → saved model
`cashflow_forecast_model`. Rebuild behavior **explicit**: only the Train model scenario retrains it.

**Inference**

- `score_series`, `prepare_forecast`: the forecast of the future days (negatives count as 0).
- `stack_cash_flows`, `join_months`: forecast, installments and last year's transactions in one table, each row
  with its month. The `source` column says where a row comes from.
- `group_days`, `window_running_totals`, `prepare_balance_path`: totals per day, running totals, balance.
- `group_months`, `join_accounts`, `prepare_monthly_outlook`: the days added up per month, risk level.
- `group_drivers`, `join_driver_months`, `window_driver_rank`, `prepare_drivers`: per month, each category and
  loan with its share, the same month last year and the change.

## The model

| Setting | Here |
|---|---|
| Target | `amount` (total of the day, 0 if nothing happened) |
| Time / step | `date`, 1 day |
| Series | `account_id` + `flow` + `category` |
| Horizon | 123 days |
| External features | `day_of_month`, `day_of_week`, `is_weekend`, `is_month_start`, `is_month_end` |

External features are known in advance: that is why `daily_series` also holds the future days, with their
calendar but an empty amount. The test hides the last 123 days and compares the forecast with what happened.

The deployed algorithm is the one with the lowest **RMSE**, which counts the big days (payroll, rent, tax: one
day a month). **MASE** (below 1 = better than "same as the day before") can look good for a forecast of 0
every day, which misses those payments: compare the two in the Lab. The model check still asks MASE < 1.

**Current build**: LightGBM is deployed (RMSE 14,964 vs DeepAR 16,839), MASE 0.93: under the limit of 1,
but weaker than the code project's 0.32. Risk: Levant Logistics HIGH every month, Tripoli Textiles MEDIUM in
October then HIGH, every other account LOW.

## Bad case and risk

Bad case (variable `bad_case_pct` = 0.15): money in 15% lower, expenses 15% higher, installments as scheduled.

- **HIGH**: the lowest balance of the month goes past the overdraft limit
- **MEDIUM**: the lowest balance or the bad-case one goes below 0
- **LOW**: otherwise

## Scenarios

| Scenario | Steps |
|---|---|
| **Train model** | check the inputs → build Data prep → train the model |
| **Daily forecast** | check the inputs → build Data prep → check the model → build Inference → metrics |

The model check stops the forecast if the model is older than 45 days, has a MASE of 1 or more, or does not
forecast 123 days. The Free Edition has no time triggers: run the scenarios by hand or from the dashboard.

## Things to try

- Change `bad_case_pct` (... > Variables) and run Daily forecast.
- Lab > Design > Algorithms: add Prophet or ETS and compare.
- Add a loan to `loan_schedule` starting next month and read `monthly_outlook`.
