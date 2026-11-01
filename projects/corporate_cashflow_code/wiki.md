## What this project does

For each of 10 corporate accounts it forecasts the rest of this month + the next 3 months: money in, expenses
and loan installments, the balance day by day and, per month, the lowest balance and a risk level. It is the
**scripted** version of `CORP_CASHFLOW`: the same data, built with Python and SQL recipes and a gradient
boosting model.

Data is synthetic: 3 years of transactions up to yesterday (30 Sep 2026). Forecast: 1 Oct 2026 - 31 Jan 2027,
months 2026-10 (month 0) to 2027-01 (month 3).

## The idea

- **Inflows** and **expenses** (payroll, suppliers, rent...) are forecast by the model.
- **Obligations** (loan installments) are taken from `loan_schedule`: they are known, so never forecast.
- `balance = yesterday's balance + running total of (inflows - expenses - obligations)`.

## The flow, zone by zone

**Data prep**
1. `transactions_clean`: each transaction gets a category, a flow (IN, EXPENSE, or OBLIGATION when it has a
   loan_id) and excluded = 1 for one-offs and internal transfers.
2. `daily_totals` (SQL): totals per account, day, flow and category, without excluded rows and obligations.
3. `daily_series`: every day of every series (0 on quiet days), with the calendar columns, plus 123 empty
   days to forecast.
4. `daily_obligations`: the installments due in the forecast window.

**Training**
5. `forecast_model` (folder) and `model_evaluation`: the model is tested on the last 123 known days against
   three simple baselines, then retrained on all of the history and saved.

**Inference**
6. `forecast`: the expected amount per series and day.
7. `daily_balance_path`: inflows, expenses, obligations, balance and bad-case balance per day.
8. `monthly_outlook`: the days added up per month, lowest balance and its date, risk level.
9. `monthly_drivers`: each category or loan per month, its share, and the same month last year.

## Scenarios

- **Train model** (about once a month): check the inputs, build Data prep, train the model.
- **Daily forecast** (every day): check the inputs, build Data prep, check the model (trained in the last 45
  days, MASE below 1, horizon 123), build Inference, compute the metrics.

## The model

- One series per account × flow × category. Features: the calendar (day of month, weekday, weekend, month
  start / end, month), the series, its category, and past values at least 123 days old (so they are known for
  every day to forecast).
- Each series is divided by its level (average day over the last year, moved along its growth), so small and
  large series count alike and a falling account keeps falling.
- Measures: **MASE** (daily error; below 1 = better than "same as the day before") and **monthly WAPE**
  (error on the monthly totals). Last test: MASE 0.32, monthly WAPE 0.16.

## Bad case and risk

- Bad case: money in 15% lower and expenses 15% higher (variable `bad_case_pct`), obligations as scheduled.
- **HIGH**: the lowest balance goes past the overdraft limit. **MEDIUM**: it, or the bad case, goes below 0.
  **LOW**: otherwise.

## Things to try

- Change `bad_case_pct` (**... > Variables**) and run **Daily forecast**: see the risk levels move.
- Add a feature in the library `cashflow_model.py`, run **Train model** and compare `model_evaluation`.
