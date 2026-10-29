## What this project does

The **scripted** version of `CORP_CASHFLOW`: same data, same goal (forecast each account's money in and
money out for every day of the next 30 days, follow the balance day by day, find its lowest point, flag
liquidity risk), built with **Python and SQL recipes**.

It forecasts another way: a **regression**. The features are built by hand (calendar + past values) and a
gradient boosting model learns the day's amount from them. `CORP_CASHFLOW` uses Dataiku's **time series
forecasting** instead, where the algorithms handle time themselves.

Data is synthetic: 10 companies, 3 years of transactions up to yesterday (30 Sep 2026). Forecast:
**1 - 30 Oct 2026** (project variable `horizon_days`, **... > Variables**).

## The flow, left to right

| Tag | Recipe | Output | What it does |
|---|---|---|---|
| 1-input | uploaded CSVs | `accounts`, `transactions`, `calendar` | The data |
| 2-clean | Python `compute_transactions_clean` | `transactions_clean` | Description rules -> category; sign -> flow (IN/OUT); excluded = internal transfer or one-off |
| 3-daily | SQL `compute_daily_totals` | `daily_totals` | Money in / out per account and day — *inside* PostgreSQL |
| 3-daily | Python `compute_daily_flows` | `daily_flows` | Every account × flow × day, 0 on quiet days, + 30 future days (empty amount) |
| 4-forecast | Python `compute_forecast` | `forecast`, `model_evaluation` | Gradient boosting: test, then forecast with a 10%-90% range |
| 5-outlook | Python `compute_balance_outlook` | `daily_balance_path`, `account_outlook` | Balance per day, lowest point and its date, risk level |

## The model

- Learns **the day's amount** of each series (account × IN/OUT) from the calendar (day of month, weekday,
  weekend, month start / end) and from past values at least 30 days old (28-day average, same weekday 5-8
  weeks before, same day last month), so every feature is known for every day to forecast.
- Tested on the last 30 known days; `model_evaluation` compares its error (MAE, MASE) with three simple
  baselines. Then retrained on the full history to forecast the next 30 days.
- Two more models (10% and 90% quantiles) give the range.

## Balance path and risk level

`balance (day) = yesterday's balance + running total of (money in - money out)`. The pessimistic balance
is the 1-in-10 bad case of that running total (the daily ranges add up as variances).

**HIGH**: the lowest balance goes past the overdraft limit. **MEDIUM**: below 0, expected or pessimistic.
**LOW**: otherwise.

## Things to try

- Open a recipe, edit the code, **Run**; use the notebook button to explore it interactively.
- Add a feature in `compute_forecast` (e.g. the same day last year) and compare `model_evaluation`.
- Change `horizon_days` and rebuild.
