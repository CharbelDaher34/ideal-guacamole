# CORP_CASHFLOW_CODE — corporate cash-flow forecast, scripted

The same data and goal as [`../corporate_cashflow`](../corporate_cashflow) (daily money in and out for the
next 30 days, balance day by day, lowest point and risk), built with **Python and SQL recipes** and a
**regression** model (gradient boosting, scikit-learn) instead of Dataiku's time series forecasting.

```bash
python3 ../corporate_cashflow/generate_data.py   # if ../corporate_cashflow/data/ does not exist yet
python3 create_project.py                         # (re)creates project CORP_CASHFLOW_CODE and builds it (~1 min)
```

Open http://localhost:10000/projects/CORP_CASHFLOW_CODE/ — the project **Wiki** walks through it.

## Flow (5 recipes)

```
transactions ─[Python]─► transactions_clean ─[SQL]─► daily_totals
daily_totals, calendar, accounts ─[Python]─► daily_flows (history + 30 future days)
daily_flows ─[Python, scikit-learn]─► forecast, model_evaluation
forecast, accounts ─[Python]─► daily_balance_path, account_outlook
```

| File | What |
|---|---|
| `create_project.py` | Project definition: datasets, recipes, tags, project variable `horizon_days`, wiki |
| `recipes/clean_transactions.py` | Description rules -> category; sign -> flow (IN/OUT); flag internal transfers and one-offs |
| `recipes/daily_totals.sql` | Money in / out per account and day (GROUP BY inside PostgreSQL) |
| `recipes/daily_flows.py` | Every account × flow × day, 0 on quiet days, the next 30 days with an empty amount |
| `recipes/forecast.py` | Gradient boosting: test on the last 30 days vs 3 baselines, then forecast the next 30 with a 10%-90% range |
| `recipes/balance_outlook.py` | Running balance, pessimistic balance, lowest point and its date, risk level |
| `wiki.md` | Project wiki home page |

## The model

One row = one series (account × IN/OUT) and one day; the model learns the day's amount from:

- the calendar: day of month, weekday, weekend, month start / end;
- past values that are **at least 30 days old**, so they are known for every day to forecast: the 28-day
  average ending 30 days before, the same weekday 5-8 weeks before, the same day last month;
- the series itself (a category: `CA1001_IN`, ...).

On the last 30 known days (MASE, below 1 = better than "same as the day before"): gradient boosting 0.50,
same day last month 0.92, last known day 0.92, same weekday of the last known week 1.32.

## Code vs visual

| | `CORP_CASHFLOW_CODE` (this) | `CORP_CASHFLOW` |
|---|---|---|
| Recipes | 4 Python + 1 SQL | 15 visual recipes + 1 Visual ML training |
| Approach | regression: features built by hand (calendar + past values) | time series forecasting: DSS handles time |
| Model | scikit-learn `HistGradientBoostingRegressor` (+ 10% / 90% quantile models) | best of 9 time series algorithms (DeepAR, ETS, Prophet, LightGBM...) |
| Horizon | 30 days | 30 days |

**Re-running `create_project.py` deletes and re-creates the project.**
