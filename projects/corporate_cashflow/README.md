# CORP_CASHFLOW — corporate cash-flow forecast (daily time series)

Forecast each corporate account's **money in and money out for every day of the next 30 days** with
Dataiku's **time series forecasting**, follow the balance day by day from yesterday's balance, and find
its **lowest point** and when it happens. No code recipes: visual recipes and Visual ML. Synthetic data only.

```bash
python3 generate_data.py     # data/: accounts.csv (10), transactions.csv (~25,600, Oct 2023 - Sep 2026), calendar.csv
python3 create_project.py    # (re)creates project CORP_CASHFLOW in DSS, trains 9 algorithms, runs the flow (~7 min)
```

The first run also creates the DSS code env `timeseries` (DSS's "Visual ML, Causal ML, and Timeseries
Forecasting (CPU)" packages: statsmodels, pmdarima, prophet, gluonts, CPU torch; ~3 GB, a few minutes).

Open http://localhost:10000/projects/CORP_CASHFLOW/ — the project **Wiki** ("Start here") walks through it.

## Data

- `transactions`: signed `amount` (+ in, - out) and a bank-style `description` (`TRF FROM ...`,
  `SALARY OCT 2026`, `INTERNAL TRF TO SAVINGS ...`). Receipts and supplier payments on business days;
  payroll (25th), rent (1st), loan (10th), utilities (15th), quarterly VAT (20th), fees (month end);
  internal transfers and a few one-off deals that the flow removes.
- `accounts`: closing `balance` on `balance_date` = yesterday (D-1 = 30 Sep 2026), overdraft limit.
- `calendar`: one row per day up to Dec 2026 with `day_of_month`, `day_of_week`, `is_weekend`,
  `is_month_start`, `is_month_end` (known in advance, used as external features).

## Flow (16 recipes, 3 zones)

```
Data prep   transactions ─[Prepare: description rules]─► transactions_clean ─[Group, excluded = 0]─► daily_totals
            calendar × accounts ─[Join cross]─► account_days ─[Join daily_totals]─► ─[Prepare: 0 on quiet days,
            fold IN/OUT]─► daily_flows_scoring (history + 30 future days)
Training    daily_flows_scoring ─[Prepare: history only]─► daily_flows
            daily_flows ─[Visual ML time series: 9 algorithms, best deployed]─► cashflow_forecast_model
Inference   daily_flows_scoring + cashflow_forecast_model ─[Score]─► scored_flows ─[Prepare]─► forecast
            forecast ─[Group]─► daily_forecast ─[Window: running totals]─► ─[Join balance]─► ─[Prepare]─► daily_balance_path
            daily_balance_path ─[Group]─► account_lowest ─[Join: day of the lowest + accounts]─► ─[Prepare: risk]─► account_outlook
```

Three scenarios (Scenarios tab > Run), one per zone; each builds only its own zone:

| Scenario | Steps |
|---|---|
| **Prepare data** | 1. check the inputs (data quality rules); 2. rebuild the Data prep zone |
| **Train model** | 1. run *Prepare data*; 2. retrain the model (Training zone) |
| **Daily forecast** | 1. run *Prepare data*; 2. **check the model**; 3. forecast, balance path, outlook (Inference zone); 4. compute the outlook metrics |

- **Input checks** (dataset > Data Quality): `transactions` not empty, no empty field, one row per
  `txn_id`, only known accounts and up to the balances' date (yesterday's data is in); `accounts` one row
  per account, balance and limit filled, one balance date; `calendar` one row per day, covers the days to
  forecast. A rule in error stops the scenario.
- **Model check**: the saved model's active version exists, was trained in the last 45 days, has a test MASE below 1
  (better than "same as the day before") and forecasts `horizon_days` days; otherwise the scenario stops
  with the reason.
- **Metrics** on `account_outlook` (dataset > Metrics, with history): HIGH and MEDIUM accounts, lowest
  balance ahead, row count.
- **Dashboard** "Cash-flow outlook": run buttons, the metrics, lowest balance per company, balance per
  day, the outlook table.

The Free Edition has no time-based triggers: run them by hand, from the dashboard buttons, or from a cron
with `dsscli scenario-run CORP_CASHFLOW DAILY_FORECAST`.

The horizon is the project variable `horizon_days` (30), read by the recipes as `${horizon_days}`.

The model is created at deploy time like you would in the UI: a Lab analysis on `daily_flows` (target
`amount`, time `date`, identifiers `account_id` + `flow`, horizon 30, calendar columns as external
features), the algorithms in `ALGORITHMS` trained and compared on the last 30 days, and the best on MASE
deployed to the flow with its Score recipe. The Lab analysis stays in the project to compare, tune and
redeploy.

Risk level: **HIGH** if the lowest balance goes past the overdraft limit, **MEDIUM** if it goes below 0
(expected, or in the pessimistic case = the 1-in-10 bad case of the running total), **LOW** otherwise.

All managed datasets are PostgreSQL tables named `CORP_CASHFLOW_<dataset>`.

| File | What |
|---|---|
| `generate_data.py` | Synthetic accounts, transactions and calendar (seeded) |
| `create_project.py` | Project definition: datasets, description rules, visual recipes, algorithms to compare, zones, scenarios, model check, wiki |
| `quality.py` | Input data quality rules, outlook metrics and dashboard (shared with the code project) |
| `wiki.md` | Project wiki home page |
| `HOW_IT_WORKS.md` | The method and the Dataiku components, explained simply |

## Results of the current build

- Test days 1 - 30 Sep 2026, MASE (below 1 = better than "same as the day before"): DeepAR 0.57 (deployed),
  feed-forward NN 0.70, ETS 0.72, Prophet 0.73, LightGBM 0.77, naive 1.14, Seasonal LOESS 1.14, seasonal
  naive 1.34, ARIMA 1.44.
- Outlook 1 - 30 Oct 2026: 2 MEDIUM (Levant Logistics: balance down to about -1,500 by 30 Oct, pessimistic
  -129k; Tripoli Textiles: pessimistic -35k), 8 LOW, no HIGH.

**Re-running `create_project.py` deletes and re-creates the project** — changes made in the UI (including
models trained in the Lab) are lost; put them in the code instead.
