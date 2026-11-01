# Cash-flow forecast: guide

Two Dataiku projects answer the same question: **will a corporate account go below 0, or past its
overdraft limit, in the next 30 days, and on which day?**

| Project | Built with | Model |
|---|---|---|
| `CORP_CASHFLOW` | visual recipes, no code | Visual ML time series: best of 9 algorithms (DeepAR, MASE 0.57) |
| `CORP_CASHFLOW_CODE` | Python and SQL recipes | scikit-learn gradient boosting on calendar + past values (MASE 0.50) |

Same data, same outputs, same scenarios, checks and dashboard.

Screen-by-screen tour of the Dataiku UI: [`docs/DSS_WALKTHROUGH.md`](docs/DSS_WALKTHROUGH.md).

---

## 1. Data

| Input | Content |
|---|---|
| `transactions` | one row per transaction up to yesterday (D-1): signed `amount` (+ in, − out), bank `description` |
| `accounts` | one row per account: `balance` on `balance_date` (= yesterday), `overdraft_limit` |
| `calendar` | one row per day, history and future: day of month, weekday, weekend, month start / end |

The description rules give each transaction a category. Internal transfers and one-off deals are removed:
they are not repeatable cash flow. After that, only **money in (IN)** and **money out (OUT)** matter.

## 2. Method

1. **Daily series**: money in and money out per account and day, 0 on quiet days. 10 accounts × IN/OUT
   = 20 series. Daily, because a month can end positive and still dip below 0 on payroll day.
2. **Forecast** each series for the next 30 days (the project variable `horizon_days`). The calendar is
   known in advance, so it is used as a feature: the 30 future days are in the data with an empty amount.
3. **Test**: the last 30 known days are hidden, forecast, and compared with what happened.
   **MASE** = error ÷ error of "same as the day before"; below 1 is better than that.
4. **Balance path**: yesterday's balance + running total of (in − out), day by day. The **pessimistic
   balance** is the 1-in-10 bad case (daily ranges added up as variances).
5. **Risk**: **HIGH** if the lowest balance goes past the overdraft limit, **MEDIUM** if it goes below 0
   (expected or pessimistic), **LOW** otherwise.

## 3. The Flow: three zones

A project has one Flow; zones group its parts. Datasets used by another zone are *shared* into it.

```
Data prep ──► Training ──► model ──► Inference ──► account_outlook ──► dashboard
    └─────────────────────────────────────┘ (daily series, read by both)
```

| Zone | Visual project | Code project |
|---|---|---|
| **Data prep** | Prepare (rules) → Group → Join × 2 → Prepare → `daily_flows_scoring` | Python → SQL → Python → `daily_flows` |
| **Training** | `daily_flows` → Lab model → saved model `cashflow_forecast_model` | Python → folder `forecast_model` (`model.pkl`, `model_info.json`) + `model_evaluation` |
| **Inference** | Score → `forecast` → Group, Window, Join, Prepare → `daily_balance_path` → `account_outlook` | Python (load model) → `forecast` → Python → `daily_balance_path`, `account_outlook` |

The model's rebuild behavior is **explicit**: building the outputs never retrains it; only a build of
the model itself does.

**Outputs**

| Dataset | One row per | Main columns |
|---|---|---|
| `forecast` | account, flow, future day | forecast + range (10% / 90%) |
| `daily_balance_path` | account, future day | inflow, outflow, balance, pessimistic_balance |
| `account_outlook` | account | opening / closing balance, total_in, total_out, lowest_balance + date, pessimistic_lowest_balance, risk_level |

## 4. Running it

A **job** is one run of the Flow. A **scenario** is a saved list of steps (each build step creates a
job); it stops at the first failed step.

| Scenario | Steps | When |
|---|---|---|
| **Prepare data** | check the inputs → build the Data prep zone | first step of the two others |
| **Train model** | Prepare data → retrain the model (Training zone) | about monthly |
| **Daily forecast** | Prepare data → check the model → build the Inference zone → outlook metrics | every day |

**Checks**

- **Input rules** (dataset > Data Quality): transactions not empty, no empty field, unique `txn_id`, only
  known accounts, up to the balances' date; accounts unique with balance and limit filled, one balance
  date; calendar unique days covering the next 30.
- **Model check**: the model exists, was trained in the last **45 days**, has **MASE < 1**, and forecasts
  `horizon_days` days.
- **Metrics** on `account_outlook` (with history): HIGH and MEDIUM accounts, lowest balance ahead.

**Dashboard** "Cash-flow outlook": run buttons, the metrics, lowest balance per company, balance per day,
the outlook table.

The Free Edition has no scheduled triggers and does not let the API or `dsscli` start a scenario: run
scenarios by hand, from the dashboard, or from the external dashboard ([`../outlook/`](../outlook/README.md)),
which opens the scenario in DSS and follows the run.

## 5. In the DSS UI

| To | Do |
|---|---|
| Run a scenario | ▷ (top bar) → Scenarios → open it → **Run**; results in **Last runs** |
| Run one step | double-click the recipe → **Run** |
| Build a dataset | right-click it → **Build…** → only this / upstream / downstream |
| Build a zone | **▶ BUILD** on the zone header |
| Clear a dataset | click it once → right panel **Actions** → **Clear data** (building refills it) |
| See why a run failed | ▷ → Jobs → the red job → the red step → logs |
| See checks and metrics | dataset → **Data Quality** / **Metrics** |
| Roll back the model (visual) | open the saved model → make another version active |

## 6. When something fails

| Message | Cause | Fix |
|---|---|---|
| A rule in error in *Prepare data* | inputs incomplete (e.g. yesterday's transactions missing) | fix or reload the input, run again |
| "No trained model…" / "no active version" | model missing or cleared | run **Train model** |
| "trained N days ago (limit 45)" | model too old | run **Train model** |
| "MASE is ≥ 1" | model no better than the naive guess | retrain; compare algorithms in the Lab |
| "forecasts N days, horizon_days is M" | horizon changed in one place only | align the Lab horizon (visual) and retrain |
| "No timeseries has enough future values" | future rows missing from the scoring data | check `calendar` covers the next 30 days |

## 7. Changing things

Everything is defined in code and pushed with `python3 create_project.py`. That **deletes and recreates
the project**: changes made only in the UI are lost.

| To change | Edit |
|---|---|
| Horizon | `HORIZON` (visual) / variable `horizon_days` (code), then retrain |
| Description rules | `RULES` in `create_project.py` (visual) / `recipes/clean_transactions.py` (code) |
| Algorithms (visual) | `ALGORITHMS` in `corporate_cashflow/create_project.py` |
| Model features (code) | `corporate_cashflow_code/library/cashflow_model.py` |
| Input rules, metrics, dashboard | `corporate_cashflow/quality.py` (both projects) |
| Model check limits (45 days, MASE 1) | `MODEL_CHECK` in each `create_project.py` |
| Zones, scenarios | end of each `create_project.py` |

Helpers for all of this (recipes, models, zones, scenarios, rules, dashboards): `projects/dssproj.py`,
described in `projects/README.md`.

## Glossary

| Term | Meaning |
|---|---|
| D-1 | yesterday: the data is complete up to it |
| Series | the daily amounts of one account and one flow (IN or OUT) |
| Horizon | how many days ahead are forecast (30) |
| External feature | a column known in advance (the calendar) |
| MASE | error ÷ error of "same as the day before"; < 1 is better |
| Quantile 10% / 90% | a range: 1 chance in 10 to fall below / above |
| Zone | a box grouping part of the Flow |
| Explicit | an item rebuilt only when built by name |
| Active version | the saved-model version used for scoring |
