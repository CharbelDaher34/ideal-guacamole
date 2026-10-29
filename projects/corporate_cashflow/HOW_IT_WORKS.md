# How it works: daily cash-flow forecasting in Dataiku

How project `CORP_CASHFLOW` goes from raw bank transactions to a 30-day, day-by-day outlook per account.

---

## Part 1: The time series process

### 1. Why daily, and why IN and OUT

A month can end positive and still go below the overdraft limit on payroll day. So we forecast **every
day**, and money **in** and money **out** separately: a net of 0 can be 500k in and 500k out.

### 2. What a time series is here

One value per day: the total money in (or out) of one account on that day. With 10 accounts × IN/OUT we
have **20 series** side by side in one table:

| account_id | flow | date | amount (example values) |
|---|---|---|---|
| CA1001 | IN | 2026-09-29 | 84,000 |
| CA1001 | OUT | 2026-09-29 | 31,500 |
| CA1001 | OUT | 2026-09-25 | 120,000 (payroll) |
| ... | ... | ... | ... |

- **Time** (`date`): when.
- **Identifiers** (`account_id` + `flow`): which series the row belongs to. Without them, the series mix
  together and training fails with "duplicate timestamps".
- **Target** (`amount`): the value to forecast. Always ≥ 0; the flow says in or out.

### 3. Prepare the data

1. **Categorize** each transaction from its description with simple rules (`SALARY` → payroll,
   `TRF FROM` → customer receipt, `INTERNAL TRF` → internal transfer, ...).
2. **Remove** internal transfers and one-off deals: they are not repeatable cash flow.
3. **Sum per account, flow and day**.
4. **Fill quiet days with 0**: one row per series per day, no gaps.

The categories are only used for cleaning; the forecast is IN and OUT.

### 4. Choose what to forecast

- **Horizon**: 30 days, from today (data is complete up to yesterday, D-1).
- **External features**: columns known in advance, here the calendar: day of month, weekday, weekend,
  month start / end. They let the models learn "payroll leaves around the 25th" or "weekends are quiet"
  without knowing any schedule.

### 5. Split into train and test

The last 30 days of each series are **hidden**. Each algorithm learns on the rest, forecasts them, and is
compared with what really happened.

```
Oct 2023 ─────────────── train ─────────────── 31 Aug 2026 │ test: 1 – 30 Sep 2026 │ forecast: 1 – 30 Oct
```

### 6. Train several algorithms

| Algorithm | Idea in one line |
|---|---|
| Naive | every future day = the last known day (the baseline to beat) |
| Seasonal naive | every future day = same weekday of the last known week |
| ETS, Seasonal LOESS, ARIMA | statistical models of level, trend and weekly pattern |
| Prophet | trend + seasonality + calendar features |
| LightGBM | gradient boosting on past values (30+ days old) + calendar features |
| Feed-forward NN, DeepAR | neural networks that learn from all 20 series together + calendar features |

### 7. Check the metrics

| Metric | Meaning |
|---|---|
| **MASE** (main) | error ÷ error of "same as the day before". **Below 1 = better than that.** |
| MAE | average error per day, in USD |
| RMSE | like MAE, but punishes big misses (a missed payroll) more |
| MSIS | how good the forecast range (10%–90%) is |

MAPE is not useful here: many days are 0, and dividing by 0 breaks it.

MASE results: **DeepAR 0.57** (best), feed-forward 0.70, ETS 0.72, Prophet 0.73, LightGBM 0.77, naive 1.14,
STL 1.14, seasonal naive 1.34, ARIMA 1.44.

### 8. Keep the best and forecast

The best model forecasts 1–30 Oct for each series, with:

- `forecast`: the expected amount of the day;
- `quantile_01` … `quantile_09`: a range (e.g. 10% chance the day is below `quantile_01`).

### 9. Follow the balance day by day

```
balance(day) = yesterday's balance + running total of (forecast in − forecast out)
```

| date | in | out | balance |
|---|---|---|---|
| yesterday | | | 150,000 |
| 1 Oct | 40,000 | 20,000 | 170,000 |
| 23 Oct | 30,000 | 120,000 (payroll) | **−35,000** ← lowest |
| 30 Oct | 60,000 | 10,000 | 25,000 |

- **Pessimistic balance**: the 1-in-10 bad case of the running total. The daily ranges add up as
  variances; we do not add every day's worst case, which would assume all 30 days go wrong at once.
- **Risk**: HIGH if the lowest balance goes past the overdraft limit; MEDIUM if it goes below 0 (expected
  or pessimistic); LOW otherwise.
- Output per account: opening and closing balance, totals in / out, **lowest balance and its date**,
  pessimistic lowest balance, risk level.

---

## Part 2: The Dataiku components

### The building blocks

| Component | What it is | In this project |
|---|---|---|
| **Project** | Container for everything | `CORP_CASHFLOW` |
| **Flow** | The pipeline: datasets → recipes → datasets | one Flow, split in 3 zones |
| **Flow zone** | A box grouping part of the Flow | Data prep, Training, Inference |
| **Dataset** | A table | 3 uploaded CSVs + PostgreSQL tables for every output |
| **Recipe** | A step that reads datasets and writes a dataset | Prepare, Group, Join, Window, Score |
| **Lab (Visual analysis)** | Workspace to train and compare models | 9 algorithms trained on `daily_flows` |
| **Saved model** (green diamond) | The deployed best model in the Flow | `cashflow_forecast_model` |
| **Code environment** | Python with the ML packages | `timeseries` (statsmodels, prophet, gluonts, torch) |
| **Wiki** | Project documentation | "Start here" page |
| **Job** | One run of the Flow | **Build** a dataset to (re)run what feeds it |
| **Scenario** | A saved list of builds, run with one click | Train model, Daily forecast |

### The Flow: one Flow, three zones

A project has **one** Flow. Training and inference are two parts of it, grouped in **zones**:

```
┌─ Data prep ──────────────────────────────────────────────────────────────────────────────┐
│ accounts, transactions, calendar ─► transactions_clean ─► daily_totals                   │
│ ─► account_days ─► account_days_totals ─► daily_flows_scoring (history + 30 days)        │
└──────────────┬──────────────────────────────────────┬────────────────────────────────────┘
┌─ Training ───┴──────────────────────┐ ┌─ Inference ─┴────────────────────────────────────┐
│ daily_flows (history only)          │ │ daily_flows_scoring + model ─[Score]─► forecast  │
│ ─[Lab]─► ◆ cashflow_forecast_model  ├─► ─► daily_balance_path ─► account_outlook         │
└─────────────────────────────────────┘ └──────────────────────────────────────────────────┘
```

- **The data prep is shared**, not copied: training and inference must prepare the data the same way.
  A dataset used by another zone is *shared* into it (right-click > Share to a flow zone).
- **The saved model is the link**: training writes a new version, inference uses the active version.
- **The model's rebuild behavior is Explicit** (saved model > Settings): building `account_outlook`
  never retrains it, only a build of the model itself does.

In the UI: **+ Zone** (Flow, top right) creates a zone; select items, right-click > **Move to a flow zone**.

### How each component is set up

**Datasets**
- Uploaded CSVs with their column types: `dateonly` for days, `double` for amounts, `int` for the calendar.
- Managed datasets: PostgreSQL tables `CORP_CASHFLOW_<name>`; schema computed from the recipe.

**Prepare recipe**: a list of steps, applied top to bottom.
- `prepare_transactions`: *Formula* `category` (the description rules, as nested `if(contains(...))`),
  `flow = if(amount > 0, "IN", "OUT")`, `excluded`; *Rename* `booking_date` → `date`.
- `prepare_daily_flows`: *Formula* `days_ahead = diff(date, balance_date, "days")`; *Keep rows*
  `days_ahead <= 30`; *Formula* 0 on quiet days; *Fold* `inflow` / `outflow` into `flow` / `amount` (one row
  per flow); *Formula* empty `amount` on the future days.
- `prepare_balance_path`, `prepare_outlook`: balance, pessimistic balance, risk level.

**Group recipe**: keys + aggregations, optional pre-filter and computed columns.
- `group_daily_totals`: keys `account_id`, `date`; pre-filter `excluded == 0`; sums of `inflow` and
  `outflow` (computed from the sign).
- `group_daily_forecast`: puts IN and OUT side by side per day; `net` and `net_variance`.
- `group_outlook`: per account, totals and lowest balances.

**Join recipe**
- `join_account_days`: **cross join** calendar × accounts = every day for every account.
- `join_daily_totals`: left join on `account_id` + `date`.
- `join_accounts`: finds the day of the lowest balance (`lowest_balance = balance`) and adds the account.

**Window recipe**: `window_running_totals`, partition `account_id`, order `date`, running sums of `net`
and `net_variance`.

**Lab: time series analysis.** From `daily_flows` → **Lab → AutoML Prediction → Time series forecasting**.
Settings under **Design**:

| Design tab | Setting | Value here |
|---|---|---|
| Target | target column | `amount` |
| Time series | time variable | `date` |
| | identifiers ("Multiple time series") | `account_id`, `flow` |
| Forecasting parameters | time step / horizon | 1 day / 30 |
| Train / Test set | test size / gap | 30 steps / 0 |
| Features handling | external features | the 5 calendar columns; any other column rejected |
| Metrics | optimize for | MASE |
| Algorithms | which ones and their settings | the 9 above (e.g. ARIMA s=7, DeepAR context 60) |
| Runtime environment | code env | `timeseries` |

**Train** → **Result** shows every model with its metrics, the forecast chart per series and the metrics
per series. Pick a model → **Deploy** puts it in the Flow as a saved model.

**Scenarios** (Scenarios tab > **Run**): each step builds items.

| Scenario | Steps | When |
|---|---|---|
| Train model | build `daily_flows`; build `cashflow_forecast_model` only (= retrain, new active version) | monthly |
| Daily forecast | build `account_outlook` and everything upstream (the explicit model is skipped) | every day |

The Free Edition has no time-based triggers: run them by hand or from a cron
(`dsscli scenario-run CORP_CASHFLOW DAILY_FORECAST`).

**Score recipe**: applies the saved model to `daily_flows_scoring`. With external features, the input
must contain the 30 future rows (empty `amount`, calendar filled), otherwise scoring fails with "No
timeseries has enough future values of external features".

### Where the parameters live

Everything is defined in [`create_project.py`](create_project.py):

| To change | Edit in `create_project.py` | Or in the UI |
|---|---|---|
| Horizon | `HORIZON` | Lab → Design → Forecasting parameters (+ the `days_ahead` filter) |
| Description rules | `RULES` | `prepare_transactions` formula |
| Algorithms and their settings | `ALGORITHMS` | Lab → Design → Algorithms |
| External features | `CALENDAR` | Lab → Design → Features handling |
| Risk rule | `prepare_outlook` formulas | Prepare recipe step |
| Zones, scenarios | `p.zone(...)`, `p.scenario(...)` | Flow > zones; Scenarios tab |

> Re-running `create_project.py` deletes and re-creates the project, so changes made only in the UI are
> lost.
