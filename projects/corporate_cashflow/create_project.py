"""Corporate banking: forecast each account's daily money in and normal expenses for the rest of this month
and the next 3 months with Dataiku's time series forecasting, add the loan installments from their schedule,
then follow the balance day by day and month by month to find its lowest point.

    python3 generate_data.py     # (re)create the synthetic CSVs in data/
    python3 create_project.py    # push project CORP_CASHFLOW into DSS and run the whole flow

No code recipes: visual recipes (Prepare, Group, Join, Window, Stack) and Visual ML time series forecasting
(a Lab analysis comparing two algorithms, the best one deployed to the flow + a Score recipe).
The Flow has three zones (Data prep, Training, Inference) and two scenarios (Train model, Daily forecast),
with data quality rules on the inputs, a check of the model before forecasting, metrics on the outlook and a
dashboard. The project wiki is wiki.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from dssproj import (Project, build_step, col, ensure_code_env, formula, keep_columns, keep_rows,  # noqa: E402
                     metrics_step, order_columns, python_step, rename, rules_step, tile_text)
from quality import (EXTRA_TOP, INPUTS, OUTLOOK_METRIC_NAMES, OUTLOOK_METRICS, add_input_rules,  # noqa: E402
                     outlook_dashboard)

CODE_ENV = "timeseries"                 # Python env with DSS's time series packages, created if missing
MONTHS = 3                              # full months after D's month: project variable horizon_months
HORIZON = 31 * (MONTHS + 1) - 1         # 123 days: the model's horizon, the longest rest of a month + 3 months
BAD_CASE_PCT = 0.15                     # bad case: money in 15% lower, expenses 15% higher (variable bad_case_pct)
CALENDAR = ["day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end"]
F, I, D = "double", "int", "dateonly"

# Description rules, checked in order: the first text found in the description gives the category.
# internal_transfer and one_off are not real, repeatable cash flow: they are removed before forecasting.
# Loan installments (loan_id set) are category loan and flow OBLIGATION: taken from the schedule, not forecast.
RULES = [
    ("INTERNAL TRF", "internal_transfer"),
    ("SALE OF", "one_off"), ("LOAN DRAWDOWN", "one_off"), ("CAPITAL INJECTION", "one_off"),
    ("SETTLEMENT PAYMENT", "one_off"),
    ("TRF FROM", "customer_receipt"), ("POS SETTLEMENT", "card_sales"),
    ("PAYMENT TO", "supplier"), ("SALARY", "payroll"), ("RENT", "rent"), ("LOAN INSTALLMENT", "loan"),
    ("BILL", "utilities"), ("VAT PAYMENT", "tax"), ("FEE", "bank_fee"),
]

# Algorithms the Lab trains and compares; the one with the lowest RMSE on the test days is deployed. RMSE, not
# MASE: payroll, rent or tax fall on one day a month, and a forecast of 0 every day has a small MASE (DeepAR
# does that) but misses the month's payroll; RMSE counts those big days. Only 2: each one is slow to train.
ALGORITHMS = {
    "gluonts_torch_deepar_timeseries": {"context_length": [60]},         # DeepAR (recurrent neural network)
    "lightgbm_regression": {"learning_rate": [0.1], "n_estimators": [200], "min_child_samples": [20],
                            "colsample_bytree": [0.9]},                  # gradient boosting on lags + calendar
}


def month_number(date):
    return f'datePart({date}, "year") * 12 + datePart({date}, "month")'


# month_index: 0 = the month of D (first forecast day = balance_date + 1), 1-3 = the next months, < 0 = history
FIRST_DAY = 'inc(balance_date, 1, "days")'
MONTH_INDEX = f"{month_number('date')} - ({month_number(FIRST_DAY)})"
OUTLOOK_DAYS = "days_ahead >= 1 && month_index <= ${horizon_months}"
LAST_YEAR = "month_index >= -12 && month_index <= ${horizon_months} - 12"     # the same months, a year earlier
# to find the day of the lowest balance: the lowest balance, the earliest day if several
LOWEST_ORDER = 'balance + (month_index * 31 + datePart(date, "day")) / 1000.0'
RISK = ('if(lowest_balance < 0 - overdraft_limit, "HIGH", '
        'if(lowest_balance < 0 || bad_case_lowest_balance < 0, "MEDIUM", "LOW"))')


def category_formula():
    """The rules as one nested if(...) formula; an installment (loan_id set) is a loan."""
    expr = '"other"'
    for text, category in reversed(RULES):
        expr = f'if(contains(description, "{text}"), "{category}", {expr})'
    return f'if(!isBlank(loan_id), "loan", {expr})'


def select(*columns):
    """Keep these columns, in this order."""
    return [keep_columns(*columns), order_columns(*columns)]


p = Project("CORP_CASHFLOW", "Corporate cash-flow forecast",
            "Forecast each corporate account's money in and out for the rest of this month and the next 3 "
            "months, loan installments included, and find the lowest balance of each month.",
            variables={"horizon_months": str(MONTHS), "bad_case_pct": str(BAD_CASE_PCT)})
p.wiki_home("Start here", (HERE / "wiki.md").read_text())


def dataset(name, description, tag):
    """Managed PostgreSQL dataset; its schema is computed by DSS from the recipe that writes it."""
    p.pg_dataset(name, [], description, tags=[tag])


# 1. inputs -----------------------------------------------------------------------------------
p.csv_dataset("accounts", HERE / "data/accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
    col("balance", F), col("balance_date", D), col("overdraft_limit", F),
], "10 corporate accounts: closing balance on balance_date (yesterday, D-1) and overdraft limit.",
    tags=["1-input"])
p.csv_dataset("transactions", HERE / "data/transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date", D), col("amount", F), col("description"), col("loan_id"),
], "Every transaction up to yesterday: signed amount (+ in, - out), bank description, loan_id on installments.",
    tags=["1-input"])
p.csv_dataset("calendar", HERE / "data/calendar.csv", [col("date", D)] + [col(c, I) for c in CALENDAR],
              "One row per day, history and days to forecast, with calendar features known in advance.",
              tags=["1-input"])
p.csv_dataset("loan_schedule", HERE / "data/loan_schedule.csv", [
    col("account_id"), col("loan_id"), col("due_date", D), col("amount", F),
], "Loan repayment schedule: one row per installment (past and future), amount > 0.", tags=["1-input"])

# 2. data prep: clean transactions, the daily series, the installments ahead ------------------------
dataset("transactions_clean", "transactions + category, flow (IN / EXPENSE / OBLIGATION) and excluded.", "2-prep")
p.prepare_recipe("prepare_transactions", "transactions", "transactions_clean", [
    formula("category", category_formula()),
    formula("flow", 'if(!isBlank(loan_id), "OBLIGATION", if(amount > 0, "IN", "EXPENSE"))'),
    formula("excluded", 'if(category == "internal_transfer" || category == "one_off", 1, 0)'),
    rename({"booking_date": "date"}),
], "Description rules give the category; an installment is OBLIGATION, otherwise the sign gives IN / EXPENSE.",
    tags=["2-prep"])

dataset("daily_totals", "Amount per account, flow, category and day, on the days with transactions.", "2-prep")
p.group_recipe("group_daily_totals", "transactions_clean", "daily_totals", ["account_id", "flow", "category", "date"],
               {"value": ["sum"]}, count=False, pre_filter='excluded == 0 && flow != "OBLIGATION"',
               computed={"value": (F, "abs(amount)")}, names={"value_sum": "amount"},
               description="Add up the transactions of each day, without installments and excluded rows.",
               tags=["2-prep"])

dataset("series", "The time series to forecast: one row per account, flow and category.", "2-prep")
p.group_recipe("group_series", "daily_totals", "series", ["account_id", "flow", "category"], {},
               names={"count": "active_days"}, description="Every account, flow and category of the history.",
               tags=["2-prep"])

dataset("account_days", "Every calendar day for every account, with its balance of yesterday.", "2-prep")
p.join_recipe("join_account_days", ["calendar", "accounts"], "account_days", [], how="CROSS",
              columns={"accounts": ["account_id", "balance_date", "balance"]},
              description="Calendar x accounts.", tags=["2-prep"])

dataset("account_calendar", "account_days up to the model's horizon, with days_ahead (1 = D), month and "
        "month_index (0 = D's month, 1-3 the next ones, < 0 history).", "2-prep")
p.prepare_recipe("prepare_account_calendar", "account_days", "account_calendar", [
    formula("days_ahead", 'diff(date, balance_date, "days")'),
    keep_rows(f"days_ahead <= {HORIZON}"),
    formula("month", 'substring(toString(date), 0, 7)'),
    formula("month_index", MONTH_INDEX),
    *select("account_id", "date", "days_ahead", "month", "month_index", "balance", *CALENDAR),
], "Number the days from yesterday's balance and the calendar months from D's month.", tags=["2-prep"])

dataset("series_days", "Every series on every day of account_calendar.", "2-prep")
p.join_recipe("join_series_days", ["account_calendar", "series"], "series_days", ["account_id"], how="INNER",
              columns={"series": ["flow", "category"]},
              description="Each account's days x its series.", tags=["2-prep"])

dataset("series_days_amounts", "series_days + the day's amount (empty when nothing happened).", "2-prep")
p.join_recipe("join_daily_totals", ["series_days", "daily_totals"], "series_days_amounts",
              ["account_id", "flow", "category", "date"], columns={"daily_totals": ["amount"]},
              description="Add each day's total.", tags=["2-prep"])

dataset("daily_series", f"The time series: amount per account, flow, category and day up to yesterday, then the "
        f"next {HORIZON} days with an empty amount, and the calendar features.", "2-prep")
p.prepare_recipe("prepare_daily_series", "series_days_amounts", "daily_series", [
    formula("amount", "if(days_ahead <= 0 && isBlank(amount), 0, amount)"),
    *select("account_id", "flow", "category", "date", "amount", *CALENDAR),
], "0 on the quiet days of the history; the future days keep an empty amount (to forecast).", tags=["2-prep"])

dataset("daily_obligations", "Loan installments due from D to the end of month horizon_months.", "2-prep")
p.join_recipe("join_obligations", ["account_calendar", "loan_schedule"], "daily_obligations",
              ["account_id", ("date", "due_date")], how="INNER", pre_filters={"account_calendar": OUTLOOK_DAYS},
              columns={"account_calendar": ["account_id", "date"], "loan_schedule": ["loan_id", "amount"]},
              description="The schedule's installments due on an outlook day.", tags=["2-prep"])

# 3. training: the time series model -----------------------------------------------------------
dataset("training_series", "daily_series up to yesterday: the training data.", "3-train")
p.prepare_recipe("prepare_training", "daily_series", "training_series", [keep_rows("!isBlank(amount)")],
                 "Keep the history (amount known).", tags=["3-train"])

p.timeseries_model("cashflow_forecast_model", "training_series", target="amount", time="date",
                   ids=["account_id", "flow", "category"], horizon=HORIZON, algorithms=ALGORITHMS, env=CODE_ENV,
                   external=CALENDAR, metric="RMSE", explicit=True, skip_reports=True,
                   description=f"Daily amount per account, flow and category, {HORIZON} days ahead, tested on the "
                               "last days of the history, calendar columns as external features. The best "
                               "algorithm (RMSE) is deployed; retrained only by the Train model scenario.",
                   tags=["3-train"])

# 4. inference: forecast, then everything that moves money on each outlook day ----------------------
dataset("scored_series", "daily_series + the forecast of the future days.", "4-forecast")
p.score_recipe("score_series", "cashflow_forecast_model", "daily_series", "scored_series",
               description=f"Forecast the next {HORIZON} days with the active model version.", tags=["4-forecast"])

dataset("forecast", "Per account, flow, category and future day: the forecast amount (>= 0).", "4-forecast")
p.prepare_recipe("prepare_forecast", "scored_series", "forecast", [
    keep_rows("!isBlank(forecast)"),
    formula("forecast", "max(forecast, 0)"),
    *select("account_id", "flow", "category", "date", "forecast"),
], "Future days only; a negative forecast counts as 0.", tags=["4-forecast"])

dataset("cash_flows", "All the amounts in one table: forecast, installments and last year's transactions "
        "(source = the dataset of the row).", "4-forecast")
p.stack_recipe("stack_cash_flows", ["forecast", "daily_obligations", "transactions_clean"], "cash_flows",
               origin="source", pre_filters={"transactions_clean": "excluded == 0"},
               description="Forecast, installments and kept transactions, one under the other.", tags=["4-forecast"])

dataset("cash_flows_months", "cash_flows of the outlook days and of the same months last year, with month, "
        "month_index and yesterday's balance.", "4-forecast")
p.join_recipe("join_months", ["cash_flows", "account_calendar"], "cash_flows_months", ["account_id", "date"],
              how="INNER", pre_filters={"account_calendar": f"({OUTLOOK_DAYS}) || ({LAST_YEAR})"},
              columns={"cash_flows": ["source", "account_id", "date", "flow", "category", "loan_id", "forecast",
                                      "amount"], "account_calendar": ["month", "month_index", "balance"]},
              description="Keep D to the end of month horizon_months (the model's extra days are dropped) and "
                          "the same months last year.", tags=["4-forecast"])

# 5. balance path and months ----------------------------------------------------------------------
dataset("daily_cash_flows", "Per account and outlook day: inflows, expenses, obligations, net and bad-case net.",
        "5-outlook")
p.group_recipe("group_days", "cash_flows_months", "daily_cash_flows",
               ["account_id", "date", "month", "month_index", "balance"],
               {"inflow": ["sum"], "expense": ["sum"], "obligation": ["sum"], "net": ["sum"], "bad_case": ["sum"]},
               count=False, pre_filter="month_index >= 0",
               computed={
                   "inflow": (F, 'if(flow == "IN", round(forecast), 0)'),
                   "expense": (F, 'if(flow == "EXPENSE", round(forecast), 0)'),
                   "obligation": (F, 'if(source == "daily_obligations", round(amount), 0)'),
                   "net": (F, 'if(flow == "IN", round(forecast), 0 - round(if(source == "forecast", forecast, '
                              'amount)))'),
                   "bad_case": (F, 'if(flow == "IN", forecast * (1 - ${bad_case_pct}), if(flow == "EXPENSE", '
                                   '0 - forecast * (1 + ${bad_case_pct}), 0 - amount))'),
               },
               names={"inflow_sum": "inflows", "expense_sum": "expenses", "obligation_sum": "obligations",
                      "net_sum": "net", "bad_case_sum": "bad_case_net"},
               description="Add up each day (balance = yesterday's balance, the same on every row of an account).",
               tags=["5-outlook"])

dataset("daily_running", "daily_cash_flows + the running totals of net and bad-case net since D.", "5-outlook")
p.window_recipe("window_running_totals", "daily_cash_flows", "daily_running", ["account_id"], ["date"],
                {"net": {"sum": True}, "bad_case_net": {"sum": True}},
                names={"net_sum": "net_running", "bad_case_net_sum": "bad_case_running"},
                description="Cumulative sums per account, in date order.", tags=["5-outlook"])

dataset("daily_balance_path", "Per account and outlook day: inflows, expenses, obligations, outflows, net, "
        "balance and bad-case balance.", "5-outlook")
p.prepare_recipe("prepare_balance_path", "daily_running", "daily_balance_path", [
    formula("outflows", "expenses + obligations"),
    formula("bad_case_balance", "round(balance + bad_case_running)"),
    formula("balance", "balance + net_running"),
    *select("account_id", "date", "month", "month_index", "inflows", "expenses", "obligations", "outflows", "net",
            "balance", "bad_case_balance"),
], "balance = yesterday's balance + running net. Bad case: money in bad_case_pct lower, expenses bad_case_pct "
   "higher, installments as scheduled.", tags=["5-outlook"])

dataset("monthly_totals", "Per account and month: days, totals, closing balance, lowest balance and its day.",
        "5-outlook")
p.group_recipe("group_months", "daily_balance_path", "monthly_totals", ["account_id", "month", "month_index"],
               {"date": ["min", "max", "count", "first"], "inflows": ["sum"], "expenses": ["sum"],
                "obligations": ["sum"], "outflows": ["sum"], "net": ["sum"], "balance": ["min", "last"],
                "bad_case_balance": ["min"]}, count=False,
               computed={"lowest_order": (F, LOWEST_ORDER)}, order={"date": "lowest_order", "balance": "date"},
               names={"date_min": "first_day", "date_max": "last_day", "date_count": "days",
                      "date_first": "lowest_balance_date", "inflows_sum": "inflows", "expenses_sum": "expenses",
                      "obligations_sum": "obligations", "outflows_sum": "outflows", "net_sum": "net",
                      "balance_min": "lowest_balance", "balance_last": "closing_balance",
                      "bad_case_balance_min": "bad_case_lowest_balance"},
               description="Add up the days of each month; closing = the last day's balance; the day of the lowest "
                           "balance (the first one if several).", tags=["5-outlook"])

dataset("monthly_totals_accounts", "monthly_totals + company name and overdraft limit.", "5-outlook")
p.join_recipe("join_accounts", ["monthly_totals", "accounts"], "monthly_totals_accounts", ["account_id"],
              columns={"accounts": ["company_name", "overdraft_limit"]},
              description="Add the account details.", tags=["5-outlook"])

dataset("monthly_outlook", "One row per account and month: totals, opening / closing / lowest balance and the "
        "risk level.", "5-outlook")
p.prepare_recipe("prepare_monthly_outlook", "monthly_totals_accounts", "monthly_outlook", [
    formula("opening_balance", "closing_balance - net"),
    formula("risk_level", RISK),
    *select("account_id", "company_name", "month", "month_index", "first_day", "last_day", "days", "inflows",
            "expenses", "obligations", "outflows", "net", "opening_balance", "closing_balance", "lowest_balance",
            "lowest_balance_date", "bad_case_lowest_balance", "overdraft_limit", "risk_level"),
], "Opening = closing - net. HIGH: the lowest balance goes past the overdraft limit; MEDIUM: below 0, expected "
   "or bad case; LOW otherwise.", tags=["5-outlook"])

# 6. drivers: what makes each month, per category and loan, vs the same month last year -----------------
dataset("driver_months", "Per account, month, flow and driver (category, or loan_id for OBLIGATION): amount "
        "this year and the same month last year.", "6-drivers")
p.group_recipe("group_drivers", "cash_flows_months", "driver_months", ["account_id", "bucket", "driver_flow", "driver"],
               {"this_year": ["sum"], "last_year": ["sum"]}, count=False,
               computed={
                   "bucket": (I, "if(month_index < 0, month_index + 12, month_index)"),
                   "driver_flow": ("string", 'if(isBlank(loan_id), flow, "OBLIGATION")'),
                   "driver": ("string", "if(isBlank(loan_id), category, loan_id)"),
                   "this_year": (F, 'if(month_index < 0, 0, round(if(source == "forecast", forecast, amount)))'),
                   "last_year": (F, "if(month_index < 0, abs(amount), null)"),
               },
               names={"this_year_sum": "amount", "last_year_sum": "last_year_total"},
               description="bucket = the outlook month a row counts for (last year's month + 12).",
               tags=["6-drivers"])

dataset("driver_details", "driver_months + the month's dates and totals from monthly_outlook.", "6-drivers")
p.join_recipe("join_driver_months", ["driver_months", "monthly_outlook"], "driver_details",
              ["account_id", ("bucket", "month_index")], how="INNER",
              columns={"driver_months": ["account_id", "driver_flow", "driver", "amount", "last_year_total"],
                       "monthly_outlook": ["month", "month_index", "days", "last_day", "inflows", "expenses",
                                           "obligations"]},
              description="Add the month and its totals.", tags=["6-drivers"])

dataset("driver_ranked", "driver_details + rownumber: 1 = the largest driver of its account, month and flow.",
        "6-drivers")
p.window_recipe("window_driver_rank", "driver_details", "driver_ranked", ["account_id", "month", "driver_flow"],
                ["amount", "driver"], {}, desc=["amount"], ranks=["rowNumber"],
                description="Rank the drivers by amount.", tags=["6-drivers"])

dataset("monthly_drivers", "One row per account, month, flow and driver: amount, share of the flow, the same "
        "month last year and the change.", "6-drivers")
p.prepare_recipe("prepare_drivers", "driver_ranked", "monthly_drivers", [
    rename({"driver_flow": "flow", "rownumber": "rank"}),
    formula("flow_total", 'if(flow == "IN", inflows, if(flow == "EXPENSE", expenses, obligations))'),
    formula("share", "if(flow_total > 0, round(amount / flow_total * 1000) / 1000, 0)"),
    # last year's month, scaled to the days of the month in the outlook (month 0 can be shorter)
    formula("last_year", 'if(last_year_total > 0, round(last_year_total * days / datePart(last_day, "day")), 0)'),
    formula("change_pct", "round((amount - last_year) / if(last_year > 0, last_year, null) * 1000) / 1000"),
    *select("account_id", "month", "month_index", "flow", "driver", "amount", "share", "last_year", "change_pct",
            "rank"),
], "share of the flow's month total (0-1); change_pct as a fraction (0.25 = +25%), empty without last year.",
    tags=["6-drivers"])

# Flow zones -----------------------------------------------------------------------------------------
p.zone("Data prep", tags=["1-input", "2-prep"])
p.zone("Training", tags=["3-train"], shared=["daily_series"])
p.zone("Inference", tags=["4-forecast", "5-outlook", "6-drivers"],
       shared=["daily_series", "cashflow_forecast_model", "accounts", "account_calendar", "daily_obligations",
               "transactions_clean"])

# Checks: data quality rules on the inputs, metrics on the outlook ----------------------------------
add_input_rules(p)
p.metrics("monthly_outlook", python=OUTLOOK_METRICS, names=OUTLOOK_METRIC_NAMES)

# Before forecasting: the model exists, is recent, beats "same as the day before" and forecasts HORIZON days.
MODEL_CHECK = f"""
import time

import dataiku

MODEL, MAX_AGE_DAYS, MAX_MASE, HORIZON = "cashflow_forecast_model", 45, 1.0, {HORIZON}
project = dataiku.api_client().get_default_project()
found = [m["id"] for m in project.list_saved_models() if m["name"] == MODEL]
if not found:
    raise Exception(f"No saved model {{MODEL}}: run the Train model scenario.")
model = project.get_saved_model(found[0])
active = model.get_active_version()
if not active:
    raise Exception(f"{{MODEL}} has no active version: run the Train model scenario.")
details = model.get_version_details(active["id"])
snippet = details.get_raw_snippet()
age_days = (time.time() * 1000 - snippet["trainDate"]) / 86400000
if age_days > MAX_AGE_DAYS:
    raise Exception(f"The model was trained {{age_days:.0f}} days ago (limit {{MAX_AGE_DAYS}}): run Train model.")
if snippet["mase"] >= MAX_MASE:
    raise Exception(f"The model's MASE is {{snippet['mase']:.2f}}: no better than 'same as the day before'.")
trained_for = details.get_raw()["coreParams"]["predictionLength"]
if trained_for != HORIZON:
    raise Exception(f"The model forecasts {{trained_for}} days, not {{HORIZON}}: retrain it in the Lab.")
print(f"Model OK: trained {{age_days:.1f}} days ago, MASE {{snippet['mase']:.3f}}, {{trained_for}} days ahead")
"""

# Scenarios (Scenarios tab > Run) ---------------------------------------------------------------------
DATA_PREP = ["daily_series", "daily_obligations"]
p.scenario("Train model", [
    rules_step("Check the inputs", INPUTS),
    build_step("Build the Data prep zone", DATA_PREP, zone_only=True),
    build_step("Train the model (Training zone)", ["cashflow_forecast_model"], zone_only=True),
], description="Retrain the deployed algorithm on the latest history: a new active version of the saved model.")
p.scenario("Daily forecast", [
    rules_step("Check the inputs", INPUTS),
    build_step("Build the Data prep zone", DATA_PREP, zone_only=True),
    python_step("Check the model", MODEL_CHECK),
    build_step("Forecast, balance path, months and drivers (Inference zone)", ["monthly_outlook", "monthly_drivers"],
               zone_only=True),
    metrics_step("Compute the outlook metrics", ["monthly_outlook"]),
], description="Forecast the next months with the active model version (the model is not retrained).")

outlook_dashboard(p, extra=[
    (tile_text("**Model comparison**: Lab > cashflow_forecast_model (MASE per algorithm, forecast charts)."),
     (0, EXTRA_TOP, 36, 2)),
])

if __name__ == "__main__":
    ensure_code_env(CODE_ENV)
    p.deploy()
