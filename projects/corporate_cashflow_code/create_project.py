"""Corporate banking, scripted version: the same data and goal as ../corporate_cashflow, built with Python and
SQL recipes and a regression model (gradient boosting) instead of Dataiku's time series forecasting.

Goal: for every account, money in and out for the rest of this month + the next 3 months, the balance day by
day, and per month the lowest balance, the risk level and what drives it. Loan installments (obligations) are
not forecast: they come from the loan schedule.

    python3 ../corporate_cashflow/generate_data.py   # the synthetic CSVs (shared with the visual project)
    python3 create_project.py                         # push project CORP_CASHFLOW_CODE into DSS and build it

recipes/ holds the recipe code (.py = Python, .sql = SQL run inside PostgreSQL), library/ the code shared by
training and scoring, wiki.md the project wiki. Data quality rules, metrics and dashboard are shared with
CORP_CASHFLOW (../corporate_cashflow/quality.py).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "corporate_cashflow" / "data"
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "corporate_cashflow")]
from dssproj import Project, build_step, col, metrics_step, python_step, rules_step, tile_table  # noqa: E402
from quality import (EXTRA_TOP, INPUTS, OUTLOOK_METRIC_NAMES, OUTLOOK_METRICS, add_input_rules,  # noqa: E402
                     outlook_dashboard)

HORIZON_MONTHS = 3
HORIZON_DAYS = 31 * (HORIZON_MONTHS + 1) - 1   # 123: the model forecasts the longest possible window
D, F, I = "dateonly", "double", "int"
CALENDAR = [col(c, I) for c in ("day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end")]

LIBRARY = (HERE / "library" / "cashflow_model.py").read_text()
LIBRARY = re.sub(r"^HORIZON_DAYS = \d+", f"HORIZON_DAYS = {HORIZON_DAYS}", LIBRARY, flags=re.M)


def code(name):
    return (HERE / "recipes" / name).read_text()


p = Project("CORP_CASHFLOW_CODE", "Corporate cash-flow forecast (code)",
            "Same data as CORP_CASHFLOW, built with Python and SQL recipes and a regression model.",
            variables={"horizon_months": str(HORIZON_MONTHS), "bad_case_pct": "0.15"})
p.wiki_home("Start here", (HERE / "wiki.md").read_text())
p.library("cashflow_model.py", LIBRARY)

# Data prep: the inputs, cleaned, as daily series to learn from, and the installments to pay ----------------
p.csv_dataset("accounts", DATA / "accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
    col("balance", F), col("balance_date", D), col("overdraft_limit", F),
], "One row per account: yesterday's closing balance and the overdraft limit.", tags=["1-input"])
p.csv_dataset("transactions", DATA / "transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date", D), col("amount", F), col("description"), col("loan_id"),
], "Every transaction up to yesterday: signed amount, bank description, loan_id on loan installments.",
    tags=["1-input"])
p.csv_dataset("calendar", DATA / "calendar.csv", [col("date", D)] + CALENDAR,
              "One row per day, history and days to forecast, with calendar features known in advance.",
              tags=["1-input"])
p.csv_dataset("loan_schedule", DATA / "loan_schedule.csv", [
    col("account_id"), col("loan_id"), col("due_date", D), col("amount", F),
], "Loan installments, past and future: one row per loan and due date.", tags=["1-input"])

p.pg_dataset("transactions_clean", [
    col("txn_id"), col("account_id"), col("date", "date"), col("amount", F), col("description"), col("loan_id"),
    col("category"), col("flow"), col("excluded", I),
], "Transactions with a category, a flow (IN, EXPENSE or OBLIGATION) and excluded = 1 for one-offs and "
   "internal transfers.", tags=["2-clean"])
p.python_recipe("compute_transactions_clean", ["transactions"], ["transactions_clean"], code("clean_transactions.py"),
                "Give each transaction a category, a flow and an excluded flag.", tags=["2-clean"])

p.pg_dataset("daily_totals", [col("account_id"), col("date", D), col("flow"), col("category"), col("amount", F)],
             "Money in and expenses per account, day and category (days with transactions only).",
             tags=["3-daily"])
p.sql_recipe("compute_daily_totals", ["transactions_clean"], ["daily_totals"], code("daily_totals.sql"),
             "Add up the transactions per account, day, flow and category, inside PostgreSQL.", tags=["3-daily"])

p.pg_dataset("daily_series", [col("account_id"), col("flow"), col("category"), col("date", "date")] + CALENDAR
             + [col("amount", F)],
             "The series to learn from: every account, flow, category and day, 0 on quiet days, empty on the "
             "days to forecast.", tags=["3-daily"])
p.python_recipe("compute_daily_series", ["daily_totals", "calendar", "accounts"], ["daily_series"],
                code("daily_series.py"), "Fill in every day of every series and add the calendar columns.",
                tags=["3-daily"])

p.pg_dataset("daily_obligations", [col("account_id"), col("loan_id"), col("date", "date"), col("amount", F)],
             "Loan installments to pay from D to the end of month 3.", tags=["3-daily"])
p.python_recipe("compute_daily_obligations", ["loan_schedule", "accounts"], ["daily_obligations"],
                code("daily_obligations.py"), "Keep the schedule's installments that fall in the forecast window.",
                tags=["3-daily"])

# Training: the model is saved in a managed folder ---------------------------------------------------------
p.managed_folder("forecast_model", "The trained model (model.pkl) and how good it is (model_info.json); rebuilt "
                 "only by the \"Train model\" scenario.", tags=["4-train"], explicit=True)
p.pg_dataset("model_evaluation", [col("method"), col("mae", F), col("mase", F), col("monthly_wape", F)],
             "The model vs three simple baselines on the last 123 known days.", tags=["4-train"])
p.python_recipe("train_forecast_model", ["daily_series"], ["forecast_model", "model_evaluation"],
                code("train_model.py"), "Test the model on the last 123 known days, then retrain it on all of "
                "the history and save it.", tags=["4-train"])

# Inference: forecast, balance path, months, drivers --------------------------------------------------------
p.pg_dataset("forecast", [col("account_id"), col("flow"), col("category"), col("date", "date"), col("forecast", F)],
             "Forecast amount per account, flow, category and day for the next 123 days.", tags=["5-forecast"])
p.python_recipe("score_forecast", ["daily_series", "forecast_model"], ["forecast"], code("score_forecast.py"),
                "Forecast the next 123 days with the saved model.", tags=["5-forecast"])

MONTH = [col("month"), col("month_index", I)]
AMOUNTS = [col(c, F) for c in ("inflows", "expenses", "obligations", "outflows", "net")]
p.pg_dataset("daily_balance_path", [col("account_id"), col("date", "date")] + MONTH + AMOUNTS
             + [col("balance", F), col("bad_case_balance", F)],
             "Per account and day from D to the end of month 3: amounts, balance and bad-case balance.",
             tags=["6-outlook"])
p.python_recipe("compute_daily_balance_path", ["forecast", "daily_obligations", "accounts"], ["daily_balance_path"],
                code("daily_balance_path.py"), "Add the obligations to the forecast and run the balance day by day.",
                tags=["6-outlook"])

p.pg_dataset("monthly_outlook", [col("account_id"), col("company_name")] + MONTH + [
    col("first_day", "date"), col("last_day", "date"), col("days", I)] + AMOUNTS + [
    col("opening_balance", F), col("closing_balance", F), col("lowest_balance", F), col("lowest_balance_date", "date"),
    col("bad_case_lowest_balance", F), col("overdraft_limit", F), col("risk_level"),
], "Per account and month: totals, balances, lowest point and risk level.", tags=["6-outlook"])
p.python_recipe("compute_monthly_outlook", ["daily_balance_path", "accounts"], ["monthly_outlook"],
                code("monthly_outlook.py"), "Add the days up per month and set the month's risk level.",
                tags=["6-outlook"])

p.pg_dataset("monthly_drivers", [col("account_id")] + MONTH + [
    col("flow"), col("driver"), col("amount", F), col("share", F), col("last_year", F), col("change_pct", F),
    col("rank", I)], "Per account, month and flow: each category or loan, its share and its change vs last year.",
    tags=["6-outlook"])
p.python_recipe("compute_monthly_drivers", ["forecast", "daily_obligations", "transactions_clean", "monthly_outlook"],
                ["monthly_drivers"], code("monthly_drivers.py"),
                "Split each month into categories and loans and compare with the same month last year.",
                tags=["6-outlook"])

# Flow zones; shared = items of another zone that this zone reads -------------------------------------------
p.zone("Data prep", tags=["1-input", "2-clean", "3-daily"])
p.zone("Training", tags=["4-train"], shared=["daily_series"])
p.zone("Inference", tags=["5-forecast", "6-outlook"],
       shared=["daily_series", "forecast_model", "accounts", "daily_obligations", "transactions_clean"])

# Checks: data quality rules on the inputs, metrics on the outlook ------------------------------------------
add_input_rules(p)
p.metrics("monthly_outlook", python=OUTLOOK_METRICS, names=OUTLOOK_METRIC_NAMES)

# Before forecasting: a model exists, is recent, beats "same as the day before" and has the right horizon.
MODEL_CHECK = f"""
import time

import dataiku

MAX_AGE_DAYS, MAX_MASE, HORIZON_DAYS = 45, 1.0, {HORIZON_DAYS}
folder = dataiku.Folder("forecast_model")
if "/model_info.json" not in folder.list_paths_in_partition():
    raise Exception("No trained model in forecast_model: run the Train model scenario.")
info = folder.read_json("model_info.json")
age_days = (time.time() - info["trained_at"]) / 86400
if age_days > MAX_AGE_DAYS:
    raise Exception(f"The model was trained {{age_days:.0f}} days ago (limit {{MAX_AGE_DAYS}}): run Train model.")
if info["mase"] >= MAX_MASE:
    raise Exception(f"The model's MASE is {{info['mase']:.2f}}: no better than 'same as the day before'.")
if info["horizon"] != HORIZON_DAYS:
    raise Exception(f"The model forecasts {{info['horizon']}} days, not {{HORIZON_DAYS}}: run Train model.")
print(f"Model OK: trained {{age_days:.1f}} days ago, MASE {{info['mase']:.3f}}, "
      f"monthly WAPE {{info['monthly_wape']:.3f}}")
"""

# Scenarios (Scenarios tab > Run); each build step stops at its zone's boundary ------------------------------
CHECK_INPUTS = rules_step("Check the inputs", INPUTS)
BUILD_DATA_PREP = build_step("Build the Data prep zone", ["daily_series", "daily_obligations"], zone_only=True)
p.scenario("Train model", [
    CHECK_INPUTS,
    BUILD_DATA_PREP,
    build_step("Train the model (Training zone)", ["forecast_model"], zone_only=True),
], description="Check the inputs, prepare the data, test and retrain the model and save it.")
p.scenario("Daily forecast", [
    CHECK_INPUTS,
    BUILD_DATA_PREP,
    python_step("Check the model", MODEL_CHECK),
    build_step("Forecast, balance path, months, drivers (Inference zone)", ["monthly_drivers"], zone_only=True),
    metrics_step("Compute the outlook metrics", ["monthly_outlook"]),
], description="Check the inputs, prepare the data, check the model, then forecast the next 3 months.")

outlook_dashboard(p, extra=[
    (tile_table("Model vs baselines on the last 123 known days (MASE < 1 = better than the day before; "
                "monthly WAPE = error on the monthly totals)", "model_evaluation"), (0, EXTRA_TOP, 36, 7)),
])

if __name__ == "__main__":
    p.deploy()
