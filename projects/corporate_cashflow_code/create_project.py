"""Corporate banking, scripted version: same data and goal as ../corporate_cashflow (daily money in and out
for the next 30 days, balance day by day, lowest point), built with Python and SQL recipes and a regression
model (gradient boosting) instead of Dataiku's time series forecasting.

    python3 ../corporate_cashflow/generate_data.py   # the synthetic CSVs (shared with the visual project)
    python3 create_project.py                         # push project CORP_CASHFLOW_CODE into DSS and build it

Recipe code lives in recipes/ (.py = Python recipe, .sql = SQL recipe run inside PostgreSQL),
the project wiki in wiki.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "corporate_cashflow" / "data"
sys.path.insert(0, str(HERE.parent))
from dssproj import Project, col  # noqa: E402

D, F, I = "dateonly", "double", "int"
CALENDAR = [col(c, I) for c in ("day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end")]


def code(name):
    return (HERE / "recipes" / name).read_text()


p = Project("CORP_CASHFLOW_CODE", "Corporate cash-flow forecast (code)",
            "Same data as CORP_CASHFLOW, built with Python and SQL recipes and a regression model.",
            variables={"horizon_days": "30"})
p.wiki_home("Start here", (HERE / "wiki.md").read_text())

# 1. input ------------------------------------------------------------------------------------
p.csv_dataset("accounts", DATA / "accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
    col("balance", F), col("balance_date", D), col("overdraft_limit", F),
], "10 corporate accounts: closing balance on balance_date (yesterday, D-1) and overdraft limit.",
    tags=["1-input"])
p.csv_dataset("transactions", DATA / "transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date", D), col("amount", F), col("description"),
], "Every transaction up to yesterday: signed amount (+ in, - out) and the bank description.", tags=["1-input"])
p.csv_dataset("calendar", DATA / "calendar.csv", [col("date", D)] + CALENDAR,
              "One row per day (history and the days to forecast) with calendar features known in advance.",
              tags=["1-input"])

# 2. clean (Python) ---------------------------------------------------------------------------
p.pg_dataset("transactions_clean", [
    col("txn_id"), col("account_id"), col("date", "date"), col("amount", F), col("description"),
    col("category"), col("flow"), col("excluded", I),
], "transactions + category (description rules), flow (IN/OUT), excluded (1 = internal transfer or one-off).",
    tags=["2-clean"])
p.python_recipe("compute_transactions_clean", ["transactions"], ["transactions_clean"], code("clean_transactions.py"),
                "Description rules -> category; sign -> flow; flag what is not repeatable cash flow.",
                tags=["2-clean"])

# 3. daily series (SQL, then Python) -----------------------------------------------------------
p.pg_dataset("daily_totals", [col("account_id"), col("date", D), col("flow"), col("amount", F)],
             "Money in / out per account and day, without the excluded rows (days with transactions only).",
             tags=["3-daily"])
p.sql_recipe("compute_daily_totals", ["transactions_clean"], ["daily_totals"], code("daily_totals.sql"),
             "GROUP BY account, day and flow inside PostgreSQL.", tags=["3-daily"])

p.pg_dataset("daily_flows", [col("account_id"), col("flow"), col("date", "date")] + CALENDAR + [col("amount", F)],
             "The time series: amount per account, flow and day (0 on quiet days) up to yesterday, plus the next "
             "horizon_days days with an empty amount and their calendar features.", tags=["3-daily"])
p.python_recipe("compute_daily_flows", ["daily_totals", "calendar", "accounts"], ["daily_flows"],
                code("daily_flows.py"), "Every account x flow x day; 0 when nothing happened; future days empty.",
                tags=["3-daily"])

# 4. forecast (Python, scikit-learn) ------------------------------------------------------------
p.pg_dataset("forecast", [col("account_id"), col("flow"), col("date", "date"), col("forecast", F),
                          col("forecast_low", F), col("forecast_high", F)],
             "Per account, flow and future day: forecast and its range (10% / 90%).", tags=["4-forecast"])
p.pg_dataset("model_evaluation", [col("method"), col("mae", F), col("mase", F)],
             "Error on the last 30 known days: the model vs three simple baselines.", tags=["4-forecast"])
p.python_recipe("compute_forecast", ["daily_flows"], ["forecast", "model_evaluation"], code("forecast.py"),
                "Gradient boosting on calendar + past values: test on the last 30 days, then forecast the next 30.",
                tags=["4-forecast"])

# 5. balance path and outlook (Python) --------------------------------------------------------
p.pg_dataset("daily_balance_path", [col("account_id"), col("date", "date"), col("inflow", F), col("outflow", F),
                                    col("balance", F), col("pessimistic_balance", F)],
             "Per account and future day: forecast in, out, balance and pessimistic balance.", tags=["5-outlook"])
p.pg_dataset("account_outlook", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"), col("opening_balance", F),
    col("overdraft_limit", F), col("total_in", F), col("total_out", F), col("closing_balance", F),
    col("lowest_balance", F), col("lowest_balance_date", "date"), col("pessimistic_lowest_balance", F),
    col("risk_level"),
], "One row per account: the next 30 days in numbers, lowest balance and its date, risk level.",
    tags=["5-outlook"])
p.python_recipe("compute_balance_outlook", ["forecast", "accounts"], ["daily_balance_path", "account_outlook"],
                code("balance_outlook.py"), "Running balance from yesterday's balance; lowest point; risk level.",
                tags=["5-outlook"])

if __name__ == "__main__":
    p.deploy()
