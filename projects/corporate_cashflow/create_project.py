"""Corporate banking: forecast each account's daily money in and money out for the next 30 days with
Dataiku's time series forecasting, then follow the balance day by day to find its lowest point.

    python3 generate_data.py     # (re)create the synthetic CSVs in data/
    python3 create_project.py    # push project CORP_CASHFLOW into DSS and run the whole flow

No code recipes: visual recipes (Prepare, Group, Join, Window) and Visual ML time series forecasting
(a Lab analysis comparing several algorithms, the best one deployed to the flow + a Score recipe).
The Flow has three zones (Data prep, Training, Inference) and two scenarios: "Train model" retrains the
saved model, "Daily forecast" scores the latest data with it. The project wiki is wiki.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dssproj import (Project, col, ensure_code_env, fold, formula, keep_columns, keep_rows,  # noqa: E402
                     rename)

CODE_ENV = "timeseries"                 # Python env with DSS's time series packages, created if missing
HORIZON = 30                            # days to forecast, from D (data is complete up to D-1)
CALENDAR = ["day_of_month", "day_of_week", "is_weekend", "is_month_start", "is_month_end"]
F, I, D = "double", "int", "dateonly"

# Description rules, checked in order: the first text found in the description gives the category.
# internal_transfer and one_off are not real, repeatable cash flow: they are removed before forecasting.
RULES = [
    ("INTERNAL TRF", "internal_transfer"),
    ("SALE OF", "one_off"), ("LOAN DRAWDOWN", "one_off"), ("CAPITAL INJECTION", "one_off"),
    ("SETTLEMENT PAYMENT", "one_off"),
    ("TRF FROM", "customer_receipt"), ("POS SETTLEMENT", "card_sales"),
    ("PAYMENT TO", "supplier"), ("SALARY", "payroll"), ("RENT", "rent"), ("LOAN INSTALLMENT", "loan"),
    ("BILL", "utilities"), ("VAT PAYMENT", "tax"), ("FEE", "bank_fee"),
]

# Algorithms the Lab trains and compares; the one with the lowest MASE on the test days is deployed.
ALGORITHMS = {
    "trivial_identity_timeseries": {},                                   # naive: repeat the last day
    "seasonal_naive_timeseries": {"season_length": 7},                   # repeat the same weekday last week
    "ets_timeseries": {"trend": ["add"], "seasonal": ["add"], "seasonal_periods": 7},  # exponential smoothing
    "seasonal_loess_timeseries": {"period": 7},                          # STL: trend + weekly pattern
    "arima_timeseries": {"p": 2, "D": 1, "s": 7},                        # ARIMA with a weekly difference
    "prophet_timeseries": {},                                            # Prophet (Meta) + calendar features
    "lightgbm_regression": {"learning_rate": [0.1], "n_estimators": [200], "min_child_samples": [20],
                            "colsample_bytree": [0.9]},                  # gradient boosting on lags + calendar
    "gluonts_torch_simple_feed_forward_timeseries": {"context_length": [60], "learning_rate": [0.001]},
    "gluonts_torch_deepar_timeseries": {"context_length": [60]},         # DeepAR (recurrent neural network)
}


def category_formula():
    """The rules as one nested if(...) formula."""
    expr = '"other"'
    for text, category in reversed(RULES):
        expr = f'if(contains(description, "{text}"), "{category}", {expr})'
    return expr


p = Project("CORP_CASHFLOW", "Corporate cash-flow forecast",
            "Forecast each corporate account's daily money in and out for the next 30 days and find the "
            "lowest balance ahead.")
p.wiki_home("Start here", (HERE / "wiki.md").read_text())


def dataset(name, description, tag):
    """Managed PostgreSQL dataset; its schema is computed by DSS from the recipe that writes it."""
    p.pg_dataset(name, [], description, tags=[tag])


# 1. input ------------------------------------------------------------------------------------
p.csv_dataset("accounts", HERE / "data/accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
    col("balance", F), col("balance_date", D), col("overdraft_limit", F),
], "10 corporate accounts: closing balance on balance_date (yesterday, D-1) and overdraft limit.",
    tags=["1-input"])
p.csv_dataset("transactions", HERE / "data/transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date", D), col("amount", F), col("description"),
], "Every transaction up to yesterday: signed amount (+ in, - out) and the bank description.", tags=["1-input"])
p.csv_dataset("calendar", HERE / "data/calendar.csv", [col("date", D)] + [col(c, I) for c in CALENDAR],
              "One row per day (history and the days to forecast) with calendar features known in advance.",
              tags=["1-input"])

# 2. clean: categories from the description, transfers and one-offs flagged ----------------------
dataset("transactions_clean", "transactions + category (description rules), flow (IN/OUT) and excluded "
        "(1 = internal transfer or one-off).", "2-clean")
p.prepare_recipe("prepare_transactions", "transactions", "transactions_clean", [
    formula("category", category_formula()),
    formula("flow", 'if(amount > 0, "IN", "OUT")'),
    formula("excluded", 'if(category == "internal_transfer" || category == "one_off", 1, 0)'),
    rename({"booking_date": "date"}),
], "Description rules -> category; sign -> flow; flag what is not repeatable cash flow.", tags=["2-clean"])

# 3. daily series: one row per account, flow and day (0 when nothing happened) ------------------
dataset("daily_totals", "Money in and money out per account and day (days with transactions only).", "3-daily")
p.group_recipe("group_daily_totals", "transactions_clean", "daily_totals", ["account_id", "date"],
               {"inflow": ["sum"], "outflow": ["sum"]}, count=False, pre_filter="excluded == 0",
               computed={"inflow": (F, "if(amount > 0, amount, 0)"), "outflow": (F, "if(amount < 0, 0 - amount, 0)")},
               names={"inflow_sum": "inflow", "outflow_sum": "outflow"},
               description="Sum the kept transactions per account and day.", tags=["3-daily"])

dataset("account_days", "Every calendar day for every account.", "3-daily")
p.join_recipe("join_account_days", ["calendar", "accounts"], "account_days", [], how="CROSS",
              columns={"accounts": ["account_id", "balance_date"]},
              description="Calendar x accounts: one row per account and day.", tags=["3-daily"])

dataset("account_days_totals", "account_days + the day's money in and out (empty when nothing happened).",
        "3-daily")
p.join_recipe("join_daily_totals", ["account_days", "daily_totals"], "account_days_totals", ["account_id", "date"],
              columns={"daily_totals": ["inflow", "outflow"]},
              description="Add the day's totals to each account and day.", tags=["3-daily"])

dataset("daily_flows_scoring", f"The time series: amount per account, flow (IN/OUT) and day, up to yesterday, "
        f"plus the next {HORIZON} days with an empty amount (to forecast) and their calendar features.", "3-daily")
p.prepare_recipe("prepare_daily_flows", "account_days_totals", "daily_flows_scoring", [
    formula("days_ahead", 'diff(date, balance_date, "days")'),
    keep_rows(f"days_ahead <= {HORIZON}"),
    formula("inflow", "if(isBlank(inflow), 0, inflow)"),
    formula("outflow", "if(isBlank(outflow), 0, outflow)"),
    fold(["inflow", "outflow"], "flow", "amount"),
    formula("flow", 'if(flow == "inflow", "IN", "OUT")'),
    formula("amount", 'if(days_ahead > 0, "", amount)'),
    keep_columns("account_id", "flow", "date", "amount", *CALENDAR),
], f"0 on days without transactions; keep up to yesterday + {HORIZON} days; one row per flow (IN/OUT); "
   "amount emptied on the future days.",
    tags=["3-daily"])

# 4. training: the time series model -----------------------------------------------------------
dataset("daily_flows", "daily_flows_scoring up to yesterday: the training data.", "4-train")
p.prepare_recipe("prepare_training", "daily_flows_scoring", "daily_flows", [keep_rows("!isBlank(amount)")],
                 "Keep the history only (amount known).", tags=["4-train"])

p.timeseries_model("cashflow_forecast_model", "daily_flows", target="amount", time="date",
                   ids=["account_id", "flow"], horizon=HORIZON, algorithms=ALGORITHMS, env=CODE_ENV,
                   external=CALENDAR, explicit=True,
                   description=f"Time series forecasting of the daily amount per account and flow, {HORIZON} days "
                               f"ahead, tested on the last {HORIZON} days, calendar columns as external features. "
                               "The Lab analysis compares the algorithms; the best (MASE) is deployed. Retrained "
                               "only by the \"Train model\" scenario (rebuild behavior: explicit).",
                   tags=["4-train"])

# 5. inference: forecast the next days with the saved model --------------------------------------
dataset("scored_flows", "daily_flows_scoring + forecast and quantiles on the future days.", "5-forecast")
p.score_recipe("score_flows", "cashflow_forecast_model", "daily_flows_scoring", "scored_flows",
               description=f"Forecast the next {HORIZON} days from the full history.", tags=["5-forecast"])

dataset("forecast", f"Per account, flow and future day: forecast and its range (quantile_01 = 10% ... "
        "quantile_09 = 90%).", "5-forecast")
p.prepare_recipe("prepare_forecast", "scored_flows", "forecast", [
    keep_rows("!isBlank(forecast)"),
    keep_columns("account_id", "flow", "date", "forecast", *[f"quantile_0{i}" for i in range(1, 10)]),
], "Keep the future days and the forecast columns.", tags=["5-forecast"])

# 6. balance path: yesterday's balance + money in - money out, day by day ----------------------
dataset("daily_forecast", "Per account and future day: forecast money in, money out, net, and the variance of "
        "net (how uncertain the day is, from the 10%-90% ranges).", "6-balance")
p.group_recipe("group_daily_forecast", "forecast", "daily_forecast", ["account_id", "date"],
               {"inflow": ["sum"], "outflow": ["sum"], "net": ["sum"], "net_variance": ["sum"]}, count=False,
               computed={
                   "inflow": (F, 'if(flow == "IN", max(forecast, 0), 0)'),
                   "outflow": (F, 'if(flow == "OUT", max(forecast, 0), 0)'),
                   "net": (F, 'if(flow == "IN", max(forecast, 0), 0 - max(forecast, 0))'),
                   # 10%-90% range = 2.563 standard deviations of a normal distribution
                   "net_variance": (F, "pow((quantile_09 - quantile_01) / 2.563, 2)"),
               },
               names={"inflow_sum": "inflow", "outflow_sum": "outflow", "net_sum": "net",
                      "net_variance_sum": "net_variance"},
               description="Put IN and OUT side by side per day (negative forecasts count as 0).",
               tags=["6-balance"])

dataset("daily_running", "daily_forecast + running totals of net and of its variance since D.", "6-balance")
p.window_recipe("window_running_totals", "daily_forecast", "daily_running", ["account_id"], ["date"],
                {"net": {"sum": True}, "net_variance": {"sum": True}},
                names={"net_sum": "net_running", "net_variance_sum": "variance_running"},
                description="Cumulative sums per account, in date order.", tags=["6-balance"])

dataset("daily_running_accounts", "daily_running + the account's balance yesterday.", "6-balance")
p.join_recipe("join_opening_balance", ["daily_running", "accounts"], "daily_running_accounts", ["account_id"],
              columns={"accounts": ["balance"]}, description="Add yesterday's balance.", tags=["6-balance"])

dataset("daily_balance_path", "Per account and future day: forecast in, out, balance and pessimistic balance.",
        "6-balance")
p.prepare_recipe("prepare_balance_path", "daily_running_accounts", "daily_balance_path", [
    formula("pessimistic_balance", "round(balance + net_running - 1.2816 * sqrt(variance_running))"),
    formula("balance", "round(balance + net_running)"),
    formula("inflow", "round(inflow)"),
    formula("outflow", "round(outflow)"),
    keep_columns("account_id", "date", "inflow", "outflow", "balance", "pessimistic_balance"),
], "balance = yesterday's balance + running net. Pessimistic: the 1-in-10 bad case of the running net "
   "(1.2816 standard deviations below).", tags=["6-balance"])

# 7. outlook: lowest point per account and risk level -----------------------------------------
dataset("account_lowest", f"Per account over the {HORIZON} days: totals in and out, lowest balance, lowest "
        "pessimistic balance.", "7-outlook")
p.group_recipe("group_outlook", "daily_balance_path", "account_lowest", ["account_id"],
               {"inflow": ["sum"], "outflow": ["sum"], "balance": ["min"], "pessimistic_balance": ["min"]},
               count=False,
               names={"inflow_sum": f"total_in_{HORIZON}d", "outflow_sum": f"total_out_{HORIZON}d",
                      "balance_min": "lowest_balance", "pessimistic_balance_min": "pessimistic_lowest_balance"},
               description="Totals and lowest balances per account.", tags=["7-outlook"])

dataset("account_lowest_details", "account_lowest + the day of the lowest balance + company, balance yesterday "
        "and overdraft limit.", "7-outlook")
p.join_recipe("join_accounts", ["account_lowest", "daily_balance_path", "accounts"], "account_lowest_details",
              {"daily_balance_path": ["account_id", ("lowest_balance", "balance")], "accounts": ["account_id"]},
              columns={"daily_balance_path": ["date"],
                       "accounts": ["company_name", "sector", "relationship_manager", "balance", "overdraft_limit"]},
              description="Find the day whose balance is the lowest one; add the account details.",
              tags=["7-outlook"])

dataset("account_outlook", f"One row per account: the next {HORIZON} days in numbers and a risk level.", "7-outlook")
p.prepare_recipe("prepare_outlook", "account_lowest_details", "account_outlook", [
    rename({"balance": "opening_balance", "date": "lowest_balance_date"}),
    formula("closing_balance", f"round(opening_balance + total_in_{HORIZON}d - total_out_{HORIZON}d)"),
    formula(f"total_in_{HORIZON}d", f"round(total_in_{HORIZON}d)"),
    formula(f"total_out_{HORIZON}d", f"round(total_out_{HORIZON}d)"),
    formula("risk_level", 'if(lowest_balance < 0 - overdraft_limit, "HIGH", '
                          'if(lowest_balance < 0 || pessimistic_lowest_balance < 0, "MEDIUM", "LOW"))'),
    keep_columns("account_id", "company_name", "sector", "relationship_manager", "opening_balance",
                 "overdraft_limit", f"total_in_{HORIZON}d", f"total_out_{HORIZON}d", "closing_balance",
                 "lowest_balance", "lowest_balance_date", "pessimistic_lowest_balance", "risk_level"),
], "HIGH: the lowest balance goes past the overdraft limit. MEDIUM: below 0, expected or pessimistic. "
   "LOW: otherwise.", tags=["7-outlook"])

# Flow zones: the shared data preparation, then training and inference side by side ------------------
p.zone("Data prep", tags=["1-input", "2-clean", "3-daily"])
p.zone("Training", tags=["4-train"], shared=["daily_flows_scoring"])
p.zone("Inference", tags=["5-forecast", "6-balance", "7-outlook"],
       shared=["daily_flows_scoring", "cashflow_forecast_model", "accounts"])

# Scenarios (Scenarios tab > Run): retrain now and then; forecast every day once yesterday's data is in.
p.scenario("Train model", [
    ("Rebuild the training data", ["daily_flows"], "RECURSIVE_FORCED_BUILD"),
    ("Retrain the saved model", ["cashflow_forecast_model"], "NON_RECURSIVE_FORCED_BUILD"),
], description="Retrain the deployed algorithm on the latest history: a new active version of the saved model.")
p.scenario("Daily forecast", [
    ("Prepare the latest data, forecast, balance path and outlook", ["account_outlook"], "RECURSIVE_FORCED_BUILD"),
], description=f"Forecast the next {HORIZON} days with the active model version (the model is not retrained).")

if __name__ == "__main__":
    ensure_code_env(CODE_ENV)
    p.deploy()
