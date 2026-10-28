"""Corporate banking: forecast each account's net cash flow for the next 3 months with Dataiku's time
series forecasting, flag liquidity risk, and have an LLM write a short note per account.

    python3 generate_data.py     # (re)create the synthetic CSVs in data/
    python3 create_project.py    # push project CORP_CASHFLOW into DSS and run the whole flow

No code recipes: visual recipes (Prepare, Group, Join, Prompt) and Visual ML time series forecasting
(a Lab analysis comparing several algorithms, the best one deployed to the flow + a Score recipe).
The project wiki is wiki.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dssproj import (Project, col, ensure_code_env, formula, keep_columns, parse_date,  # noqa: E402
                     truncate_date)

LLM_ID = "openai:openai:gpt-6-luna"     # LLM Mesh model used by the Prompt recipe
CODE_ENV = "timeseries"                 # Python env with DSS's time series packages, created if missing
F = "double"

# Algorithms the Lab trains and compares; the one with the lowest MASE on the test months is deployed.
ALGORITHMS = {
    "trivial_identity_timeseries": {},                                   # naive: repeat the last month
    "seasonal_naive_timeseries": {"season_length": 12},                  # repeat the same month last year
    "ets_timeseries": {"trend": ["add"], "seasonal": ["add"]},           # exponential smoothing (Holt-Winters)
    "seasonal_loess_timeseries": {"period": 12},                         # STL: trend + seasonality
    "arima_timeseries": {"p": 1, "D": 1, "s": 12},                       # ARIMA with a yearly difference
    "autoarima_timeseries": {"m": 12},                                   # ARIMA that picks its own orders
    "prophet_timeseries": {"changepoint_prior_scale": 0.05},             # Prophet (Meta)
    "gluonts_torch_simple_feed_forward_timeseries": {},                  # small neural network
    "gluonts_torch_deepar_timeseries": {},                               # DeepAR (recurrent neural network)
}

p = Project("CORP_CASHFLOW", "Corporate cash-flow forecast",
            "Forecast the next 3 months of net cash flow per corporate account, flag liquidity risk, write a note.")
p.wiki_home("Start here", (HERE / "wiki.md").read_text())


def dataset(name, description, tag):
    """Managed PostgreSQL dataset; its schema is computed by DSS from the recipe that writes it."""
    p.pg_dataset(name, [], description, tags=[tag])


# 1. input ------------------------------------------------------------------------------------
p.csv_dataset("accounts", HERE / "data/accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
    col("current_balance", F), col("overdraft_limit", F),
], "10 corporate accounts: company, sector, balance today (end of Sep 2026), overdraft limit.", tags=["1-input"])
p.csv_dataset("transactions", HERE / "data/transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date"), col("amount", F), col("direction"), col("category"),
], "Every transaction Oct 2023 - Sep 2026 (CREDIT = money in, DEBIT = money out).", tags=["1-input"])

# 2. monthly net cash flow = the time series --------------------------------------------------
dataset("transactions_prepared", "Transactions with their month and a signed amount.", "2-monthly")
p.prepare_recipe("prepare_transactions", "transactions", "transactions_prepared", [
    parse_date("booking_date"),
    truncate_date("booking_date", "MONTH", "month"),
    formula("net", 'if(direction == "CREDIT", amount, 0 - amount)'),
], "Add the month and the signed amount (money in +, money out -).", tags=["2-monthly"])

dataset("monthly_net", "The time series: net cash flow per account (series id) and month (time).", "2-monthly")
p.group_recipe("group_by_month", "transactions_prepared", "monthly_net", ["account_id", "month"],
               {"net": ["sum"]}, count=False, names={"net_sum": "net"},
               description="Sum the signed amounts per account and month.", tags=["2-monthly"])

# 3. time series forecasting ------------------------------------------------------------------
p.timeseries_model("cashflow_forecast_model", "monthly_net", target="net", time="month", ids=["account_id"],
                   horizon=3, algorithms=ALGORITHMS, env=CODE_ENV,
                   description="Time series forecasting of net per account, 3 months ahead, tested on the last "
                               "3 months. The Lab analysis compares the algorithms; the best (MASE) is deployed.",
                   tags=["3-forecast"])

dataset("forecast", "monthly_net + 3 new months per account (Oct - Dec 2026) with the forecast and its "
        "quantiles (quantile_01 = 10% ... quantile_09 = 90%). Chart it: history and forecast together.",
        "3-forecast")
p.score_recipe("forecast_next_months", "cashflow_forecast_model", "monthly_net", "forecast",
               description="Forecast Oct - Dec 2026 from the full history.", tags=["3-forecast"])

# 4. outlook ----------------------------------------------------------------------------------
dataset("forecast_3m", "Per account: forecast net over the 3 months, worst month, pessimistic total (10%).",
        "4-outlook")
p.group_recipe("group_forecast_3m", "forecast", "forecast_3m", ["account_id"],
               {"forecast": ["sum", "min"], "quantile_01": ["sum"]}, count=False, pre_filter="!isBlank(forecast)",
               names={"forecast_sum": "forecast_3m_net", "forecast_min": "worst_month_net",
                      "quantile_01_sum": "pessimistic_3m_net"},
               description="Keep the forecast months and sum them per account.", tags=["4-outlook"])

dataset("forecast_accounts", "forecast_3m + the account's company, balance and overdraft limit.", "4-outlook")
p.join_recipe("join_accounts", ["forecast_3m", "accounts"], "forecast_accounts", ["account_id"],
              description="Add the account details.", tags=["4-outlook"])

dataset("account_outlook", "Balance today, 3-month forecast and projected balance per account, with a risk level.",
        "4-outlook")
p.prepare_recipe("prepare_outlook", "forecast_accounts", "account_outlook", [
    formula("forecast_3m_net", "round(forecast_3m_net)"),
    formula("worst_month_net", "round(worst_month_net)"),
    formula("pessimistic_3m_net", "round(pessimistic_3m_net)"),
    formula("projected_balance", "round(current_balance + forecast_3m_net)"),
    formula("pessimistic_balance", "round(current_balance + pessimistic_3m_net)"),
    formula("risk_level", 'if(projected_balance < 0 - overdraft_limit, "HIGH", '
                          'if(projected_balance < 0 || pessimistic_balance < 0, "MEDIUM", "LOW"))'),
    keep_columns("account_id", "company_name", "sector", "relationship_manager", "current_balance",
                 "overdraft_limit", "forecast_3m_net", "worst_month_net", "pessimistic_3m_net",
                 "projected_balance", "pessimistic_balance", "risk_level"),
], "projected balance (end of Dec) = balance today + 3-month forecast. HIGH: past the overdraft limit, "
   "MEDIUM: below 0, expected or in the pessimistic case.", tags=["4-outlook"])

# 5. GenAI ------------------------------------------------------------------------------------
dataset("account_notes", "account_outlook + llm_output: the note written by the LLM.", "5-genai")
p.prompt_recipe("prompt_account_notes", "account_outlook", "account_notes", LLM_ID, """\
Write a cash-flow note (max 80 words, plain text) for the relationship manager of {{company_name}} ({{sector}}).
Facts in USD, forecast for Oct - Dec 2026: balance today {{current_balance}}, overdraft limit {{overdraft_limit}},
net cash flow forecast over the 3 months {{forecast_3m_net}} (pessimistic case {{pessimistic_3m_net}}),
worst month {{worst_month_net}}, projected balance end of December {{projected_balance}}
(pessimistic case {{pessimistic_balance}}), risk level {{risk_level}}.
Say what is happening and suggest one action.""",
                system="You are a corporate-banking credit analyst. Use only the facts given.",
                description="One LLM call per account.", tags=["5-genai"])

if __name__ == "__main__":
    ensure_code_env(CODE_ENV)
    p.deploy()
