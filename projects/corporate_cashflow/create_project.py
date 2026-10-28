"""Corporate banking: forecast each account's net cash flow for next month, flag liquidity risk, and
have an LLM write a short note per account.

    python3 generate_data.py     # (re)create the synthetic CSVs in data/
    python3 create_project.py    # push project CORP_CASHFLOW into DSS and run the whole flow

No code recipes: visual recipes (Prepare, Group, Window, Split, Join, Prompt) and Visual ML
(a Lab model deployed to the flow + a Score recipe). The project wiki is wiki.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dssproj import Project, col, formula, keep_columns, parse_date, truncate_date  # noqa: E402

LLM_ID = "openai:openai:gpt-6-luna"     # LLM Mesh model used by the Prompt recipe
TEST_FROM = 202606                      # months from Jun 2026 test the model, earlier ones train it
F = "double"

p = Project("CORP_CASHFLOW", "Corporate cash-flow forecast",
            "Forecast next month's net cash flow per corporate account, flag liquidity risk, write a note.")
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
], "Every transaction Oct 2024 - Sep 2026 (CREDIT = money in, DEBIT = money out).", tags=["1-input"])

# 2. monthly net cash flow --------------------------------------------------------------------
dataset("transactions_prepared", "Transactions with their month and a signed amount.", "2-monthly")
p.prepare_recipe("prepare_transactions", "transactions", "transactions_prepared", [
    parse_date("booking_date"),
    truncate_date("booking_date", "MONTH", "month"),
    formula("net", 'if(direction == "CREDIT", amount, 0 - amount)'),
], "Add the month and the signed amount (money in +, money out -).", tags=["2-monthly"])

dataset("monthly_net", "Net cash flow per account and month.", "2-monthly")
p.group_recipe("group_by_month", "transactions_prepared", "monthly_net", ["account_id", "month"],
               {"net": ["sum"]}, count=False, names={"net_sum": "net"},
               description="Sum the signed amounts per account and month.", tags=["2-monthly"])

# 3. machine learning -------------------------------------------------------------------------
dataset("features", "monthly_net + what the model learns from: last 3 months average, the same month "
        "last year, and the value to predict (next month).", "3-ml")
p.window_recipe("window_features", "monthly_net", "features", ["account_id"], ["month"],
                {"net": {"lag": [11], "lead": [1], "avg": True}}, frame=(2, 0),
                names={"net_avg": "net_avg_3m", "net_lag11": "net_last_year", "net_lead": "net_next_month"},
                description="Per account, ordered by month: 3-month average, next month 1 year ago (lag 11), "
                            "next month (lead 1).", tags=["3-ml"])

dataset("features_roles", "features + role: train, test, forecast or warm_up (no last-year value yet).", "3-ml")
p.prepare_recipe("prepare_roles", "features", "features_roles", [
    formula("period", 'datePart(month, "year") * 100 + datePart(month, "month")'),
    formula("role", f'if(isBlank(net_next_month), "forecast", if(isBlank(net_last_year), "warm_up", '
                    f'if(period >= {TEST_FROM}, "test", "train")))'),
], "Label each row: the latest month is forecast, Jun-Aug 2026 test the model, older months train it.",
    tags=["3-ml"])

for name, what in (("ml_train", "Rows the model learns from."), ("ml_test", "Rows the model is tested on."),
                   ("ml_forecast", "The latest month of each account, to forecast.")):
    dataset(name, what, "3-ml")
p.split_recipe("split_by_role", "features_roles", "role",
               {"train": "ml_train", "test": "ml_test", "forecast": "ml_forecast"},
               description="One dataset per role (warm_up rows are dropped).", tags=["3-ml"])

p.ml_model("next_month_model", "ml_train", "ml_test", "net_next_month",
           reject=["account_id", "month", "period", "role"], algorithm="leastsquare_regression",
           description="Linear regression: next month's net from this month, the 3-month average and the "
                       "same month last year. Open it for the metrics and coefficients.", tags=["3-ml"])

dataset("forecast", "ml_forecast + the model's prediction for next month.", "3-ml")
p.score_recipe("score_next_month", "next_month_model", "ml_forecast", "forecast",
               description="Apply the model to the latest month.", tags=["3-ml"])

# 4. outlook ----------------------------------------------------------------------------------
dataset("forecast_accounts", "forecast + the account's company, balance and overdraft limit.", "4-outlook")
p.join_recipe("join_accounts", ["forecast", "accounts"], "forecast_accounts", ["account_id"],
              columns={"forecast": ["account_id", "month", "net", "net_avg_3m", "net_last_year", "prediction"],
                       "accounts": ["company_name", "sector", "relationship_manager", "current_balance",
                                    "overdraft_limit"]},
              description="Add the account details.", tags=["4-outlook"])

dataset("account_outlook", "Balance today, forecast and projected balance per account, with a risk level.",
        "4-outlook")
p.prepare_recipe("prepare_outlook", "forecast_accounts", "account_outlook", [
    formula("forecast_month", 'inc(month, 1, "month")'),
    formula("forecast_net", "round(prediction)"),
    formula("projected_balance", "round(current_balance + prediction)"),
    formula("risk_level", 'if(projected_balance < 0 - overdraft_limit, "HIGH", '
                          'if(projected_balance < 0, "MEDIUM", "LOW"))'),
    formula("last_month_net", "round(net)"),
    formula("avg_3m_net", "round(net_avg_3m)"),
    keep_columns("account_id", "company_name", "sector", "relationship_manager", "current_balance",
                 "overdraft_limit", "last_month_net", "avg_3m_net", "forecast_month", "forecast_net",
                 "projected_balance", "risk_level"),
], "projected balance = balance today + forecast. HIGH: past the overdraft limit, MEDIUM: below 0.",
    tags=["4-outlook"])

# 5. GenAI ------------------------------------------------------------------------------------
dataset("account_notes", "account_outlook + llm_output: the note written by the LLM.", "5-genai")
p.prompt_recipe("prompt_account_notes", "account_outlook", "account_notes", LLM_ID, """\
Write a cash-flow note (max 80 words, plain text) for the relationship manager of {{company_name}} ({{sector}}).
Facts in USD: balance today {{current_balance}}, overdraft limit {{overdraft_limit}},
net cash flow last month {{last_month_net}}, average of the last 3 months {{avg_3m_net}},
forecast for next month {{forecast_net}}, projected balance {{projected_balance}}, risk level {{risk_level}}.
Say what is happening and suggest one action.""",
                system="You are a corporate-banking credit analyst. Use only the facts given.",
                description="One LLM call per account.", tags=["5-genai"])

if __name__ == "__main__":
    p.deploy()
