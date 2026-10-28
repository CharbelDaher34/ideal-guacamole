"""Corporate banking, scripted version: same use case as ../corporate_cashflow, built with code recipes.

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

S, B, D, F = "string", "bigint", "date", "double"


def code(name):
    return (HERE / "recipes" / name).read_text()


p = Project("CORP_CASHFLOW_CODE", "Corporate cash-flow forecast (code)",
            "Same as CORP_CASHFLOW, built with Python and SQL recipes.",
            variables={"llm_id": "openai:openai:gpt-6-luna"})
p.wiki_home("Start here", (HERE / "wiki.md").read_text())

# 1. ingest -----------------------------------------------------------------------------------
p.csv_dataset("accounts_raw", DATA / "accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("segment"), col("country"), col("currency"),
    col("relationship_manager"), col("opening_date"), col("opening_balance", F), col("overdraft_limit", F),
], "Corporate accounts: company, sector, segment, relationship manager, opening balance, overdraft limit.",
    tags=["1-ingest"])
p.csv_dataset("transactions_raw", DATA / "transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date"), col("amount", F), col("direction"),
    col("category"), col("counterparty"),
], "Every booked transaction since Oct 2023 (CREDIT = money in, DEBIT = money out).", tags=["1-ingest"])

# 2. prepare ----------------------------------------------------------------------------------
p.pg_dataset("transactions_prepared", [
    col("txn_id"), col("account_id"), col("booking_date", D), col("month", D), col("direction"),
    col("category"), col("counterparty"), col("amount", F), col("signed_amount", F),
], "Clean transactions with signed amount and month.", tags=["2-prepare"])
p.pg_dataset("accounts", [
    col("account_id"), col("company_name"), col("sector"), col("segment"), col("country"), col("currency"),
    col("relationship_manager"), col("opening_date", D), col("opening_balance", F), col("overdraft_limit", F),
], "Clean account reference data.", tags=["2-prepare"])
p.python_recipe("compute_transactions_prepared", ["transactions_raw", "accounts_raw"],
                ["transactions_prepared", "accounts"], code("prepare_data.py"),
                "Types, de-duplication, signed amounts.", tags=["2-prepare"])

# 3. aggregate (SQL in PostgreSQL) ------------------------------------------------------------
p.pg_dataset("monthly_cashflow", [
    col("account_id"), col("month", D), col("inflow", F), col("outflow", F), col("net_cashflow", F),
    col("n_transactions", B), col("customer_receipts", F), col("payroll", F), col("supplier_payments", F),
    col("debt_and_tax", F),
], "Inflow, outflow and net cash flow per account and month.", tags=["3-aggregate"])
p.sql_recipe("compute_monthly_cashflow", ["transactions_prepared"], ["monthly_cashflow"],
             code("monthly_cashflow.sql"), "Monthly aggregation, executed in PostgreSQL.", tags=["3-aggregate"])

# 4-5. machine learning -----------------------------------------------------------------------
p.pg_dataset("training_set", [col("account_id"), col("month", D)],
             "One row per account and month: features known at that month + actual values 1-3 months later.",
             tags=["4-ml"])
p.python_recipe("compute_training_set", ["monthly_cashflow", "accounts"], ["training_set"],
                code("build_training_set.py"), "Feature engineering: lags, averages, seasonality, targets.",
                tags=["4-ml"])
p.pg_dataset("cashflow_forecast", [col("account_id"), col("forecast_month", D)],
             "Forecast inflow / outflow / net per account for the next 3 months, with a 10-90% range.",
             tags=["4-ml"])
p.pg_dataset("model_evaluation", [col("method")],
             "Backtest error of the model vs two simple baselines, per horizon.", tags=["4-ml"])
p.pg_dataset("feature_importance", [col("feature")], "What the models rely on.", tags=["4-ml"])
p.pg_dataset("backtest_predictions", [col("account_id")],
             "Predicted vs actual for the last 6 months (for charts).", tags=["4-ml"])
p.python_recipe("compute_cashflow_forecast", ["training_set"],
                ["cashflow_forecast", "model_evaluation", "feature_importance", "backtest_predictions"],
                code("train_and_forecast.py"), "Train LightGBM, backtest against baselines, forecast 3 months.",
                tags=["4-ml"])

# 6. insights (SQL) ---------------------------------------------------------------------------
p.pg_dataset("account_outlook", [
    col("account_id"), col("company_name"), col("sector"), col("segment"), col("relationship_manager"),
    col("overdraft_limit", F), col("current_balance", F), col("last_3m_net", F),
    col("forecast_3m_inflow", F), col("forecast_3m_outflow", F), col("forecast_3m_net", F),
    col("projected_balance", F), col("projected_min_balance", F), col("worst_case_min_balance", F),
    col("risk_level"),
], "Current and projected balances per account with a HIGH / MEDIUM / LOW liquidity risk.", tags=["5-insights"])
p.sql_recipe("compute_account_outlook", ["accounts", "monthly_cashflow", "cashflow_forecast"], ["account_outlook"],
             code("account_outlook.sql"), "Balances + forecast => risk level.", tags=["5-insights"])

# 7. report (GenAI) ---------------------------------------------------------------------------
p.pg_dataset("cashflow_report", [
    col("scope"), col("account_id"), col("company_name"), col("risk_level"), col("report_markdown"),
    col("generator"), col("generated_at", D),
], "The report: one PORTFOLIO summary row + one row per account (markdown).", tags=["6-genai"])
p.python_recipe("compute_cashflow_report", ["account_outlook", "cashflow_forecast", "model_evaluation"],
                ["cashflow_report"], code("generate_report.py"),
                "Writes the report with the LLM in the project variable llm_id.", tags=["6-genai"])

if __name__ == "__main__":
    p.deploy()
