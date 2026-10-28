"""Corporate banking, scripted version: same data as ../corporate_cashflow, built with code recipes and a
regression model (the visual project uses time series forecasting).

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

S, D, F = "string", "date", "double"


def code(name):
    return (HERE / "recipes" / name).read_text()


p = Project("CORP_CASHFLOW_CODE", "Corporate cash-flow forecast (code)",
            "Same data as CORP_CASHFLOW, built with Python and SQL recipes and a regression model.",
            variables={"llm_id": "openai:openai:gpt-6-luna"})
p.wiki_home("Start here", (HERE / "wiki.md").read_text())

OUTLOOK = [col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
           col("current_balance", F), col("overdraft_limit", F), col("last_month_net", F), col("avg_3m_net", F),
           col("forecast_month", D), col("forecast_net", F), col("projected_balance", F), col("risk_level")]

# 1. input ------------------------------------------------------------------------------------
p.csv_dataset("accounts", DATA / "accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("relationship_manager"),
    col("current_balance", F), col("overdraft_limit", F),
], "10 corporate accounts: company, sector, balance today (end of Sep 2026), overdraft limit.", tags=["1-input"])
p.csv_dataset("transactions", DATA / "transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date"), col("amount", F), col("direction"), col("category"),
], "Every transaction Oct 2023 - Sep 2026 (CREDIT = money in, DEBIT = money out).", tags=["1-input"])

# 2. monthly net cash flow (Python) -----------------------------------------------------------
p.pg_dataset("monthly_net", [col("account_id"), col("month", D), col("net", F)],
             "Net cash flow per account and month.", tags=["2-monthly"])
p.python_recipe("compute_monthly_net", ["transactions"], ["monthly_net"], code("monthly_net.py"),
                "Signed amounts summed per account and month.", tags=["2-monthly"])

# 3. machine learning (SQL features, Python model) ---------------------------------------------
p.pg_dataset("features", [
    col("account_id"), col("month", D), col("net", F), col("net_avg_3m", F), col("net_last_year", F),
    col("net_next_month", F), col("role"),
], "monthly_net + 3-month average, same month last year, next month (to predict) and the row's role.",
    tags=["3-ml"])
p.sql_recipe("compute_features", ["monthly_net"], ["features"], code("features.sql"),
             "Window functions in PostgreSQL: AVG, LAG, LEAD.", tags=["3-ml"])

p.pg_dataset("account_outlook", OUTLOOK,
             "Balance today, forecast and projected balance per account, with a risk level.", tags=["3-ml"])
p.pg_dataset("model_evaluation", [col("method"), col("mae", F)],
             "Average error on the test months: the model vs two simple baselines.", tags=["3-ml"])
p.python_recipe("compute_forecast", ["features", "accounts"], ["account_outlook", "model_evaluation"],
                code("forecast.py"), "Linear regression (scikit-learn): train, test, forecast, risk level.",
                tags=["3-ml"])

# 4. GenAI (Python + LLM Mesh) ----------------------------------------------------------------
p.pg_dataset("account_notes", OUTLOOK + [col("note")], "account_outlook + the note written by the LLM.",
             tags=["4-genai"])
p.python_recipe("compute_account_notes", ["account_outlook"], ["account_notes"], code("account_notes.py"),
                "One LLM call per account, with the LLM in the project variable llm_id.", tags=["4-genai"])

if __name__ == "__main__":
    p.deploy()
