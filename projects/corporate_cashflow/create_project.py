"""Corporate banking: forecast each account's next 3 months of cash flow and write a GenAI report.

    python3 generate_data.py     # (re)create the synthetic CSVs in data/
    python3 create_project.py    # push project CORP_CASHFLOW into DSS and run the whole flow

No code recipes: the flow uses Dataiku's visual recipes (Prepare, Group, Window, Join, Split, Stack,
Prompt) and Visual ML (a Lab model deployed to the flow, Evaluate and Score recipes).
The project wiki is wiki.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dssproj import (Project, col, formula, keep_columns, parse_date, remove_empty,  # noqa: E402
                     rename, step, truncate_date)

LLM_ID = "openai:openai:gpt-6-luna"     # LLM Mesh model used by the Prompt recipes
FLOWS = ("inflow", "outflow", "net_cashflow")
S, F = "string", "double"

p = Project("CORP_CASHFLOW", "Corporate cash-flow forecast",
            "Forecast the next 3 months of cash flow per corporate account, flag liquidity risk, write a report.")
p.wiki_home("Start here", (HERE / "wiki.md").read_text())


def dataset(name, description, tag):
    """Managed PostgreSQL dataset; its schema is computed by DSS from the recipe that writes it."""
    p.pg_dataset(name, [], description, tags=[tag])


# 1. ingest -----------------------------------------------------------------------------------
p.csv_dataset("accounts_raw", HERE / "data/accounts.csv", [
    col("account_id"), col("company_name"), col("sector"), col("segment"), col("country"), col("currency"),
    col("relationship_manager"), col("opening_date"), col("opening_balance", F), col("overdraft_limit", F),
], "Corporate accounts: company, sector, segment, relationship manager, opening balance, overdraft limit.",
    tags=["1-ingest"])
p.csv_dataset("transactions_raw", HERE / "data/transactions.csv", [
    col("txn_id"), col("account_id"), col("booking_date"), col("amount", F), col("direction"),
    col("category"), col("counterparty"),
], "Every booked transaction since Oct 2023 (CREDIT = money in, DEBIT = money out).", tags=["1-ingest"])

# 2. prepare ----------------------------------------------------------------------------------
dataset("accounts", "Clean account reference data.", "2-prepare")
p.prepare_recipe("prepare_accounts", "accounts_raw", "accounts", [parse_date("opening_date")],
                 "Parse the opening date.", tags=["2-prepare"])

dataset("transactions_prepared", "Clean transactions: month, signed amount, and the amount split by "
        "direction and category (ready to be summed per month).", "2-prepare")
p.prepare_recipe("prepare_transactions", "transactions_raw", "transactions_prepared", [
    remove_empty("account_id", "booking_date", "amount"),
    parse_date("booking_date"),
    truncate_date("booking_date", "MONTH", "month"),
    formula("direction", "toUppercase(trim(direction))"),
    formula("amount", "abs(amount)"),
    formula("signed_amount", 'if(direction == "CREDIT", amount, 0 - amount)'),
    formula("inflow", 'if(direction == "CREDIT", amount, 0)'),
    formula("outflow", 'if(direction == "DEBIT", amount, 0)'),
    formula("customer_receipts", 'if(category == "CUSTOMER_RECEIPT", amount, 0)'),
    formula("payroll", 'if(category == "PAYROLL", amount, 0)'),
    formula("supplier_payments", 'if(category == "SUPPLIER_PAYMENT", amount, 0)'),
    formula("debt_and_tax", 'if(category == "LOAN_REPAYMENT" || category == "TAX", amount, 0)'),
], "Clean and type the transactions, add the month and the amounts by direction / category.", tags=["2-prepare"])

# 3. aggregate --------------------------------------------------------------------------------
SUMMED = ("inflow", "outflow", "customer_receipts", "payroll", "supplier_payments", "debt_and_tax")
dataset("monthly_cashflow", "Inflow, outflow and net cash flow per account and month.", "3-aggregate")
p.group_recipe("group_monthly_cashflow", "transactions_prepared", "monthly_cashflow", ["account_id", "month"],
               {c: ["sum"] for c in ("signed_amount",) + SUMMED},
               names={"signed_amount_sum": "net_cashflow", "count": "n_transactions",
                      **{f"{c}_sum": c for c in SUMMED}},
               description="Sum the transactions per account and month.", tags=["3-aggregate"])

dataset("account_balance", "Per account: total net cash flow since the opening balance, and its latest month.",
        "3-aggregate")
p.group_recipe("group_account_balance", "monthly_cashflow", "account_balance", ["account_id"],
               {"net_cashflow": ["sum"], "month": ["max"]}, count=False,
               names={"net_cashflow_sum": "hist_net", "month_max": "last_month"},
               description="Total net cash flow of the whole history, and the latest month.", tags=["3-aggregate"])

# 4. features ---------------------------------------------------------------------------------
# Window over each account's months: previous months (lag), same month last year (lag 9-11),
# 12-month average, and the actual values 1-3 months later (lead) = what the model learns to predict.
LAGS = {1: "m1", 2: "m2", 11: "ly_h1", 10: "ly_h2", 9: "ly_h3"}
dataset("monthly_history", "monthly_cashflow + for each month: previous months, same month last year, "
        "12-month average, and the next 3 months.", "4-features")
p.window_recipe("window_monthly_history", "monthly_cashflow", "monthly_history", ["account_id"], ["month"],
                {c: {"lag": list(LAGS), "lead": [1, 2, 3], "avg": True} for c in FLOWS}, frame=(11, 0),
                names={**{f"{c}_lag{k}": f"{c}_{v}" for c in FLOWS for k, v in LAGS.items()},
                       **{f"{c}_lead{h}": f"{c}_next_{h}" for c in FLOWS for h in (1, 2, 3)},
                       **{f"{c}_avg": f"{c}_avg12" for c in FLOWS}},
                description="Per account, ordered by month: lags, leads and a 12-month moving average.",
                tags=["4-features"])

dataset("monthly_history_accounts", "monthly_history + the account's sector, segment and latest month.",
        "4-features")
p.join_recipe("join_account_segment", ["monthly_history", "accounts", "account_balance"],
              "monthly_history_accounts", ["account_id"],
              columns={"accounts": ["sector", "segment"], "account_balance": ["last_month"]},
              description="Add sector and segment (the model uses them) and the account's latest month.",
              tags=["4-features"])

TRAINING_COLUMNS = (["account_id", "month", "month_of_year", "months_to_latest"]
                    + [f"{c}_{x}" for c in FLOWS
                       for x in ("m0", "m1", "m2", "avg3", "avg12", "ly_h1", "next_1", "ly_h2", "next_2",
                                 "ly_h3", "next_3")]
                    + ["scale", "sector", "segment", "row_role"])
dataset("training_set", "One row per account and month: what was known at that month + the actual values 1-3 "
        "months later.", "4-features")
p.prepare_recipe("prepare_training_set", "monthly_history_accounts", "training_set", [
    rename({c: f"{c}_m0" for c in FLOWS}),
    *[formula(f"{c}_avg3", f"({c}_m0 + {c}_m1 + {c}_m2) / 3") for c in FLOWS],
    formula("month_of_year", 'datePart(month, "month")'),
    formula("months_to_latest", '(datePart(last_month, "year") - datePart(month, "year")) * 12 '
                                '+ datePart(last_month, "month") - datePart(month, "month")'),
    formula("scale", "inflow_avg12"),
    remove_empty("inflow_ly_h1"),
    formula("row_role", 'if(isBlank(net_cashflow_next_1), "FORECAST_ORIGIN", '
                        'if(isBlank(net_cashflow_next_3), "PARTIAL", "TRAIN"))'),
    keep_columns(*TRAINING_COLUMNS),
], "3-month averages, month of year, months before the latest month, scale; keep months with a full year of "
   "history; mark each row TRAIN (3 future months known), PARTIAL, or FORECAST_ORIGIN (latest month).",
    tags=["4-features"])

# 5. machine learning (Visual ML) -------------------------------------------------------------
# One model for every account, flow (inflow / outflow / net) and horizon (1-3 months ahead):
# training_set is unfolded to one row per account x month x flow x horizon. Amounts are divided by the
# account's average monthly inflow ("scale") so small and large companies can share the model.
PREFIX = {"inflow": "inflow", "outflow": "outflow", "net": "net_cashflow"}
SLOTS = [(f, h) for f in PREFIX for h in (1, 2, 3)]
MONEY_FEATURES = [f"{c}_{x}" for c in FLOWS for x in ("m0", "m1", "m2", "avg3", "avg12", "ly_h1", "ly_h2", "ly_h3")]
TARGETS = [f"{c}_next_{h}" for c in FLOWS for h in (1, 2, 3)]


def by_slot(template):
    """Nested if() picking, for the row's flow and horizon, the column named by template."""
    expr = "null"
    for f, h in reversed(SLOTS):
        expr = f'if(slot == "slot_{f}_{h}", {template.format(c=PREFIX[f], h=h)}, {expr})'
    return expr


def by_flow(suffix):
    return (f'if(flow == "inflow", inflow_{suffix}, if(flow == "outflow", outflow_{suffix}, '
            f'net_cashflow_{suffix}))')


ML_COLUMNS = (["account_id", "month", "scale", "sector", "segment", "month_of_year", "target_month_of_year",
               "flow", "horizon"] + [f"own_{x}" for x in ("m0", "m1", "m2", "avg3", "avg12", "last_year")]
              + MONEY_FEATURES + ["target", "role"])
dataset("ml_dataset", "One row per account, month, flow and horizon: features (in units of the account's "
        "monthly inflow), target, and role TRAIN / TEST / FORECAST.", "5-ml")
p.prepare_recipe("prepare_ml_dataset", "training_set", "ml_dataset", [
    *[formula(c, f"{c} / scale") for c in MONEY_FEATURES + TARGETS],
    *[formula(f"slot_{f}_{h}", "1") for f, h in SLOTS],
    step("MultiColumnFold", columns=[f"slot_{f}_{h}" for f, h in SLOTS], foldNameColumn="slot",
         foldValueColumn="slot_value", foldRemoveFoldedColumns=True),
    formula("flow", 'if(startsWith(slot, "slot_inflow"), "inflow", '
                    'if(startsWith(slot, "slot_outflow"), "outflow", "net"))'),
    formula("horizon", 'if(endsWith(slot, "_1"), 1, if(endsWith(slot, "_2"), 2, 3))'),
    formula("target_month_of_year", "if(month_of_year + horizon > 12, month_of_year + horizon - 12, "
                                    "month_of_year + horizon)"),
    *[formula(f"own_{x}", by_flow(x)) for x in ("m0", "m1", "m2", "avg3", "avg12")],
    formula("own_last_year", by_slot("{c}_ly_h{h}")),
    formula("target", by_slot("{c}_next_{h}")),
    formula("role", 'if(months_to_latest == 0, "FORECAST", if(isBlank(target), "UNKNOWN", '
                    'if(months_to_latest - horizon < 6, "TEST", "TRAIN")))'),
    keep_columns(*ML_COLUMNS),
], "Scale amounts, unfold to one row per flow and horizon, pick the target, and set the role: TEST = target "
   "in the last 6 months of data (held out to measure the model), TRAIN = older, FORECAST = latest month.",
    tags=["5-ml"])

for name, desc in [("ml_train", "Rows the model learns from."),
                   ("ml_test", "Rows held out to measure the model (target in the last 6 months)."),
                   ("ml_forecast", "Latest month: the rows to forecast from.")]:
    dataset(name, desc, "5-ml")
p.split_recipe("split_ml_dataset", "ml_dataset", "role",
               {"TRAIN": "ml_train", "TEST": "ml_test", "FORECAST": "ml_forecast"},
               description="Split the rows by role.", tags=["5-ml"])

p.ml_model("cashflow_model", "ml_train", "ml_test", "target", reject=["account_id", "month", "scale", "role"],
           params={"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 10,
                   "colsample_bytree": 0.8, "random_state": 0},
           interval=0.8,
           description="LightGBM regression, trained on ml_train and tested on ml_test, with 10%-90% "
                       "prediction intervals. Open the model for metrics, feature importance, explanations.",
           tags=["5-ml"])

dataset("test_predictions", "ml_test + the model's prediction and error per row.", "5-ml")
dataset("model_metrics", "Metrics of the model on ml_test (one row per evaluation).", "5-ml")
p.evaluate_recipe("evaluate_cashflow_model", "cashflow_model", "ml_test", "test_predictions", "model_metrics",
                  description="Score the held-out rows and compute the model's metrics.", tags=["5-ml"])

dataset("test_errors", "Absolute errors (USD) of the model and of two simple baselines, per flow and horizon.",
        "5-ml")
p.group_recipe("group_test_errors", "test_predictions", "test_errors", ["flow", "horizon"], {
    "err_model": ["sum"], "err_last_year": ["sum"], "err_avg3": ["sum"], "actual": ["sum"]},
    computed={"err_model": (F, "abs(prediction - target) * scale"),
              "err_last_year": (F, "abs(own_last_year - target) * scale"),
              "err_avg3": (F, "abs(own_avg3 - target) * scale"),
              "actual": (F, "abs(target) * scale")},
    names={"count": "n_points", **{f"{c}_sum": c for c in ("err_model", "err_last_year", "err_avg3", "actual")}},
    description="Sum the errors of the model, of 'same month last year' and of 'average of last 3 months'.",
    tags=["5-ml"])

dataset("model_evaluation", "Error (WAPE) of the model vs the two baselines on the held-out months.", "5-ml")
p.prepare_recipe("prepare_model_evaluation", "test_errors", "model_evaluation", [
    formula("model_wape", "err_model / actual"),
    formula("last_year_wape", "err_last_year / actual"),
    formula("avg3_wape", "err_avg3 / actual"),
    keep_columns("flow", "horizon", "n_points", "model_wape", "last_year_wape", "avg3_wape"),
], "WAPE = total absolute error / total actual amount (lower is better).", tags=["5-ml"])

dataset("forecast_scored", "ml_forecast + the model's prediction and 10%-90% interval.", "5-ml")
p.score_recipe("score_forecast", "cashflow_model", "ml_forecast", "forecast_scored",
               description="Apply the model to the latest month.", tags=["5-ml"])

dataset("forecast_amounts", "Forecast per account, flow and month, back in USD.", "5-ml")
p.prepare_recipe("prepare_forecast_amounts", "forecast_scored", "forecast_amounts", [
    formula("forecast_month", 'inc(month, horizon, "month")'),
    formula("amount", 'if(flow == "net", prediction, max(prediction, 0)) * scale'),
    formula("amount_lower", "toNumber(prediction_interval_lower) * scale"),
    formula("amount_upper", "toNumber(prediction_interval_upper) * scale"),
    keep_columns("account_id", "month", "horizon", "forecast_month", "flow", "amount", "amount_lower",
                 "amount_upper"),
], "Multiply back by the account's scale, compute the forecast month.", tags=["5-ml"])

dataset("forecast_by_month", "Forecast per account and month, one column per flow.", "5-ml")
p.group_recipe("group_forecast_by_month", "forecast_amounts", "forecast_by_month",
               ["account_id", "month", "horizon", "forecast_month"],
               {c: ["sum"] for c in ("inflow", "outflow", "net", "net_low", "net_high")}, count=False,
               computed={"inflow": (F, 'if(flow == "inflow", amount, 0)'),
                         "outflow": (F, 'if(flow == "outflow", amount, 0)'),
                         "net": (F, 'if(flow == "net", amount, 0)'),
                         "net_low": (F, 'if(flow == "net", amount_lower, 0)'),
                         "net_high": (F, 'if(flow == "net", amount_upper, 0)')},
               names={"month": "origin_month", "inflow_sum": "inflow_forecast", "outflow_sum": "outflow_forecast",
                      "net_sum": "net_model", "net_low_sum": "net_model_lower", "net_high_sum": "net_model_upper"},
               description="Put inflow, outflow and net side by side.", tags=["5-ml"])

dataset("cashflow_forecast", "Forecast inflow / outflow / net per account for the next 3 months, with a 10-90% "
        "range.", "5-ml")
p.prepare_recipe("prepare_cashflow_forecast", "forecast_by_month", "cashflow_forecast", [
    formula("net_forecast", "inflow_forecast - outflow_forecast"),
    formula("net_lower", "net_forecast - (net_model - net_model_lower)"),
    formula("net_upper", "net_forecast + (net_model_upper - net_model)"),
    formula("model", '"cashflow_model (Visual ML, LightGBM)"'),
    keep_columns("account_id", "origin_month", "horizon", "forecast_month", "inflow_forecast", "outflow_forecast",
                 "net_forecast", "net_lower", "net_upper", "model"),
], "Net = inflow - outflow; its 10-90% range comes from the model's interval on net.", tags=["5-ml"])

# 6. insights ---------------------------------------------------------------------------------
dataset("forecast_path", "cashflow_forecast + cumulative net (expected and pessimistic) month after month.",
        "6-insights")
p.window_recipe("window_forecast_path", "cashflow_forecast", "forecast_path", ["account_id"], ["forecast_month"],
                {"net_forecast": {"sum": True}, "net_lower": {"sum": True}},
                names={"net_forecast_sum": "cum_net", "net_lower_sum": "cum_low"},
                description="Running total of the forecast net cash flow per account.", tags=["6-insights"])

dataset("forecast_summary", "Per account: 3-month forecast totals, lowest cumulative net, monthly detail.",
        "6-insights")
# computed columns of a Group recipe running in PostgreSQL are translated to SQL: build text with concat()
MONTH = ('datePart(forecast_month, "year"), "-", if(datePart(forecast_month, "month") < 10, "0", ""), '
         'datePart(forecast_month, "month")')
p.group_recipe("group_forecast_summary", "forecast_path", "forecast_summary", ["account_id"], {
    "inflow_forecast": ["sum"], "outflow_forecast": ["sum"], "net_forecast": ["sum"],
    "cum_net": ["min"], "cum_low": ["min"], "forecast_month": ["min", "max"], "month_line": ["concat"]},
    count=False,
    computed={"month_line": (S, f'concat({MONTH}, ": inflow ", round(inflow_forecast), ", outflow ", '
                                'round(outflow_forecast), ", net ", round(net_forecast), " (10-90% range ", '
                                'round(net_lower), " to ", round(net_upper), ")")')},
    names={"inflow_forecast_sum": "forecast_3m_inflow", "outflow_forecast_sum": "forecast_3m_outflow",
           "net_forecast_sum": "forecast_3m_net", "cum_net_min": "min_cum_net", "cum_low_min": "min_cum_low",
           "forecast_month_min": "forecast_from", "forecast_month_max": "forecast_to",
           "month_line_concat": "monthly_forecast"},
    description="3-month totals and lowest point of the forecast path, per account.", tags=["6-insights"])

dataset("outlook_inputs", "Account + balance history + last 3 months + forecast summary, one row per account.",
        "6-insights")
p.join_recipe("join_outlook_inputs", ["accounts", "account_balance", "training_set", "forecast_summary"],
              "outlook_inputs", ["account_id"], how="INNER",
              columns={"accounts": ["account_id", "company_name", "sector", "segment", "relationship_manager",
                                    "opening_balance", "overdraft_limit"],
                       "account_balance": ["hist_net"],
                       "training_set": ["net_cashflow_m0", "net_cashflow_m1", "net_cashflow_m2"]},
              pre_filters={"training_set": 'row_role == "FORECAST_ORIGIN"'},
              description="Bring together what the outlook needs (training_set: only the latest month).",
              tags=["6-insights"])

MONEY = ("overdraft_limit", "current_balance", "last_3m_net", "forecast_3m_inflow", "forecast_3m_outflow",
         "forecast_3m_net", "projected_balance", "projected_min_balance", "worst_case_min_balance")
dataset("account_outlook", "Current and projected balances per account with a HIGH / MEDIUM / LOW liquidity "
        "risk.", "6-insights")
p.prepare_recipe("prepare_account_outlook", "outlook_inputs", "account_outlook", [
    formula("current_balance", "opening_balance + hist_net"),
    formula("last_3m_net", "net_cashflow_m0 + net_cashflow_m1 + net_cashflow_m2"),
    formula("projected_balance", "current_balance + forecast_3m_net"),
    formula("projected_min_balance", "current_balance + min(min_cum_net, 0)"),
    formula("worst_case_min_balance", "current_balance + min(min_cum_low, 0)"),
    *[formula(c, f"round({c})") for c in MONEY],
    formula("risk_level", 'if(projected_min_balance < 0 - overdraft_limit, "HIGH", '
                          'if(projected_min_balance < 0 || worst_case_min_balance < 0 - overdraft_limit, '
                          '"MEDIUM", "LOW"))'),
    keep_columns("account_id", "company_name", "sector", "segment", "relationship_manager", *MONEY,
                 "risk_level", "forecast_from", "forecast_to", "monthly_forecast"),
], "Balances and risk. HIGH: the expected path goes beyond the overdraft limit. MEDIUM: it dips below zero, "
   "or the pessimistic path (10% band) breaks the limit. LOW: otherwise.", tags=["6-insights"])

# 7. report (GenAI) ---------------------------------------------------------------------------
dataset("account_reports", "account_outlook + the LLM's commentary per account (llm_output).", "7-genai")
p.prompt_recipe("prompt_account_reports", "account_outlook", "account_reports", LLM_ID,
                system="You are a corporate-banking credit analyst. Amounts are in USD.",
                template="""Write a short cash-flow outlook (max 120 words, markdown) for the relationship manager of
this client, starting with the heading '### {{company_name}} ({{account_id}}) - risk {{risk_level}}'.
Cover: the 3-month forecast vs the last 3 months, the weakest month, the lowest projected balance vs the
overdraft limit, and one concrete suggested action. Use only these facts:
- Sector / segment: {{sector}}, {{segment}}. Relationship manager: {{relationship_manager}}
- Current balance: {{current_balance}}. Overdraft limit: {{overdraft_limit}}
- Net cash flow over the last 3 months: {{last_3m_net}}
- Forecast for the next 3 months: inflows {{forecast_3m_inflow}}, outflows {{forecast_3m_outflow}}, net {{forecast_3m_net}}
- Forecast by month:
{{monthly_forecast}}
- Projected balance at the end: {{projected_balance}}. Lowest projected balance: {{projected_min_balance}}.
  Pessimistic lowest balance (10% case): {{worst_case_min_balance}}
- Risk level: {{risk_level}} (HIGH = expected balance goes beyond the overdraft limit; MEDIUM = it dips below
  zero, or the pessimistic path breaks the limit; LOW = otherwise)""",
                description="One LLM call per account (LLM Mesh, connection openai).", tags=["7-genai"])

dataset("portfolio_facts", "One row: the portfolio's risk mix, totals and accounts needing attention.", "7-genai")
p.group_recipe("group_portfolio_facts", "account_outlook", "portfolio_facts", [], {
    "is_high": ["sum"], "is_medium": ["sum"], "is_low": ["sum"], "forecast_3m_net": ["sum"],
    "last_3m_net": ["sum"], "forecast_from": ["min"], "forecast_to": ["max"], "attention_line": ["concat"]},
    computed={"is_high": ("bigint", 'if(risk_level == "HIGH", 1, 0)'),
              "is_medium": ("bigint", 'if(risk_level == "MEDIUM", 1, 0)'),
              "is_low": ("bigint", 'if(risk_level == "LOW", 1, 0)'),
              "attention_line": (S, 'if(risk_level == "LOW", null, concat("- ", company_name, " (", risk_level, '
                                    '"): lowest projected balance ", projected_min_balance, ", pessimistic ", '
                                    'worst_case_min_balance, ", overdraft limit ", overdraft_limit, '
                                    '", 3-month net ", forecast_3m_net))')},
    names={"count": "n_accounts", "is_high_sum": "n_high", "is_medium_sum": "n_medium", "is_low_sum": "n_low",
           "forecast_3m_net_sum": "portfolio_forecast_net", "last_3m_net_sum": "portfolio_last_3m_net",
           "forecast_from_min": "forecast_from", "forecast_to_max": "forecast_to",
           "attention_line_concat": "accounts_needing_attention"},
    description="Aggregate the whole portfolio into one row.", tags=["7-genai"])

dataset("model_accuracy", "One row: average error (%) on monthly inflows of the model and of the two baselines.",
        "7-genai")
p.group_recipe("group_model_accuracy", "model_evaluation", "model_accuracy", [], {
    "model_error": ["avg"], "last_year_error": ["avg"], "avg3_error": ["avg"]}, count=False,
    pre_filter='flow == "inflow"',
    computed={"model_error": (F, "model_wape * 100"), "last_year_error": (F, "last_year_wape * 100"),
              "avg3_error": (F, "avg3_wape * 100")},
    names={"model_error_avg": "model_inflow_error_pct", "last_year_error_avg": "last_year_inflow_error_pct",
           "avg3_error_avg": "avg3_inflow_error_pct"},
    description="Average the inflow errors over the 3 horizons.", tags=["7-genai"])

dataset("portfolio_inputs", "portfolio_facts + model_accuracy in one row.", "7-genai")
p.join_recipe("join_portfolio_inputs", ["portfolio_facts", "model_accuracy"], "portfolio_inputs", [], how="CROSS",
              description="Put the two one-row datasets side by side.", tags=["7-genai"])

dataset("portfolio_report", "portfolio_inputs + the LLM's executive summary (llm_output).", "7-genai")
p.prompt_recipe("prompt_portfolio_report", "portfolio_inputs", "portfolio_report", LLM_ID,
                system="You are head of corporate-banking risk. Amounts are in USD.",
                template="""Write an executive summary (max 200 words, markdown) of this 3-month cash-flow forecast for
the corporate portfolio, starting with the heading '## Corporate cash-flow outlook, <period>' where <period>
is written like 'Oct 2026 - Dec 2026'. Mention the risk mix, the accounts needing attention and how
reliable the model was. Facts:
- Forecast period: first month {{forecast_from}}, last month {{forecast_to}}
- Accounts: {{n_accounts}} ({{n_high}} high risk, {{n_medium}} medium, {{n_low}} low)
- Portfolio net cash flow: forecast next 3 months {{portfolio_forecast_net}}, last 3 months {{portfolio_last_3m_net}}
- Accounts needing attention:
{{accounts_needing_attention}}
- Backtest error on monthly inflows: model {{model_inflow_error_pct}}%, baseline 'same month last year'
  {{last_year_inflow_error_pct}}%, baseline 'average of last 3 months' {{avg3_inflow_error_pct}}%""",
                description="One LLM call for the portfolio summary.", tags=["7-genai"])

dataset("report_rows", "Portfolio summary + account sections, all columns.", "7-genai")
p.stack_recipe("stack_report", ["portfolio_report", "account_reports"], "report_rows",
               description="Portfolio row first, then one row per account.", tags=["7-genai"])

dataset("cashflow_report", "The report: one PORTFOLIO summary row + one row per account (markdown).", "7-genai")
p.prepare_recipe("prepare_cashflow_report", "report_rows", "cashflow_report", [
    formula("scope", 'if(isBlank(account_id), "PORTFOLIO", "ACCOUNT")'),
    formula("company_name", 'if(isBlank(account_id), "All accounts", company_name)'),
    rename({"llm_output": "report_markdown"}),
    formula("generator", f'if(isBlank(llm_error_message), "llm:{LLM_ID}", concat("LLM error: ", llm_error_message))'),
    formula("generated_at", "now()"),
    keep_columns("scope", "account_id", "company_name", "risk_level", "report_markdown", "generator",
                 "generated_at"),
], "Keep the report columns and say which model wrote each row.", tags=["7-genai"])

if __name__ == "__main__":
    p.deploy()
