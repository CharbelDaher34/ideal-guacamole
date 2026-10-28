# Write the cash-flow report with an LLM: one portfolio summary + one section per account.
#
# The LLM is the project variable `llm_id` (Project > ... > Variables), an LLM Mesh model such as
# "openai:openai:gpt-6-luna" (connection `openai` in Administration > Connections).
# No fallback: if the variable is empty or a call fails, the recipe fails.
import json
from datetime import datetime

import dataiku
import pandas as pd

LLM_ID = (dataiku.get_custom_variables().get("llm_id") or "").strip()
if not LLM_ID:
    raise ValueError("Set the project variable llm_id to an LLM Mesh model id")
RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def naive(s):
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


outlook = dataiku.Dataset("account_outlook").get_dataframe()
fc = dataiku.Dataset("cashflow_forecast").get_dataframe()
ev = dataiku.Dataset("model_evaluation").get_dataframe()
fc["forecast_month"] = naive(fc["forecast_month"])
outlook = outlook.sort_values(["risk_level", "projected_min_balance"],
                              key=lambda s: s.map(RISK_ORDER) if s.name == "risk_level" else s)
months = sorted(fc["forecast_month"].unique())
period = f"{pd.Timestamp(months[0]):%b %Y} - {pd.Timestamp(months[-1]):%b %Y}"
model_err = ev[ev["method"].str.startswith("LightGBM")]["inflow_wape"].mean()
base_err = ev[ev["method"].str.contains("last year")]["inflow_wape"].mean()
llm = dataiku.api_client().get_default_project().get_llm(LLM_ID)


def account_facts(r):
    f = fc[fc["account_id"] == r.account_id].sort_values("forecast_month")
    return {
        "company": r.company_name, "account_id": r.account_id, "sector": r.sector, "segment": r.segment,
        "risk_level": r.risk_level, "current_balance": round(r.current_balance), "overdraft_limit": round(r.overdraft_limit),
        "last_3_months_net": round(r.last_3m_net), "forecast_3_months_net": round(r.forecast_3m_net),
        "projected_balance_end": round(r.projected_balance), "projected_lowest_balance": round(r.projected_min_balance),
        "pessimistic_lowest_balance": round(r.worst_case_min_balance),
        "monthly_forecast": [{"month": f"{pd.Timestamp(m.forecast_month):%b %Y}", "inflow": round(m.inflow_forecast),
                              "outflow": round(m.outflow_forecast), "net": round(m.net_forecast),
                              "net_range": [round(m.net_lower), round(m.net_upper)]} for m in f.itertuples()],
    }


def ask_llm(prompt):
    resp = llm.new_completion().with_message(prompt, role="user").execute()
    if not resp.success:
        raise RuntimeError(f"LLM call to {LLM_ID} failed")
    return resp.text


ACCOUNT_PROMPT = """You are a corporate-banking credit analyst. Write a short cash-flow outlook (max 120 words, markdown)
for the relationship manager of this client, starting with a '### <company> (<account>) - risk <level>' heading.
Cover: the 3-month forecast vs recent months, the weakest month, the lowest projected balance vs the overdraft
limit, and one concrete suggested action. Use only these facts (amounts in USD):
{facts}"""

PORTFOLIO_PROMPT = """You are head of corporate-banking risk. Write an executive summary (max 200 words, markdown,
starting with '## Corporate cash-flow outlook, {period}') of this 3-month cash-flow forecast for the portfolio.
Mention the risk mix, the accounts needing attention and how reliable the model was. Facts:
{facts}"""

now, generator = datetime.utcnow(), f"llm:{LLM_ID}"
portfolio_facts = {
    "period": period, "risk_counts": outlook["risk_level"].value_counts().to_dict(),
    "portfolio_forecast_net": round(outlook["forecast_3m_net"].sum()),
    "portfolio_last_3_months_net": round(outlook["last_3m_net"].sum()),
    "model_inflow_error": round(model_err, 3), "baseline_inflow_error": round(base_err, 3),
    "accounts_needing_attention": [account_facts(r) for r in outlook[outlook["risk_level"] != "LOW"].itertuples()],
}
rows = [{"scope": "PORTFOLIO", "account_id": "", "company_name": "All accounts", "risk_level": "",
         "report_markdown": ask_llm(PORTFOLIO_PROMPT.format(period=period, facts=json.dumps(portfolio_facts, indent=1))),
         "generator": generator, "generated_at": now}]
for r in outlook.itertuples():
    rows.append({"scope": "ACCOUNT", "account_id": r.account_id, "company_name": r.company_name,
                 "risk_level": r.risk_level,
                 "report_markdown": ask_llm(ACCOUNT_PROMPT.format(facts=json.dumps(account_facts(r), indent=1))),
                 "generator": generator, "generated_at": now})

dataiku.Dataset("cashflow_report").write_with_schema(pd.DataFrame(rows))
