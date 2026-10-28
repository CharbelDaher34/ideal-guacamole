# One LLM-written note per account. The LLM is the project variable `llm_id` (Project > ... > Variables),
# an LLM Mesh model of the connection `openai`. No fallback: if a call fails, the recipe fails.
import dataiku
import pandas as pd

LLM_ID = dataiku.get_custom_variables()["llm_id"]
llm = dataiku.api_client().get_default_project().get_llm(LLM_ID)

PROMPT = """Write a cash-flow note (max 80 words, plain text) for the relationship manager of {company_name} ({sector}).
Facts in USD: balance today {current_balance:.0f}, overdraft limit {overdraft_limit:.0f},
net cash flow last month {last_month_net:.0f}, average of the last 3 months {avg_3m_net:.0f},
forecast for next month {forecast_net:.0f}, projected balance {projected_balance:.0f}, risk level {risk_level}.
Say what is happening and suggest one action."""

outlook = dataiku.Dataset("account_outlook").get_dataframe()
notes = []
for row in outlook.to_dict("records"):
    resp = (llm.new_completion()
            .with_message("You are a corporate-banking credit analyst. Use only the facts given.", role="system")
            .with_message(PROMPT.format(**row), role="user")
            .execute())
    if not resp.success:
        raise RuntimeError(f"LLM call to {LLM_ID} failed for {row['company_name']}")
    notes.append(resp.text)

outlook["note"] = notes
dataiku.Dataset("account_notes").write_with_schema(outlook)
