# CORP_CASHFLOW — corporate cash-flow forecast

Forecast each corporate account's net cash flow for next month, flag liquidity risk, and have an LLM
write a short note per account. No code recipes: Dataiku visual recipes and Visual ML. Synthetic data only.

```bash
python3 generate_data.py     # data/accounts.csv (10 accounts), data/transactions.csv (1,760 rows, Oct 2024 - Sep 2026)
python3 create_project.py    # (re)creates project CORP_CASHFLOW in DSS, runs every recipe, trains the model (~1 min)
```

Open http://localhost:10000/projects/CORP_CASHFLOW/ — the project **Wiki** ("Start here") walks through it.

## Flow (10 recipes)

```
transactions ─[Prepare]─► transactions_prepared ─[Group]─► monthly_net ─[Window]─► features
features ─[Prepare]─► features_roles ─[Split]─► ml_train, ml_test, ml_forecast
ml_train (+ ml_test) ─[Visual ML]─► next_month_model
ml_forecast ─[Score]─► forecast ─[Join + accounts]─► forecast_accounts ─[Prepare]─► account_outlook
account_outlook ─[Prompt]─► account_notes
```

The model is created at deploy time like you would in the UI: a Lab analysis on `ml_train` (test set
`ml_test`), trained, then deployed to the flow with its Score recipe. The Lab analysis stays in the
project (**Lab** menu) to try other algorithms.

All managed datasets are PostgreSQL tables named `CORP_CASHFLOW_<dataset>`; visual recipes on them run
inside the database (Dataiku generates the SQL).

| File | What |
|---|---|
| `generate_data.py` | Synthetic accounts and transactions (seeded) |
| `create_project.py` | Project definition: datasets, visual recipes, model settings, prompt, tags, wiki |
| `wiki.md` | Project wiki home page |

## Results of the current build

- Test months (Jun - Aug 2026): average error $35.5k for the model, vs $45.9k for "same month last year"
  and $52.8k for "average of the last 3 months". R² 0.89.
- Risk for Oct 2026: 2 HIGH (Levant Logistics, Tripoli Textiles), 1 MEDIUM (Sidon Port Services), 7 LOW.

## GenAI

The **Prompt** recipe `prompt_account_notes` calls `LLM_ID` in `create_project.py`
(`openai:openai:gpt-6-luna`, LLM Mesh connection `openai`): 10 calls per run, with each account's
(synthetic) figures. `{{column}}` in the template is replaced by the row's value. A failed call shows in
the `llm_error_message` column.

**Re-running `create_project.py` deletes and re-creates the project** — changes made in the UI are lost;
put them in the code instead.
