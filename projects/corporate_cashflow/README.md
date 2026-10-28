# CORP_CASHFLOW — corporate cash-flow forecast

Forecast each corporate account's next 3 months of cash flow, flag liquidity risk, and have an LLM write
a report. No code recipes: Dataiku visual recipes and Visual ML. Synthetic data only.

```bash
python3 generate_data.py     # data/accounts.csv (30 accounts), data/transactions.csv (~17k rows, Oct 2023 - Sep 2026)
python3 create_project.py    # (re)creates project CORP_CASHFLOW in DSS and runs every recipe and trains the model (~6 min)
```

Open http://localhost:10000/projects/CORP_CASHFLOW/ — the project **Wiki** ("Start here") walks through it.

## Flow

```
transactions_raw ─[Prepare]─► transactions_prepared ─[Group]─► monthly_cashflow ─[Group]─► account_balance
accounts_raw ─────[Prepare]─► accounts
monthly_cashflow ─[Window]─► monthly_history ─[Join + accounts, account_balance]─► ─[Prepare]─► training_set

training_set ─[Prepare: unfold flow x horizon]─► ml_dataset ─[Split]─► ml_train, ml_test, ml_forecast
ml_train (+ ml_test) ─[Visual ML training]─► saved model cashflow_model
ml_test ─[Evaluate]─► test_predictions, model_metrics ─► [Group] ─► [Prepare] ─► model_evaluation
ml_forecast ─[Score]─► forecast_scored ─[Prepare]─[Group]─[Prepare]─► cashflow_forecast

cashflow_forecast ─[Window]─[Group]─► forecast_summary ─┐
accounts, account_balance, training_set ────────────────┴─[Join]─[Prepare]─► account_outlook
account_outlook ─[Prompt]─► account_reports ─────────────────────────────────┐
account_outlook ─[Group]─► portfolio_facts ─┐                                ├─[Stack]─[Prepare]─► cashflow_report
model_evaluation ─[Group]─► model_accuracy ─┴─[Join]─[Prompt]─► portfolio_report
```

The model is created at deploy time like you would in the UI: a Lab analysis on `ml_train` (test set
`ml_test`), trained, then deployed to the flow with its Evaluate and Score recipes. The Lab analysis stays
in the project (**Lab** menu) to compare other algorithms.

Visual recipes on PostgreSQL datasets run inside the database (Dataiku generates the SQL). In their
formulas, build text with `concat(a, b, ...)`, not `a + b` (SQL `+` is numeric only).

All managed datasets are PostgreSQL tables named `CORP_CASHFLOW_<dataset>`.

| Folder / file | What |
|---|---|
| `generate_data.py` | Synthetic accounts and transactions (seeded) |
| `create_project.py` | Project definition: datasets, visual recipes (steps, aggregations, joins, prompts), model settings, tags, wiki |
| `wiki.md` | Project wiki home page |

## Results of the current build

- Held-out months (target in the last 6 months): model error on monthly inflows 13-15% (WAPE) vs 17%
  for "same month last year" and 14-22% for "average of last 3 months"; the model wins on every flow and
  horizon (`model_evaluation`).
- Risk: 3 HIGH (Tripoli Textiles, Levant Logistics, Union Electronics), 2 MEDIUM (Prime Freight Lines,
  Phoenicia Retail Group), 25 LOW.

## GenAI

Two **Prompt** recipes (`prompt_account_reports`, `prompt_portfolio_report`) call `LLM_ID` in
`create_project.py`, now `openai:openai:gpt-6-luna` (LLM Mesh connection `openai`, Administration >
Connections). The prompt templates are in `create_project.py`; `{{column}}` is replaced by the row's value.
Each run sends each account's figures (synthetic) to OpenAI: 31 calls. The `generator` column of
`cashflow_report` shows which model wrote each row, or the LLM error.

The API key lives only in the DSS connection and in `dataiku/.openai_key` (mode 600).

**Re-running `create_project.py` deletes and re-creates the project** — changes made in the UI
(including prompt edits) are lost; put them in the code instead.
