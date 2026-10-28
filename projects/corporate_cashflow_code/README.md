# CORP_CASHFLOW_CODE — corporate cash-flow forecast, scripted

The same use case as [`../corporate_cashflow`](../corporate_cashflow) (visual recipes + Visual ML),
built with **Python and SQL recipes**. Same synthetic data.

```bash
python3 ../corporate_cashflow/generate_data.py   # if ../corporate_cashflow/data/ does not exist yet
python3 create_project.py                         # (re)creates project CORP_CASHFLOW_CODE and builds it (~2 min)
```

Open http://localhost:10000/projects/CORP_CASHFLOW_CODE/ — the project **Wiki** walks through it.

## Flow

```
accounts_raw, transactions_raw ─[Python] prepare─► transactions_prepared, accounts
transactions_prepared ─[SQL] monthly─► monthly_cashflow ─[Python] training set─► training_set
training_set ─[Python] train + backtest + forecast─► cashflow_forecast, model_evaluation,
                                                     feature_importance, backtest_predictions
accounts, monthly_cashflow, cashflow_forecast ─[SQL] outlook─► account_outlook
account_outlook, cashflow_forecast, model_evaluation ─[Python] LLM report─► cashflow_report
```

| File | What |
|---|---|
| `create_project.py` | Project definition: datasets, recipes, tags, variables, wiki |
| `recipes/*.py` | Python recipes |
| `recipes/*.sql` | SQL recipes, run inside PostgreSQL (`${projectKey}` = table prefix) |
| `wiki.md` | Project wiki home page |

## Code vs visual: what differs

| | `CORP_CASHFLOW_CODE` (this) | `CORP_CASHFLOW` |
|---|---|---|
| Recipes | 4 Python + 2 SQL | 27 visual recipes |
| Model | 6 LightGBM models (flow x horizon), in code | 1 Visual ML LightGBM model (flow and horizon as features) |
| Evaluation | rolling backtest over 6 month-ends | held-out last 6 months (Evaluate recipe) |
| Final model | retrained on all history | trained on data up to 6 months ago |
| 10-90% range | from backtest errors | Visual ML prediction intervals |
| Report | Python + LLM Mesh API | Prompt recipes |

The LLM is the project variable `llm_id` (`openai:openai:gpt-6-luna`); no fallback, the report recipe
fails if the LLM does. **Re-running `create_project.py` deletes and re-creates the project.**
