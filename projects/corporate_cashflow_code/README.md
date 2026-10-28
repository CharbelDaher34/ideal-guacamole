# CORP_CASHFLOW_CODE — corporate cash-flow forecast, scripted

The same data as [`../corporate_cashflow`](../corporate_cashflow) (visual recipes + time series forecasting),
built with **Python and SQL recipes** and a **regression** on hand-made features instead. Forecasts next month.

```bash
python3 ../corporate_cashflow/generate_data.py   # if ../corporate_cashflow/data/ does not exist yet
python3 create_project.py                         # (re)creates project CORP_CASHFLOW_CODE and builds it (~1 min)
```

Open http://localhost:10000/projects/CORP_CASHFLOW_CODE/ — the project **Wiki** walks through it.

## Flow (4 recipes)

```
transactions ─[Python]─► monthly_net ─[SQL]─► features
features, accounts ─[Python]─► account_outlook, model_evaluation
account_outlook ─[Python + LLM]─► account_notes
```

| File | What |
|---|---|
| `create_project.py` | Project definition: datasets, recipes, tags, variables, wiki |
| `recipes/monthly_net.py` | Net cash flow per account and month (pandas) |
| `recipes/features.sql` | 3-month average, same month last year, next month, row role (window functions in PostgreSQL; test = Jun - Aug 2026) |
| `recipes/forecast.py` | Linear regression (scikit-learn): train, test vs baselines, forecast, risk level |
| `recipes/account_notes.py` | One LLM note per account (LLM Mesh) |
| `wiki.md` | Project wiki home page |

## Code vs visual

| | `CORP_CASHFLOW_CODE` (this) | `CORP_CASHFLOW` |
|---|---|---|
| Recipes | 3 Python + 1 SQL | 7 visual recipes + 1 Visual ML training |
| Approach | regression: features built by hand (SQL window functions) | time series forecasting: DSS handles time |
| Model | scikit-learn `LinearRegression` | best of 9 time series algorithms (ARIMA, ETS, Prophet, DeepAR...) |
| Horizon | next month | next 3 months |
| Report | Python + LLM Mesh API | Prompt recipe |

The LLM is the project variable `llm_id` (`openai:openai:gpt-6-luna`); no fallback, the notes recipe
fails if the LLM does. **Re-running `create_project.py` deletes and re-creates the project.**
