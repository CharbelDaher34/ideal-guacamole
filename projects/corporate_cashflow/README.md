# CORP_CASHFLOW — corporate cash-flow forecast (time series)

Forecast each corporate account's net cash flow for the next 3 months with Dataiku's **time series
forecasting**, flag liquidity risk, and have an LLM write a short note per account. No code recipes:
visual recipes and Visual ML. Synthetic data only.

```bash
python3 generate_data.py     # data/accounts.csv (10 accounts), data/transactions.csv (2,640 rows, Oct 2023 - Sep 2026)
python3 create_project.py    # (re)creates project CORP_CASHFLOW in DSS, trains 9 algorithms, runs the flow (~1 min)
```

The first run also creates the DSS code env `timeseries` (DSS's "Visual ML, Causal ML, and Timeseries
Forecasting (CPU)" packages: statsmodels, pmdarima, prophet, gluonts, CPU torch; ~3 GB, a few minutes).

Open http://localhost:10000/projects/CORP_CASHFLOW/ — the project **Wiki** ("Start here") walks through it.

## Flow (8 recipes)

```
transactions ─[Prepare]─► transactions_prepared ─[Group]─► monthly_net        (the time series)
monthly_net ─[Visual ML time series: 9 algorithms, best deployed]─► cashflow_forecast_model
monthly_net ─[Score]─► forecast ─[Group]─► forecast_3m ─[Join + accounts]─► ─[Prepare]─► account_outlook
account_outlook ─[Prompt]─► account_notes
```

The model is created at deploy time like you would in the UI: a Lab analysis on `monthly_net` (target
`net`, time `month`, identifier `account_id`, horizon 3), the algorithms in `ALGORITHMS` trained and
compared on the last 3 months, and the best on MASE deployed to the flow with its Score recipe. The Lab
analysis stays in the project to compare, tune and redeploy.

All managed datasets are PostgreSQL tables named `CORP_CASHFLOW_<dataset>`.

| File | What |
|---|---|
| `generate_data.py` | Synthetic accounts and transactions (seeded) |
| `create_project.py` | Project definition: datasets, visual recipes, algorithms to compare, prompt, tags, wiki |
| `wiki.md` | Project wiki home page |

## Results of the current build

- Test months Jul - Sep 2026, MASE (below 1 = better than "same as last month"): ARIMA 0.68 (deployed),
  ETS 0.70, AutoARIMA 0.78, Seasonal LOESS 0.79, feed-forward NN 0.86, Prophet 0.89, naive 0.90,
  DeepAR 0.91, seasonal naive 0.98.
- Risk at end of Dec 2026: 2 HIGH (Levant Logistics, Tripoli Textiles), 3 MEDIUM (Mount Lebanon Hotels,
  Byblos Construction, Sidon Port Services), 5 LOW.

## GenAI

The **Prompt** recipe `prompt_account_notes` calls `LLM_ID` in `create_project.py`
(`openai:openai:gpt-6-luna`, LLM Mesh connection `openai`): 10 calls per run, with each account's
(synthetic) figures. A failed call shows in the `llm_error_message` column.

**Re-running `create_project.py` deletes and re-creates the project** — changes made in the UI (including
models trained in the Lab) are lost; put them in the code instead.
