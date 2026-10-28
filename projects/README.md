# Projects as code

Define a Dataiku project (datasets + recipes) in Python and push it into the local DSS.
The project then shows up in the DSS UI with its full flow.

The Free Edition has no public REST API / API keys, so `dssproj.py` writes a project
export archive, loads it with `dsscli project-import` inside the `dataiku-dss` container,
then runs the recipes one by one in flow order with `dsscli build`. Before running a visual
recipe it asks DSS (internal UI API, admin/admin) to compute the recipe's output schema, as the
UI does when you save a recipe. Visual ML models are created through the same calls as the UI:
Lab analysis, training, deploy to the flow, then their Evaluate / Score recipes.

## Projects

| Folder | Project key | What |
|---|---|---|
| `demo/` | `DEV_DEMO` | Minimal example: CSV -> Python -> SQL |
| `corporate_cashflow/` | `CORP_CASHFLOW` | Corporate-banking next-month cash-flow forecast + GenAI notes, 10 visual recipes and Visual ML, no code (see its README) |
| `corporate_cashflow_code/` | `CORP_CASHFLOW_CODE` | Same use case and data, scripted: 4 Python / SQL recipes, scikit-learn model (see its README) |

## Run the demo

```bash
python3 demo/create_demo.py
```

Creates project `DEV_DEMO`:

```
orders_raw (uploaded CSV) -> [Python] compute_orders_clean -> orders_clean (Postgres)
                          -> [SQL]    compute_customer_totals -> customer_totals (Postgres)
```

Tables land in PostgreSQL as `DEV_DEMO_orders_clean` and `DEV_DEMO_customer_totals`.

## Write your own

```python
from dssproj import Project, col, formula, parse_date

p = Project("MY_PROJECT", "My project")
p.csv_dataset("raw", "data.csv", [col("id", "bigint"), col("day"), col("amount", "double")])
p.pg_dataset("clean", [])             # schema computed by DSS from the recipe
p.prepare_recipe("prepare_raw", "raw", "clean", [parse_date("day"), formula("amount_k", "amount / 1000")])
p.pg_dataset("daily", [])
p.group_recipe("group_daily", "clean", "daily", ["day"], {"amount": ["sum", "avg"]})
p.deploy()
```

Datasets: `csv_dataset`, `pg_dataset` (table `<PROJECTKEY>_<name>` on `postgres_local`).

Visual recipes (prefer these; open them in the UI to see the settings):

| Helper | Dataiku recipe | Main arguments |
|---|---|---|
| `prepare_recipe` | Prepare | list of steps: `formula`, `parse_date`, `truncate_date`, `remove_empty`, `keep_rows`, `rename`, `keep_columns`, `delete_columns`, or any processor with `step(type, **params)` |
| `group_recipe` | Group | keys, `{column: [aggregations]}`, `computed=` pre-computed columns, `names=` output renames |
| `window_recipe` | Window | partition, order, `{column: {"lag": [..], "lead": [..], "avg": True}}`, `frame=(preceding, following)` |
| `join_recipe` | Join | inputs, key columns, `how=` LEFT / INNER / CROSS, `columns=`, `pre_filters=` |
| `stack_recipe` | Stack | inputs |
| `split_recipe` | Split | column, `{value: output dataset}` |
| `prompt_recipe` | Prompt (LLM Mesh) | LLM id, prompt template with `{{column}}` placeholders |

Visual ML: `ml_model(name, train, test, target, reject=, params=, interval=)` (Lab analysis trained
and deployed to the flow as saved model `name`), `evaluate_recipe`, `score_recipe`.

Code recipes: `python_recipe`, `sql_recipe`. All recipes take `description=` and `tags=`;
`wiki_home` and `Project(..., variables={...})` for the wiki and project variables.
Formulas are Dataiku formulas (GREL); on PostgreSQL they are translated to SQL, so build text
with `concat(...)`. Column types: `string`, `bigint`, `int`, `double`, `boolean`, `date`.

**`deploy()` deletes and re-imports a project with the same key**: anything changed in the
UI for that project is lost. Treat the Python file as the source of truth, or use a new key.
