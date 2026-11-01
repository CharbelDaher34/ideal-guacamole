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

The two cash-flow projects are explained in two documents: how the forecast works,
[`docs/HOW_FORECASTING_WORKS.html`](docs/HOW_FORECASTING_WORKS.html), and how to work with them in
Dataiku, [`docs/DSS_WALKTHROUGH.md`](docs/DSS_WALKTHROUGH.md).

| Folder | Project key | What |
|---|---|---|
| `corporate_cashflow/` | `CORP_CASHFLOW` | Corporate banking: money in and expenses forecast day by day for the rest of the month + 3 months, loan installments from their schedule, balance per day and per month, risk and drivers. Visual recipes and Visual ML **time series** (DeepAR vs LightGBM), no code (see its wiki) |
| `corporate_cashflow_code/` | `CORP_CASHFLOW_CODE` | Same data, goal and outputs, scripted: Python + SQL recipes, **gradient boosting** (scikit-learn) on calendar + past values (see its wiki) |

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
| `prepare_recipe` | Prepare | list of steps: `formula`, `parse_date`, `truncate_date`, `remove_empty`, `keep_rows`, `fold`, `rename`, `keep_columns`, `delete_columns`, or any processor with `step(type, **params)` |
| `group_recipe` | Group | keys, `{column: [aggregations]}`, `computed=` pre-computed columns, `names=` output renames |
| `window_recipe` | Window | partition, order, `{column: {"lag": [..], "lead": [..], "sum": True}}` (default frame: running total), `frame=(preceding, following)` |
| `join_recipe` | Join | inputs, key columns (or `{dataset: keys}`, a key can be `(left, right)` when names differ), `how=` LEFT / INNER / CROSS, `columns=`, `pre_filters=` |
| `stack_recipe` | Stack | inputs |
| `split_recipe` | Split | column, `{value: output dataset}` |
| `prompt_recipe` | Prompt (LLM Mesh) | LLM id, prompt template with `{{column}}` placeholders |

Visual ML: `ml_model(name, train, test, target, reject=, params=, interval=, explicit=)` (regression: Lab analysis
trained and deployed to the flow as saved model `name`), `timeseries_model(name, input, target, time, ids,
horizon, algorithms, env, external=, explicit=)` (time series forecasting: trains every algorithm in `algorithms`,
deploys the best on MASE; `external` = columns known in advance, then the dataset to score must also hold
the future rows with an empty target), `evaluate_recipe`, `score_recipe`. `ensure_code_env(name)` creates a DSS code env with the
Visual ML + time series packages if it does not exist (time series models need it).

`explicit=True`: the saved model is only retrained when it is built by name (a scenario step), never by
building what is downstream of it.

Code recipes: `python_recipe`, `sql_recipe`; `managed_folder(name, explicit=)` for files (e.g. a pickled
model, `dataiku.Folder(name)` in Python); `library(filename, code)` adds a module to the project library,
importable from every Python recipe. All recipes take `description=` and `tags=`;
`wiki_home` and `Project(..., variables={...})` for the wiki and project variables.

Training and inference in one project: `zone(name, tags=, items=, shared=)` groups the items with those tags
into a flow zone (`shared` = inputs from another zone, shown in both), and `scenario(name, steps)` saves a list
of steps to run from the Scenarios tab: `build_step(name, items, mode, zone_only=)`, `rules_step`,
`metrics_step`, `python_step` (raise to stop the scenario), `scenario_step` (run another one). The Free Edition
has no time-based triggers and no API or `dsscli` scenario runs: run them in the UI. Zones and scenarios are created
at the end of `deploy()`, once the saved models exist.

Checks and results: `data_quality(dataset, rules)` with `rule_record_count`, `rule_not_empty`, `rule_unique`,
`rule_python`; `metrics(dataset, python=, names=)` (record count + a Python probe, ids from `python_metric`);
`dashboard(name, [(tile, (left, top, width, height))])` with `tile_text`, `tile_table`, `tile_chart`,
`tile_metric`, `tile_scenario` on a 36-column grid.
Formulas are Dataiku formulas (GREL); on PostgreSQL they are translated to SQL, so build text
with `concat(...)`. Column types: `string`, `bigint`, `int`, `double`, `boolean`, `date` (timestamp), `dateonly` (a day, e.g.
`2026-09-30` in a CSV).

**`deploy()` deletes and re-imports a project with the same key**: anything changed in the
UI for that project is lost. Treat the Python file as the source of truth, or use a new key.
