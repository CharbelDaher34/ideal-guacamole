import time

import pandas as pd

flow = project.get_flow()
if project.list_recipes():
    raise RuntimeError("This project already has a flow. This notebook sets up an empty project only; "
                       "change the flow in Dataiku, or create a new project to set it up again.")
lib = {}
exec(LIBRARY_CODE, lib)     # the shared settings (FEATURES, ...) are needed here too, for the visual models
exec(WAREHOUSE_CODE, lib)   # the curated warehouse tables (TABLES, COLUMNS, staging_sql)
RAW = [warehouse_table for warehouse_table, _ in lib["TABLES"].values()] + [lib["LOAD_RECORD"]]

# recipe: (inputs, outputs, flow zone); "models" is a managed folder, everything else a dataset
FLOW = {
    "load_check": (RAW, ["load_status"], "Staging"),
    "categorise": (["monthly_flows_by_origin", "category_rules"], ["monthly_flows"], "Data prep"),
    "obligations": (["load_status", "scheduled_payments", "category_rules"], ["monthly_obligations"], "Data prep"),
    "build_rows": (["load_status", "monthly_flows", "monthly_obligations", "clients", "monthly_balances"],
                   ["training_rows", "scoring_rows"], "Data prep"),
    "train_models": (["load_status", "training_rows"], ["models"], "Training (code)"),
    "test_models": (["load_status", "training_rows", "models"], ["test_predictions", "test_results"],
                    "Training (code)"),
    "ml_rows": (["load_status", "training_rows", "scoring_rows"], ["ml_train", "ml_test", "ml_scoring"],
                "Training (visual)"),
    "score": (["scoring_rows", "models"], ["predictions"], "Inference"),
}
# created after the visual models exist (they read their Evaluate and Score outputs)
FLOW_AFTER_VISUAL = {
    "visual_predictions": (["scored_ev_visual", "scored_p10_visual", "scored_p90_visual"], ["predictions_visual"],
                           "Inference"),
    "compare_models": (["test_predictions", "eval_ev_visual_scored", "eval_p10_visual_scored",
                        "eval_p90_visual_scored"], ["model_comparison"], "Inference"),
    "final_forecast": (["load_status", "predictions", "predictions_visual", "monthly_obligations"], ["final_forecast"],
                       "Inference"),
}

variables = project.get_variables()
variables["standard"]["model_source"] = MODEL_SOURCE
project.set_variables(variables)

library = project.get_library()
python_folder = library.get_folder("python") or library.add_folder("python")
for file_name, code in (("cashflow.py", LIBRARY_CODE), ("warehouse.py", WAREHOUSE_CODE)):
    (python_folder.get_file(file_name) or python_folder.add_file(file_name)).write(code)


def new_dataset(name):
    """Managed dataset of the project on CONNECTION; on a SQL connection its table is <PROJECTKEY>_<name>."""
    dataset = project.new_managed_dataset(name).with_store_into(CONNECTION).create()
    settings = dataset.get_settings()
    if "table" in settings.get_raw()["params"]:
        settings.get_raw()["params"]["table"] = "${projectKey}_" + name
        settings.save()


# the curated warehouse tables: read-only, through the EDW_AI connection
for name in RAW:
    dataset = project.create_dataset(name, "PostgreSQL", params={
        "connection": EDW_CONNECTION, "mode": "table", "schema": EDW_SCHEMA, "table": name})
    dataset.autodetect_settings().save()
print(f"{len(RAW)} warehouse tables linked (read-only, {EDW_CONNECTION})")

# category rules: an editable dataset (Flow > category_rules > Edit), filled with the default rules
project.create_dataset("category_rules", "Inline", params={}, formatType="", formatParams={})
dataiku.Dataset("category_rules").write_with_schema(pd.DataFrame(
    [(n, *rule) for n, rule in enumerate(lib["DEFAULT_CATEGORY_RULES"], 1)],
    columns=["rule_order", "origin_type", "txn_type_tag", "direction", "category"]))

models = project.create_managed_folder("models", connection_name=FOLDER_CONNECTION)
folder_settings = models.get_settings()
folder_settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"   # retrained only when asked
folder_settings.save()
ref = lambda name: models.id if name == "models" else name
existing = {d["name"] for d in project.list_datasets()}


def recipe(kind, name, inputs, outputs, code):
    """A Python or SQL recipe (SQL runs inside the database)."""
    for out in outputs:
        if out != "models" and out not in existing:
            new_dataset(out)
            existing.add(out)
    builder = project.new_recipe("python" if kind == "python" else "sql_query", name)
    for i in inputs:
        builder.with_input(ref(i))
    for o in outputs:
        builder.with_output(ref(o))
    created = builder.create()
    settings = created.get_settings()
    settings.set_payload(code)
    if kind == "python":
        settings.set_code_env(code_env=CODE_ENV)
    settings.save()
    if kind == "sql" and all(i in RAW for i in inputs):    # other SQL recipes get their schema on their first run
        created.compute_schema_updates().apply()
    print(f"{kind} recipe", name, "->", ", ".join(outputs))


# staging: one SQL recipe per warehouse table, warehouse names back to the standard names
for table, (warehouse_table, _) in lib["TABLES"].items():
    recipe("sql", f"stage_{table}", [warehouse_table], [table], lib["staging_sql"](table, EDW_SCHEMA))
for name, (inputs, outputs, _) in FLOW.items():
    recipe("sql" if name in SQL_RECIPES else "python", name, inputs, outputs, SQL_RECIPES.get(name) or RECIPES[name])
