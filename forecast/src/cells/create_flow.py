import json
import time

import pandas as pd

# Safe to run again in the same project: what already exists is kept (datasets and their data, the category
# mapping, folders, saved models and their versions, zones), and the code of the Python recipes and the library
# is updated.
flow = project.get_flow()
lib = {}
for code in (LIBRARY_CODE, WAREHOUSE_CODE, TRACKING_CODE):   # the shared code is needed here too
    exec(code, lib)
RAW = [warehouse_table for warehouse_table, _ in lib["TABLES"].values()]
CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}

# Python recipes: (inputs, outputs, flow zone). Every dataset is a CSV file; experiments, forecast_archive and
# try_model_uploads are folders. The visual recipes (Join, Group, Score) are created further down.
# Training and inference share the data prep, then split: training_rows (Training zone) and scoring_rows (Inference
# zone) are built by two recipes, so the nightly batch never touches the training data.
FLOW = {
    "stage_tables": (RAW, list(lib["TABLES"]), "Staging"),
    "obligations": (["scheduled_with_category"], ["monthly_obligations"], "Data prep"),
    "build_training_rows": (["monthly_flows", "monthly_obligations", "clients", "monthly_balances"],
                            ["training_rows"], "Training"),
    "train_and_test": (["training_rows"], ["experiments", "test_results", "model_reference"], "Training"),
    "champion_challenger": (["training_rows", "experiments"], ["promotion_log"], "Training"),
    "build_scoring_rows": (["monthly_flows", "monthly_obligations", "clients", "monthly_balances"],
                           ["scoring_rows"], "Inference"),
    "final_forecast": (["scored_ev", "scored_p10", "scored_p90", "monthly_obligations", "promotion_log"],
                       ["final_forecast"], "Inference"),
    "archive_forecast": (["final_forecast"], ["forecast_archive", "inference_log"], "Inference"),
    # upload CSV files of clients to the folder, build try_model_forecast: the production model's forecast for them
    "try_the_model": (["try_model_uploads", "experiments"], ["try_model_forecast", "try_model_charts"],
                      "Try the model"),
}
FOLDERS = ["experiments", "forecast_archive", "try_model_uploads", "try_model_charts"]
REPLACED = ["build_rows"]                           # recipes of earlier versions, deleted (their outputs are kept)
APPENDED = ["promotion_log", "inference_log"]       # logs: each run adds a row
BUILT_BY_NAME = ["experiments", "promotion_log"]    # never rebuilt by building what comes after them

try:                                                # needs the project's admin right
    metadata = project.get_metadata()
    metadata["label"] = PROJECT_NAME
    project.set_metadata(metadata)
except Exception as e:
    if not any(w in str(e).lower() for w in ("unauthorized", "forbidden", "permission", "not allowed")):
        raise
    print(f"Warning: could not name the project {PROJECT_NAME!r} (no admin right on the project): name it by hand")

library = project.get_library()
python_folder = library.get_folder("python") or library.add_folder("python")
for file_name, code in (("cashflow.py", LIBRARY_CODE), ("warehouse.py", WAREHOUSE_CODE),
                        ("tracking.py", TRACKING_CODE)):
    (python_folder.get_file(file_name) or python_folder.add_file(file_name)).write(code)


# what the project already has (read once, kept up to date below)
datasets = {d["name"] for d in project.list_datasets()}
recipes = {r["name"] for r in project.list_recipes()}
for name in REPLACED:
    if name in recipes:
        project.get_recipe(name).delete()
        recipes.discard(name)
        print("deleted recipe", name, "(replaced)")


def new_dataset(name):
    """Managed dataset of the project: a CSV file (comma-separated, with a header row) on CONNECTION. Kept if it
    exists."""
    if name in datasets:
        return name
    datasets.add(name)
    dataset = project.new_managed_dataset(name).with_store_into(CONNECTION).create()
    settings = dataset.get_settings()
    settings.get_raw()["formatType"] = "csv"
    settings.get_raw()["formatParams"].update(CSV)
    if name in BUILT_BY_NAME:
        settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"
    settings.save()
    return name


# the curated warehouse tables and the load record: already here if the warehouse simulator wrote them (sandbox),
# otherwise CSV files in the warehouse team's landing folder, read-only (production)
simulated = [name for name in RAW + [lib["LOAD_RECORD"]] if name in datasets]
for name in RAW + [lib["LOAD_RECORD"]]:
    if name not in datasets:
        datasets.add(name)
        dataset = project.create_dataset(name, "Filesystem", params={"connection": LANDING_CONNECTION,
                                                                     "path": "/" + name},
                                          formatType="csv", formatParams=CSV)
        dataset.autodetect_settings().save()
print(f"{len(simulated)} warehouse tables already in the project, {len(RAW) + 1 - len(simulated)} linked from "
      f"{LANDING_CONNECTION}")

# category mapping: a CSV dataset with the default mapping, written only when it is created (a mapping already
# here, maybe edited, is kept). To change it: a notebook reads it, changes rows and writes it back.
if "category_mapping" not in datasets:
    dataiku.Dataset(new_dataset("category_mapping")).write_with_schema(pd.DataFrame(
        lib["DEFAULT_CATEGORY_MAPPING"],
        columns=["map_origin_type", "map_txn_type_tag", "map_direction", "mapped_category"]))
    print("category_mapping: default mapping written")

existing_folders = {f["name"]: f["id"] for f in project.list_managed_folders()}
folders = {}
for name in FOLDERS:
    if name in existing_folders:
        folders[name] = project.get_managed_folder(existing_folders[name])
        continue
    folders[name] = project.create_managed_folder(name, connection_name=FOLDER_CONNECTION)
    if name in BUILT_BY_NAME:
        folder_settings = folders[name].get_settings()
        folder_settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"
        folder_settings.save()
ref = lambda name: folders[name].id if name in folders else name
existing_models = {m["name"]: m["id"] for m in project.list_saved_models()}
saved_models = {name: project.get_saved_model(existing_models[name]) if name in existing_models
                else project.create_mlflow_pyfunc_model(name, "REGRESSION") for name in lib["SAVED_MODELS"].values()}

# visual recipes: categories by Join with category_mapping (+ a formula for anything not mapped), then Group
NO_FILTER = {"enabled": False, "distinct": False}


def join_mapping(name, source, output):
    if name in recipes:
        recipe = project.get_recipe(name)
    else:
        builder = project.new_recipe("join", name)
        builder.with_input(source)
        builder.with_input("category_mapping")
        builder.with_existing_output(new_dataset(output))
        recipe = builder.create()
        recipes.add(name)
    settings = recipe.get_settings()
    if not settings.get_recipe_inputs():          # the Join creator does not always register its inputs
        settings.add_input("main", source)
        settings.add_input("main", "category_mapping")
    settings.set_payload(json.dumps({
        "virtualInputs": [{"index": i, "preFilter": NO_FILTER, "computedColumns": [],
                           "outputColumnsSelectionMode": "AUTO_NON_CONFLICTING"} for i in (0, 1)],
        "joins": [{"table1": 0, "table2": 1, "type": "LEFT", "conditionsMode": "AND", "outerJoinOnTheLeft": True,
                   "on": [{"column1": {"table": 0, "name": c}, "column2": {"table": 1, "name": "map_" + c},
                           "type": "EQ"} for c in ("origin_type", "txn_type_tag", "direction")]}],
        "selectedColumns": [], "resolvedSelectedColumns": [], "postFilter": NO_FILTER,
        "computedColumns": [{"name": "category", "type": "string", "mode": "GREL", "expr": lib["CATEGORY_FORMULA"]}],
        "enableAutoCastInJoinConditions": False}))
    settings.save()
    print("join recipe", name, "->", output)


join_mapping("join_flows_category", "monthly_flows_by_origin", "flows_with_category")
join_mapping("join_scheduled_category", "scheduled_payments", "scheduled_with_category")
if "group_monthly_flows" in recipes:
    group = project.get_recipe("group_monthly_flows").get_settings()
else:
    builder = project.new_recipe("grouping", "group_monthly_flows")
    builder.with_input("flows_with_category")
    builder.with_existing_output(new_dataset("monthly_flows"))
    group = builder.create().get_settings()
    recipes.add("group_monthly_flows")
summed = ("amount", "known_1", "known_2", "known_3", "postings")
group.set_payload(json.dumps({
    "keys": [{"column": k} for k in ("client_no", "month", "category", "direction")],
    "values": [{"column": c, "sum": True} for c in summed], "globalCount": False,
    "preFilter": NO_FILTER, "postFilter": NO_FILTER, "computedColumns": [],
    "outputColumnNameOverrides": {f"{c}_sum": c for c in summed}, "enlargeYourBits": True}))
group.save()
print("group recipe group_monthly_flows -> monthly_flows")

for name, (inputs, outputs, _) in FLOW.items():
    for item in inputs + outputs:      # inputs made by recipes created later (Score) get their dataset now
        if item not in folders:
            new_dataset(item)
    if name in recipes:                # already there: its code and code env are updated, missing inputs / outputs added
        recipe = project.get_recipe(name)
        settings = recipe.get_settings()
        listed = lambda roles: {item["ref"] for role in roles.values() for item in role["items"]}
        for i in inputs:
            if ref(i) not in listed(settings.get_recipe_inputs()):
                settings.add_input("main", ref(i))
        for o in outputs:
            if ref(o) not in listed(settings.get_recipe_outputs()):
                settings.add_output("main", ref(o), append_mode=o in APPENDED)
        settings.save()
    else:
        builder = project.new_recipe("python", name)
        for i in inputs:
            builder.with_input(ref(i))
        for o in outputs:
            builder.with_output(ref(o), append=o in APPENDED)
        recipe = builder.create()
        recipes.add(name)
    settings = recipe.get_settings()
    settings.set_payload(RECIPES[name])
    settings.set_code_env(code_env=CODE_ENV)
    settings.save()
    print("python recipe", name, "->", ", ".join(outputs))

# Score recipes: the active version of each saved model
for short, name in lib["SAVED_MODELS"].items():
    if f"score_{short}" in recipes:
        continue
    builder = project.new_recipe("prediction_scoring", f"score_{short}")
    builder.with_input_model(saved_models[name].id)
    builder.with_input("scoring_rows")
    builder.with_existing_output(new_dataset(f"scored_{short}"))
    builder.create().rename(f"score_{short}")        # Dataiku names a Score recipe after its input
    print("score recipe", f"score_{short}", "->", f"scored_{short}")

ZONES = {"Raw data": RAW + [lib["LOAD_RECORD"]], "Staging": list(lib["TABLES"]),
         "Data prep": ["category_mapping", "flows_with_category", "monthly_flows", "scheduled_with_category"],
         "Training": [], "Inference": ["scored_ev", "scored_p10", "scored_p90"], "Try the model": ["try_model_uploads"]}
for _, outputs, zone_name in FLOW.values():
    ZONES[zone_name] += [o for o in outputs if o not in ZONES[zone_name]]
# the raw data goes in Dataiku's Default zone, renamed (so there is no empty zone). It exists once another zone does.
existing_zones = {z.name: z for z in flow.list_zones()}
zones = {z: existing_zones.get(z) or flow.create_zone(z) for z in ZONES if z != "Raw data"}
zones["Raw data"] = existing_zones.get("Raw data") or flow.get_default_zone()
if zones["Raw data"].name != "Raw data":
    zone_settings = zones["Raw data"].get_settings()
    zone_settings.name = "Raw data"
    zone_settings.save()


def safe_add(zone, item):
    """Put an item in a zone (already there: nothing to do)."""
    try:
        zone.add_item(item)
    except Exception as e:
        if "already" not in str(e).lower():
            raise


for zone_name, items in ZONES.items():
    for item in items:
        safe_add(zones[zone_name], folders[item] if item in folders else project.get_dataset(item))
for model in saved_models.values():       # next to the Score recipes that use them
    safe_add(zones["Inference"], model)
print("Flow ready:", ", ".join(zones))
