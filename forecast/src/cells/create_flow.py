import json
import sys
import time
import types

import pandas as pd

# Safe to run again in the same project: what already exists is kept (datasets and their data, the category
# mapping, folders, saved models and their versions, zones, the project variables forecaster and lab_forecaster,
# forecasters.py), and the code of the Python recipes and of the rest of the library is updated.
flow = project.get_flow()


def load_module(name, code):
    """A library file as a module of this notebook: the code it writes to the project library."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


cashflow = load_module("cashflow", LIBRARY_CODE)
warehouse = load_module("warehouse", WAREHOUSE_CODE)
tracking = load_module("tracking", TRACKING_CODE)
RAW = [warehouse_table for warehouse_table, _ in warehouse.TABLES.values()]
CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}

# Python recipes: (inputs, outputs, flow zone). Every dataset is a CSV file; experiments, forecast_archive and
# lab_experiments are folders. The visual recipes (Join, Group, Score) are created further down.
# Each zone has an owner, and the zones only meet through three interfaces:
#   1. the model rows (cashflow.model_rows): data prep -> training_rows (Training) and scoring_rows (Inference)
#   2. the forecaster (models.py): lab -> training. The project variable forecaster names the one Retrain trains;
#      the Lab (try_forecaster, and the notebook forecast_lab) compares one with the production model, untouched.
#   3. the saved models' active version: training -> inference. Inference reads nothing else from training.
FLOW = {
    "stage_tables": (RAW, list(warehouse.TABLES), "Staging"),
    "obligations": (["scheduled_with_category"], ["monthly_obligations"], "Data prep"),
    "build_training_rows": (["monthly_flows", "monthly_obligations", "clients", "monthly_balances"],
                            ["training_rows"], "Training"),
    "train_and_test": (["training_rows"], ["experiments", "test_results", "model_reference"], "Training"),
    "champion_challenger": (["training_rows", "experiments"], ["promotion_log"], "Training"),
    "build_scoring_rows": (["monthly_flows", "monthly_obligations", "clients", "monthly_balances"],
                           ["scoring_rows"], "Inference"),
    "final_forecast": (["scored_ev", "scored_p10", "scored_p90", "monthly_obligations"], ["final_forecast"],
                       "Inference"),
    "archive_forecast": (["final_forecast"], ["forecast_archive", "inference_log"], "Inference"),
    "try_forecaster": (["training_rows", "experiments"], ["lab_experiments", "lab_results"], "Lab"),
}
FOLDERS = ["experiments", "forecast_archive", "lab_experiments"]
APPENDED = ["promotion_log", "inference_log", "lab_results"]      # logs: each run adds rows
BUILT_BY_NAME = ["experiments", "promotion_log", "lab_experiments", "lab_results"]   # only built when asked by name

try:                                                # needs the project's admin right
    metadata = project.get_metadata()
    metadata["label"] = PROJECT_NAME
    project.set_metadata(metadata)
except Exception as e:
    if not any(w in str(e).lower() for w in ("unauthorized", "forbidden", "permission", "not allowed")):
        raise
    print(f"Warning: could not name the project {PROJECT_NAME!r} (no admin right on the project): name it by hand")

# the project library: shared (cashflow.py, warehouse.py), training and lab (models.py, tracking.py), inference
# (forecast.py). forecasters.py is the data scientist's: written only when it is created, never overwritten.
library = project.get_library()
python_folder = library.get_folder("python") or library.add_folder("python")
for file_name, code in (("cashflow.py", LIBRARY_CODE), ("warehouse.py", WAREHOUSE_CODE), ("models.py", MODELS_CODE),
                        ("tracking.py", TRACKING_CODE), ("forecast.py", FORECAST_CODE)):
    (python_folder.get_file(file_name) or python_folder.add_file(file_name)).write(code)
if not python_folder.get_file("forecasters.py"):
    python_folder.add_file("forecasters.py").write(FORECASTERS_CODE)
    print("forecasters.py created (the data scientist's forecasters)")

# project variables, kept if they are already set: forecaster (the one Retrain trains), lab_forecaster (the one the
# recipe try_forecaster tries)
variables = project.get_variables()
defaults = {"forecaster": "catboost", "lab_forecaster": "catboost_deeper"}
missing = {k: v for k, v in defaults.items() if not variables["standard"].get(k)}
if missing:
    variables["standard"].update(missing)
    project.set_variables(variables)
print("Forecasters: Retrain trains {forecaster}, the lab tries {lab_forecaster}".format(**variables["standard"]))


# what the project already has (read once, kept up to date below)
datasets = {d["name"] for d in project.list_datasets()}
recipes = {r["name"] for r in project.list_recipes()}


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
simulated = [name for name in RAW + [warehouse.LOAD_RECORD] if name in datasets]
for name in RAW + [warehouse.LOAD_RECORD]:
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
        cashflow.DEFAULT_CATEGORY_MAPPING,
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
                else project.create_mlflow_pyfunc_model(name, "REGRESSION") for name in cashflow.SAVED_MODELS.values()}

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
        "computedColumns": [{"name": "category", "type": "string", "mode": "GREL", "expr": cashflow.CATEGORY_FORMULA}],
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
    if name in recipes:                # already there: only its code and code env are updated
        recipe = project.get_recipe(name)
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
for short, name in cashflow.SAVED_MODELS.items():
    if f"score_{short}" in recipes:
        continue
    builder = project.new_recipe("prediction_scoring", f"score_{short}")
    builder.with_input_model(saved_models[name].id)
    builder.with_input("scoring_rows")
    builder.with_existing_output(new_dataset(f"scored_{short}"))
    builder.create().rename(f"score_{short}")        # Dataiku names a Score recipe after its input
    print("score recipe", f"score_{short}", "->", f"scored_{short}")

ZONES = {"Raw data": RAW + [warehouse.LOAD_RECORD], "Staging": list(warehouse.TABLES),
         "Data prep": ["category_mapping", "flows_with_category", "monthly_flows", "scheduled_with_category"],
         "Training": [], "Inference": ["scored_ev", "scored_p10", "scored_p90"], "Lab": []}
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
