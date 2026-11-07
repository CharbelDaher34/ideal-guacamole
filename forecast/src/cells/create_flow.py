import json
import sys
import time
import types

import pandas as pd

# Safe to run again in the same project: what already exists is kept (datasets and their data, folders, the saved
# model and its versions, zones, the project variables, forecasters.py), and the code of the Python recipes and of the
# rest of the library is updated.
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
# lab_experiments are folders; cash_forecast is the saved model.
# Each zone has an owner, and the zones only meet through three interfaces:
#   1. the data: daily_flows and clients (Staging) -> Training, Lab and Inference
#   2. the forecaster (models.py): lab -> training. The project variable forecaster names the one Retrain trains;
#      the Lab (try_forecaster, and the notebook forecast_lab) compares one with the production model, untouched.
#   3. the saved model's active version: training -> inference. Inference reads nothing else from training.
DATA = ["daily_flows", "clients"]
FLOW = {
    "stage_tables": (RAW, warehouse.STAGED, "Staging"),
    "train_and_test": (DATA, ["experiments", "test_results"], "Training"),
    "champion_challenger": (DATA + ["experiments"], ["promotion_log"], "Training"),
    "final_forecast": (DATA + [cashflow.SAVED_MODEL], ["final_forecast"], "Inference"),
    "archive_forecast": (["final_forecast"], ["forecast_archive", "inference_log"], "Inference"),
    "try_forecaster": (DATA + ["experiments"], ["lab_experiments", "lab_results"], "Lab"),
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
defaults = {"forecaster": "recent_average", "lab_forecaster": "recent_average_6m", **tracking.PROMOTION_RULES}
missing = {k: v for k, v in defaults.items() if variables["standard"].get(k) in (None, "")}
if missing:
    variables["standard"].update(missing)
    project.set_variables(variables)
print("Forecasters: Retrain trains {forecaster}, the lab tries {lab_forecaster}".format(**variables["standard"]))
print("Promotion rules:", {k: variables["standard"][k] for k in tracking.PROMOTION_RULES})


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


# the 2 warehouse tables: already here if the warehouse simulator wrote them (sandbox, CSV), otherwise read-only from
# the warehouse as they are (production): tables of WAREHOUSE_SQL_SCHEMA through the database connection
# WAREHOUSE_SQL_CONNECTION, or CSV files in the landing folder (LANDING_CONNECTION). Nothing is created there.
simulated = [name for name in RAW if name in datasets]
source = WAREHOUSE_SQL_CONNECTION or LANDING_CONNECTION
if WAREHOUSE_SQL_CONNECTION and len(simulated) < len(RAW):
    try:
        database = client.get_connection(WAREHOUSE_SQL_CONNECTION).get_info().get_type()   # Oracle, Teradata, ...
    except Exception as e:
        raise RuntimeError(f"Cannot use the connection {WAREHOUSE_SQL_CONNECTION!r} ({e}). Ask a Dataiku admin for a "
                           "read-only database connection to the warehouse, usable by your group, or check its "
                           "name in WAREHOUSE_SQL_CONNECTION.") from None
for name in RAW:
    if name in datasets:
        continue
    datasets.add(name)
    if WAREHOUSE_SQL_CONNECTION:
        project.create_sql_table_dataset(name, database, WAREHOUSE_SQL_CONNECTION, name, WAREHOUSE_SQL_SCHEMA)
    else:
        dataset = project.create_dataset(name, "Filesystem", params={"connection": LANDING_CONNECTION,
                                                                     "path": "/" + name},
                                          formatType="csv", formatParams=CSV)
        dataset.autodetect_settings().save()
print(f"{len(simulated)} warehouse tables already in the project, {len(RAW) - len(simulated)} read from {source}")

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
existing_models = {m["name"]: m["id"] for m in project.list_saved_models()}
saved_model = (project.get_saved_model(existing_models[cashflow.SAVED_MODEL]) if cashflow.SAVED_MODEL in existing_models
               else project.create_mlflow_pyfunc_model(cashflow.SAVED_MODEL, "REGRESSION"))
ref = lambda name: (folders[name].id if name in folders else saved_model.id if name == cashflow.SAVED_MODEL
                    else name)

for name, (inputs, outputs, _) in FLOW.items():
    for item in inputs + outputs:
        if item not in folders and item != cashflow.SAVED_MODEL:
            new_dataset(item)
    if name in recipes:                # already there: its inputs and outputs changed? then made again
        settings = project.get_recipe(name).get_settings()
        if (set(settings.get_flat_input_refs()) != set(map(ref, inputs))
                or set(settings.get_flat_output_refs()) != set(map(ref, outputs))):
            project.get_recipe(name).delete()
            recipes.discard(name)
            print("recipe", name, "has new inputs or outputs: made again")
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

ZONES = {"Raw data": RAW, "Staging": list(warehouse.STAGED), "Training": [], "Inference": [], "Lab": []}
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
safe_add(zones["Inference"], saved_model)    # next to the recipe that uses it
print("Flow ready:", ", ".join(zones))
