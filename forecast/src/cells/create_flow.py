import json
import re
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

# the input, checked before anything is built: the 2 warehouse tables, read-only through EDW_CONNECTION (SQL only)
try:
    database = client.get_connection(EDW_CONNECTION).get_info().get_type()      # Oracle, PostgreSQL, ...
except Exception as e:
    raise RuntimeError(f"Cannot use the connection {EDW_CONNECTION!r} ({e}). Check EDW_CONNECTION, or ask a Dataiku "
                       "admin for a read-only database connection to the warehouse, usable by your group.") from None
print(f"Input: {database} connection {EDW_CONNECTION}, read-only")
for table, (name, _) in warehouse.TABLES.items():
    columns = warehouse.read_columns(table)
    query = warehouse.table_query(table, f'"{EDW_SCHEMA}"."{name}"', end=pd.Timestamp.now())
    try:
        dataiku.SQLExecutor2(connection=EDW_CONNECTION).query_to_df(f"SELECT * FROM ({query}) q WHERE 1 = 0")
    except Exception as e:
        found = re.search(r"(ORA-\d+[^\\\n']*|ERROR: [^\\\n']*)", str(e))    # the database's own message
        reason = found.group(1).strip() if found else str(e).splitlines()[0][:200]
        raise RuntimeError(f"Cannot read {EDW_SCHEMA}.{name} through {EDW_CONNECTION} ({reason}). Check EDW_SCHEMA, "
                           f"that the table exists with the columns {', '.join(columns)}, and that the connection's "
                           "user may read it.") from None
    print(f"   {EDW_SCHEMA}.{name}: can be read ({len(columns)} columns used)")
print(f"Output: datasets on {OUTPUT_CONNECTION}, folders on {OUTPUT_FOLDER_CONNECTION}")
CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}
COMPRESSED = ["final_forecast"]       # the big one: compressed CSV (gz)

# Python recipes: (inputs, outputs, flow zone). Every dataset is a CSV file (compressed for the big one); daily_flows
# (Parquet files, read far faster than a dataset), experiments, forecast_archive and lab_experiments are folders;
# cash_forecast is the saved model.
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
    "final_forecast": (DATA + [cashflow.SAVED_MODEL], ["final_forecast", "forecast_archive", "inference_log"],
                       "Inference"),
    "try_forecaster": (DATA + ["experiments"], ["lab_experiments", "lab_results"], "Lab"),
}
FOLDERS = ["daily_flows", "experiments", "forecast_archive", "lab_experiments"]
# logs: each run adds rows
APPENDED = ["staging_log", "promotion_log", "inference_log", "lab_results"]
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
# recipe try_forecaster tries), inference_clients_per_chunk (how many clients final_forecast forecasts at a time)
variables = project.get_variables()
defaults = {"forecaster": "quantile_boosting", "lab_forecaster": "recent_average_6m",
            "inference_clients_per_chunk": 5000, **tracking.PROMOTION_RULES}
missing = {k: v for k, v in defaults.items() if variables["standard"].get(k) in (None, "")}
if missing:
    variables["standard"].update(missing)
    project.set_variables(variables)
# history_years: always the setup's setting HISTORY_YEARS (staging's first read and the scenario Full rebuild)
variables["standard"]["history_years"] = "" if cashflow.history_years(HISTORY_YEARS) is None else HISTORY_YEARS
project.set_variables(variables)
print("History read the first time (and by Full rebuild):",
      "the whole history" if HISTORY_YEARS is None else f"the last {HISTORY_YEARS} years up to the cut-off day")
print("Forecasters: Retrain trains {forecaster}, the lab tries {lab_forecaster}; inference forecasts "
      "{inference_clients_per_chunk} clients at a time".format(**variables["standard"]))
print("Promotion rules:", {k: variables["standard"][k] for k in tracking.PROMOTION_RULES})


# what the project already has (read once, kept up to date below)
datasets = {d["name"] for d in project.list_datasets()}
recipes = {r["name"] for r in project.list_recipes()}



def new_dataset(name):
    """Managed dataset of the project: a CSV file (comma-separated, with a header row) on OUTPUT_CONNECTION. Kept if it
    exists."""
    if name in datasets:
        return name
    datasets.add(name)
    dataset = project.new_managed_dataset(name).with_store_into(OUTPUT_CONNECTION).create()
    settings = dataset.get_settings()
    settings.get_raw()["formatType"] = "csv"
    settings.get_raw()["formatParams"].update(CSV, compress="gz" if name in COMPRESSED else "")
    if name in BUILT_BY_NAME:
        settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"
    settings.save()
    return name


# the 2 warehouse tables: datasets linked read-only to the tables of EDW_SCHEMA through EDW_CONNECTION (nothing is
# created or written there). One already in the project is kept only if it is that table (a run of the setup again);
# anything else (fake data, CSV, an old project's table) stops the setup before it is used.
for name in RAW:
    if name in datasets:
        raw = project.get_dataset(name).get_settings().get_raw()
        params = raw.get("params", {})
        if (params.get("connection"), params.get("schema"), params.get("table")) != (EDW_CONNECTION, EDW_SCHEMA, name):
            raise RuntimeError(f"The project already has a dataset {name} that is not the table {EDW_SCHEMA}.{name} "
                               f"of {EDW_CONNECTION} (it is {raw.get('type')} on {params.get('connection')}): fake or "
                               f"old data. Delete it (Flow > {name} > Delete), or use a new empty project, then run "
                               "the setup again.")
        print(f"{name}: already linked to {EDW_SCHEMA}.{name} of {EDW_CONNECTION}")
        continue
    datasets.add(name)
    dataset = project.create_sql_table_dataset(name, database, EDW_CONNECTION, name, EDW_SCHEMA)
    dataset.autodetect_settings().save()           # the table's columns, read from the database
    if not dataset.get_settings().get_raw().get("schema", {}).get("columns"):
        # staging reads with its own SELECT, so it still works; only Dataiku's view of the table would fail
        print(f"WARNING: Dataiku could not read the columns of {EDW_SCHEMA}.{name}: opening the dataset in Dataiku "
              "will fail. Fix: the dataset's Settings > Test table & get schema > Save.")
    print(f"{name}: linked to {EDW_SCHEMA}.{name} of {EDW_CONNECTION}")
override = variables["standard"].get("cutoff_override")
if override:
    print(f"WARNING: the project variable cutoff_override is {override}: the forecast uses that day instead of "
          "yesterday. Delete it (Project > Variables) unless you mean it.")

existing_folders = {f["name"]: f["id"] for f in project.list_managed_folders()}
folders = {}
for name in FOLDERS:
    if name in existing_folders:
        folders[name] = project.get_managed_folder(existing_folders[name])
        continue
    folders[name] = project.create_managed_folder(name, connection_name=OUTPUT_FOLDER_CONNECTION)
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
