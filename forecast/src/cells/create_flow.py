import importlib.util
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
forecast = load_module("forecast", FORECAST_CODE)  # its SQL table helpers (no polars needed)


def shown(schema, table):
    return f"{schema}.{table}" if schema else table


# the 2 warehouse tables, as the settings name them: their Flow dataset (warehouse.TABLES: the name the code uses,
# whatever the table) -> (settings' prefix, schema, table)
SOURCES = {warehouse.TABLES["clients"][0]: ("CLIENTS", CLIENTS_SCHEMA.strip(), CLIENTS_TABLE.strip()),
           warehouse.TABLES["transactions"][0]: ("POSTINGS", POSTINGS_SCHEMA.strip(), POSTINGS_TABLE.strip())}
RAW = list(SOURCES)

# the input, checked before anything is built: the 2 warehouse tables, read-only through EDW_CONNECTION (SQL only)
try:
    database = client.get_connection(EDW_CONNECTION).get_info().get_type()      # Oracle, PostgreSQL, ...
except Exception as e:
    raise RuntimeError(f"Cannot use the connection {EDW_CONNECTION!r} ({e}). Check EDW_CONNECTION, or ask a Dataiku "
                       "admin for a read-only database connection to the warehouse, usable by your group.") from None
print(f"Input: {database} connection {EDW_CONNECTION}, read-only")
for table, (name, _) in warehouse.TABLES.items():
    prefix, schema, table_in = SOURCES[name]
    columns = warehouse.read_columns(table)
    quoted = (f'"{schema}".' if schema else "") + f'"{table_in}"'
    query = warehouse.table_query(table, quoted, end=pd.Timestamp.now())
    try:
        dataiku.SQLExecutor2(connection=EDW_CONNECTION).query_to_df(f"SELECT * FROM ({query}) q WHERE 1 = 0")
    except Exception as e:
        found = re.search(r"(ORA-\d+[^\\\n']*|ERROR: [^\\\n']*)", str(e))    # the database's own message
        reason = found.group(1).strip() if found else str(e).splitlines()[0][:200]
        raise RuntimeError(f"Cannot read {shown(schema, table_in)} through {EDW_CONNECTION} ({reason}). Check "
                           f"{prefix}_SCHEMA and {prefix}_TABLE (the name as the database shows it, capitals "
                           f"included), that the table exists with the columns {', '.join(columns)}, and that the "
                           "connection's user may read it.") from None
    print(f"   {shown(schema, table_in)}: can be read ({len(columns)} columns used), the dataset {name}")

# the output, checked and made before anything is built: the forecast's table on FORECAST_CONNECTION, created here
# with SQL if it is not there (with the name of the settings, not one Dataiku would pick)
try:
    output_type = client.get_connection(FORECAST_CONNECTION).get_info().get_type()
except Exception as e:
    raise RuntimeError(f"Cannot use the connection {FORECAST_CONNECTION!r} ({e}). Check FORECAST_CONNECTION: a "
                       "database connection your group may write into (never the warehouse), whose \"Details "
                       "readable by\" includes your group (ask a Dataiku admin).") from None
FORECAST = forecast.table_name(FORECAST_SCHEMA.strip(), FORECAST_TABLE.strip())    # SCHEMA.TABLE, unquoted
print(f"Output: the forecast into {FORECAST} on the {output_type} connection {FORECAST_CONNECTION} (the dataset "
      f"final_forecast)")
FORECAST_COLUMNS = forecast.output_columns({"forecaster": "string", "load_id": "string"})
try:
    print("   " + forecast.ensure_table(FORECAST_CONNECTION, FORECAST, FORECAST_COLUMNS))
except Exception as e:
    found = re.search(r"(ORA-\d+[^\\\n']*|ERROR: [^\\\n']*)", str(e))
    reason = found.group(1).strip() if found else str(e).splitlines()[0][:200]
    print(forecast.create_table_sql(FORECAST, FORECAST_COLUMNS, "oracle" if output_type == "Oracle" else "other"))
    raise RuntimeError(f"Cannot create or read the table {FORECAST} on {FORECAST_CONNECTION} ({reason}). Check "
                       "FORECAST_SCHEMA (it must exist) and that the connection's user may create tables there, or "
                       "ask the database's owner to create it (the CREATE TABLE above), then run again.") from None
if output_type == "Oracle":                     # the recipe load_forecast_sql writes into Oracle with oracledb
    check_code_env(CODE_ENV, ("oracledb",))
# the forecast's writer: on Oracle, the connection's login must be readable by your group, and oracledb in the code
# env. This notebook's Python may have no oracledb (the recipes run on CODE_ENV): then only the login is checked here.
driver = importlib.util.find_spec("oracledb") is not None
try:
    done = forecast.check_table_writer(FORECAST_CONNECTION, driver=driver)
except RuntimeError as e:
    raise RuntimeError(f"The forecast cannot be written into {FORECAST} on {FORECAST_CONNECTION}: {e}") from None
if done:
    print(f"   {done}" + ("" if driver else f" (oracledb: in {CODE_ENV}, checked above; the recipe logs in with it)"))
print(f"Other outputs: datasets on {OUTPUT_CONNECTION}, folders on {OUTPUT_FOLDER_CONNECTION}")
CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}
SQL_OUTPUT = "final_forecast"         # linked to the table FORECAST below; every other dataset is a CSV file

# Python recipes: (inputs, outputs, flow zone). daily_flows (Parquet files, read far faster than a dataset),
# experiments and forecast (Parquet, one subfolder per night) are folders; cash_forecast is the saved model.
# Each zone has an owner, and the zones only meet through three interfaces:
#   1. the data: daily_flows and clients (Staging) -> Training and Inference (and the data scientist's lab notebook)
#   2. the forecaster (models.py, forecasters.py): the project variable forecaster names the one Retrain trains. The
#      data scientist tries forecasters first in the notebook forecast_lab, outside the flow, without touching
#      production.
#   3. the saved model's active version: training -> inference. Inference reads nothing else from training.
# The load test is a project of its own (build_load_test_project.ipynb): nothing of it is in this project.
DATA = ["daily_flows", "clients"]
FLOW = {
    "stage_tables": (RAW, warehouse.STAGED, "Staging"),
    "train_and_test": (DATA, ["experiments", "test_results"], "Training"),
    "champion_challenger": (DATA + ["experiments"], ["promotion_log"], "Training"),
    "nightly_forecast": (DATA + [cashflow.SAVED_MODEL], ["forecast", "inference_log"], "Inference"),
    "load_forecast_sql": (["forecast"], ["final_forecast"], "Inference"),
}
FOLDERS = ["daily_flows", "experiments", "forecast"]
# logs: each run adds rows
APPENDED = ["staging_log", "promotion_log", "inference_log"]
BUILT_BY_NAME = ["experiments", "promotion_log"]   # only built when asked by name

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

# project variables, kept if they are already set: forecaster (the one Retrain trains), inference_workers (how many
# groups of clients the nightly forecast runs at a time)
variables = project.get_variables()
variables["standard"].pop("inference_clients_per_chunk", None)   # the older setting
defaults = {"forecaster": "daily_catboost", "inference_workers": 4, **tracking.PROMOTION_RULES}
missing = {k: v for k, v in defaults.items() if variables["standard"].get(k) in (None, "")}
if variables["standard"].get("forecaster") == "quantile_boosting":          # the earlier placeholder, removed
    missing["forecaster"] = "daily_catboost"
    print("Project variable forecaster: quantile_boosting (removed) replaced by daily_catboost")
if missing:
    variables["standard"].update(missing)
    project.set_variables(variables)
# history_months, clients_limit: always the setup's settings (what staging reads: the first time and Full rebuild)
months = cashflow.whole_number("HISTORY_MONTHS", HISTORY_MONTHS)
limit = cashflow.whole_number("CLIENTS_LIMIT", CLIENTS_LIMIT)
variables["standard"].pop("history_years", None)                 # the older setting
variables["standard"].update(history_months="" if months is None else months, clients_limit="" if limit is None else limit)
# forecast_sql_*: always the setup's settings too (where the recipe load_forecast_sql writes the forecast)
variables["standard"].update(forecast_sql_connection=FORECAST_CONNECTION, forecast_sql_schema=FORECAST_SCHEMA.strip(),
                             forecast_sql_table=FORECAST_TABLE.strip())
project.set_variables(variables)
print("History read the first time (and by Full rebuild):",
      "the whole history" if months is None else f"the last {months} months up to the cut-off day",
      "| clients:", "all the business clients" if limit is None else f"the first {limit:,} business clients")
if months is not None and months < cashflow.MIN_TRAINING_MONTHS:
    print(f"WARNING: with {months} months of history, the first run only stages the data: training, promotion and "
          f"the forecast need at least {cashflow.MIN_TRAINING_MONTHS} months")
print("Models: Retrain trains {forecaster}; the nightly forecast runs {inference_workers} groups of clients at a "
      "time".format(**variables["standard"]))
print("Promotion rules:", {k: variables["standard"][k] for k in tracking.PROMOTION_RULES})


# what the project already has (read once, kept up to date below)
datasets = {d["name"] for d in project.list_datasets()}
recipes = {r["name"] for r in project.list_recipes()}



def new_dataset(name):
    """Managed dataset of the project: a CSV file (comma-separated, with a header row) on OUTPUT_CONNECTION. Kept if it
    exists. (final_forecast is not one: it is linked to its table below.)"""
    if name in datasets:
        return name
    datasets.add(name)
    dataset = project.new_managed_dataset(name).with_store_into(OUTPUT_CONNECTION).create()
    settings = dataset.get_settings()
    settings.get_raw()["formatType"] = "csv"
    settings.get_raw()["formatParams"].update(CSV)
    if name in BUILT_BY_NAME:
        settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"
    settings.save()
    return name


def linked_to(name):
    """(connection, schema, table) of a dataset already in the project, and what it is (for a message)."""
    raw = project.get_dataset(name).get_settings().get_raw()
    params = raw.get("params", {})
    where = (params.get("connection"), params.get("schema") or "", params.get("table"))
    if raw.get("managed"):
        return where, f"a managed {raw.get('type')} dataset on {where[0]}, its table named by Dataiku"
    return where, f"{raw.get('type')} on {where[0]}" + (f", table {shown(*where[1:])}" if where[2] else "")


# the 2 warehouse tables: datasets linked read-only to the settings' tables through EDW_CONNECTION (nothing is created
# or written there). Their Flow names stay DIM_CUSTOMER and FCT_STATEMENT_ENTRY whatever the tables (the code uses
# them; staging reads each dataset's own table). One already in the project is kept only if it is that table (a run
# of the setup again); anything else (fake data, CSV, an old project's table, another table) stops the setup before
# it is used.
for name in RAW:
    prefix, schema, table_in = SOURCES[name]
    if name in datasets:
        where, what = linked_to(name)
        if where != (EDW_CONNECTION, schema, table_in):
            raise RuntimeError(f"The project already has a dataset {name} that is not the table "
                               f"{shown(schema, table_in)} of {EDW_CONNECTION} ({prefix}_SCHEMA, {prefix}_TABLE): it "
                               f"is {what}. Delete it (Flow > {name} > Delete, without deleting its data: only the "
                               "link goes), or use a new empty project, then run the setup again.")
        print(f"{name}: already linked to {shown(schema, table_in)} of {EDW_CONNECTION}")
        continue
    datasets.add(name)
    dataset = project.create_sql_table_dataset(name, database, EDW_CONNECTION, table_in, schema or None)
    dataset.autodetect_settings().save()           # the table's columns, read from the database
    if not dataset.get_settings().get_raw().get("schema", {}).get("columns"):
        # staging reads with its own SELECT, so it still works; only Dataiku's view of the table would fail
        print(f"WARNING: Dataiku could not read the columns of {shown(schema, table_in)}: opening the dataset in "
              "Dataiku will fail. Fix: the dataset's Settings > Test table & get schema > Save.")
    print(f"{name}: linked to {shown(schema, table_in)} of {EDW_CONNECTION}")

# final_forecast: a dataset linked to the forecast's table, made above. Dataiku quotes the names it is given, while
# the table was made with unquoted names, which Oracle stores in capitals and PostgreSQL in small letters: so the link
# uses the names as the database stores them. A final_forecast already in the project is kept only if it is that
# table; an older managed one (Dataiku named its table) or one linked elsewhere stops the setup.
stored = str.upper if output_type == "Oracle" else str.lower
target = (FORECAST_CONNECTION, stored(FORECAST_SCHEMA.strip()), stored(FORECAST_TABLE.strip()))
if SQL_OUTPUT in datasets:
    where, what = linked_to(SQL_OUTPUT)
    if where != target:
        raise RuntimeError(
            f"The project already has a dataset {SQL_OUTPUT} that is not the table {shown(*target[1:])} of "
            f"{FORECAST_CONNECTION} (FORECAST_SCHEMA, FORECAST_TABLE): it is {what}, from an older setup or made by "
            f"hand. Delete it (Flow > {SQL_OUTPUT} > Delete, without deleting its data: its old table stays, drop it "
            f"later by hand if no one uses it), then run the setup again: it links {SQL_OUTPUT} to "
            f"{shown(*target[1:])} (the table is ready) and loads the forecast there.")
    print(f"{SQL_OUTPUT}: already linked to {shown(*target[1:])} of {FORECAST_CONNECTION}")
    dataset = project.get_dataset(SQL_OUTPUT)
else:
    datasets.add(SQL_OUTPUT)
    dataset = project.create_sql_table_dataset(SQL_OUTPUT, output_type, FORECAST_CONNECTION, target[2],
                                               target[1] or None)
    print(f"{SQL_OUTPUT}: linked to {shown(*target[1:])} of {FORECAST_CONNECTION}")
dataset.autodetect_settings().save()               # the table's columns (new ones too), read from the database
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
# the project's default Python: CODE_ENV, so anything that inherits it (a scenario step, a new notebook or recipe)
# has polars, mlflow, ... and not Dataiku's built-in Python
try:
    project_settings = project.get_settings()
    project_settings.set_python_code_env(CODE_ENV)
    project_settings.save()
    print(f"The project's default Python: the code env {CODE_ENV}")
except Exception as e:          # needs the project's admin right: the recipes and scenario steps name CODE_ENV anyway
    print(f"WARNING: could not make {CODE_ENV} the project's default Python ({e}). Set it by hand: Settings > Code "
          f"env > Python > {CODE_ENV}")

ZONES = {"Raw data": RAW, "Staging": list(warehouse.STAGED), "Training": [], "Inference": []}
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
