import json
import sys
import time
import types

import pandas as pd

if FINAL_LOAD not in ("csv", "sql"):
    raise RuntimeError(f"FINAL_LOAD is {FINAL_LOAD!r}: set it to \"csv\" or \"sql\" in the settings cell above.")
placeholders = [name for name, value in (("CODE_ENV", CODE_ENV), ("FINAL_CONNECTION", FINAL_CONNECTION),
                                         ("FINAL_TABLE", FINAL_TABLE))
                if (not str(value).strip() or str(value).startswith("YOUR_"))
                and (name == "CODE_ENV" or FINAL_LOAD == "sql")]       # FINAL_*: only for the final load "sql"
if placeholders:
    raise RuntimeError(f"Fill in {' and '.join(placeholders)} in the settings cell above, then run again.")


def load_module(name, code):
    """A library file as a module of this notebook: the code it writes to the project library."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


for name, code in (("cashflow", LIBRARY_CODE), ("warehouse", WAREHOUSE_CODE), ("forecast", FORECAST_CODE),
                   ("loadtest", LOADTEST_CODE)):
    load_module(name, code)             # to check the tests and make the SQL tables below (this notebook has no polars)
CHECKED = sys.modules["loadtest"].check_tests(TESTS)          # checked before anything is built (as a list)

# the final load "sql": the table FINAL_SCHEMA.FINAL_TABLE, made here with SQL (forecast.ensure_table) with the
# columns of the forecast's rows. Oracle: written by the recipe with oracledb (no dataset); else: through a dataset
forecast = sys.modules["forecast"]
final_sql = forecast.table_name(FINAL_SCHEMA, FINAL_TABLE) if FINAL_LOAD == "sql" else None
final_oracle = FINAL_LOAD == "sql" and forecast.sql_dialect(FINAL_CONNECTION.strip()) == "oracle"
check_code_env(CODE_ENV, ("polars", "catboost") + (("oracledb",) if final_oracle else ()))
if FINAL_LOAD == "sql":
    final_columns = forecast.output_columns({"forecaster": "string", "load_id": "string"})
    try:
        print(f"Final load: {forecast.ensure_table(FINAL_CONNECTION.strip(), final_sql, final_columns)}")
    except Exception as e:
        print("For a DBA, the table to make:\n" + forecast.create_table_sql(final_sql, final_columns,
                                                                           "oracle" if final_oracle else "other"))
        raise RuntimeError(f"The table {final_sql} on {FINAL_CONNECTION} could not be made or read ({e}). Check "
                           "FINAL_CONNECTION (a SQL connection of this Dataiku) and FINAL_SCHEMA, and that the "
                           "connection's user may create the table; or ask a DBA to run the CREATE TABLE above; or set "
                           "FINAL_LOAD = \"csv\".") from None
    if final_oracle:            # the login; oracledb itself is checked in the code env above (not this notebook's)
        print("Final load: Oracle, written with oracledb; "
              + forecast.check_table_writer(FINAL_CONNECTION.strip(), driver=False))

try:                                                # needs the project's admin right
    metadata = project.get_metadata()
    metadata["label"] = PROJECT_NAME
    project.set_metadata(metadata)
except Exception as e:
    if not any(w in str(e).lower() for w in ("unauthorized", "forbidden", "permission", "not allowed")):
        raise
    print(f"Warning: could not name the project {PROJECT_NAME!r} (no admin right on the project): name it by hand")

# the project library: the forecast project's own files (the same code), and loadtest.py
library = project.get_library()
python_folder = library.get_folder("python") or library.add_folder("python")
for file_name, code in (("cashflow.py", LIBRARY_CODE), ("warehouse.py", WAREHOUSE_CODE), ("models.py", MODELS_CODE),
                        ("forecast.py", FORECAST_CODE), ("loadtest.py", LOADTEST_CODE)):
    (python_folder.get_file(file_name) or python_folder.add_file(file_name)).write(code)
print("Library: cashflow.py, warehouse.py, models.py, forecast.py (the forecast project's code) and loadtest.py")

# the flow, in 2 zones: Data prep (the fake data) and Load test (the test, its log and the report).
# Python recipes: (inputs, outputs, zone); load_test_log is a CSV; load_test_final, the dataset linked to the table of
# FINAL_LOAD = "sql" when it is not on Oracle (Oracle: no dataset; "csv": a file in the folder load_test_forecast)
final_dataset = FINAL_LOAD == "sql" and not final_oracle
FLOW = {"make_fake_data": ([], ["load_test_flows"], "Data prep"),
        "run_load_test": (["load_test_flows"], ["load_test_forecast", "load_test_log"]
                          + (["load_test_final"] if final_dataset else []), "Load test"),
        "load_test_report": (["load_test_log"], ["load_test_report"], "Load test")}
FOLDERS = ["load_test_flows", "load_test_forecast", "load_test_report"]
CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}
datasets = {d["name"] for d in project.list_datasets()}
recipes = {r["name"] for r in project.list_recipes()}
for old in ("run_test_plan",):                       # the earlier recipe, replaced by run_load_test
    if old in recipes:
        project.get_recipe(old).delete()
        recipes.discard(old)
existing_folders = {f["name"]: f["id"] for f in project.list_managed_folders()}
folders = {name: (project.get_managed_folder(existing_folders[name]) if name in existing_folders
                  else project.create_managed_folder(name, connection_name=OUTPUT_FOLDER_CONNECTION))
           for name in FOLDERS}
# the fake data is only made when asked by name (step 2 of the scenario), never by building what follows it
folder_settings = folders["load_test_flows"].get_settings()
folder_settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"
folder_settings.save()
variables = project.get_variables()                  # what run_load_test reads: the final load, csv or sql
variables["standard"].update(final_load=FINAL_LOAD, final_connection=FINAL_CONNECTION.strip(),
                             final_schema=FINAL_SCHEMA.strip(), final_table=FINAL_TABLE.strip())  # loadtest.final_table
variables["standard"].update(log_sql_connection=LOG_SQL_CONNECTION.strip(), log_sql_schema=LOG_SQL_SCHEMA.strip(),
                             log_sql_table=LOG_SQL_TABLE.strip())    # the SQL copy of load_test_log (log_to_sql)
project.set_variables(variables)
print(f"Final load of each test: {FINAL_LOAD}"
      + (f", the table {final_sql} on {FINAL_CONNECTION}" if final_sql else ""))
if LOG_SQL_CONNECTION.strip() and LOG_SQL_TABLE.strip():
    loadtest = sys.modules["loadtest"]
    log_sql = forecast.table_name(LOG_SQL_SCHEMA, LOG_SQL_TABLE)
    try:                                             # made now, so a missing right shows before the tests run
        print(f"SQL copy of the log: {loadtest.ensure_log_table(LOG_SQL_CONNECTION.strip(), log_sql)}")
    except Exception as e:
        print(f"WARNING: the SQL copy of the log, {log_sql} on {LOG_SQL_CONNECTION}, could not be made or read: {e}\n"
              "The tests still run (their rows go to the dataset load_test_log); the job run_load_test will fail at "
              "its end until the table can be written. To make it by hand (Oracle):")
        print(forecast.create_table_sql(log_sql, loadtest.LOG_COLUMNS))
else:
    print("SQL copy of the log: none (LOG_SQL_CONNECTION is empty)")
if final_dataset:   # load_test_final: a dataset linked to the table made above (not a managed one: no new table)
    schema, table = FINAL_SCHEMA.strip().lower(), FINAL_TABLE.strip().lower()  # as PostgreSQL stores unquoted names
    try:                                                                        # (Dataiku quotes them)
        kind = client.get_connection(FINAL_CONNECTION.strip()).get_info().get_type()
    except Exception:
        kind = "PostgreSQL"
    if "load_test_final" in datasets:
        raw = project.get_dataset("load_test_final").get_settings().get_raw()
        params = raw.get("params", {})
        linked = (params.get("connection"), params.get("table"), params.get("schema") or "")
        if raw.get("managed") or linked != (FINAL_CONNECTION.strip(), table, schema):
            raise RuntimeError(f"The dataset load_test_final is not linked to the table {final_sql} on "
                               f"{FINAL_CONNECTION} (for example the managed dataset of an older version, whose "
                               "table Dataiku named): delete it (Flow > load_test_final > Delete, without its data), "
                               "then run again.")
    if "load_test_final" not in datasets:
        project.create_sql_table_dataset("load_test_final", kind, FINAL_CONNECTION.strip(), table, schema)
    project.get_dataset("load_test_final").autodetect_settings().save()        # its columns, read from the table
    print(f"Dataset load_test_final: linked to the table {final_sql} on {FINAL_CONNECTION} (Dataiku's writer)")
if "load_test_log" not in datasets:
    settings = project.new_managed_dataset("load_test_log").with_store_into(OUTPUT_CONNECTION).create().get_settings()
    settings.get_raw()["formatType"] = "csv"
    settings.get_raw()["formatParams"].update(CSV)
    settings.save()
ref = lambda name: folders[name].id if name in folders else name
# first, the recipes whose inputs or outputs changed are removed (made again below), so no output is held by an old one;
# an output no longer wanted (load_test_final, for FINAL_LOAD = "csv") is only taken off the recipe (nothing to create)
for name, (inputs, outputs, _) in FLOW.items():
    if name in recipes:
        settings = project.get_recipe(name).get_settings()
        extra = set(settings.get_flat_output_refs()) - set(map(ref, outputs))
        if extra and set(settings.get_flat_output_refs()) - extra == set(map(ref, outputs)):
            for role in settings.get_recipe_outputs().values():
                role["items"] = [item for item in role.get("items", []) if item.get("ref") not in extra]
            settings.save()
            print("recipe", name, "no longer writes", ", ".join(sorted(extra)))
        if (set(settings.get_flat_input_refs()) != set(map(ref, inputs))
                or set(settings.get_flat_output_refs()) != set(map(ref, outputs))):
            project.get_recipe(name).delete()
            recipes.discard(name)
            print("recipe", name, "has new inputs or outputs: made again")
for name, (inputs, outputs, _) in FLOW.items():
    if name not in recipes:
        builder = project.new_recipe("python", name)
        for i in inputs:
            builder.with_input(ref(i))
        for o in outputs:
            builder.with_output(ref(o), append=o == "load_test_log")
        builder.create()
    settings = project.get_recipe(name).get_settings()
    settings.set_payload(RECIPES[name])
    settings.set_code_env(code_env=CODE_ENV)
    settings.save()
    print("python recipe", name, "->", ", ".join(outputs))

# the zones: Dataiku's Default zone becomes Data prep (so there is no empty zone). It exists once another zone does
flow = project.get_flow()
existing_zones = {z.name: z for z in flow.list_zones()}
zones = {"Load test": existing_zones.get("Load test") or flow.create_zone("Load test")}
zones["Data prep"] = existing_zones.get("Data prep") or flow.get_default_zone()
if zones["Data prep"].name != "Data prep":
    zone_settings = zones["Data prep"].get_settings()
    zone_settings.name = "Data prep"
    zone_settings.save()
for _, outputs, zone_name in FLOW.values():
    for item in outputs:
        try:
            zones[zone_name].add_item(folders[item] if item in folders else project.get_dataset(item))
        except Exception as e:
            if "already" not in str(e).lower():
                raise
print("Flow ready: zones Data prep (the fake data) and Load test (the test, its log, the report)")

# the scenario, run by hand (no trigger). Its step 1 sets the tests (Scenarios > Run the load test > Steps > 1): the
# settings above when the scenario is made; running this notebook again keeps the tests already set there.
def numbers_step(label, variables):
    return {"id": "step_0", "name": label, "enabled": True, "runConditionType": "RUN_IF_STATUS_MATCH",
            "runConditionStatuses": ["SUCCESS", "WARNING"], "type": "define_vars",
            "params": {"variables": variables, "definitions": []}}


def build_step(n, label, items, even_if_failed=False):
    return {"id": f"step_{n}", "name": label, "enabled": True, "runConditionType": "RUN_IF_STATUS_MATCH",
            "runConditionStatuses": ["SUCCESS", "WARNING"] + (["FAILED"] if even_if_failed else []),
            "type": "build_flowitem", "params": {
                "builds": [{"type": "MANAGED_FOLDER", "itemId": folders[i].id, "partitionsSpec": ""} if i in folders
                           else {"type": "DATASET", "itemId": i, "partitionsSpec": ""} for i in items],
                "jobType": "NON_RECURSIVE_FORCED_BUILD", "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}


SCENARIO = "Run the load test"
DESCRIPTION = (
    "Run by hand. Step 1 (Steps > 1. Set the tests): tests, run one after the other, a grid "
    '{"clients": [...], "months": ... (at least 16), "postings_per_month": ..., "clients_per_group": [...], '
    '"workers": [...]} (each a number or a list: every combination is a test) or a list of tests. '
    "It makes the fake data of each size if it is not there yet (zone Data prep, not timed, kept for the next runs), "
    "runs each test (the forecast, then its final load: a CSV file or the SQL table), then the report: each client count on its own, the settings "
    "compared at the same count (dashboard Load test, folder load_test_report), even when a test failed")
STEPS = [numbers_step("Set the tests", {"tests": TESTS}),
         build_step(1, "The fake data of each size (not timed)", ["load_test_flows"]),
         build_step(2, "The tests", ["load_test_log"]),
         build_step(3, "The report", ["load_test_report"], even_if_failed=True)]
existing_scenarios = {s["name"]: s["id"] for s in project.list_scenarios()}


def first_numbers(name):
    """The variables in step 1 of an existing scenario."""
    first = (project.get_scenario(existing_scenarios[name]).get_settings().get_raw()["params"].get("steps") or [{}])[0]
    return first.get("params", {}).get("variables", {}) if first.get("type") == "define_vars" else {}


# tests already set: kept (and the earlier version's single test, its numbers directly in step 1, as the only test)
before = {}
for name in ("Create the fake data", SCENARIO):
    if name in existing_scenarios:
        before.update({k.replace("fake_", ""): v for k, v in first_numbers(name).items()})
variables = STEPS[0]["params"]["variables"]
if isinstance(before.get("tests"), (list, dict)) and before["tests"]:
    variables["tests"] = before["tests"]
elif any(k in before for k in CHECKED[0]):
    variables["tests"] = [{k: before.get(k, v) for k, v in CHECKED[0].items()}]
for old in ("Create the fake data", "Run the test plan"):           # the earlier scenarios, now this one
    if old in existing_scenarios:
        project.get_scenario(existing_scenarios.pop(old)).delete()
        print(f"Scenario {old}: removed (its numbers are now in step 1 of {SCENARIO})")
if SCENARIO in existing_scenarios:
    settings = project.get_scenario(existing_scenarios[SCENARIO]).get_settings()
    settings.get_raw()["description"] = DESCRIPTION
    settings.get_raw()["params"]["steps"] = STEPS
    settings.save()
else:
    project.create_scenario(SCENARIO, "step_based", {"active": False, "description": DESCRIPTION,
                                                     "params": {"steps": STEPS}})
print(f"Scenario ready: {SCENARIO} (run it by hand). Step 1, Set the tests: {json.dumps(variables)}")
for n, test in enumerate(sys.modules["loadtest"].check_tests(variables["tests"]), 1):
    print(f"    test {n}: {json.dumps(test)}")
