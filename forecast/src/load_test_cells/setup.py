import json
import sys
import time
import types

import pandas as pd

placeholders = [name for name, value in (("CODE_ENV", CODE_ENV), ("OUTPUT_SQL_CONNECTION", OUTPUT_SQL_CONNECTION))
                if str(value).startswith("YOUR_")]
if placeholders:
    raise RuntimeError(f"Fill in {' and '.join(placeholders)} in the settings cell above, then run again.")
check_code_env(CODE_ENV, ("polars", "catboost"))


def load_module(name, code):
    """A library file as a module of this notebook: the code it writes to the project library."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


for name, code in (("cashflow", LIBRARY_CODE), ("warehouse", WAREHOUSE_CODE), ("loadtest", LOADTEST_CODE)):
    load_module(name, code)                         # to check the numbers below (this notebook has no polars)
SIZE = {"fake_clients": FAKE_CLIENTS, "fake_months": FAKE_MONTHS, "fake_postings_per_month": FAKE_POSTINGS_PER_MONTH,
        "fake_clients_per_group": FAKE_CLIENTS_PER_GROUP}
size = sys.modules["loadtest"].data_size(SIZE)                    # checked before anything is built
plan = sys.modules["loadtest"].read_plan(TEST_PLAN)

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

# the numbers are set in step 1 of each scenario (below), not as project variables: those of an earlier setup go
variables = project.get_variables()
old = [k for k in ("fake_clients", "fake_months", "fake_postings_per_month", "load_test_plan", "bank_clients")
       if k in variables["standard"]]
if old:
    for k in old:
        variables["standard"].pop(k)
    project.set_variables(variables)
    print(f"Project variables {', '.join(old)} removed: these numbers are now set in step 1 of the scenarios")

# the flow, in 2 zones: Data prep (the fake data) and Load test (the tests, their log and the report).
# Python recipes: (inputs, outputs, zone); load_test_final is a table on OUTPUT_SQL_CONNECTION, load_test_log a CSV
FLOW = {"make_fake_data": ([], ["load_test_flows"], "Data prep"),
        "run_test_plan": (["load_test_flows"], ["load_test_forecast", "load_test_final", "load_test_log"],
                          "Load test"),
        "load_test_report": (["load_test_log"], ["load_test_report"], "Load test")}
FOLDERS = ["load_test_flows", "load_test_forecast", "load_test_report"]
CSV = {"style": "excel", "charset": "utf8", "separator": ",", "quoteChar": '"', "escapeChar": "\\",
       "dateSerializationFormat": "ISO", "arrayMapFormat": "json", "parseHeaderRow": True, "compress": ""}
datasets = {d["name"] for d in project.list_datasets()}
recipes = {r["name"] for r in project.list_recipes()}
existing_folders = {f["name"]: f["id"] for f in project.list_managed_folders()}
folders = {name: (project.get_managed_folder(existing_folders[name]) if name in existing_folders
                  else project.create_managed_folder(name, connection_name=OUTPUT_FOLDER_CONNECTION))
           for name in FOLDERS}
# the fake data is only made when asked by name (scenario Create the fake data): the load test never makes it again
folder_settings = folders["load_test_flows"].get_settings()
folder_settings.get_raw()["flowOptions"]["rebuildBehavior"] = "EXPLICIT"
folder_settings.save()
if "load_test_final" not in datasets:
    project.new_managed_dataset("load_test_final").with_store_into(OUTPUT_SQL_CONNECTION).create()
if "load_test_log" not in datasets:
    settings = project.new_managed_dataset("load_test_log").with_store_into(OUTPUT_CONNECTION).create().get_settings()
    settings.get_raw()["formatType"] = "csv"
    settings.get_raw()["formatParams"].update(CSV)
    settings.save()
ref = lambda name: folders[name].id if name in folders else name
# first, the recipes whose inputs or outputs changed are removed (made again below), so no output is held by an old one
for name, (inputs, outputs, _) in FLOW.items():
    if name in recipes:
        settings = project.get_recipe(name).get_settings()
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
print("Flow ready: zones Data prep (the fake data) and Load test (the tests, their log, the report)")

# the 2 scenarios, run by hand (no trigger): one makes the data (not timed), the other runs the load test on it.
# Step 1 of each sets its numbers (Scenarios > the scenario > Steps > 1): the settings above when the scenario is
# made; running this notebook again keeps the numbers already set there (a new number gets its default).
def numbers_step(label, variables):
    return {"id": "step_0", "name": label, "enabled": True, "runConditionType": "RUN_IF_STATUS_MATCH",
            "runConditionStatuses": ["SUCCESS", "WARNING"], "type": "define_vars",
            "params": {"variables": variables, "definitions": []}}


def build_step(n, label, items):
    return {"id": f"step_{n}", "name": label, "enabled": True, "runConditionType": "RUN_IF_STATUS_MATCH",
            "runConditionStatuses": ["SUCCESS", "WARNING"], "type": "build_flowitem", "params": {
                "builds": [{"type": "MANAGED_FOLDER", "itemId": folders[i].id, "partitionsSpec": ""} if i in folders
                           else {"type": "DATASET", "itemId": i, "partitionsSpec": ""} for i in items],
                "jobType": "NON_RECURSIVE_FORCED_BUILD", "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}


SCENARIOS = {
    "Create the fake data": (
        "Run by hand, first, and again after changing its size: the fake daily flows the load test reads, zone Data "
        "prep. Not timed. Set the size in step 1 (Steps > 1. Set the fake data's size): fake_clients, fake_months "
        "(at least 16), fake_postings_per_month, fake_clients_per_group. Data of the same size already there is kept", [
            numbers_step("Set the fake data's size",
                         {name: size[key] for key, name in sys.modules["loadtest"].SIZE.items()}),
            build_step(1, "Make the fake daily flows", ["load_test_flows"])]),
    "Run the load test": (
        "Run by hand, after Create the fake data: every test of the test plan, one after the other, on all the fake "
        "data's clients (the forecast, then its SQL load), then the report (dashboard Load test, folder load_test_report). Set the plan in step 1 "
        "(Steps > 1. Set the test plan): load_test_plan, one line per test with test (a name) and workers; "
        "bank_clients, the number of clients the report gives the time for", [
            numbers_step("Set the test plan", {"load_test_plan": plan, "bank_clients": int(BANK_CLIENTS)}),
            build_step(1, "Every test of the plan", ["load_test_log"]),
            build_step(2, "The report", ["load_test_report"])]),
}
existing_scenarios = {s["name"]: s["id"] for s in project.list_scenarios()}
if "Run the test plan" in existing_scenarios:                       # the earlier name
    project.get_scenario(existing_scenarios.pop("Run the test plan")).delete()
for name, (description, steps) in SCENARIOS.items():
    if name in existing_scenarios:     # run again: its numbers set on the scenario's page are kept
        settings = project.get_scenario(existing_scenarios[name]).get_settings()
        before = (settings.get_raw()["params"].get("steps") or [{}])[0]
        if before.get("type") == "define_vars":
            old = before["params"].get("variables", {})
            steps[0]["params"]["variables"] = {k: old.get(k, v) for k, v in steps[0]["params"]["variables"].items()}
        settings.get_raw()["description"] = description
        settings.get_raw()["params"]["steps"] = steps
        settings.save()
    else:
        project.create_scenario(name, "step_based", {"active": False, "description": description,
                                                     "params": {"steps": steps}})
    numbers = steps[0]["params"]["variables"]
    print(f"Scenario ready: {name} (run it by hand). Step 1, {steps[0]['name']}:")
    for k, v in numbers.items():
        print(f"    {k} = {json.dumps(v)}")
