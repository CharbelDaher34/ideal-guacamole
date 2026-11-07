# Production runs: two independent scenarios. Nightly batch (inference) forecasts with the active version of the saved
# models; Retrain, test and promote (training) adds a version and activates it if it is better, and never forecasts.
# Each starts by setting the cut-off day (yesterday) and checking the warehouse load of that day is finished (Python
# steps; the check is a mock to replace with the bank's); the staging recipe adds the new days to daily_flows and
# stops if the postings do not reach the cut-off day. A third one, Full rebuild of the daily flows, has no trigger:
# run it by hand to read the whole history again. The tracked models (folder experiments) and the promotion log are only rebuilt when built by name, so the
# nightly batch never retrains by accident.
def step(n, label, what, mode=None):
    common = {"id": f"step_{n}", "name": label, "enabled": True, "runConditionType": "RUN_IF_STATUS_MATCH",
              "runConditionStatuses": ["SUCCESS", "WARNING"]}
    if isinstance(what, str):           # a Python step: library function to call
        module = "warehouse" if what in ("set_cutoff", "check_warehouse_loaded", "clear_daily_flows") else "tracking"
        return {**common, "type": "custom_python", "params": {
            "script": f"from {module} import {what}\n\n{what}()\n", "envSelection": {"envMode": "INHERIT"},
            "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}
    kind = lambda i: ("MANAGED_FOLDER", folders[i].id) if i in folders else ("DATASET", i)
    builds = [{"type": t, "itemId": item, "partitionsSpec": ""} for t, item in map(kind, what)]
    return {**common, "type": "build_flowitem", "params": {
        "builds": builds, "jobType": mode, "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}


START = [("Set the cut-off day (yesterday)", "set_cutoff"),
         ("Check the warehouse load is finished", "check_warehouse_loaded")]
NIGHTLY = [
    *START,
    ("Forecast with the production model, archive and log it", ["inference_log"], "RECURSIVE_FORCED_BUILD"),
]
SCENARIOS = {
    "Nightly batch": ("After each warehouse load: forecast the next 30, 60 and 90 days with the production model (the active "
                      "version of the saved models), archive it and log the run", "daily", NIGHTLY),
    "Retrain, test and promote": (
        "Train and test a new model with the forecaster the project variable forecaster names (tracked in Experiment "
        "Tracking, deployed as a saved model version), and activate it if it beats the production model. It does not "
        "forecast: the Nightly batch uses the active version", "monthly", [
            *START,
            ("Prepare the data (daily flows, clients)", ["daily_flows"], "RECURSIVE_FORCED_BUILD"),
            ("Train, test and deploy a new model", ["experiments"], "NON_RECURSIVE_FORCED_BUILD"),
            ("Champion / challenger", ["promotion_log"], "NON_RECURSIVE_FORCED_BUILD"),
            ("Promote if better", "apply_promotion"),
        ]),
    "Full rebuild of the daily flows": (
        "Run by hand: empties daily_flows and staging_log, then reads the whole history from the warehouse again "
        "(the nightly runs only read the new days). For example after a reversal cancels a posting of a day already "
        "staged. It does not forecast: run the Nightly batch after it", None, [
            *START,
            ("Empty the daily flows", "clear_daily_flows"),
            ("Read the whole history again", ["daily_flows"], "RECURSIVE_FORCED_BUILD"),
        ]),
}
existing_scenarios = {s["name"]: s["id"] for s in project.list_scenarios()}
for name, (description, schedule, steps) in SCENARIOS.items():
    if name in existing_scenarios:     # run again: update its steps, keep its triggers and run history
        settings = project.get_scenario(existing_scenarios[name]).get_settings()
        settings.get_raw()["description"] = description
        settings.get_raw()["params"]["steps"] = [step(n, *s) for n, s in enumerate(steps)]
        settings.save()
        print(f"Scenario updated: {name}")
        continue
    scenario = project.create_scenario(name, "step_based", {
        "active": False, "description": description,
        "params": {"steps": [step(n, *s) for n, s in enumerate(steps)]}})
    if schedule is None:                           # run by hand only
        print(f"Scenario ready: {name} (no trigger: run it by hand)")
        continue
    settings = scenario.get_settings()
    if schedule == "daily":
        settings.add_daily_trigger(hour=6, timezone="Asia/Riyadh")
    else:
        settings.add_monthly_trigger(day=5, hour=7, timezone="Asia/Riyadh")
    settings.raw_triggers[-1]["active"] = False    # switch on in production (Automation node)
    try:
        settings.save()
        note = "trigger added, switched off"
    except Exception as e:                         # licences without time-based triggers (Free Edition)
        if "License" not in str(e):
            raise
        note = "no trigger (not in this licence): add it on the Automation node"
    print(f"Scenario ready: {name} ({note})")
