# Production runs. Each scenario starts with the load check (a Python step): it stops the scenario if the latest
# warehouse load is incomplete, and sets the cut-off month. The tracked models (folder experiments) and the
# promotion log are only rebuilt when built by name, so the nightly batch never retrains by accident.
def step(n, label, what, mode=None):
    common = {"id": f"step_{n}", "name": label, "enabled": True, "runConditionType": "RUN_IF_STATUS_MATCH",
              "runConditionStatuses": ["SUCCESS", "WARNING"]}
    if isinstance(what, str):           # a Python step: library function to call
        module = "warehouse" if what == "check_latest_load" else "tracking"
        return {**common, "type": "custom_python", "params": {
            "script": f"from {module} import {what}\n\n{what}()\n", "envSelection": {"envMode": "INHERIT"},
            "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}
    kind = lambda i: ("MANAGED_FOLDER", folders[i].id) if i in folders else ("DATASET", i)
    builds = [{"type": t, "itemId": item, "partitionsSpec": ""} for t, item in map(kind, what)]
    return {**common, "type": "build_flowitem", "params": {
        "builds": builds, "jobType": mode, "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}


NIGHTLY = [
    ("Check the latest warehouse load", "check_latest_load"),
    ("Forecast with the production model, archive and log it", ["inference_log"], "RECURSIVE_FORCED_BUILD"),
]
SCENARIOS = {
    "Nightly batch": ("After each warehouse load: forecast the next 3 months with the production model (the active "
                      "version of the saved models), archive it and log the run", "daily", NIGHTLY),
    "Retrain, test and promote": (
        "Train and test a new model (tracked in Experiment Tracking, deployed as a saved model version), activate it if "
        "it beats the production model, then forecast with the production model", "monthly", [
            ("Check the latest warehouse load", "check_latest_load"),
            ("Prepare the training data", ["training_rows"], "RECURSIVE_FORCED_BUILD"),
            ("Train, test and deploy a new model", ["experiments"], "NON_RECURSIVE_FORCED_BUILD"),
            ("Champion / challenger", ["promotion_log"], "NON_RECURSIVE_FORCED_BUILD"),
            ("Promote if better", "apply_promotion"),
        ] + NIGHTLY[1:]),
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
