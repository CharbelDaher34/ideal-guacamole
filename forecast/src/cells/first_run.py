def build(step, items, mode):
    """Runs a Dataiku job and waits for it, showing its PROGRESS lines here (also in the job's log) every 30 s."""
    print(f"\n{step} ...", flush=True)
    builder = project.new_job(mode).with_auto_update_schema_before_each_recipe_run(True)
    for item in items:
        if item in folders:
            builder.with_output(folders[item].id, object_type="MANAGED_FOLDER")
        else:
            builder.with_output(item)
    job, shown, began = builder.start(), 0, time.time()
    while True:
        state = job.get_status().get("baseStatus", {}).get("state", "")
        try:
            lines = [l[l.index("PROGRESS ") + 9:] for l in job.get_log().splitlines() if "PROGRESS " in l]
        except Exception:
            lines = []
        for line in lines[shown:]:
            print("   ", line, flush=True)
        shown = max(shown, len(lines))
        if state in ("DONE", "FAILED", "ABORTED"):
            break
        time.sleep(30 if time.time() - began > 60 else 5)
    if state != "DONE":
        raise RuntimeError(f"{step}: the job {state}. Open it in Jobs (id {job.id}) to read its log.")
    print(f"{step}: done in {(time.time() - began) / 60:.1f} min", flush=True)


started = time.time()
warehouse.set_cutoff()                                          # the first 2 steps of every scenario
warehouse.check_warehouse_loaded()
build("1/4 Staging: read the warehouse (clients and daily flows)", ["daily_flows"], "RECURSIVE_FORCED_BUILD")
staged_only = months is not None and months < cashflow.MIN_TRAINING_MONTHS
if staged_only:                                                # too little history to train: stop after staging
    print(f"\nStaging done in {(time.time() - started) / 60:.1f} minutes. Steps 2-4 (training, promotion, first "
          f"forecast) are skipped: they need at least {cashflow.MIN_TRAINING_MONTHS} months of history, and "
          f"HISTORY_MONTHS is {months}.")
else:
    build("2/4 Train, test, track and deploy the first model", ["experiments"], "NON_RECURSIVE_FORCED_BUILD")
    build("3/4 Champion / challenger", ["promotion_log"], "NON_RECURSIVE_FORCED_BUILD")
    production = tracking.apply_promotion()                    # activates its version of cash_forecast
    build("4/4 Forecast every active client, archive and log the run", ["inference_log"], "RECURSIVE_FORCED_BUILD")
    print(f"First run done in {(time.time() - started) / 60:.1f} minutes: production model {production}")
