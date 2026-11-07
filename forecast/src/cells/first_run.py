def build(items, mode):
    job = project.new_job(mode).with_auto_update_schema_before_each_recipe_run(True)
    for item in items:
        if item in folders:
            job.with_output(folders[item].id, object_type="MANAGED_FOLDER")
        else:
            job.with_output(item)
    job.start_and_wait()


started = time.time()
warehouse.set_cutoff()                                          # the first step of both scenarios
build(["daily_flows"], "RECURSIVE_FORCED_BUILD")             # staging: clients and daily flows
build(["experiments"], "NON_RECURSIVE_FORCED_BUILD")         # train, test, track and deploy the first model
build(["promotion_log"], "NON_RECURSIVE_FORCED_BUILD")       # champion / challenger
production = tracking.apply_promotion()                        # activates its version of cash_forecast
build(["inference_log"], "RECURSIVE_FORCED_BUILD")           # forecast, archive, log the run
print(f"First run done in {(time.time() - started) / 60:.1f} minutes: production model {production}")
