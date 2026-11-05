def build(items, mode):
    job = project.new_job(mode).with_auto_update_schema_before_each_recipe_run(True)
    for item in items:
        if item in folders:
            job.with_output(folders[item].id, object_type="MANAGED_FOLDER")
        else:
            job.with_output(item)
    job.start_and_wait()


started = time.time()
warehouse.check_latest_load()                                   # the first step of both scenarios
build(["training_rows"], "RECURSIVE_FORCED_BUILD")
build(["experiments"], "NON_RECURSIVE_FORCED_BUILD")         # train, test, track and deploy the first model
build(["promotion_log"], "NON_RECURSIVE_FORCED_BUILD")       # champion / challenger
production = tracking.apply_promotion()                        # activates it in the three saved models
build(["inference_log"], "RECURSIVE_FORCED_BUILD")           # score, forecast, archive, log the run
print(f"First run done in {(time.time() - started) / 60:.1f} minutes: production model {production}")
