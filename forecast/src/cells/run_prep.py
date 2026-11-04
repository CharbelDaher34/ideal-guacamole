started = time.time()
lib["check_latest_load"]()      # the same check that starts each scenario: complete load, sets cutoff_month
job = project.new_job("RECURSIVE_FORCED_BUILD").with_auto_update_schema_before_each_recipe_run(True)
for name in ["ml_train", "ml_test", "ml_scoring"] + list(lib["TABLES"]):
    job.with_output(name)
job.start_and_wait()
print(f"Staging and data prep built in {(time.time() - started) / 60:.1f} minutes")
