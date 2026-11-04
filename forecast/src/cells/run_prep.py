started = time.time()
job = project.new_job("RECURSIVE_FORCED_BUILD").with_auto_update_schema_before_each_recipe_run(True)
for name in ["ml_train", "ml_test", "ml_scoring"] + list(lib["TABLES"]):
    job.with_output(name)
job.start_and_wait()
status = dataiku.Dataset("load_status").get_dataframe()
print(f"Load {status['load_id'][0]} checked: as of {status['as_of_date'][0]:%Y-%m-%d}, "
      f"cut-off month {status['cutoff_month'][0]:%Y-%m}")
print(f"Staging and data prep built in {(time.time() - started) / 60:.1f} minutes")
