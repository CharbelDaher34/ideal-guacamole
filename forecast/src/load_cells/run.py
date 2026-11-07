# Runs the recipe final_forecast alone (the forecast, its archive and the inference log), as the Nightly batch does
# after staging, and shows its row of inference_log: rows, chunks, seconds per step and the most memory it used.
variables = project.get_variables()
variables["standard"]["inference_clients_per_chunk"] = CLIENTS_PER_CHUNK
project.set_variables(variables)
started = time.time()
project.new_job("NON_RECURSIVE_FORCED_BUILD").with_output("inference_log").start_and_wait()
print(f"final_forecast took {(time.time() - started) / 60:.1f} minutes (with Dataiku's job start)")
log = dataiku.Dataset("inference_log").get_dataframe()
display(log.sort_values("run_at").tail(1).T)
