if RUN_NOW:      # what the 2 scenarios run, as jobs (the recipes read the numbers of each scenario's step 1)
    build("1/2 Create the fake data (not timed)", ["load_test_flows"], "NON_RECURSIVE_FORCED_BUILD")
    build("2/2 Run the load test: every test of the plan", ["load_test_log"], "NON_RECURSIVE_FORCED_BUILD")
    build("The report", ["load_test_report"], "NON_RECURSIVE_FORCED_BUILD")
    log = dataiku.Dataset("load_test_log").get_dataframe()
    batch = log[log["batch_id"] == log["batch_id"].max()].sort_values("test_no")
    print(f"Batch {batch['batch_id'].iloc[0]}, on {int(batch['clients'].iloc[0]):,} fake clients:")
    display(batch[["test", "status", "workers", "seconds_forecast", "seconds_sql_load", "clients_per_second",
                   "peak_memory_mb", "minutes_at_bank_size", "error"]].set_index("test"))
    print("The full report: Dashboards > Load test, or the folder load_test_report > latest_report.html")
else:
    print("Set up. Then: Scenarios > Create the fake data (its size: Steps > 1), Run; then Scenarios > Run the load "
          "test (its plan: Steps > 1), Run")
