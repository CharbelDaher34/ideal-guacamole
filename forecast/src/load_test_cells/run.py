if RUN_NOW:      # what the scenario Run the load test runs, as jobs (the recipes read its step 1)
    build("1/3 The fake data of each size (not timed; kept if already there)", ["load_test_flows"],
          "NON_RECURSIVE_FORCED_BUILD")
    try:
        build("2/3 The tests", ["load_test_log"], "NON_RECURSIVE_FORCED_BUILD")
    finally:                                         # the report even when a test failed
        build("3/3 The report", ["load_test_report"], "NON_RECURSIVE_FORCED_BUILD")
        log = dataiku.Dataset("load_test_log").get_dataframe().sort_values("run_at")
        print("The last tests (the newest last):")
        display(log[["batch", "test", "status", "workers", "clients", "clients_per_group", "seconds_forecast",
                     "seconds_sql_load", "clients_per_second", "peak_memory_mb", "error"]].tail(10).set_index("batch"))
        print("The full report: Dashboards > Load test, or the folder load_test_report > latest_report.html")
else:
    print("Set up. Then: Scenarios > Run the load test (its tests: Steps > 1), Run")
