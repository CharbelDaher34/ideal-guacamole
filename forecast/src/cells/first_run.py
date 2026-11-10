started = time.time()
warehouse.set_cutoff()                                          # the first 2 steps of every scenario
warehouse.check_warehouse_loaded()
build("1/5 Staging: read the warehouse (clients and daily flows)", ["daily_flows"], "RECURSIVE_FORCED_BUILD")
staged_only = months is not None and months < cashflow.MIN_TRAINING_MONTHS
if staged_only:                                                # too little history to train: stop after staging
    print(f"\nStaging done in {(time.time() - started) / 60:.1f} minutes. Steps 2-5 (training, promotion, first "
          f"forecast) are skipped: they need at least {cashflow.MIN_TRAINING_MONTHS} months of history, and "
          f"HISTORY_MONTHS is {months}.")
else:
    build("2/5 Train, test, track and deploy the first model", ["experiments"], "NON_RECURSIVE_FORCED_BUILD")
    build("3/5 Champion / challenger", ["promotion_log"], "NON_RECURSIVE_FORCED_BUILD")
    production = tracking.apply_promotion()                    # activates its version of cash_forecast
    build("4/5 Forecast every active client and log the run", ["inference_log"], "RECURSIVE_FORCED_BUILD")
    build("5/5 Load the forecast into the SQL table final_forecast", ["final_forecast"], "NON_RECURSIVE_FORCED_BUILD")
    print(f"First run done in {(time.time() - started) / 60:.1f} minutes: production model {production}")
