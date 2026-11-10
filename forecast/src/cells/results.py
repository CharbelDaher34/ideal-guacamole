if staged_only:                                                 # only staging ran (HISTORY_MONTHS too short)
    clients_staged = dataiku.Dataset("clients").get_dataframe()
    print(f"Staged: {len(clients_staged):,} business clients, {cashflow.read_manifest()[1]['rows']:,} "
          "daily_flows rows (per client, day and category)")
    print("Staging log (one row per staging run: what was read from the warehouse):")
    display(dataiku.Dataset("staging_log").get_dataframe().sort_values("run_at"))
else:
    print("Production model (active version of the saved model cash_forecast):", tracking.production_model_id())
    print("Promotion log:")
    display(dataiku.Dataset("promotion_log").get_dataframe())
    print("Test on the last 6 periods of 30 days: TOTAL rows (net per client), category rows, and ALL together, by window (1 = days 1-30, 2 = 31-60, 3 = 61-90):")
    display(dataiku.Dataset("test_results").get_dataframe()[
        ["level", "window", "rows", "wape_model", "wape_baseline", "coverage_p10_p90", "below_p10", "above_p90"]])
    final = dataiku.Dataset("final_forecast").get_dataframe()
    print(f"Final forecast: one row per active client, {len(final)} rows, {final.shape[1]} columns. Status:",
          final["status"].value_counts().to_dict())
    one = final[final["status"] == "OK"].iloc[0]
    print(f"One client, {one['client_no']} (TOTAL and the first categories; empty: a category it does not have):")
    display(one.dropna().head(40).to_frame("value"))
    print("Inference log (one row per forecast run):")
    display(dataiku.Dataset("inference_log").get_dataframe().sort_values("run_at"))
    print("Staging log (one row per staging run: what was read from the warehouse):")
    display(dataiku.Dataset("staging_log").get_dataframe().sort_values("run_at"))
