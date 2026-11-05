print("Production model (active version of forecast_ev, forecast_p10, forecast_p90):", tracking.production_model_id())
print("Promotion log:")
display(dataiku.Dataset("promotion_log").get_dataframe())
print("Test on the last 6 months, net cash flow per client (by months ahead, and ALL):")
display(dataiku.Dataset("test_results").get_dataframe()[
    ["months_ahead", "rows", "wape_model", "wape_baseline", "coverage_p10_p90", "below_p10", "above_p90"]])
print("Inference log (one row per forecast run):")
display(dataiku.Dataset("inference_log").get_dataframe())
