print("Production model (active version of forecast_ev, forecast_p10, forecast_p90):", lib["production_model_id"]())
print("Promotion log:")
display(dataiku.Dataset("promotion_log").get_dataframe())
results = dataiku.Dataset("test_results").get_dataframe()
print("Test on the last 6 months (all months ahead together):")
display(results[results["months_ahead"].astype(str) == "ALL"][
    ["category_name", "wape_model", "wape_baseline", "coverage_p10_p90", "below_p10", "above_p90"]])
print("Inference log (one row per forecast run):")
display(dataiku.Dataset("inference_log").get_dataframe())
