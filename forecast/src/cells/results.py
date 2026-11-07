print("Production model (active version of the saved model cash_forecast):", tracking.production_model_id())
print("Promotion log:")
display(dataiku.Dataset("promotion_log").get_dataframe())
print("Test on the last 6 months: TOTAL rows (net per client), category rows, and ALL together, by months ahead:")
display(dataiku.Dataset("test_results").get_dataframe()[
    ["level", "months_ahead", "rows", "wape_model", "wape_baseline", "coverage_p10_p90", "below_p10", "above_p90"]])
final = dataiku.Dataset("final_forecast").get_dataframe()
one = final.loc[final["status"] == "OK", "client_no"].iloc[0]
print(f"Final forecast: {final['client_no'].nunique()} clients, {len(final)} rows. One client, {one}:")
display(final[final["client_no"] == one])
print("Inference log (one row per forecast run):")
display(dataiku.Dataset("inference_log").get_dataframe())
