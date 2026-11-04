comparison = dataiku.Dataset("model_comparison").get_dataframe()
print("Code vs visual models on the last 6 months (wape: lower is better; coverage: close to 0.8 is best):")
display(comparison[["category_name", "wape_code", "wape_visual", "coverage_p10_p90_code",
                    "coverage_p10_p90_visual", "better_ev"]])

forecast = dataiku.Dataset("final_forecast").get_dataframe()
example = forecast.loc[forecast["status"] == "OK", "client_no"].iloc[0]
print(f"Forecast for client {example} (SAR, models: {MODEL_SOURCE}):")
display(forecast[forecast["client_no"] == example]
        .pivot_table(index="category_name", columns="months_ahead", values=["p10", "ev", "p90"], sort=False)
        .round(0))
