# An example upload, the first time only: two clients of the data (the last 24 months), renamed EXAMPLE_1 and
# EXAMPLE_2, in the format the folder try_model_uploads takes. Then forecast them with the production model.
uploads = dataiku.Folder("try_model_uploads")
if not uploads.list_paths_in_partition():
    forecast = dataiku.Dataset("final_forecast").get_dataframe()
    picked = forecast.loc[forecast["status"] == "OK", "client_no"].astype(str).drop_duplicates().head(2).tolist()
    names = {c: f"EXAMPLE_{i + 1}" for i, c in enumerate(picked)}
    flows = lib["read_dataset"]("monthly_flows").astype({"client_no": str})
    flows["month"] = lib["month_start"](flows["month"])
    flows = flows[flows["client_no"].isin(picked) & (flows["month"] > flows["month"].max() - pd.DateOffset(months=24))]
    balances = lib["read_dataset"]("monthly_balances").astype({"client_no": str})
    balances["month"] = lib["month_start"](balances["month"])
    profile = dataiku.Dataset("clients").get_dataframe().astype({"client_no": str})
    example = (flows[["client_no", "month", "category", "amount"]]
               .merge(profile[["client_no", "segment", "sector", "region", "last_annual_sales", "number_of_employees"]],
                      on="client_no", how="left")
               .merge(balances[["client_no", "month", "balance_end"]], on=["client_no", "month"], how="left"))
    example["client_no"] = example["client_no"].map(names)
    example["month"] = example["month"].dt.strftime("%Y-%m")
    uploads.upload_data("example_clients.csv", example.round(2).to_csv(index=False).encode())
    print(f"Example upload written: example_clients.csv ({len(example)} rows, {len(names)} clients)")
build(["try_model_forecast"], "NON_RECURSIVE_FORCED_BUILD")
tried = dataiku.Dataset("try_model_forecast").get_dataframe()
print("Forecast of the uploaded clients (months ahead 1, net cash flow):")
display(tried[(tried["months_ahead"] == 1) & (tried["category"] == "NET")])
