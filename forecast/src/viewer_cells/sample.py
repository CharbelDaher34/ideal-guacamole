first = read_parquet(folder, files["file"].iloc[0]).to_pandas()
print(f"{files['file'].iloc[0]}: {len(first):,} rows, {first['client_no'].nunique():,} clients. Columns:")
print(first.dtypes.to_string())
display(first.head(SAMPLE_ROWS))
