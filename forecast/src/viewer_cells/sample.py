first = read_file(paths[0])
print(f"{paths[0]}: {len(first):,} rows, {first['client_no'].nunique():,} clients. Columns:")
print(first.dtypes.to_string())
display(first.head(SAMPLE_ROWS))
