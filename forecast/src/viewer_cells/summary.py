# Totals over the first SUMMARY_FILES files (each client's rows are all in one file), read one file at a time.
chosen = paths if SUMMARY_FILES in (-1, None) else paths[:SUMMARY_FILES]
started, parts, clients_read = time.time(), [], 0
for i, path in enumerate(chosen, 1):
    d = read_file(path)
    d["net"] = d["money_in"] - d["money_out"]
    clients_read += d["client_no"].nunique()
    parts.append(d.groupby("category").agg(rows=("client_no", "size"), clients=("client_no", "nunique"),
                                           postings=("postings", "sum"), money_in=("money_in", "sum"),
                                           money_out=("money_out", "sum"), net=("net", "sum"),
                                           first_day=("day", "min"), last_day=("day", "max")))
    if i % 10 == 0 or i == len(chosen):
        print(f"read {i} of {len(chosen)} files ({time.time() - started:.0f} s)")
by_category = pd.concat(parts).groupby(level=0).agg({"rows": "sum", "clients": "sum", "postings": "sum",
                                                     "money_in": "sum", "money_out": "sum", "net": "sum",
                                                     "first_day": "min", "last_day": "max"})
print(f"{'All' if len(chosen) == len(paths) else f'First {len(chosen)} of {len(paths)}'} files: "
      f"{by_category['rows'].sum():,} rows, {clients_read:,} clients, {by_category['postings'].sum():,} postings, "
      f"from {by_category['first_day'].min():%Y-%m-%d} to {by_category['last_day'].max():%Y-%m-%d}")
display(by_category.round(0).sort_values("rows", ascending=False))
