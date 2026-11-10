# Totals over the first SUMMARY_GROUPS groups of clients (each client's rows are all in its group), one group at a time.
groups = sorted(int(g) for g in manifest["files"])
chosen = groups if SUMMARY_GROUPS in (-1, None) else groups[:SUMMARY_GROUPS]
started, parts = time.time(), []
for i, group in enumerate(chosen, 1):
    parts.append(read_group(folder, manifest, group).group_by("category").agg(
        rows=pl.len(), clients=pl.col("client_no").n_unique(), postings=pl.col("postings").sum(),
        money_in=pl.col("money_in").sum(), money_out=pl.col("money_out").sum(),
        first_day=pl.col("day").min(), last_day=pl.col("day").max()))
    if i % 10 == 0 or i == len(chosen):
        print(f"read {i} of {len(chosen)} groups ({time.time() - started:.0f} s)")
by_category = (pl.concat(parts).group_by("category").agg(
    pl.col("rows", "clients", "postings", "money_in", "money_out").sum(), pl.col("first_day").min(),
    pl.col("last_day").max()).with_columns(net=pl.col("money_in") - pl.col("money_out")).sort("rows", descending=True))
print(f"{'All' if len(chosen) == len(groups) else f'The first {len(chosen)} of {len(groups)}'} groups: "
      f"{by_category['rows'].sum():,} rows, {by_category['clients'].sum():,} client-categories")
display(by_category.to_pandas().round(0))
