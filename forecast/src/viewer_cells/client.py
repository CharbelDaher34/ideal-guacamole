# One client's daily rows and its net cash flow per month (money in, money out, net): only its group's files are read.
wanted = str(CLIENT_NO).strip() or first["client_no"].iloc[0]
group = client_groups([wanted], manifest["groups"])[0]
rows = read_group(folder, manifest, group).filter(pl.col("client_no") == wanted).sort("day", "category").to_pandas()
if rows.empty:
    print(f"Client {wanted} is not in {FOLDER}")
else:
    print(f"Client {wanted} (group {group}): {len(rows):,} rows, {rows['day'].min():%Y-%m-%d} to "
          f"{rows['day'].max():%Y-%m-%d}, {rows['postings'].sum():,} postings")
    display(rows.tail(SAMPLE_ROWS))
    monthly = (rows.assign(month=rows["day"].dt.to_period("M").dt.to_timestamp())
               .groupby("month")[["money_in", "money_out"]].sum())
    monthly["net"] = monthly["money_in"] - monthly["money_out"]
    display(monthly.round(0))
    ax = monthly.plot(figsize=(12, 4), title=f"Client {wanted}: money in, money out and net per month (SAR)")
    ax.axhline(0, color="grey", linewidth=0.8)
