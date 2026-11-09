# One client's daily rows and its net cash flow per month (money in, money out, net). Its rows are all in one file:
# the files are read until it is found.
wanted = str(CLIENT_NO).strip() or first["client_no"].iloc[0]
rows = pd.DataFrame()
for path in paths:
    rows = read_file(path, filters=[("client_no", "=", wanted)])
    if len(rows):
        break
if rows.empty:
    print(f"Client {wanted} is not in {FOLDER}")
else:
    rows = rows.sort_values(["day", "category"]).reset_index(drop=True)
    print(f"Client {wanted}: {len(rows):,} rows, {rows['day'].min():%Y-%m-%d} to {rows['day'].max():%Y-%m-%d}, "
          f"{rows['postings'].sum():,} postings, in {path}")
    display(rows.tail(SAMPLE_ROWS))
    monthly = (rows.assign(month=rows["day"].dt.to_period("M").dt.to_timestamp())
               .groupby("month")[["money_in", "money_out"]].sum())
    monthly["net"] = monthly["money_in"] - monthly["money_out"]
    display(monthly.round(0))
    ax = monthly.plot(figsize=(12, 4), title=f"Client {wanted}: money in, money out and net per month (SAR)")
    ax.axhline(0, color="grey", linewidth=0.8)
