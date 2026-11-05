# Each month once per client and category (months ahead 1): the amount (actual), the part already scheduled at the
# cut-off (obligations_known) and what the models predict (target = actual - obligations_known). The net cash flow
# (money in minus money out) is left out here: its ratios mix signs.
monthly = rows[(rows["months_ahead"] == 1) & (rows["category"] != "NET")].assign(
    category_name=lambda d: d["category"].map(names))
by_category = monthly.groupby("category_name")
series = monthly.groupby(["category_name", "client_no"])["actual"]
volatility = (series.std() / series.mean().abs()).replace(np.inf, np.nan)
overview = pd.DataFrame({
    "clients": by_category["client_no"].nunique(),
    "months": by_category.size(),
    "average_month_sar": by_category["actual"].mean().round(0),
    "zero_months": by_category["actual"].apply(lambda s: (s.abs() < 1).mean()).round(3),
    "known_at_cutoff": (by_category["obligations_known"].sum() / by_category["actual"].sum()).round(3),
    "volatility": volatility.groupby(level="category_name").median().round(2),
}).sort_values("average_month_sar", ascending=False)
display(overview)
print("Clients by segment:", rows.drop_duplicates("client_no")["segment"].value_counts().to_dict())
