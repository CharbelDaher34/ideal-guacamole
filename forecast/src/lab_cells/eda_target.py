# What the models learn: target / series_scale (the client's usual amount in that category), per category
scaled = rows.assign(category_name=rows["category"].map(names), y=rows["target"] / rows["series_scale"])
display(scaled.groupby("category_name")["y"].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).round(2))

# How much each input moves with it (Spearman correlation, inputs divided by the series scale too)
inputs = ["last_amount", "avg_3m", "same_month_last_year", "free_last", "free_avg_3m", "oblig_target_month"]
correlation = {}
for name, part in scaled.groupby("category_name"):
    x = part[inputs].div(part["series_scale"], axis=0)
    x = x.loc[:, x.nunique() > 1]                   # an input that never changes has no correlation (blank)
    correlation[name] = x.corrwith(part["y"], method="spearman")
display(pd.DataFrame(correlation).T.reindex(columns=inputs).round(2))
