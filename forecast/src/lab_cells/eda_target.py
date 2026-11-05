# What the models learn: the net cash flow divided by the client's scale (its usual month), overall and by segment
scaled = rows.assign(y=rows["target"] / rows["scale"])
display(scaled.groupby("segment")["y"].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).round(2))

# How much each input moves with it (Spearman correlation, inputs divided by the scale too)
inputs = ["last_amount", "avg_3m", "same_month_last_year", "client_in_avg_3m", "client_out_avg_3m"]
x = scaled[inputs].div(scaled["scale"], axis=0)
display(x.loc[:, x.nunique() > 1].corrwith(scaled["y"], method="spearman").round(2).to_frame("correlation"))
