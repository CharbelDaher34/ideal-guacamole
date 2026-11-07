# Per category: how many clients use it, money in and out per month, and how often a client's month is negative
d = daily.assign(month=cashflow.month_start(daily["day"]).values)
per_month = d.groupby(["category", "client_no", "month"])[["money_in", "money_out"]].sum().reset_index()
overview = per_month.groupby("category").agg(clients=("client_no", "nunique"), money_in=("money_in", "mean"),
                                             money_out=("money_out", "mean"))
overview["negative_months"] = (monthly[monthly["category"] != TOTAL].assign(neg=lambda m: m["net"] < 0)
                               .groupby("category")["neg"].mean())
display(overview.round({"money_in": 0, "money_out": 0, "negative_months": 3}).sort_values("clients", ascending=False))

# Per segment, the client's whole net cash flow (TOTAL)
total = monthly[monthly["category"] == TOTAL].merge(clients[["client_no", "segment"]], on="client_no")
display(total.groupby("segment").agg(clients=("client_no", "nunique"), months=("month", "size"),
                                     net=("net", "mean"), negative_months=("net", lambda s: (s < 0).mean()))
        .round({"net": 0, "negative_months": 3}))
