# One client, day by day: what the models see (daily net, all categories), and the monthly totals they predict
client_no = monthly[monthly["category"] == TOTAL].groupby("client_no")["net"].size().idxmax()   # longest history
one = daily[daily["client_no"] == client_no].assign(net=lambda d: d["money_in"] - d["money_out"])
one = one.groupby(pd.to_datetime(one["day"]))["net"].sum().loc[lambda s: s.index > cutoff - pd.DateOffset(months=6)]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.5))
(one / 1e3).plot(ax=a, style=".", title=f"Client {client_no}: daily net, last 6 months (thousand SAR)")
months = monthly[(monthly["client_no"] == client_no) & (monthly["category"] == TOTAL)].set_index("month")["net"]
(months.tail(24) / 1e3).plot.bar(ax=b, title="Monthly net, last 24 months (thousand SAR)")
b.set_xticklabels([f"{m:%y-%m}" for m in months.tail(24).index], rotation=90)
plt.tight_layout()
plt.show()
