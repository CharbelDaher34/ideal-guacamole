# All clients per month: the net cash flow of each category (million SAR)
by_category = (monthly[monthly["category"] != TOTAL].pivot_table(index="month", columns="category", values="net",
                                                                  aggfunc="sum") / 1e6)
ax = by_category.plot(figsize=(12, 4.5), title="Net cash flow per category, all clients (million SAR)")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# Seasonality without the trend: money in and out each month / the average of the 12 months before (1 = usual)
d = daily.assign(month=cashflow.month_start(daily["day"]).values)
totals = d.groupby("month")[["money_in", "money_out"]].sum()
ratio = totals / totals.rolling(12).mean().shift(1)
profile = ratio.groupby(ratio.index.month).mean()
profile.index = [calendar.month_abbr[m] for m in profile.index]
ax = profile.plot.bar(figsize=(10, 3.5), title="Seasonality: month / the 12 months before, all clients (1 = usual)")
ax.axhline(1, color="grey", linewidth=0.5)
plt.show()
