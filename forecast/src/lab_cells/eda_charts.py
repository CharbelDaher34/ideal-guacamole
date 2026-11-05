# Totals over all clients per month: money in, money out and net (million SAR)
totals = monthly.groupby("month")[["IN", "OUT", "NET"]].sum() / 1e6
ax = totals.plot(figsize=(12, 4), title="All clients per month (million SAR)")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("")
plt.show()

# Seasonality without the trend: each month's total divided by the average of the 12 months before it (1 = usual)
ratio = totals[["IN", "OUT"]] / totals[["IN", "OUT"]].rolling(12).mean().shift(1)
profile = ratio.groupby(ratio.index.month).mean()
profile.index = [calendar.month_abbr[m] for m in profile.index]
ax = profile.plot.bar(figsize=(10, 3.5), title="Seasonality: month / the 12 months before, all clients (1 = usual)")
ax.axhline(1, color="grey", linewidth=0.5)
plt.show()
