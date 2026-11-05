# Monthly totals over all clients, money in and money out by category
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for direction, ax in zip(("IN", "OUT"), axes):
    totals = monthly[monthly["direction"] == direction].pivot_table(
        index="target_month", columns="category_name", values="actual", aggfunc="sum")
    (totals / 1e6).plot(ax=ax, title=f"Money {direction.lower()} per month, all clients (million SAR)")
    ax.set_xlabel("")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# Seasonality without the trend: all clients' amount in a calendar month divided by their average of the 12 months
# before it (1 = a usual month). Totals, not a median: most months of lumpy categories are zero.
seasonal = monthly.assign(calendar_month=monthly["target_month"].dt.month).pivot_table(
    index="category_name", columns="calendar_month", values=["actual", "avg_12m"], aggfunc="sum")
profile = seasonal["actual"] / seasonal["avg_12m"]
fig, ax = plt.subplots(figsize=(10, 4.5))
image = ax.imshow(profile.values, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(profile.columns)))
ax.set_xticklabels([calendar.month_abbr[m] for m in profile.columns])
ax.set_yticks(range(len(profile.index)))
ax.set_yticklabels(profile.index)
ax.set_title("Seasonality: month / the 12 months before, all clients (1 = usual)")
fig.colorbar(image, ax=ax)
plt.show()
