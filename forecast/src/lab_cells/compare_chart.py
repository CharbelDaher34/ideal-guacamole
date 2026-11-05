errors = result["errors"].drop(index="All categories")
if "simple_guess" in errors:                         # the same EV as the baseline: shown once
    errors = errors.drop(columns="simple_guess_baseline")
ax = errors.plot.barh(figsize=(10, 8), title="EV error per category (lower is better)")
ax.set_xlim(0, min(errors.max().max() * 1.05, 3))
ax.set_ylabel("")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
