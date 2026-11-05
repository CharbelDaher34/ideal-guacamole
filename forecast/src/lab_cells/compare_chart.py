errors = result["errors"].drop(index="ALL")
if "simple_guess" in errors:                         # the same EV as the baseline: shown once
    errors = errors.drop(columns="simple_guess_baseline")
ax = errors.plot.bar(figsize=(10, 4), title="EV error per months ahead (lower is better)")
ax.set_xlabel("months ahead")
plt.tight_layout()
plt.show()
