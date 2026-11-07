# Train each on the same months, test it on the last 6, next to the production model. quantile_boosting is the
# production recipe trained again: when the production model was trained on the same cut-off day, their scores match
# (a check that the test is the same).
RECORD = True       # record each try in Experiment Tracking (experiment cashflow_lab)
result = lab_compare([QuantileBoosting(), QuantileBoostingByCategory(), RecentAverage(), RecentAverage6()],
                     record=RECORD)
print("Net cash flow per client, category and TOTAL. EV error: lower is better. P10-P90 coverage: aim 80%.")
display(result["board"])
print("EV error per window (ALL: together):")
display(result["errors"])
print("P10-P90 coverage per window:")
display(result["coverage"])
