# Train each on the same months, test it on the last 6, next to the production model. recent_average is the
# production recipe trained again: it should match the production model's scores (a check that the test is the same).
RECORD = True       # record each try in Experiment Tracking (experiment cashflow_lab)
result = lab_compare([RecentAverage(), RecentAverage6(), LagBoosting()], record=RECORD)
print("Net cash flow per client, category and TOTAL. EV error: lower is better. P10-P90 coverage: aim 80%.")
display(result["board"])
print("EV error per months ahead (ALL: together):")
display(result["errors"])
print("P10-P90 coverage per months ahead:")
display(result["coverage"])
