FOLDER = "daily_flows"       # the folder to read: daily_flows (the real data) or load_test_flows (the load test's)
SUMMARY_FILES = 10           # how many Parquet files the summary reads (-1: all of them; slower on the full data)
SAMPLE_ROWS = 20             # rows shown from the first file
CLIENT_NO = ""               # a client number to show day by day (empty: the first client of the first file)
