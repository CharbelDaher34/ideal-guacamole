# Rows per month (all the groups together), from the manifest: no file is read here.
display(files.groupby("month").agg(files=("file", "size"), rows=("rows", "sum")))
display(files.head(SAMPLE_ROWS))
