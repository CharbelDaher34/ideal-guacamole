# Reads only: nothing in this notebook changes the project.
import time

import pandas as pd
import polars as pl

from cashflow import client_groups, read_group, read_manifest, read_parquet

folder, manifest = read_manifest(FOLDER)
if manifest is None:
    raise RuntimeError(f"Nothing in the folder {FOLDER}: run staging (or, in the load test project, the scenario Create "
                       "the fake data) first")
files = pd.DataFrame([(int(g), month, path, rows) for g, months in manifest["files"].items()
                      for month, (path, rows) in months.items()],
                     columns=["group", "month", "file", "rows"]).sort_values(["group", "month"]).reset_index(drop=True)
print(f"{FOLDER}: {manifest['rows']:,} rows in {len(files):,} files: {manifest['groups']:,} groups of clients (a client "
      f"never changes group), one file per group and month, {files['month'].min()} to {files['month'].max()}")
