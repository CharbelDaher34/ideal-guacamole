# Reads only: nothing in this notebook changes the project.
import io
import time

import dataiku
import pandas as pd
import pyarrow.parquet as pq

from cashflow import CURRENT, _daily_files, _plain

folder, paths = _daily_files(FOLDER)
if not paths:
    raise RuntimeError(f"No Parquet file in the folder {FOLDER}: run staging (or the scenario Load test) first")
with folder.get_download_stream(CURRENT) as stream:
    version = stream.read().decode().strip()


def read_file(path, columns=None, filters=None):
    """One Parquet file of the folder as a DataFrame."""
    with folder.get_download_stream(path) as stream:
        data = stream.read()
    return _plain(pq.read_table(io.BytesIO(data), columns=columns, filters=filters).to_pandas())


print(f"{FOLDER}: current version {version}, {len(paths)} Parquet files")
