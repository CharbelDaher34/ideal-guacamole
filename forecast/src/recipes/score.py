# Score: P10, EV and P90 of the part the models predict, for each of the next 3 months (final models).
# Clients with less than 12 months of history get no prediction; a category the client did not use in the
# last 12 months gets 0.
import pickle

import dataiku
import pandas as pd

from cashflow import predict, read_dataset

rows = read_dataset("scoring_rows")
folder = dataiku.Folder("models")
models = {}
for kind in ("ev", "p10_p90"):
    with folder.get_download_stream(f"final_{kind}.pkl") as s:
        models[kind] = pickle.loads(s.read())

ok = rows["enough_history"].astype(bool) & rows["active"].astype(bool)
rows = rows.join(predict(models, rows[ok]))
unused = rows["enough_history"].astype(bool) & ~rows["active"].astype(bool)
rows.loc[unused, ["p10", "ev", "p90"]] = 0.0
cols = ["client_no", "category", "direction", "cutoff_month", "months_ahead", "target_month", "history_months",
        "enough_history", "p10", "ev", "p90"]
dataiku.Dataset("predictions").write_with_schema(rows[cols])
