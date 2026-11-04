# Train the two models (EV, and P10/P90 together) twice:
#   backtest: only rows whose predicted month is before the last 6 months (used by the test step)
#   final:    all rows (used to forecast the next 3 months)
import json
import pickle

import dataiku
import pandas as pd

from cashflow import (CATBOOST, FEATURES, HOLDOUT_MONTHS, QUANTILES, add_months, cutoff_month, fit_models,
                      month_start, read_dataset)

last_month = cutoff_month()
holdout_start = add_months(last_month, -(HOLDOUT_MONTHS - 1))
rows = read_dataset("training_rows")
rows["target_month"] = month_start(rows["target_month"])
folder = dataiku.Folder("models")

info = {"last_month": str(last_month.date()), "holdout_start": str(holdout_start.date()), "features": FEATURES,
        "quantiles": list(QUANTILES), "catboost": CATBOOST}
for name, part in (("backtest", rows[rows["target_month"] < holdout_start]), ("final", rows)):
    models = fit_models(part)
    for kind, model in models.items():
        with folder.get_writer(f"{name}_{kind}.pkl") as w:
            pickle.dump(model, w)
    importance = models["ev"].get_feature_importance(prettified=True)
    info[name] = {"rows": len(part), "clients": int(part["client_no"].nunique()),
                  "target_months": [str(part["target_month"].min().date()), str(part["target_month"].max().date())],
                  "ev_feature_importance": dict(zip(importance["Feature Id"], importance["Importances"].round(2)))}
    print(name, info[name]["rows"], "rows")
with folder.get_writer("info.json") as w:
    w.write(json.dumps(info, indent=2).encode())
