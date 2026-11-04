# Compare the two ways of training on the same last 6 months (both trained without them):
#   EV error (WAPE) = sum |actual - EV| / sum |actual|, and the share of actual months between P10 and P90.
# The project variable model_source picks which one feeds the final forecast.
import dataiku
import numpy as np
import pandas as pd

from cashflow import CATEGORIES, finish, read_dataset

keys = ["client_no", "category", "cutoff_month", "months_ahead"]
code = read_dataset("test_predictions")[keys + ["target", "p10", "ev", "p90"]].assign(source="code")

visual = read_dataset("eval_ev_visual_scored")[keys + ["series_scale", "target", "prediction"]]
visual = visual.rename(columns={"prediction": "ev"})
for q in ("p10", "p90"):
    scored = read_dataset(f"eval_{q}_visual_scored")[keys + ["prediction"]].rename(columns={"prediction": q})
    visual = visual.merge(scored, on=keys)
s = visual["series_scale"]
visual[["p10", "ev", "p90"]] = finish(visual["ev"] * s, visual["p10"] * s, visual["p90"] * s, visual["category"],
                                      visual.index)
both = pd.concat([code, visual.drop(columns="series_scale").assign(source="visual")], ignore_index=True)


def scores(d):
    size = d["target"].abs().sum()
    return pd.Series({"rows": len(d),
                      "wape": round((d["target"] - d["ev"]).abs().sum() / size, 4) if size else np.nan,
                      "coverage_p10_p90": round(((d["target"] >= d["p10"]) & (d["target"] <= d["p90"])).mean(), 4),
                      "below_p10": round((d["target"] < d["p10"]).mean(), 4),
                      "above_p90": round((d["target"] > d["p90"]).mean(), 4)})


per_category = both.groupby(["category", "source"]).apply(scores).reset_index()
overall = both[both["category"] != "NET"].groupby("source").apply(scores).reset_index().assign(category="ALL")
table = pd.concat([overall, per_category], ignore_index=True)
table["category_name"] = table["category"].map({c: v[0] for c, v in CATEGORIES.items()}).fillna("All categories")
wide = table.pivot(index=["category", "category_name"], columns="source",
                   values=["wape", "coverage_p10_p90", "below_p10", "above_p90", "rows"])
wide.columns = [f"{metric}_{source}" for metric, source in wide.columns]
wide = wide.reset_index()
wide["better_ev"] = np.where(wide["wape_code"] <= wide["wape_visual"], "code", "visual")
dataiku.Dataset("model_comparison").write_with_schema(wide)
