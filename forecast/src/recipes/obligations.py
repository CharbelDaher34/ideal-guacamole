# Obligations: the scheduled payments (from the warehouse, past and future) per client, category, month due and
# month known. Their category comes from category_mapping (visual Join recipe scheduled_with_category).
import dataiku
import pandas as pd

from cashflow import add_months, cutoff_month, month_start, read_dataset

horizon_end = add_months(cutoff_month(), 4)     # needed up to 3 months after the cut-off
items = read_dataset("scheduled_with_category")
items = items[pd.to_datetime(items["due_date"]) < horizon_end]
items["due_month"] = month_start(items["due_date"])
items["known_month"] = month_start(items["known_date"])
monthly_obligations = (items.groupby(["client_no", "category", "direction", "due_month", "known_month"], as_index=False)
                       .agg(amount=("amount", "sum"), items=("origin_ref", "count")))
dataiku.Dataset("monthly_obligations").write_with_schema(monthly_obligations)
