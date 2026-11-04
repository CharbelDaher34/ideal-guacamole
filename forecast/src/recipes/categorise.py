# Monthly flows: money in and out per client, month and category. The warehouse gives them by origin type and
# transaction type tag (own-account transfers and reversals already left out); the editable category_rules turn
# those into categories.
import dataiku

from cashflow import apply_category_rules, month_start, read_dataset

flows = read_dataset("monthly_flows_by_origin")
flows["month"] = month_start(flows["month"])
flows["category"] = apply_category_rules(flows, read_dataset("category_rules"))
monthly = (flows.groupby(["client_no", "month", "category", "direction"], as_index=False)
           [["amount", "known_1", "known_2", "known_3", "postings"]].sum())
dataiku.Dataset("monthly_flows").write_with_schema(monthly.round(2))
