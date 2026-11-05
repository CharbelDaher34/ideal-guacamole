# Staging: the warehouse tables with the standard column names and values, and the daily postings summed per month.
#   clients, scheduled_payments: CFF_CLIENT and CFF_SCHEDULED_PYMT renamed
#   monthly_flows_by_origin: FCT_STATEMENT_ENTRY summed per client, month, direction, T24 module and transaction type
#   (business clients, the last 60 months up to the cut-off, no reversals, no transfers between own accounts). On a
#   database connection the sum runs in the warehouse (one Oracle query, read-only): only the totals come back.
import dataiku

from cashflow import cutoff_month
from warehouse import TABLES, is_sql, monthly_flows, monthly_flows_sql, to_standard

clients = to_standard(dataiku.Dataset(TABLES["clients"][0]).get_dataframe(), "clients")
dataiku.Dataset("clients").write_with_schema(clients)
scheduled = to_standard(dataiku.Dataset(TABLES["scheduled_payments"][0]).get_dataframe(), "scheduled_payments")
dataiku.Dataset("scheduled_payments").write_with_schema(scheduled)

entries = dataiku.Dataset(TABLES["transactions"][0])
if is_sql(entries):
    table = lambda d: d.get_location_info()["info"]["quotedResolvedTableName"]
    query = monthly_flows_sql(table(entries), table(dataiku.Dataset(TABLES["clients"][0])), cutoff_month())
    flows = to_standard(dataiku.SQLExecutor2(dataset=entries).query_to_df(query), "monthly_flows_by_origin")
else:
    flows = monthly_flows(to_standard(entries.get_dataframe(), "transactions"), clients["client_no"], cutoff_month())
dataiku.Dataset("monthly_flows_by_origin").write_with_schema(flows)
print(f"{len(clients)} clients, {len(flows):,} monthly totals of postings, {len(scheduled):,} scheduled payments")
