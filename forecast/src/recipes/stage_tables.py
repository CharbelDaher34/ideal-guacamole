# Staging: the 3 curated warehouse tables (CSV files in the landing folder) with the standard column names.
import dataiku

from warehouse import TABLES, to_standard

for table, (warehouse_table, _) in TABLES.items():
    raw = dataiku.Dataset(warehouse_table).get_dataframe()
    dataiku.Dataset(table).write_with_schema(to_standard(raw, table))
