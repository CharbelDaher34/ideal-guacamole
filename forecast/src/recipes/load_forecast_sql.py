# The Nightly batch's last step: the current forecast (folder forecast, written by nightly_forecast) loaded into the
# SQL table of the setup's FORECAST_SCHEMA and FORECAST_TABLE on FORECAST_CONNECTION (the project variables
# forecast_sql_*; in the Flow, the dataset final_forecast), replacing the previous one: one row per active client.
# Oracle: written with oracledb; another database: through the dataset final_forecast. If it fails, the forecast
# itself is complete in the folder: run this step again alone (build final_forecast).
import dataiku

from forecast import load_to_table, table_name

variables = dataiku.get_custom_variables()
connection = str(variables.get("forecast_sql_connection") or "").strip()
schema, name = (str(variables.get(f"forecast_sql_{k}") or "").strip() for k in ("schema", "table"))
if not connection or not name:
    raise ValueError("The project variables forecast_sql_connection and forecast_sql_table are not set: run the setup "
                     "notebook again (it sets them from FORECAST_CONNECTION, FORECAST_SCHEMA and FORECAST_TABLE)")
table = table_name(schema, name)
rows, version = load_to_table(connection, table, dataset="final_forecast")
print(f"{table} on {connection} (final_forecast): {rows:,} rows loaded from {version}")
