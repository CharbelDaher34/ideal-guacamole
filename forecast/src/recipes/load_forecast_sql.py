# The Nightly batch's last step: the current forecast (folder forecast, written by nightly_forecast) loaded into the
# SQL table final_forecast, replacing the previous one: one row per active client. If it fails, the forecast itself is
# complete in the folder: run this step again alone (build final_forecast).
from forecast import load_to_sql

rows, version = load_to_sql("final_forecast")
print(f"final_forecast: {rows:,} rows loaded from {version}")
