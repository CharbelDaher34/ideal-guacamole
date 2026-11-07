# The real data back, as the scenario Full rebuild of the daily flows does: empty daily_flows and staging_log, read
# the whole history from the warehouse again (clients and daily_flows), then forecast the real clients.
from warehouse import clear_daily_flows

clear_daily_flows()
folder_id = next(f["id"] for f in project.list_managed_folders() if f["name"] == "daily_flows")
project.new_job("RECURSIVE_FORCED_BUILD").with_output(folder_id, object_type="MANAGED_FOLDER").start_and_wait()
project.new_job("NON_RECURSIVE_FORCED_BUILD").with_output("inference_log").start_and_wait()
print("Real data and forecast back:", len(dataiku.Dataset("clients").get_dataframe()), "clients")
