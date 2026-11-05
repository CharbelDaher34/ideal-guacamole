import dataiku

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
# The simulated warehouse tables are datasets of this project (CSV files): the forecast setup uses them as its Raw
# data. In the bank, the project reads the real tables from the warehouse instead (read-only).
