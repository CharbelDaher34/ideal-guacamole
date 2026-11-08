import dataiku

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
# The simulated warehouse tables are written into a local test database (EDW_CONNECTION): the forecast setup then
# reads them through SQL, exactly as it reads the bank's warehouse (read-only).
