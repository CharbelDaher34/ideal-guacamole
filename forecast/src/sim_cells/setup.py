import dataiku

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
# The simulated warehouse tables are datasets of this project (CSV files): the forecast setup uses them as its Raw
# data. In the bank, the real tables come from the warehouse team's landing folder instead.
