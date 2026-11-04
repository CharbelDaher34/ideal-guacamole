import copy

import dataiku
from dataiku import SQLExecutor2

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())

# Sandbox only: in the bank, the schema and the EDW_AI connection already exist. Here both connections point at
# the same PostgreSQL server and credentials as BASE_CONNECTION, schema edw_ai.
SQLExecutor2(connection=BASE_CONNECTION).query_to_df(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}", post_queries=["COMMIT"])
base = client.get_connection(BASE_CONNECTION).get_settings().get_raw()
for name, writable in ((LOADER_CONNECTION, True), (READ_CONNECTION, False)):
    if name in client.list_connections():
        print(f"Connection {name} already exists")
        continue
    params = copy.deepcopy(base["params"])
    params["namingRule"].update(schemaName=SCHEMA, tableNameDatasetNamePrefix="")
    params["defaultSchema"] = SCHEMA
    connection = client.create_connection(name, "PostgreSQL", params=params)
    settings = connection.get_settings()
    settings.get_raw().update(allowWrite=writable, allowManagedDatasets=writable, allowManagedFolders=False)
    settings.save()
    print(f"Connection {name} created ({'read and write' if writable else 'read-only'}, schema {SCHEMA})")
