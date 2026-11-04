import pandas as pd

N_CLIENTS = 300                    # sample business clients
SEED = 7                           # same seed = same sample data
SIM_END = "2027-04"                # the sample data is generated up to this month, then published month by month
AS_OF = "2026-10-31"               # publish the warehouse as it was at the end of this day (move it on to simulate the next load)
BASE_CONNECTION = "postgres_local"    # an existing PostgreSQL connection: the two warehouse connections copy its server
LOADER_CONNECTION = "EDW_AI_LOADER"   # the warehouse team's connection (writes the tables)
READ_CONNECTION = "EDW_AI"            # the connection the forecast project reads with (read-only)
SCHEMA = "edw_ai"                     # schema of the warehouse tables
