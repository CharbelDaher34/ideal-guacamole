import pandas as pd

N_CLIENTS = 40                     # sample business clients (small, for a quick sandbox run; 300 for realistic scores)
SEED = 7                           # same seed = same sample data
SIM_END = "2027-04"                # the sample data is generated up to this month, then published month by month
AS_OF = "2026-10-31"               # publish the warehouse as it was at the end of this day (move it on to simulate the next load)
CONNECTION = "filesystem_managed"  # where the simulated warehouse tables are written (CSV files)
WRITE_DETAILED_TABLES = False      # also write the 17 detailed tables (the forecast only reads the 4 curated ones)
