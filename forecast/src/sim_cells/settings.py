import pandas as pd

N_CLIENTS = 40                     # sample business clients (small, for a quick sandbox run; 300 for realistic scores)
SEED = 7                           # same seed = same sample data
SIM_END = "2027-04"                # the sample data is generated up to this month, then published month by month
AS_OF = "2026-10-31"               # publish the warehouse as it was at the end of this day (move it on to simulate the next load)
EDW_CONNECTION = "EDW_LOCAL"       # a local TEST database the simulator may write to (never the bank's warehouse);
                                   # the setup notebook's EDW_CONNECTION must name the same one
EDW_SCHEMA = "EDW"                 # the schema the 2 tables are written into (it must exist)
