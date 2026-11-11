# Load test, zone Data prep (not timed): the fake daily flows the tests read, in load_test_flows, one subfolder per
# size of fake data (clients, months, postings_per_month, clients_per_group) used by the tests in step 1 of the
# scenario Run the load test, each laid out like the forecast project's daily_flows (one Parquet file per group of
# clients and month). A size already there is kept (whatever day it was made); a size no longer listed is removed.
import pandas as pd

from loadtest import prepare_data, step1_tests

cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)          # yesterday, for new data
for prefix, info in prepare_data(step1_tests(), cutoff).items():
    print(f"PROGRESS fake data ready, {prefix[1:]}: up to {info['cutoff']}, {info['daily_rows']:,} daily rows, "
          f"{info['data_mb']} MB", flush=True)
