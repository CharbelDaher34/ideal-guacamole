# Load test, zone Data prep (scenario "Create the fake data"; not timed): the fake daily flows the load test reads,
# in the folder load_test_flows, of the size step 1 of the scenario sets: fake_clients, fake_months (months of
# history up to yesterday), fake_postings_per_month. Parquet files exactly like the forecast project's daily_flows (one row
# per client, day and category: money_in, money_out, postings; each client's rows together), and data.json (what was
# made, up to which day, how long it took). Data of the same size already there is kept; a new size replaces it.
import dataiku
import pandas as pd

from loadtest import CREATE, SIZE, data_size, fake_data, scenario_numbers

size = data_size(scenario_numbers(CREATE, list(SIZE.values())))
cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)          # yesterday
info = fake_data(size, cutoff)
print(f"PROGRESS fake data ready: {info['clients']:,} clients, {info['months']} months, {info['postings_per_month']} "
      f"postings a month, up to {info['cutoff']}: {info['daily_rows']:,} daily rows, {info['data_mb']} MB", flush=True)
