N_CLIENTS = 420_000          # fake business clients
MIN_MONTHS, MAX_MONTHS = 6, 36   # each client's history: a random number of 30-day periods up to the cut-off day (some
                                 # under 12, so a few get NOT_ENOUGH_HISTORY)
ROWS_PER_MONTH = 16          # daily_flows rows per client and 30-day period (days with postings x categories)
CLIENTS_PER_CHUNK = 5000     # sets the project variable inference_clients_per_chunk for the test
