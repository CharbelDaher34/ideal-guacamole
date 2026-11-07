# Fake clients (copies of the real ones' rows with new client numbers, so the columns are the same) and fake daily
# flows (random amounts), written straight to clients and daily_flows (Parquet files, as staging writes them), a
# batch of clients per file. staging_log is emptied: the next staging run then stops and asks for the scenario Full
# rebuild, so fake and real data never mix.
started = time.time()
rng = np.random.default_rng(7)
categories = np.array(sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER}))
base = read_dataset("clients")
fake = base.sample(N_CLIENTS, replace=True, random_state=7).reset_index(drop=True)
fake["client_no"] = [f"LT{i:07d}" for i in range(N_CLIENTS)]
dataiku.Dataset("clients").write_with_schema(fake)
project.get_dataset("staging_log").clear()
last = np.datetime64(f"{cutoff:%Y-%m-%d}", "D")


def batches():
    """The fake daily flows, 3,000 clients at a time, each client's rows together."""
    for start in range(0, N_CLIENTS, 3000):
        numbers = fake["client_no"].to_numpy()[start:start + 3000]
        n = len(numbers)
        per_client = rng.integers(MIN_MONTHS, MAX_MONTHS + 1, n) * ROWS_PER_MONTH
        who = np.repeat(np.arange(n), per_client)
        position = np.arange(len(who)) - np.repeat(np.cumsum(per_client) - per_client, per_client)
        back = (position // ROWS_PER_MONTH) * PERIOD_DAYS                # 30-day periods back from the cut-off
        size = rng.lognormal(9, 1.2, n)[who]                       # each client's typical amount (SAR)
        batch = pd.DataFrame({
            "client_no": numbers[who],
            "day": (last - back - rng.integers(0, PERIOD_DAYS, len(who))).astype("datetime64[ns]"),
            "category": rng.choice(categories, len(who)),
            "money_in": np.where(rng.random(len(who)) < 0.6, size * rng.lognormal(0, 0.6, len(who)), 0.0),
            "money_out": np.where(rng.random(len(who)) < 0.6, size * rng.lognormal(0, 0.6, len(who)), 0.0),
            "postings": rng.integers(1, 6, len(who))})
        yield (batch.groupby(["client_no", "day", "category"], as_index=False).sum()
               .round({"money_in": 2, "money_out": 2}))


rows = write_daily(batches())
print(f"{N_CLIENTS:,} fake clients, {rows:,} daily_flows rows written in {(time.time() - started) / 60:.1f} minutes")
