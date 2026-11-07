# Fake clients (copies of the real ones' rows with new client numbers, so the columns are the same) and fake daily
# flows (random amounts), written straight to clients and daily_flows, in batches of clients. staging_log is emptied:
# the next staging run then stops and asks for the scenario Full rebuild, so fake and real data never mix.
started = time.time()
rng = np.random.default_rng(7)
categories = np.array(sorted(set(CATEGORY_OF_TYPE.values()) | {OTHER}))
base = read_dataset("clients")
fake = base.sample(N_CLIENTS, replace=True, random_state=7).reset_index(drop=True)
fake["client_no"] = [f"LT{i:07d}" for i in range(N_CLIENTS)]
dataiku.Dataset("clients").write_with_schema(fake)
for name in ("daily_flows", "staging_log"):
    project.get_dataset(name).clear()

daily_flows = dataiku.Dataset("daily_flows")
last = np.datetime64(f"{cutoff:%Y-%m}", "M")
rows, writer = 0, None
for start in range(0, N_CLIENTS, 10_000):
    numbers = fake["client_no"].to_numpy()[start:start + 10_000]
    n = len(numbers)
    per_client = rng.integers(MIN_MONTHS, MAX_MONTHS + 1, n) * ROWS_PER_MONTH
    who = np.repeat(np.arange(n), per_client)
    position = np.arange(len(who)) - np.repeat(np.cumsum(per_client) - per_client, per_client)
    month = (last - position // ROWS_PER_MONTH).astype("datetime64[D]")
    size = rng.lognormal(9, 1.2, n)[who]                       # each client's typical amount (SAR)
    batch = pd.DataFrame({
        "client_no": numbers[who],
        "day": (month + rng.integers(0, 28, len(who))).astype("datetime64[ns]"),
        "category": rng.choice(categories, len(who)),
        "money_in": np.where(rng.random(len(who)) < 0.6, size * rng.lognormal(0, 0.6, len(who)), 0.0),
        "money_out": np.where(rng.random(len(who)) < 0.6, size * rng.lognormal(0, 0.6, len(who)), 0.0),
        "postings": rng.integers(1, 6, len(who))})
    batch = (batch.groupby(["client_no", "day", "category"], as_index=False).sum()
             .round({"money_in": 2, "money_out": 2}))
    if writer is None:
        daily_flows.write_schema_from_dataframe(batch)
        writer = daily_flows.get_writer()
    writer.write_dataframe(batch)
    rows += len(batch)
writer.close()
print(f"{N_CLIENTS:,} fake clients, {rows:,} daily_flows rows written in {(time.time() - started) / 60:.1f} minutes")
