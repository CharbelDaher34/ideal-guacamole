# Plan: scale the pipeline to ~420k clients

Status: **built and tested locally (2026-11-07).** Details are in [README.md](README.md); this file keeps what was
decided and what is left.

## What's decided (2026-11-07)

- The bank has Dataiku Enterprise. Any read-only SQL query on the warehouse is allowed (nothing is ever written
  there).
- Keep it simple: no buckets and no partitions for now. The flow looks the same as today.
- The bank's cores and RAM are unknown until a benchmark there; the load test notebook is that benchmark.

## What was built

1. **Staging reads only the new days.**
   - The first run reads the whole history. After that, each run reads only the postings after the last day already
     in `daily_flows` (`WHERE VAL_DT >= …`, up to the end of the cut-off month) and adds them to `daily_flows`.
   - `staging_log` gets one row per run (`FULL`, `NEW_DAYS`, `NOTHING_NEW`).
   - Tested: the nightly reads give exactly the same `daily_flows` as a full read, through CSV and through a
     PostgreSQL connection (where the warehouse returned only the new month's 1,924 postings out of 58,113).

2. **"Full rebuild of the daily flows"**, a scenario run by hand: empties `daily_flows` and `staging_log`, then reads
   the whole history again.

3. **Inference in chunks** of `inference_clients_per_chunk` clients (project variable, 5,000). The forecast, its
   compressed archive and `inference_log` (now with seconds per step and peak memory) are written chunk by chunk.
   Tested: chunks of 7 clients give exactly the same forecast as one chunk.

4. **`inference_load_test.ipynb`** (sandbox only): fake data for 420k clients, times the forecast, puts the real data
   back. Result on the local Docker Dataiku (48 cores, but the recipe uses one):

   | Clients | Daily rows read | Forecast rows | Chunks | Reading | Forecasting | Total | Peak memory |
   |---|---|---|---|---|---|---|---|
   | 20,000 | 6.5M | 0.66M | 4 | 17 s | 19 s | 36 s | 807 MB |
   | 420,000 | 137M | 13.9M | 84 | 5.2 min | 6.3 min | 11.5 min | 961 MB |

   Memory barely moves with 21 times more clients; the time grows in line with the data.

5. **No Parquet:** Dataiku can't write Parquet on a plain file connection (only with Hadoop/Spark storage), so
   `daily_flows` and `final_forecast` are compressed CSV instead.

## Later, only if needed

- **Partitions, or running chunks in parallel:** if inference is still too slow after the bank's benchmark.
- **Training on a sample of clients:** when training on all clients gets too heavy (the data scientist's side).
