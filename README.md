# Digital Cash Management Forecast

- **How it works, end to end:** [`docs/how_it_works.md`](docs/how_it_works.md)
- **How to run it in the bank's Dataiku:** [`forecast/README.md`](forecast/README.md)
- **The 2 warehouse tables (columns, what we know):** [`docs/warehouse_tables.md`](docs/warehouse_tables.md)

The rest of this page is about the local test Dataiku (Docker sandbox), not the bank.

## Local sandbox: Dataiku DSS — Free Edition

Dataiku DSS 14.2.0 running in Docker (rootless) for testing development work: the closest image to the bank's
14.2.2 (there is no 14.2.2 image). The earlier 15.2.0 data folder is kept in `dss_data_15.2_backup/` (not in git).

- Image: `dataiku/dss:14.2.0` (Python 3.9 only)
- URL (server-local only): http://127.0.0.1:10000
- Data dir (projects, code envs, config, logs): `./dss_data` → `/home/dataiku/dss` in the container
- Code env `cashflow_forecast` (scikit-learn, MLflow): created once by an admin; the forecast setup notebook
  checks it (**Administration → Code Envs**).

### PostgreSQL

On the test machine it plays the bank's warehouse: `warehouse_simulator.ipynb` writes the fake `DIM_CUSTOMER` and
`FCT_STATEMENT_ENTRY` into its schema `EDW`, and the forecast reads them through the DSS connection `EDW_LOCAL` with
the same SQL as at the bank (set `EDW_CONNECTION = "EDW_LOCAL"` in the setup notebook). Oracle, the bank's database,
can't be used here: the Free Edition licence forbids Oracle datasets.

A `postgres:17` container (`dataiku-postgres`) runs next to DSS, with data in `./pg_data`.
It is **not exposed to the server** — only reachable on the Docker network as `postgres:5432`.

- Credentials: `.env` (`POSTGRES_USER` / `POSTGRES_DB` / `POSTGRES_PASSWORD`, file is mode 600, not in git).
  On a fresh clone: `cp .env.example .env`, set a password, `chmod 600 .env`
- DSS connection: `EDW_LOCAL` (created by `make rebuild`).
- psql shell: `docker exec -it dataiku-postgres psql -U dataiku -d dataiku`

### Start / stop

```bash
make start   # start DSS + PostgreSQL, wait until ready
make stop    # stop both (keeps data)
```

Login: **admin / admin** — `make start` resets the admin password to `admin` every time.
Port 10000 is bound to 127.0.0.1 only; reach it with your own SSH tunnel.

### Registration

Registered (2026-10-28) as a Free Edition (Community) license for charbel.michel@eurisko.net —
instance id `lyx3aosk`, no expiry. Registration was done from the server because the browser
form's call to `https://my.dataiku.com` was blocked on the client side. Wiping `dss_data` means
registering again. **Administration → License** shows which features the license enables.

DSS backend log: `dss_data/run/backend.log` (read it with `docker exec dataiku-dss tail -f /home/dataiku/dss/run/backend.log`).

`dss_data` is owned by the container's `dataiku` user (rootless Docker maps it to a sub-UID),
so edit or delete its contents through `docker exec`, not directly from the host.

### What to expect from the Free Edition

Works: Python/R/SQL/Shell recipes, Jupyter notebooks, code environments, project libraries,
Git integration, plugin development, Dash/Bokeh/Shiny webapps, visual ML,
files/SFTP/cloud storage + PostgreSQL/MySQL connections.

Not available: public REST API / API keys (so no VS Code extension or external `dataikuapi`),
scheduled scenarios & automation, API node / deployer, Spark / Kubernetes (and therefore Code Studios),
Snowflake/BigQuery/Oracle/etc. connectors, partitioning. Use the 14-day Dataiku Cloud trial to test those.

### Reset from scratch

Everything of the sandbox is rebuilt from nothing by scripts in `sandbox/` (run from this server, not from
Dataiku; nothing of it is in the forecast notebooks):

```bash
make reset     # = make wipe, then make rebuild
make wipe      # delete the 2 containers, their network, dss_data, pg_data and the 2 images; the Free Edition
               # licence is saved first in sandbox/keep/license.json (not in git)
make rebuild   # pull the images, start, put the licence back, create the schema EDW in PostgreSQL, the connection
               # EDW_LOCAL, the code env cashflow_forecast (scikit-learn, mlflow 2.22.0, catboost) with its
               # Jupyter kernel, the project DCM_FORECAST, then run warehouse_simulator.ipynb and
               # build_forecast_project.ipynb (local settings) and import forecast_lab.ipynb and daily_flows_viewer.ipynb
```

`make rebuild` takes about 15 minutes the first time (the DSS image is 12.7 GB, the code env a few minutes).
Without `sandbox/keep/license.json`, register the Free Edition in the browser first, then `python3 sandbox/rebuild.py`.
`sandbox/dss.py` also runs a notebook or a scenario from the server:
`python3 sandbox/dss.py run-scenario DCM_FORECAST Load_test`.
