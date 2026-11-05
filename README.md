# Dataiku DSS — Free Edition (sandbox)

The cash-flow forecast project (CSV files, no database) is built by a notebook: see [`forecast/`](forecast/README.md).

Dataiku DSS 15.2.0 running in Docker (rootless) for testing development work.

- Image: `dataiku/dss:15.2.0`
- URL (server-local only): http://127.0.0.1:10000
- Data dir (projects, code envs, config, logs): `./dss_data` → `/home/dataiku/dss` in the container
- Code env `cashflow_forecast` (CatBoost, MLflow): created by the forecast notebook if missing
  (**Administration → Code Envs**).

## PostgreSQL

Not used by the cash-flow forecast (it stores CSV files); kept for other work.

A `postgres:17` container (`dataiku-postgres`) runs next to DSS, with data in `./pg_data`.
It is **not exposed to the server** — only reachable on the Docker network as `postgres:5432`.

- Credentials: `.env` (`POSTGRES_USER` / `POSTGRES_DB` / `POSTGRES_PASSWORD`, file is mode 600, not in git).
  On a fresh clone: `cp .env.example .env`, set a password, `chmod 600 .env`
- DSS connection: `postgres_local` (pre-created in `dss_data/config/connections.json`).
  After logging in, check it under **Administration → Connections → postgres_local → Test**.
- psql shell: `docker exec -it dataiku-postgres psql -U dataiku -d dataiku`

## Start / stop

```bash
make start   # start DSS + PostgreSQL, wait until ready
make stop    # stop both (keeps data)
```

Login: **admin / admin** — `make start` resets the admin password to `admin` every time.
Port 10000 is bound to 127.0.0.1 only; reach it with your own SSH tunnel.

## Registration

Registered (2026-10-28) as a Free Edition (Community) license for charbel.michel@eurisko.net —
instance id `lyx3aosk`, no expiry. Registration was done from the server because the browser
form's call to `https://my.dataiku.com` was blocked on the client side. Wiping `dss_data` means
registering again. **Administration → License** shows which features the license enables.

DSS backend log: `dss_data/run/backend.log` (read it with `docker exec dataiku-dss tail -f /home/dataiku/dss/run/backend.log`).

`dss_data` is owned by the container's `dataiku` user (rootless Docker maps it to a sub-UID),
so edit or delete its contents through `docker exec`, not directly from the host.

## What to expect from the Free Edition

Works: Python/R/SQL/Shell recipes, Jupyter notebooks, code environments, project libraries,
Git integration, plugin development, Dash/Bokeh/Shiny webapps, visual ML,
files/SFTP/cloud storage + PostgreSQL/MySQL connections.

Not available: public REST API / API keys (so no VS Code extension or external `dataikuapi`),
scheduled scenarios & automation, API node / deployer, Spark / Kubernetes (and therefore Code Studios),
Snowflake/BigQuery/Oracle/etc. connectors, partitioning. Use the 14-day Dataiku Cloud trial to test those.

## Reset from scratch

```bash
docker compose down
docker run --rm -u 0 -v "$PWD/dss_data:/d" -v "$PWD/pg_data:/p" --entrypoint sh dataiku/dss:15.2.0 -c 'rm -rf /d/* /d/.[!.]* /p/* /p/.[!.]*'
docker compose up -d
```

This also wipes the `postgres_local` connection; re-add it in the DSS UI afterwards
(host `postgres`, port `5432`, credentials from `.env`).
