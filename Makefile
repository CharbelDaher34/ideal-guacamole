IMAGE := dataiku/dss:14.2.0
PG_IMAGE := postgres:17
URL   := http://127.0.0.1:10000

.PHONY: start stop wipe rebuild reset

start: ## Start DSS + PostgreSQL, wait until ready, login admin/admin
	@mkdir -p dss_data pg_data
	@docker run --rm -u 0 -v "$(CURDIR)/dss_data:/d" --entrypoint chown $(IMAGE) 1000:1000 /d
	docker compose up -d
	@printf "Waiting for DSS"; \
	for i in $$(seq 1 120); do \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' $(URL)/dip/api/get-configuration); \
	  if [ "$$code" = 200 ]; then break; fi; \
	  printf "."; sleep 5; \
	done; \
	[ "$$code" = 200 ] || { echo " not ready after 10 min, see: docker compose logs"; exit 1; }
	@for i in $$(seq 1 24); do \
	  docker exec dataiku-dss /home/dataiku/dss/bin/dsscli user-edit admin --password admin >/dev/null 2>&1 && break; \
	  sleep 5; \
	done  # a fresh install answers before its backend is ready: retried
	@echo " ready: $(URL)  (login admin / admin)"

stop: ## Stop DSS + PostgreSQL (keeps data)
	docker compose stop

wipe: ## Delete everything of the sandbox: containers, network, data (dss_data, pg_data) and images. Keeps the licence
	@mkdir -p sandbox/keep
	@if docker ps -q -f name=^dataiku-dss$$ | grep -q .; then \
	  docker cp dataiku-dss:/home/dataiku/dss/config/license.json sandbox/keep/license.json && echo "licence saved in sandbox/keep"; fi
	docker compose down --remove-orphans
	docker run --rm -u 0 -v "$(CURDIR)/dss_data:/d" -v "$(CURDIR)/pg_data:/p" --entrypoint sh $(IMAGE) \
	  -c 'rm -rf /d/* /d/.[!.]* /p/* /p/.[!.]*'
	-docker rmi $(IMAGE) $(PG_IMAGE)
	@echo "sandbox wiped (the licence is kept in sandbox/keep/license.json)"

rebuild: start ## Fresh DSS from nothing: licence, test database, connection, code env, project, first run
	python3 sandbox/rebuild.py

reset: wipe rebuild ## wipe, then rebuild
