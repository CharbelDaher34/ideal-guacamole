IMAGE := dataiku/dss:15.2.0
URL   := http://127.0.0.1:10000

.PHONY: start stop dashboard

start: ## Start DSS + PostgreSQL, wait until ready, login admin/admin
	@mkdir -p dss_data pg_data
	@docker run --rm -u 0 -v "$(CURDIR)/dss_data:/d" --entrypoint chown $(IMAGE) 1000:1000 /d
	docker compose up -d
	@printf "Waiting for DSS"; \
	for i in $$(seq 1 60); do \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' $(URL)/dip/api/get-configuration); \
	  if [ "$$code" = 200 ]; then break; fi; \
	  printf "."; sleep 5; \
	done; \
	[ "$$code" = 200 ] || { echo " not ready after 5 min, see: docker compose logs"; exit 1; }
	@docker exec dataiku-dss /home/dataiku/dss/bin/dsscli user-edit admin --password admin >/dev/null 2>&1
	@echo " ready: $(URL)  (login admin / admin)"

stop: ## Stop DSS + PostgreSQL (keeps data)
	docker compose stop

dashboard: ## Start the external cash-flow dashboard on http://127.0.0.1:8060 (needs .dss_api_key)
	cd outlook && uvicorn app:app --host 127.0.0.1 --port 8060
