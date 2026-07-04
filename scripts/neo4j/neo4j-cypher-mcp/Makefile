COMPOSE := docker-compose.yml
ENV_FILE ?= ../../../.env
NETWORK := nutanix-net

.PHONY: nutanix-net
nutanix-net:
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create $(NETWORK)

.PHONY: up
up: nutanix-net
	@echo "Starting Neo4j Cypher MCP server..."
	docker compose -f $(COMPOSE) --env-file $(ENV_FILE) up -d

.PHONY: down
down:
	@echo "Stopping Neo4j Cypher MCP server..."
	docker compose -f $(COMPOSE) --env-file $(ENV_FILE) down -v

.PHONY: logs
logs:
	docker compose -f $(COMPOSE) --env-file $(ENV_FILE) logs -f

.PHONY: ps
ps:
	docker compose -f $(COMPOSE) --env-file $(ENV_FILE) ps

.PHONY: pull
pull:
	docker compose -f $(COMPOSE) --env-file $(ENV_FILE) pull
