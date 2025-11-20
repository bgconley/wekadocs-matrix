PHASE?=1
up:
	docker compose up -d
down:
	docker compose down -v
test-phase-%: up
	bash scripts/test/run_phase.sh $* || (echo "Phase $* failed" && exit 1)

# Phase 6 specific targets
test-phase-6: up
	bash scripts/test/run_phase.sh 6 || (echo "Phase 6 failed" && exit 1)

# Neo4j Cypher MCP server controls
NEO4J_CYPHER_MCP_COMPOSE := scripts/neo4j/neo4j-cypher-mcp/docker-compose.yml

.PHONY: weka-net
weka-net:
	@docker network inspect weka-net >/dev/null 2>&1 || docker network create weka-net

.PHONY: neo4j-cypher-mcp-up
neo4j-cypher-mcp-up: weka-net
	@echo "Starting Neo4j Cypher MCP server..."
	docker compose -f $(NEO4J_CYPHER_MCP_COMPOSE) --env-file .env up -d

.PHONY: neo4j-cypher-mcp-down
neo4j-cypher-mcp-down:
	@echo "Stopping Neo4j Cypher MCP server..."
	docker compose -f $(NEO4J_CYPHER_MCP_COMPOSE) down -v

.PHONY: neo4j-cypher-mcp-logs
neo4j-cypher-mcp-logs:
	docker compose -f $(NEO4J_CYPHER_MCP_COMPOSE) logs -f

.PHONY: neo4j-cypher-mcp-ps
neo4j-cypher-mcp-ps:
	docker compose -f $(NEO4J_CYPHER_MCP_COMPOSE) ps
