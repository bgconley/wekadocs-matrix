# 08 — Open Questions

## Questions Requiring Maintainer Confirmation

### Q1 — Security Architecture
**Question:** Is the MCP server intended to be publicly accessible without authentication or rate limiting?
**Context:** `src/mcp_server/security/auth.py` (JWT middleware) and `src/mcp_server/security/rate_limiter.py` exist but are marked DEAD and never wired into `main.py`.
**Impact:** If the server should be authenticated, this is a CRITICAL security gap. If it's intentionally unauthenticated (e.g., behind a network firewall), this should be documented.
**Answer method:** Ask security team / review deployment architecture.

### Q2 — Canonical Neo4j Schema
**Question:** Which Neo4j schema DDL file is the canonical source of truth?
**Context:** `scripts/neo4j/` contains 6+ schema files spanning v2.0 through v2.2. The runtime schema is managed by `src.shared.schema.create_schema()` which is config-driven, but the Cypher files serve as backups and migration scripts.
**Impact:** Developers may apply the wrong schema version during recovery or migration.
**Answer method:** Review git history of schema files. Check which file matches the live Neo4j schema dump (`neo4j_schema_dump.cypher`).

### Q3 — `src/neo/contract_checks.py` Usage
**Question:** Is `GraphContractChecker` from `contract_checks.py` actually used by the ingestion worker, or is the import vestigial?
**Context:** The module is marked `@status: DEAD` but is imported by `worker.py` at line 260 inside the processing loop.
**Impact:** If used, removing it breaks ingestion. If not used, the import is dead code that creates confusion.
**Answer method:** Add logging to the import site in worker.py and observe whether `GraphContractChecker` is actually instantiated or called.

### Q4 — Markdown Parser Selection
**Question:** Which markdown parser is the active path — `markdown.py` (legacy) or `markdown_it_parser.py` (active)?
**Context:** Both exist. `markdown_it_parser.py` is marked ACTIVE, `markdown.py` is marked DEPRECATED. The router in `parsers/__init__.py` determines which is used.
**Impact:** Different parsers produce different AST structures, which affects chunking, entity extraction, and structural edges.
**Answer method:** Check `parsers/__init__.py` routing logic. Add logging to both parsers and observe which is called during ingestion.

### Q5 — Production Config Differences
**Question:** Does production use different configuration than development?
**Context:** `config/production.yaml` is identical to `config/development.yaml`. The CI workflow deploys to staging and production using the same Kustomize base.
**Impact:** If production needs differ (different embedding profiles, cache sizes, feature flags), the current setup uses development config in production.
**Answer method:** Compare running configs in staging vs development. Check K8s ConfigMap values.

### Q6 — Cross-Document Linking Activation
**Question:** Is cross-document linking (`CrossDocLinker`) enabled in the current deployment?
**Context:** `_create_cross_doc_links()` is called at the end of `ingest_document_atomic()`, but it depends on config flags and embedding similarity computation.
**Impact:** If enabled, ingestion is slower (additional similarity computation). If disabled, RELATED_TO edges between documents are not created.
**Answer method:** Check `config/development.yaml` for cross-document linking config. Observe ingestion worker logs for cross-doc link creation.

### Q7 — Reranker Activation
**Question:** Which reranker is active — local mxbai-reranker (port 9006) or Jina AI API?
**Context:** `src/providers/rerank/` has three implementations: Jina, local service, and noop. The active one depends on config.
**Impact:** Different rerankers have different latency, quality, and cost characteristics.
**Answer method:** Check `config/development.yaml` for reranker config. Observe which provider is created by `ProviderFactory`.

### Q8 — Embedding Profile Matrix
**Question:** Which embedding profiles are actually deployed and which are development-only?
**Context:** `config/embedding_profiles.yaml` defines multiple profiles (bge_m3, snowflake_arctic, etc.). The `@status: ACTIVE` annotation on many provider files doesn't mean they're all deployed.
**Impact:** Some providers may require specific API keys, GPU resources, or network access that aren't available in all environments.
**Answer method:** Check which profiles are referenced in running configs. Verify which providers are created by `ProviderFactory` in each environment.

## Questions Answerable by Running the App/Tests

### Q9 — Parser Routing Verification
**Command:** `python -c "from src.ingestion.parsers import parse_markdown; import inspect; print(inspect.getsourcefile(parse_markdown))"`
**Question:** Which parser module does the router actually use?

### Q10 — Chunk Assembler Selection
**Command:** `python -c "from src.ingestion.chunk_assembler import get_chunk_assembler; from src.shared.config import get_config; config = get_config()[0]; print(get_chunk_assembler(config.ingestion.chunk_assembly))"`
**Question:** Which chunk assembler is selected by default config?

### Q11 — Provider Factory Resolution
**Command:** `python -c "from src.providers.factory import ProviderFactory; from src.shared.config import get_config; config = get_config()[0]; pf = ProviderFactory(config); print(pf.create_embedding_provider_for_role('dense'))"`
**Question:** Which embedding provider is created for the 'dense' role?

### Q12 — Dead Import Verification
**Command:** `python scripts/ci/check_dead_imports.py`
**Question:** Does the dead import checker report any violations?

### Q13 — Schema Application
**Command:** `python scripts/init_schema.py`
**Question:** Does schema initialization succeed without errors?

### Q14 — Ingestion Worker Startup
**Command:** `docker compose up -d ingestion-worker && docker compose logs -f ingestion-worker --tail=50`
**Question:** Does the worker start without import errors?

### Q15 — MCP Server Health
**Command:** `curl -s http://localhost:8000/health | python -m json.tool`
**Question:** Does the health check return 200 with all services connected?

### Q16 — Broken Script Verification
**Command:** `python scripts/apply_complete_schema_v2_1.py`
**Question:** Does this script fail with a file-not-found error?

## Questions Answerable by Production/Deployment Inspection

### Q17 — K8s Overlay Usage
**Question:** Are the empty `deploy/k8s/overlays/staging/` and `deploy/k8s/overlays/production/` directories intentional?
**Evidence:** CI workflow references `kubectl apply -k deploy/k8s/overlays/staging` but the directory is empty.
**Answer method:** Check if the staging deployment actually uses the base config or if there's a separate deployment mechanism.

### Q18 — New Relic Key Usage
**Question:** Is the hardcoded New Relic license key in `.env.production` a real key or a placeholder?
**Evidence:** `NEW_RELIC_LICENSE_KEY=c71e093ba36af151d238c1e443093261FFFFNRAL`
**Answer method:** Check New Relic account for this key. Verify if it's a test key or production key.

### Q19 — GPU Gateway Availability
**Question:** Is the unified GPU gateway at `10.25.0.50:8080` always available, or is it a temporary development resource?
**Evidence:** All embedding/reranker providers route through this gateway in `.env.production`.
**Answer method:** Check if this IP is in a production network range. Verify with infrastructure team.

### Q20 — Terraform State Location
**Question:** Where is the Terraform state for the LGTM VM stored?
**Evidence:** `infra/terraform/environments/dev/backend.tf` has commented-out GCS backend option.
**Answer method:** Check `terraform state list` in the infra directory. Ask infrastructure team.

### Q21 — Staging vs Production Deployment
**Question:** How does the staging deployment differ from production?
**Evidence:** CI has separate `deploy-staging` and `deploy-production` jobs, but K8s overlays are empty.
**Answer method:** Check actual K8s cluster configurations. Compare staging and production deployments.
