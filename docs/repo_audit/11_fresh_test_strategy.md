# Fresh Test Strategy

## Principle

Existing tests are untrusted by instruction. Do not repair architecture to satisfy stale tests. Inventory them, quarantine stale ones, then write new tests from the active Nutanix contract.

## Test Tiers

1. Config resolution tests
   - Given env + YAML + domain profile, assert the effective provider/model/chunker/tool/domain config.
   - Fail if WEKA instructions appear while `DOMAIN=nutanix`.

2. Ingestion stage contract tests
   - Parser returns normalized document/sections.
   - Chunker enforces token bounds without silent truncation.
   - Entity/reference enrichment produces domain-specific metadata.
   - Embedding stage produces required vector fields or explicit degradation metadata.
   - Saga write behavior rolls back/compensates correctly with faked Neo4j/Qdrant clients.

3. Retrieval unit tests
   - Query plan selects expected channels for procedural/troubleshooting/reference queries.
   - Qdrant Query API fallback reports degradation.
   - Reranker/ColBERT missing-vector paths do not silently masquerade as full-quality results.

4. MCP contract tests
   - Tool profile lists only intended tools.
   - Tool calls cannot bypass profile restrictions if that is the intended production policy.
   - Evidence responses include citations, scores, limitations, and fallback flags.

5. Nutanix golden evals
   - A small curated corpus with expected answer-support passages.
   - Queries across installation, cluster admin, networking/storage concepts, troubleshooting, APIs/CLI, and version-specific behavior.
   - Assertions on recall@k, citation precision, and answer abstention.

6. Integration smoke
   - Local non-production compose with disposable Neo4j/Qdrant/Redis volumes.
   - Ingest a tiny Nutanix fixture.
   - Query through MCP/FastAPI and verify exact source excerpts.

## What To Quarantine

- Tests importing `HybridSearchEngine` or legacy `ranking.py` unless explicitly testing compatibility.
- Tests with WEKA-only expectations once Nutanix is the target domain.
- Tests that pass by mocking away provider/config resolution.

## First Three Tests To Write

1. `test_domain_profile_controls_runtime_instructions`: Nutanix domain produces no WEKA instruction strings in MCP/reranker/embedding prompts.
2. `test_effective_config_report_matches_env`: env overrides produce the same resolved model IDs the service will use.
3. `test_query_trace_marks_degraded_rerank_or_colbert`: missing reranker/ColBERT vectors are visible in trace metadata.
