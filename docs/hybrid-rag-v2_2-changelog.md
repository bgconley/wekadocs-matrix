# Hybrid RAG v2.2 Change Log / ADR

> **Purpose**: Capture the concrete code changes, rationale, and affected files for the Hybrid RAG v2.2 rollout so future engineers can audit what landed without diff-mining the repo. This entry complements the canonical spec and architecture docs.

---

## 1. Chunk Assembly & Semantic Hooking
- **Files**: `src/ingestion/chunk_assembler.py`, `src/ingestion/semantic.py`, `src/shared/config.py`, `config/development.yaml`.
- **Changes (granular)**:
  - `StructuredChunker` replaces Greedy fallback logic: `_build_blocks`, `_balance_small_tails`, `_ensure_microdoc_neighbor`, and `_collapse_document` now honor `doc_is_microdoc`, emit stub chunks, and keep deterministic ordering.
  - All chunk emission points (`assemble`, split loop, balance pass, stub emission, doc collapse) now call the new `_enrich_chunk` hook before `post_process_chunk`, ensuring semantic metadata is consistently attached.
  - Added `SemanticEnrichmentConfig` plus YAML stanza so chunk assembly can toggle semantic providers; introduced `src/ingestion/semantic.py` (stub provider + factory).
  - Instrumented enrichment with Prometheus metrics (`semantic_enrichment_total`, `semantic_enrichment_latency_ms`) and tests covering enabled/disabled paths.
- **Rationale**: Restore adjacency guarantees, deprecate silent doc collapse, and leave a fully instrumented hook for upcoming semantic chunking/NER work.

## 2. Ingestion Graph / Qdrant Integration
- **Files**: `src/ingestion/build_graph.py`, `src/ingestion/reconcile.py`, `src/ingestion/incremental.py`, `src/ingestion/auto/orchestrator.py`, `src/shared/chunk_utils.py`, `src/shared/connections.py`.
- **Changes (granular)**:
  - GraphBuilder now sets both `document_id`/`doc_id`, `doc_tag`, `tenant`, `lang`, `version`, `doc_is_microdoc`, `is_microdoc_stub`, text/shingle hashes, semantic placeholders, and runs typed relationship builders with `IS NOT NULL` guards.
  - Embedding/upsert path switched to named vectors: `_upsert_to_qdrant` now builds `PointStruct(id=…, vector={…})`, with `auto/orchestrator.py` and `incremental.py` updated to validate dimensions when vectors come as dicts.
  - `src/shared/connections.py` gained `_extract_point_vectors` so reconciler/upsert validation handles both single vector and named-vector payloads.
  - Qdrant reconciliation/cleanup respects named vectors and semantic metadata, preventing drift when payloads change.
- **Rationale**: Align ingestion with the Neo4j v2.2 schema & Qdrant multi-vector architecture while keeping downstream filters and reconciler logic accurate.

## 3. Retrieval & Ranking Pipeline
- **Files**: `src/query/hybrid_retrieval.py`, `src/query/ranking.py`, `src/query/response_builder.py`, `src/mcp_server/query_service.py`, `src/query/hybrid_retrieval.py` tests.
- **Changes (granular)**:
  - `QdrantWeightedSearcher` now issues named-vector searches (`query_vector={"name":…, "vector":…}`), captures per-field scores, and populates `ChunkResult` fields (`doc_is_microdoc`, `is_microdoc_stub`, `title_vec_score`, `entity_vec_score`).
  - Fusion + continuity boost code records diagnostic snapshots, filters stubs before packing, updates metrics (`seed_count`, `microdoc_extras`, `expansion_count`), and preserves reranker hook placeholders.
  - Graph expansion ensures neighbors respect doc tags and microdoc status; response builder / MCP service propagate the new metadata to clients.
- **Rationale**: Deliver true multi-vector hybrid retrieval without losing the existing BM25+flow while preparing for reranking/semantic stitching phases.

## 4. Configuration & Observability
- **Files**: `src/shared/config.py`, `config/development.yaml`, `src/shared/observability/metrics.py`, `src/monitoring/health.py`.
- **Changes (granular)**:
  - Config tree now includes explicit chunk assembly knobs plus ingestion/search defaults aligned with the spec; YAML was updated with structured chunker defaults and semantic provider stub entries.
  - Prometheus module gained semantic-enrichment metrics and existing dashboards/logging were pointed at the new counters.
  - Health/monitoring wiring references schema/version flags so services fail fast if v2.2 markers are missing.
- **Rationale**: Make the v2.2 behavior reproducible via config, expose telemetry for the new hook, and keep observability aligned with the spec’s Section 8 requirements.

## 5. Tooling & Scripts
- **Files**: `scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher`, `scripts/qdrant and helpers/qdrant_setup_chunks_multi-1546.py`, `scripts/cleanup-databases.py`, `reports/cleanup/*.json`.
- **Changes (granular)**:
  - Neo4j schema guard rails include dual ID properties, typed relationships (NEXT/PREV/SAME_HEADING/etc.), and guard clauses compatible with environments that lack APOC.
  - Qdrant helper script mirrors production collection settings (named vectors, optimizers_config, payload indexes) so bootstrap matches ingestion assumptions.
  - Cleanup script enumerates Neo4j labels/Qdrant collections, emits JSON reports (dry run + live), and preserves schema metadata nodes/collections on wipe.
- **Rationale**: Provide deterministic bootstrap/reset steps so dev/test environments can be wiped safely while keeping schema + collection scaffolding intact.

## 6. Tests & Documentation
- **Files**: `tests/v2_2/…`, `docs/hybrid-rag-v2_2-*.md`, `docs/cdx-outputs/context-04.md`, `docs/cdx-outputs/context-0*.md`.
- **Changes (granular)**:
  - New chunking unit tests verify microdoc handling and semantic enrichment metrics; integration suites cover multi-vector ingestion, retrieval, cleanup, reconciliation, and ranking behaviors.
  - Testing doc now spells out how to run semantic guardrails; context files record the full engineering narrative for future sessions.
  - Architecture/spec/handoff/testing docs were updated to describe structured chunking, multi-vector retrieval, cleanup requirements, and outstanding TODOs.
- **Rationale**: Demonstrate v2.2 compliance without mocks, ensure regressions surface quickly, and give future engineers a canonical reference for the entire migration.

---

## Open Follow-Ups (tracked separately)
1. Replace the deterministic test embedder with the real Jina provider before release certification.
2. Populate `semantic_metadata` + `entity` vectors once the semantic chunking pipeline is ready.
3. Wire `_apply_reranker` to Jina reranker v3 and surface reranker metrics in Grafana.

This change log should be updated again if additional sections of the v2.2 plan land or if new ADR decisions supersede the work summarized here.
