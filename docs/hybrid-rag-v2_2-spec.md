# Hybrid RAG v2.2 Integration Plan (Canonical Spec)

This document captures the full working spec for upgrading WekaDocs Matrix to the Neo4j v2.2 schema and Qdrant multi-vector architecture. All implementation tasks, testing, and documentation deliverables trace back to this plan.

---

## 1. Objectives & Scope
- Adopt Neo4j schema v2.2 (dual doc IDs, additive indexes, typed chunk relationships) and ensure services validate the schema version on startup.
- Move hybrid retrieval to Qdrant `chunks_multi` with named vectors while preserving existing Phase‑7E BM25 + vector fusion behavior.
- Keep Jina v3 tokenizer/token limits, context assembly, and MCP response flows unchanged.
- Lay groundwork for semantic chunking/NER metadata and Jina reranker v3 without major refactors later.
- Provide production-grade code, tests, observability, and documentation suitable for the current empty dev database.

---

## 2. High-Level Architecture
(See `docs/hybrid-rag-v2_2-architecture.md` for diagrams and component breakdowns.)

Key flows:
1. **Ingestion**: Parsing → Chunk assembly → Neo4j writes → Embedding + Qdrant multi-vector upsert → Cache invalidation/reconciliation.
2. **Retrieval**: BM25 + multi-vector ANN → Fusion → (future reranker) → Graph expansion + microdocs → Context assembly → MCP output.

---

## 3. Neo4j Schema Integration
- Apply `scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher` during environment bootstrap.
- Update config (`schema.version = "v2.2"`) and add startup checks in ingestion/retrieval services to verify schema markers (e.g., `SchemaVersion` node).
- Extend `GraphBuilder` to run Part‑8 relationship builders (SAME_HEADING, CHILD_OF, etc.) after document upsert, using vanilla Cypher with `exists(c.text)` guards (no APOC).
- Ensure ingestion sets both `document_id` and `doc_id` for Sections/Chunks, and populates newly indexed properties (doc_tag, lang, version, text_hash, shingle_hash, tenant, etc.).

**Action**: Update `scripts/cleanup-databases.py` at the start of this phase so dev/test environments can be reset while preserving schema metadata.

---

## 4. Qdrant Multi-Vector Integration
- Replace `_ensure_qdrant_collection` logic with the configuration used in `scripts/qdrant and helpers/qdrant_setup_chunks_multi-1546.py` (named vectors + tuned HNSW/optimizer settings).
- In `_process_embeddings`, compute multiple vectors per chunk (content/title, optional entity) using the existing Jina provider.
- `_upsert_to_qdrant` writes `PointStruct` objects with `vectors={"content":…, "title":…, "entity":…}` and the canonical payload used by retrieval. Reserve the `entity` slot even if it mirrors content for now.
- Update `src/ingestion/reconcile.py` to operate against `chunks_multi` (scroll by `payload["node_id"]`, detect drift, delete/reinsert).

---

## 5. Ingestion Pipeline Enhancements
- Extend metadata helpers in `src/shared/chunk_utils.py` to produce the full schema-required field set and maintain both doc IDs.
- Introduce a configurable `StructuredChunker` that anchors on H1/H2 blocks, enforces deterministic ordering, and never collapses documents. When total tokens fall below the microdoc threshold the chunker sets `doc_is_microdoc` (and, when necessary, emits a lightweight adjacency stub) so retrieval can stitch context without rewriting chunks.
- `post_process_chunk(chunk)` remains the hook for semantic enrichment so future NER/semantic chunking passes can extend metadata without touching structural logic.
- Add `_extract_entities` stub in `GraphBuilder` that currently returns empty results but feeds into Neo4j properties/Qdrant payload (`entities`, `topics`, etc.) when implemented.
- Preserve tokenizer guardrails (Jina v3) and existing chunk-assembly behavior.

---

## 6. Retrieval Pipeline Updates
- Replace `VectorRetriever.search()` with `QdrantWeightedSearcher` to query named vectors (`content`, `title`, `entity`). Convert returned `Candidate`s into `ChunkResult`s before fusion so downstream logic stays intact.
- Keep existing BM25 fusion and scoring; extend config to expose per-field ANN weights, ef, and top-k.
- Maintain `_bounded_expansion` (NEXT_CHUNK). Add optional typed-relationship expansion using `expand_graph_neighbors` + `apply_graph_scores` gated by config; both outputs feed into `ChunkResult` to keep `ContextAssembler` unchanged.
- Insert `_apply_reranker` hook after fusion; currently identity but wired for future Jina reranker v3 integration. Log `reranker_time_ms` even if disabled.

---

## 7. Future Capability Groundwork
1. **Semantic Chunking / NER**
   - **What’s implemented now**
     - Structured chunker emits `doc_is_microdoc`, `is_microdoc_stub`, `semantic_metadata` placeholders, and routes every chunk through `_enrich_chunk`.
     - `src/ingestion/semantic.py` exposes a pluggable `SemanticEnricher` interface with a stub enricher plus Prometheus metrics (`semantic_enrichment_total`, `semantic_enrichment_latency_ms`).
     - `GraphBuilder._extract_semantic_metadata` forwards `section["semantic_metadata"]` into Neo4j/Qdrant payloads; `chunks_multi` already reserves the `entity` vector slot.
   - **What remains**
     - Swap the stub provider for a real service (e.g., HuggingFace NER or Jina’s semantic chunker), populate entities/topics/summaries, and persist them via `_extract_semantic_metadata`.
     - Add integration tests that enable `chunk_assembly.semantic.enabled=true`, confirm metadata persists through ingestion and retrieval, and monitor metric spikes.
     - Document provider-specific config (API keys, timeout, retries) once chosen.
   - **How to enable later**
     1. Update `config/*.yaml` or env overrides with `chunk_assembly.semantic.enabled: true`, `provider: <provider-name>`, and any model/timeout settings.
     2. Implement a new class in `src/ingestion/semantic.py` returning `SemanticEnrichmentResult` with filled metadata and register it in `PROVIDERS`.
     3. Optional: extend `GraphBuilder` to compute entity vectors (e.g., using Jina entity embeddings) and upsert them into the reserved `entity` vector slot.

2. **Jina Reranker v3**
   - **What’s implemented now**
     - `search.hybrid` config includes `reranker` placeholders; `HybridRetriever._apply_reranker` is wired but returns inputs untouched.
     - Metrics/telemetry capture `metrics["reranker_applied"]` (currently `False`) and `metrics["reranker_time_ms"]` for future latency tracking.
     - `src/providers/rerank` contains the abstraction for plugging in external rerankers.
   - **What remains**
     - Wire the Jina reranker client (using `JINA_API_KEY`) so `_apply_reranker` reorders candidates and updates metrics/ChunkResult scores.
     - Extend tests to assert reranked ordering for known inputs and ensure the reranker gracefully falls back when disabled.
     - Add dashboard panels for reranker latency/applied ratio.
   - **How to enable later**
     1. Set `search.hybrid.reranker.enabled=true` (and model/provider fields) in config.
     2. Implement `JinaReranker` in `src/providers/rerank/jina.py` (or similar) and have `_apply_reranker` call it when enabled, setting `metrics["reranker_applied"] = True` and updating `ChunkResult.fused_score`.
     3. Update integration tests to run with `RERANK_PROVIDER=jina-ai` plus credentials, verifying metrics/logs show reranker activity.

---

## 8. Observability & Operations
- Extend ingestion metrics to tag schema version and Qdrant collection; log relationship-building/semantic hook events.
- Retrieval metrics to include per-vector-field hit counts, graph expansion stats, reranker latency.
- Provide bootstrap scripts or Make targets that: apply schema, configure Qdrant, ingest sample docs, run smoke-test queries.
- Update `scripts/cleanup-databases.py` to reset Neo4j/Qdrant/Redis (preserving schema metadata) for rapid dev iteration.

---

## 9. Deliverables
1. **Code Implementation**: schema validation hooks, ingestion metadata/Qdrant updates, retrieval multi-vector search integration, config additions, cleanup script refresh.
2. **Change Accounting**: written summary (README/ADR) listing all files touched and rationale (schema, ingestion, retrieval, config, tooling).
3. **Testing Suite (no mocks)**:
   - Ingestion integration tests that run real Neo4j/Qdrant instances, verifying nodes/relationships/properties and multi-vector payloads.
   - Retrieval tests executing actual hybrid queries, confirming fused ranking, graph expansion fallback, context budgets.
   - Reconciliation test that introduces drift and confirms cleanup.
4. **Documentation**:
   - Architecture doc (already created) + this canonical spec; README/ops guides describing bootstrap steps, config knobs, and new behavior.
   - Notes on semantic chunking/NER and reranker groundwork.
5. **Future Capability Notes**: sections describing how to activate semantic chunking metadata and reranker features once ready.

---

## 10. Rollout & Success Criteria
- **Sequence**: bootstrap schema + Qdrant → implement ingestion changes → swap retrieval vector stage → run cleanup + reconciliation → execute integration tests → document deliverables.
- **Success Metrics**: ingestion populates both stores without errors; retrieval returns meaningful results using multi-vector ANN; metrics/logging reflect new data; cleanup script enables fast resets; future hooks documented.
- **Change Control**: add an ADR explaining why we kept the existing Phase‑7E fusion pipeline instead of adopting the helper’s `example_retrieve_pack`, preserving context for future reviewers.

---

This spec, together with `docs/hybrid-rag-v2_2-architecture.md`, forms the single source of truth for the v2.2 integration effort. All subsequent commits and documentation should reference these files for scope and rationale.
