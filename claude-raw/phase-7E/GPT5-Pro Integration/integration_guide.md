---
Title: Integration Guide — GraphRAG v2.1 (Jina v3, 1024‑D)
Generated: 2025-10-28T21:20:21.345087Z
Status: draft
---

This guide walks an agentic coder through integrating the **Phase 7E** capabilities into the existing codebase, referencing exact files/lines discovered during the audit.

## Canonical Non‑negotiables

- Embeddings: `jina-embeddings-v3` at **1024‑D**, provider `jina-ai`.
- Persist `embedding_version` (not `embedding_model`) on `:Section/:Chunk` with dual labels.
- Qdrant collection `chunks` with named vector `content` (size=1024, distance=cosine).
- Neo4j vector indexes `section_embeddings_v2` and `chunk_embeddings_v2` on `vector_embedding` (1024‑D, cosine).

## File-by-file Actions

### `config/development.yaml`
- **L16** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L23** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `docker-compose.yml`
- **L140** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L150** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L222** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L231** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L299** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L308** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `docker/ingestion-service.Dockerfile`
- **L20** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L24** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L24** (info): HF tokenizer for Jina v3 — `from_pretrained('jinaai/jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `docker/ingestion-worker.Dockerfile`
- **L19** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L23** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L23** (info): HF tokenizer for Jina v3 — `from_pretrained('jinaai/jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `docker/mcp-server.Dockerfile`
- **L19** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L23** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L23** (info): HF tokenizer for Jina v3 — `from_pretrained('jinaai/jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `requirements.txt`
- **L74** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L74** (warn): Use of tiktoken (NON-canonical for Jina v3) — `tiktoken`
  - _Action_: Replace tiktoken with the **HuggingFace tokenizer** for Jina v3 (XLM‑RoBERTa family).

### `scripts/apply_complete_schema_v2_1.py`
- **L79** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `scripts/backfill_document_tokens.py`
- **L51** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L67** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L138** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L53** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L56** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L120** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L141** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `scripts/baseline_distribution_analysis.py`
- **L50** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L51** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L52** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L131** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L134** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L143** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L143** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L162** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L166** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L182** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L182** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L199** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L199** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L282** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `scripts/dev/seed_minimal_graph.py`
- **L60** (info): Qdrant cosine distance — `Distance.COSINE`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L66** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L181** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L198** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L199** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L199** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L204** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L215** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L270** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L280** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L290** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L300** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L310** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L351** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L351** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L214** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L218** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `scripts/eval/run_eval.py`
- **L56** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L468** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `scripts/neo4j/create_schema.cypher`
- **L19** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L65** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L68** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L71** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L100** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L101** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L102** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L103** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L104** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L105** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `scripts/neo4j/create_schema_v2_1.cypher`
- **L148** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L148** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L182** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L21** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L27** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L28** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L29** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L64** (info): Vector index creation — `CREATE VECTOR INDEX section_embeddings_v2`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L65** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L66** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L75** (info): Vector index creation — `CREATE VECTOR INDEX chunk_embeddings_v2`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L76** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L77** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L126** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L129** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L132** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L132** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L157** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L159** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L188** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L189** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L190** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L192** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L193** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L69** (warn): Neo4j vector dimensions — ``vector.dimensions`: 1024`
  - _Action_: Set Neo4j vector index `vector.dimensions` to 1024 for all `Section/Chunk` vector indexes.
- **L80** (warn): Neo4j vector dimensions — ``vector.dimensions`: 1024`
  - _Action_: Set Neo4j vector index `vector.dimensions` to 1024 for all `Section/Chunk` vector indexes.

### `scripts/neo4j/create_schema_v2_1_complete.cypher`
- **L231** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L308** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L308** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L318** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L318** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L363** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L28** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L49** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L52** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L114** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L115** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L117** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L118** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L147** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L150** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L153** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L211** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L212** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L213** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L217** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L220** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L223** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L223** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L246** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L252** (info): Vector index creation — `CREATE VECTOR INDEX section_embeddings_v2`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L253** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L254** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L263** (info): Vector index creation — `CREATE VECTOR INDEX chunk_embeddings_v2`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L264** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L265** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L276** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L276** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L277** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L279** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L283** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L283** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L286** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L288** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L292** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L293** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L294** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L331** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L333** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L379** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L380** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L381** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L383** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L384** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L257** (warn): Neo4j vector dimensions — ``vector.dimensions`: 1024`
  - _Action_: Set Neo4j vector index `vector.dimensions` to 1024 for all `Section/Chunk` vector indexes.
- **L268** (warn): Neo4j vector dimensions — ``vector.dimensions`: 1024`
  - _Action_: Set Neo4j vector index `vector.dimensions` to 1024 for all `Section/Chunk` vector indexes.

### `scripts/perf/test_traversal_latency.py`
- **L43** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L43** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `scripts/run_baseline_queries.py`
- **L109** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `scripts/test/debug_explain.py`
- **L21** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L26** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `scripts/test_jina_integration.py`
- **L27** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L27** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `scripts/test_jina_payload_limits.py`
- **L30** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L85** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L146** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `scripts/validate_token_accounting.py`
- **L65** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L181** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L66** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L71** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L80** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L152** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L182** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L185** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L248** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `src/ingestion/api.py`
- **L11** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L23** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L23** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L12** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L24** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L24** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/ingestion/auto/cli.py`
- **L418** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L828** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/ingestion/auto/orchestrator.py`
- **L433** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L435** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L350** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L449** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L450** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L794** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L832** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L850** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L869** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L874** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L875** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L876** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L885** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L888** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L892** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L893** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L836** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L874** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L880** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L892** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L897** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/auto/report.py`
- **L32** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L137** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L170** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L170** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L177** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L178** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L179** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L182** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L190** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L190** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L199** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L199** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L230** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/ingestion/auto/verification.py`
- **L43** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L99** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L100** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L103** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L121** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L122** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L123** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L126** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/ingestion/build_graph.py`
- **L499** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L880** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L905** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L907** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L911** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L44** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L141** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L158** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L167** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L167** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L170** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L170** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L218** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L259** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L276** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L396** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L529** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L530** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L592** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L610** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L612** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L616** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L666** (info): Qdrant cosine distance — `Distance.COSINE`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L745** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L747** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L749** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L774** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L775** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L784** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L794** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L797** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L805** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L830** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L832** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L833** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L843** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L844** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L844** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L845** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L845** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L847** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L847** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L848** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L848** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L857** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L858** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L858** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L860** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L860** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L861** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L861** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L869** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L881** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L913** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L915** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L919** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L228** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L248** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L250** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L362** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L396** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L721** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L737** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/extract/__init__.py`
- **L33** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/extract/commands.py`
- **L257** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L258** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L28** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L33** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L39** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L44** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L54** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L54** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L63** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L84** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L92** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L103** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L120** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L126** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L137** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L158** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L164** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L263** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L267** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L267** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L272** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/extract/configs.py`
- **L252** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L253** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L26** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L29** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L34** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L39** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L46** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L57** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L57** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L65** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L95** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L102** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L128** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L134** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L158** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L165** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L189** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L208** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L258** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L262** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L262** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L267** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/extract/procedures.py`
- **L103** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L127** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L155** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/incremental.py`
- **L28** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L38** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L44** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L55** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L140** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L162** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L167** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L169** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L196** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/ingestion/parsers/html.py`
- **L219** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L220** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L201** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L208** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/parsers/markdown.py`
- **L252** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L253** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L234** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L240** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/parsers/notion.py`
- **L211** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L212** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L193** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L200** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/ingestion/reconcile.py`
- **L17** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L27** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L36** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L46** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L64** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L67** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L69** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L70** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L87** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L125** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L182** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L210** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L240** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L260** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/mcp_server/query_service.py`
- **L61** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L364** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L247** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L252** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L253** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L255** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L282** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/providers/embeddings/base.py`
- **L42** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/providers/embeddings/jina.py`
- **L3** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L150** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L186** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/providers/embeddings/sentence_transformers.py`
- **L47** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `src/providers/factory.py`
- **L80** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L199** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L200** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L212** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `src/providers/tokenizer_service.py`
- **L4** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L87** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L95** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L104** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L286** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L5** (warn): Use of tiktoken (NON-canonical for Jina v3) — `tiktoken`
  - _Action_: Replace tiktoken with the **HuggingFace tokenizer** for Jina v3 (XLM‑RoBERTa family).
- **L383** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L397** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L404** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L432** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L439** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L456** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L471** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L471** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L478** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L489** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.

### `src/query/hybrid_search.py`
- **L156** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L156** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L168** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L469** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L615** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L614** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L615** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L617** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L632** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L633** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/query/planner.py`
- **L269** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/query/response_builder.py`
- **L39** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L262** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L263** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L264** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L339** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L340** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L344** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L347** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L347** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L362** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L362** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L402** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L402** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L723** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L724** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/query/session_tracker.py`
- **L340** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L429** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L489** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L499** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L323** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L340** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L429** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L438** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/query/templates/advanced/troubleshooting_path.cypher`
- **L49** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L51** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/query/templates/explain.cypher`
- **L29** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L33** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `src/query/templates/search.cypher`
- **L6** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L15** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L28** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/query/templates/troubleshoot.cypher`
- **L18** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/registry/index_registry.py`
- **L236** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L54** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/shared/cache.py`
- **L9** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L210** (info): Pattern-scan deletion (fallback) — `SCAN`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L246** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L255** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L259** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L259** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L289** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L295** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L377** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L384** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L390** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L390** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L416** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L416** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/shared/config.py`
- **L26** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L69** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L324** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `src/shared/connections.py`
- **L237** (info): Qdrant cosine distance — `Distance.COSINE`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L241** (info): Qdrant cosine distance — `Distance.COSINE`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `src/shared/schema.py`
- **L128** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L237** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L302** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L303** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L304** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L305** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L306** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L307** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/e2e/test_golden_set.py`
- **L189** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L398** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `tests/fixtures/baseline_query_set.yaml`
- **L9** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/integration/test_jina_large_batches.py`
- **L31** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L42** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L268** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L357** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/integration/test_phase7c_integration.py`
- **L133** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L214** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L468** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L468** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L476** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L477** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L479** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L480** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L495** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L523** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L548** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L548** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L556** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L557** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L559** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L560** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L586** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L622** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L715** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L742** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L742** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L750** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L751** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L753** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L754** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L785** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L910** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L922** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L922** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L930** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L931** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L933** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L934** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L952** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L977** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L997** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1015** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1028** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1028** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1036** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1037** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1039** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1040** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1072** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1100** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1130** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L443** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L452** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L482** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L483** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L498** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L591** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L608** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L609** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L715** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L719** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L735** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L736** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L742** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L744** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L756** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L762** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L765** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L765** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L767** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L775** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L784** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L790** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L910** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L913** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L917** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L922** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L924** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L936** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L939** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L939** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L941** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L952** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L956** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L996** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1000** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1015** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1018** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1022** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1023** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1028** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1030** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1042** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1045** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1045** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1047** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1072** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1080** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L1129** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L1136** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/integration/test_session_tracking.py`
- **L230** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L239** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L535** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/p1_t3_test.py`
- **L129** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L156** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L172** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L126** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L136** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L144** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L153** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L157** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L163** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L164** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L172** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/p2_t2_test.py`
- **L69** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L145** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L154** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L167** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L177** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L211** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L231** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L242** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p2_t3_test.py`
- **L336** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L340** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L342** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L347** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L347** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/p2_t4_test.py`
- **L87** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L92** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/p3_t2_test.py`
- **L54** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L263** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/p3_t3_integration_test.py`
- **L80** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L135** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L174** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L240** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L244** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L245** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L260** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p3_t3_test.py`
- **L101** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L130** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L166** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L216** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L248** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L250** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L265** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L278** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L280** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L101** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L104** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L130** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L133** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L167** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L174** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L217** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L220** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L249** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L253** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L279** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L283** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/p3_t4_integration_test.py`
- **L59** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L117** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L151** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L306** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L307** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L344** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L375** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L118** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L152** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L308** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L313** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/p3_t4_test.py`
- **L54** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L140** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L141** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L144** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/p4_t2_perf_test.py`
- **L42** (info): MERGE Document by id (canonical) — `MERGE (d:Document {
                id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L51** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L232** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L289** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L289** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L290** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/p4_t2_test.py`
- **L48** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L49** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L50** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L139** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L149** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L158** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L159** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L187** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L256** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L282** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L307** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L358** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p4_t3_test.py`
- **L219** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p4_t4_test.py`
- **L27** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L172** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L261** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L317** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L353** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L441** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L261** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/p5_t2_test.py`
- **L170** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p5_t3_test.py`
- **L48** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L83** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p6_t1_test.py`
- **L66** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L70** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L616** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p6_t2_test.py`
- **L343** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L361** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L436** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L476** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L550** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L562** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L600** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L824** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L902** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L986** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L1098** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p6_t3_test.py`
- **L64** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L68** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L265** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L274** (info): Pattern-scan deletion (fallback) — `scan_iter`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/p6_t4_test.py`
- **L69** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L524** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L542** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/test_integration_prephase7.py`
- **L47** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L59** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L271** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L291** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L293** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L295** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L300** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L302** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L360** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L379** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L389** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L449** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L453** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L454** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L461** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L463** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/test_jina_adaptive_batching.py`
- **L32** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L49** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L72** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L97** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L155** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L190** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L236** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L390** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L405** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L422** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L467** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L499** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/test_phase1_foundation.py`
- **L27** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L57** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `tests/test_phase2_provider_wiring.py`
- **L138** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L140** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L142** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L147** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L149** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L151** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L158** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L163** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L165** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/test_phase5_response_schema.py`
- **L103** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/test_phase7c_dual_write.py`
- **L208** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L220** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L267** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L268** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L270** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L271** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L195** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L196** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L207** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L219** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L267** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L274** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/test_phase7c_index_registry.py`
- **L31** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L41** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L149** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L215** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L179** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.

### `tests/test_phase7c_ingestion.py`
- **L43** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.
- **L5** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L97** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L117** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L120** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L136** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L137** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L138** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L140** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L141** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L152** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L155** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L161** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L164** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L192** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L195** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L196** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L273** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L274** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L279** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L280** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L284** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L286** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L328** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L330** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L331** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L333** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L334** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L426** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L106** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L109** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L136** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L144** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

### `tests/test_phase7c_provider_factory.py`
- **L26** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L33** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L111** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L118** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L128** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L135** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L141** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L152** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.

### `tests/test_phase7c_reranking.py`
- **L220** (error): Use of deprecated property name 'embedding_model' (use 'embedding_version') — `embedding_model`
  - _Action_: Rename property `embedding_model` → `embedding_version` everywhere in ingestion and persistence paths.

### `tests/test_phase7c_schema_v2_1.py`
- **L388** (error): Found reference to deprecated model v4 (should be v3) — `jina-embeddings-v4`
  - _Action_: Replace all occurrences of `jina-embeddings-v4` with `jina-embeddings-v3` and ensure 1024‑D throughout.
- **L63** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L67** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L75** (info): MERGE Document by id (canonical) — `MERGE (d:Document {id: `
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L77** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L77** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L86** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L87** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L89** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L90** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L157** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L157** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L165** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L166** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L168** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L169** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L177** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L178** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L179** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L181** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L182** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L193** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L378** (info): Presence of :Chunk label (dual-label support) — `:Chunk`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L378** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L387** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L388** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L390** (info): Use of canonical 'embedding_timestamp' — `embedding_timestamp`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L391** (info): Use of canonical 'embedding_dimensions' — `embedding_dimensions`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L399** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L400** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L424** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L432** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L432** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L438** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L445** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L446** (info): Use of canonical 'vector_embedding' (Neo4j) — `vector_embedding`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L447** (info): Use of canonical 'embedding_version' — `embedding_version`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L462** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L234** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L242** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L243** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.

### `tests/test_phase7e_phase0.py`
- **L511** (info): Presence of :Section label — `:Section`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L514** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.
- **L514** (warn): Use of non-canonical 'doc_id' (should be 'document_id') — `doc_id`
  - _Action_: Rename payload key to `document_id`.

### `tests/test_tokenizer_service.py`
- **L15** (info): Reference to canonical model v3 — `jina-embeddings-v3`
  - _Action_: Confirm this matches the canonical spec and invariants.
- **L274** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L278** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L293** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.
- **L300** (warn): Use of non-canonical 'chunk_index' (should be 'order') — `chunk_index`
  - _Action_: Rename payload key to `order`.
- **L494** (warn): Use of non-canonical 'section_id' (should be 'id') — `section_id`
  - _Action_: Migrate to canonical `id` for :Section/:Chunk; maintain deterministic order-preserving IDs.

## Final Checklist
- [ ] All references use `jina-embeddings-v3` @ 1024‑D.
- [ ] `embedding_version` persisted; no `embedding_model` remains.
- [ ] Neo4j vector indexes at 1024; Qdrant vectors size=1024, distance=cosine.
- [ ] Document primary key is `:Document.id`.
- [ ] Canonical property names in all payloads and upserts.
- [ ] Hybrid retrieval (RRF default) implemented; BM25 + Vector fused.
- [ ] Bounded adjacency expansion in place and tested.
- [ ] Epoch-based cache invalidation wired on re‑ingest.


## Migration Sequence & Commands

**Order of operations (safe, low-risk):**
1. **Property rename sweep** — Replace `embedding_model` → `embedding_version` (ingestion, persistence, tests).
2. **Model pin** — Replace any `jina-embeddings-v4` → `jina-embeddings-v3`; confirm **1024‑D** in Neo4j & Qdrant configs.
3. **Tokenizer correction** — Replace `tiktoken` usage with HuggingFace tokenizer for Jina v3 (XLM‑RoBERTa family).
4. **Full‑text index** — Create Neo4j FTS on `:Chunk(text, heading)` and add BM25 retrieval adapter.
5. **Hybrid fusion** — Implement RRF (k=60) default; keep weighted fusion as option (`alpha`, default 0.6).
6. **Adjacency expansion** — Use `:NEXT_CHUNK` edges; enable ±1 expansion under conditions (query ≥12 tokens OR close top‑2 scores).
7. **Epoch‑based caching** — Add `doc_epoch`/`chunk_epoch` and bump on re‑ingest; keep pattern‑scan fallback.
8. **QA** — Run fusion A/B harness (Hit@k, MRR@10, nDCG@10), verify non‑regression; enforce context budget (≤4,500 tokens).

### Cypher: Full‑text Index for `:Chunk`
```cypher
CALL db.index.fulltext.createNodeIndex('chunk_text_index', ['Chunk'], ['text','heading']);
```

### Cypher: Verify Vector Index Dimensions (should be 1024)
```cypher
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties
WHERE type = 'VECTOR'
RETURN name, labelsOrTypes, properties
ORDER BY name;
```

### Qdrant: Ensure 1024‑D Named Vector
- Collection: `chunks`
- Named vector: `content`
- VectorParams: `size=1024`, `distance=cosine`

### Python: RRF fusion (sketch)
```python
def rrf_fuse(vec_ids, kw_ids, k=60):
    v_rank = {cid: i+1 for i, cid in enumerate(vec_ids)}
    b_rank = {cid: i+1 for i, cid in enumerate(kw_ids)}
    ids = set(v_rank) | set(b_rank)
    return sorted(
        ((cid, 1/(k+v_rank.get(cid, 9999)) + 1/(k+b_rank.get(cid, 9999))) for cid in ids),
        key=lambda x: x[1],
        reverse=True
    )
```

### Redis: Epoch bump (Python)
```python
r.hincrby("rag:v2.1:doc_epoch", document_id, 1)
pipe = r.pipeline()
for cid in updated_chunk_ids:
    pipe.hincrby("rag:v2.1:chunk_epoch", cid, 1)
pipe.execute()
```

_Last updated: 2025-10-28T21:24:53.035646Z_
