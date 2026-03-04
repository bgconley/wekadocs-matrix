# WekaDocs Matrix — End-to-End Architecture Reference

**Date:** 2026-03-04
**Branch:** `multi-embedder-reranker`
**Schema Version:** v4.0
**Status:** Active reference — update when design decisions change

---

## Reading Paths

This document covers three audiences. Jump to what you need:

- **Quick re-orientation (returning engineer):** System Overview, Model Stack, then the specific section you care about.
- **New engineer onboarding:** Read every section in order. The Appendix diagrams help.
- **AI agent coding context:** System Overview, Model Stack, Infrastructure, then jump directly to the part of the pipeline you are modifying. The Appendix score-field survival table is critical for any work touching retrieval or evidence extraction.

---

## System Overview

WekaDocs Matrix is a GraphRAG (Graph-augmented Retrieval-Augmented Generation) documentation system purpose-built for WEKA distributed file system documentation. It ingests Markdown source files, builds a hybrid knowledge store (Neo4j graph + Qdrant vector database), and exposes a retrieval interface over the Model Context Protocol (MCP) so that AI coding assistants can retrieve precise, evidence-backed answers.

The system has two independent pipelines that share the same graph and vector stores:

**Ingestion pipeline** (offline / async): Watches a directory for new Markdown files, parses them into sections, extracts named entities via GLiNER, chunks them semantically, computes three types of embeddings (dense, sparse, late-interaction), writes to Neo4j for graph structure and Qdrant for vectors, then builds structural edges and cross-document RELATED_TO links — all inside an atomic saga that rolls back if any step fails.

**Retrieval pipeline** (online / per-request): Receives a natural language question over MCP, rewrites it (with an LLM if available, otherwise heuristic), runs parallel BM25 and multi-vector search, fuses with RRF, boosts with structural and entity signals, reranks with a cross-encoder, expands into graph neighbors, and returns an Evidence Pack — a set of ranked, source-attributed quotes with coverage metadata.

The three production MCP tools are:
- `kb.retrieve_evidence` — the primary tool; returns the Evidence Pack
- `kb.read_excerpt` — given a `passage_id` from a prior evidence result, returns full text
- `graph.expand` — given `node_ids`, traverses Neo4j neighbors for structural context

All six AI models are served through a single unified GPU gateway at `10.25.0.50:8080`.

```
[Markdown files on disk]
        |
        v
[FileSystemWatcher] --> Redis queue --> [AtomicIngestionCoordinator]
                                                |
                          +---------------------+---------------------+
                          |                     |                     |
                    [Parser]             [GLiNER NER]          [SemanticChunker]
                          |                     |                     |
                          +---------------------+---------------------+
                                                |
                                    [EmbeddingService: dense+sparse+colbert]
                                                |
                          +---------------------+---------------------+
                          |                                           |
                    [Neo4j graph]                            [Qdrant vectors]
                          |                                           |
                          +---------------------+---------------------+
                                                |
                                    [Cross-doc RELATED_TO edges]

[AI Client] --> MCP --> [kb.retrieve_evidence]
                                |
                    [Query reformulation (Qwen2.5-1.5B)]
                                |
              [BM25 (Neo4j)] + [Multi-vector (Qdrant)]
                                |
                          [RRF Fusion k=60]
                                |
                    [Entity boost + Structural boost]
                                |
                    [Qwen3-Reranker-4B cross-encoder]
                                |
                    [Graph neighbor expansion (Cypher)]
                                |
                    [Evidence pack: ranked quotes + coverage]
```

---

## Model Stack

All models are served through the unified GPU gateway at `http://10.25.0.50:8080`. Total estimated VRAM: ~14.5 GB of 24 GB on an RTX 3090.

| Model | Role | Dims | Context | VRAM | Gateway Endpoint |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | Dense semantic embeddings | 1024 | 32K tokens | ~1.5 GB | `POST /v1/embeddings` |
| SPLADEv3 (`naver/splade-v3`) | Learned sparse retrieval | 30522 (vocab) | 512 tokens | ~0.5 GB | `POST /v1/embeddings/sparse` |
| ColBERTv2 (`colbert-ai/colbertv2.0`) | Late-interaction multi-vector | 128 dims/token | 512 tokens | ~0.5 GB | `POST /v1/embeddings/colbert` |
| Qwen3-Reranker-4B | Cross-encoder reranking | N/A (score) | 8K tokens | ~8 GB | `POST /v1/rerank` |
| GLiNER Medium v2.1 (`urchade/gliner_medium-v2.1`) | Named entity recognition | N/A | — | ~1 GB | `POST /v1/extract` (or `http://host.docker.internal:9002`) |
| Qwen2.5-1.5B-Instruct | Query reformulation (LLM) | N/A | 32K tokens | ~3 GB | `POST /v1/chat/completions` |

**Important:** Qwen2.5-1.5B-Instruct is not yet deployed to the GPU gateway (as of 2026-03-04). The reformulation path falls back to the heuristic rewriter in `query_service.py`.

**Provider naming:** The code uses `embedding-service` as the canonical provider name for all three embedding models. Legacy aliases (`bge-m3-service`, `bge-m3`, `bge_m3`) are normalized to `embedding-service` in `src/providers/factory.py:54-77`.

**Query instruction (Qwen3-Embedding-0.6B):** Applied to query vectors only (not document vectors) to implement asymmetric retrieval:
```
Instruct: Given a technical query about WEKA distributed file system, retrieve relevant
documentation passages covering configuration, administration, troubleshooting, and CLI commands
Query: <query text>
```

---

## Infrastructure

| Component | Technology | Ports | Data |
|---|---|---|---|
| Graph database | Neo4j 2025.10.1 Community | 7474 (HTTP), 7687 (Bolt) | 262 Documents, 3944 Chunks, schema v4.0 |
| Vector database | Qdrant v1.16.0 | 6333 (HTTP), 6334 (gRPC) | `chunks_multi_qwen3_0_6b` (empty, awaiting re-ingestion); `chunks_multi_bge_m3` (stale) |
| Queue / Cache | Redis 7.2-alpine | 6379 | Ingestion job queue, ScratchStore backup, cache epochs |
| GPU gateway | Unified endpoint (bare metal) | 8080 | 6 models, RTX 3090 24 GB |
| MCP server | FastAPI + uvicorn + MCP SDK | 8000 (via Tailscale sidecar) | Stateless per-request |
| Telemetry | Grafana Alloy v1.4.0 | 4317 (OTLP gRPC), 4318 (OTLP HTTP), 12345 (UI) | Forwards to GCP LGTM stack |

**Neo4j memory:** heap initial 1280m, heap max 2560m, page cache 2GB. Transactions timeout at 30s.

**Qdrant collection naming:** The collection name suffix derives from the embedding profile. Profile `qwen3_0_6b` → collection `chunks_multi_qwen3_0_6b`. The config field `search.vector.qdrant.collection_name` defaults to `chunks_multi` (base) with the profile suffix appended at runtime.

**Redis persistence:** AOF enabled with `appendfsync everysec`. Max memory 768MB (configurable via `REDIS_MAXMEMORY`), eviction policy `allkeys-lru`.

**Network topology:** Application containers (`weka-mcp-server`, `weka-ingestion-worker`) use `network_mode: service:ts-mcp-server` / `service:ts-ingestion-worker` — they share the Tailscale sidecar's network namespace rather than the `weka-net` bridge. This gives them Tailscale MagicDNS access to the GPU gateway. Infrastructure services (Neo4j, Qdrant, Redis, Alloy) attach directly to `weka-net`.

---

## Part 1: Ingestion Pipeline

The ingestion pipeline transforms raw Markdown files into Neo4j graph nodes/edges and Qdrant vector points. The pipeline is queue-driven and atomic: each document either completes fully or rolls back entirely.

### Document Parsing

**File:** `src/ingestion/parsers/markdown_it_parser.py`
**Entry function:** `parse_markdown(source_uri: str, raw_text: str) -> dict`
**Return shape:** `{"document": {...}, "sections": [...]}`

The parser uses `MarkdownIt("gfm-like")` (GitHub-Flavored Markdown with table plugin). It processes documents in two phases:

**Phase 1 — Frontmatter extraction:** A regex scans for YAML between `---` markers at the start of the file. The YAML is parsed and stored as document-level metadata.

**Phase 2 — AST traversal:** `markdown-it-py` produces a `SyntaxTreeNode` AST. The parser walks the AST maintaining a `heading_stack` that tracks nesting depth. Each time a heading is encountered, the stack is updated and a new section begins. The `parent_path` field is built as `" > ".join(titles)` from the current stack — for example, `"Installation > Prerequisites > Docker"`.

**Block type handling:**
- Code fences are wrapped: `[CODE]\n{code}\n[/CODE]` and contribute to `code_ratio`
- Tables are rendered as pipe-delimited text and set `has_table = True`
- `block_types` list records all block types seen in the section (paragraph, code, table, list)

**ID formulas:**
- Section ID: `SHA-256(f"{source_uri}#{anchor}|{checksum}")` — anchor is the heading slug, checksum is a content hash
- Document ID: `SHA-256(normalized_source_uri)`

**Metadata fields produced per section:** `id`, `document_id`, `source_uri`, `title`, `text`, `parent_path`, `parent_path_depth`, `line_start`, `line_end`, `block_types`, `code_ratio`, `has_table`, `has_code`, `block_type` (primary)

### Entity Extraction

Entity extraction runs in two layers:

**Structural extractors (`src/ingestion/extract/`):**
- `commands.py` — regex/pattern matching for WEKA CLI commands (e.g., `weka fs`, `weka nfs`)
- `configs.py` — configuration parameter extraction
- `procedures.py` — step-sequence detection
- `references.py` — hyperlink and "see also" cross-reference extraction

**GLiNER NER (`src/ingestion/extract/ner_gliner.py`):**
- Entry function: `enrich_chunks_with_entities(chunks: list) -> None` (modifies in-place)
- Uses GLiNER Medium v2.1 via the service at `http://host.docker.internal:9002` (falls back to local model if unavailable)
- Confidence threshold: 0.45
- Entity labels (10 types): `COMMAND`, `PARAMETER`, `COMPONENT`, `PROTOCOL`, `CLOUD_PROVIDER`, `STORAGE_CONCEPT`, `VERSION`, `PROCEDURE_STEP`, `ERROR`, `CAPACITY_METRIC`
- After enrichment, the chunk receives a `_embedding_text` field containing:
  ```
  {title}

  {text}

  [Context: {entity_context}]
  ```
  where `entity_context` is the extracted entity names joined with context. This enriched text is used as input to the sparse and dense embedders (not the raw text).

### Chunking

**File:** `src/ingestion/semantic_chunker.py`
**Class:** `SemanticChunkerAssembler`

The chunker takes the sections produced by the parser and assembles them into retrieval-optimal chunks. The configured mode in `development.yaml` is `chunk_assembly.assembler: "semantic"`.

**Token limits (from environment / docker-compose):**
- `BGE_M3_MAX_INPUT_TOKENS` (hard limit): 8192
- `BGE_M3_SAFE_INPUT_TOKENS` (practical limit): 7500
- Guard overlap: 200 tokens
- Fallback overlap: 80 tokens

**Guard-split logic:** Sections exceeding `BGE_M3_SAFE_INPUT_TOKENS` (7500) are split into token-window chunks first, then semantic chunking is applied to the windows. This is the "guard-split" that prevents oversized inputs from reaching the embedder.

**Code-heavy bypass:** Sections where `code_ratio > 0.5` skip semantic chunking entirely and use fixed-size token windows instead. This prevents the semantic boundary detector from splitting code blocks at semantically inappropriate points.

**Semantic chunking parameters (from `development.yaml`):**
- `similarity_threshold`: 0.35 — split when cosine similarity between adjacent segments drops below 35%
- `target_tokens`: 400
- `min_tokens`: 100
- `max_tokens`: 512
- Embedding adapter: `qwen3_4b` (used for Chonkie boundary detection)

**Post-processing:** `_merge_tiny_chunks()` merges fragments below the minimum token count into adjacent chunks.

**Chunk ID formula:**
```
f"{document_id}_chunk_{index}_{content_hash}"
```
where `content_hash = SHA-256(chunk_text)[:16]`

**Chonkie adapter:** `src/providers/embeddings/chonkie_adapter.py` — bridges Chonkie's chunking API to the project's embedding provider abstraction.

### Embedding Generation

**Called from:** `AtomicIngestionCoordinator._compute_embeddings()` in `src/ingestion/atomic.py`

Three distinct embedding types are computed per chunk:

**Dense (Qwen3-Embedding-0.6B):**
- Input text: GLiNER-enriched `_embedding_text` if available, else `heading + "\n\n" + text`
- Produces three named vectors: `content` (1024-D), `title` (1024-D), `doc_title` (1024-D)
- `title` uses only the section heading as input
- `doc_title` uses the document-level title for cross-document linking

**Sparse (SPLADEv3):**
- Token limit: 512 (hard model limit)
- Produces four named sparse vectors: `text-sparse`, `doc_title-sparse`, `title-sparse`, `entity-sparse`
- `entity-sparse` input: entity names joined as a space-separated string (e.g., `"NFS POSIX weka fs --cores"`)

**Late-interaction (ColBERTv2):**
- Token limit: 512
- Produces `late-interaction` as a multi-vector: one 128-D vector per input token
- Stored in Qdrant as a multi-vector field

**Batching strategy:** Token-budgeted batching with `EMBED_BATCH_MAX_TOKENS=7000`. Each batch is processed independently so that a single embedding failure does not abort the entire document.

**Text preparation priority:**
1. `chunk["_embedding_text"]` (GLiNER-enriched text with entity context) if present
2. `chunk["heading"] + "\n\n" + chunk["text"]` (structural text) as fallback

### Neo4j Graph Construction

All writes occur within a single explicit Neo4j transaction. The transaction commits only after Qdrant upsert succeeds (see Atomic Saga Protocol below).

**Node types and write patterns:**

```cypher
-- Document node
MERGE (d:Document {id: $id}) SET d += $props

-- Chunk nodes (one per chunk)
MERGE (c:Chunk {id: section.id}) SET c += section

-- Document-to-chunk relationship
MERGE (d)-[:HAS_CHUNK]->(c)

-- Entity nodes (label is dynamic, e.g., :Entity:COMMAND)
MERGE (e:Entity:{Label} {id: entity.id}) SET e += entity

-- Chunk-to-entity mention
MERGE (c)-[r:MENTIONS]->(e) SET r.confidence = entity.confidence, r.source = entity.source
```

**Cross-document references:** When a chunk references another document (via extracted hyperlinks or "see also" patterns), the resolver attempts to match the reference to an existing Document node. If the target document exists, a `REFERENCES` edge is created from the Chunk to the Document. If the target is unresolvable, a `GhostDocument` node is created with a `PendingReference` edge to be resolved when the target document is later ingested.

### Structural Edges

**File:** `src/ingestion/structural_edges.py`
**Function:** `build_structural_edges_in_tx(tx, document_id: str) -> StructuralEdgeStats`

Called inside the active Neo4j transaction, after all chunk nodes are written. Builds the navigation graph for a single document.

**Steps executed in order:**
1. Normalize `parent_path` values on all chunks for the document
2. Compute `parent_chunk_id` for each chunk (identifies the chunk representing the parent heading)
3. Clear existing structural edges for the document (idempotent re-ingestion)
4. Create `NEXT_CHUNK` edges: `MERGE (chunks[i])-[:NEXT_CHUNK]->(chunks[i+1])` ordered by `chunk.order`
5. Create `PARENT_HEADING` / `CHILD_OF` / `PARENT_OF` edges using the `parent_chunk_id` mapping
6. Create `NEXT` sibling edges (chunks sharing the same `parent_chunk_id`, ordered within the parent)

**Stats returned** (all field names in `StructuralEdgeStats`): `parent_path_normalized`, `parent_chunk_id_computed`, `edges_cleared`, `next_chunk_created`, `parent_heading_created`, `child_of_created`, `parent_of_created`, `next_sibling_created`

### Qdrant Vector Storage

**8 named vector fields per point:**

| Field Name | Type | Dimensions | Description |
|---|---|---|---|
| `content` | dense | 1024 | Semantic content embedding (Qwen3) |
| `title` | dense | 1024 | Section heading embedding |
| `doc_title` | dense | 1024 | Document title embedding |
| `late-interaction` | multi-vector | 128/token | ColBERT per-token embeddings |
| `text-sparse` | sparse | 30522 | SPLADE on enriched chunk text |
| `doc_title-sparse` | sparse | 30522 | SPLADE on document title |
| `title-sparse` | sparse | 30522 | SPLADE on section heading |
| `entity-sparse` | sparse | 30522 | SPLADE on entity names |

**Point ID:** UUID5 derived from `section["id"]` (deterministic, idempotent)

**Payload fields (~40 total):** `doc_tag`, `source_uri`, `document_id`, `section_id`, `title`, `text`, `parent_path`, `parent_path_depth`, `parent_path_norm`, `entity_metadata`, `block_type`, `block_types`, `has_code`, `has_table`, `code_ratio`, `embedding_version`, `line_start`, `line_end`, and additional provenance fields.

**Pre-write purge:** Before upserting a document's vectors, all existing points for the same `source_uri` + `embedding_version` are deleted. This ensures clean re-ingestion when document content changes.

### Atomic Saga Protocol

The ingestion coordinator (`src/ingestion/atomic.py`, class `AtomicIngestionCoordinator`) implements a two-phase commit pattern across Neo4j and Qdrant.

**Execution order:**
1. Begin Neo4j explicit transaction
2. Write all Document, Chunk, and Entity nodes and edges within the transaction
3. Build structural edges within the same transaction (`build_structural_edges_in_tx`)
4. Upsert all Qdrant vectors (outside the transaction — Qdrant has no transaction protocol)
5. Commit the Neo4j transaction

**Failure compensation:**
- If Qdrant upsert fails: rollback the Neo4j transaction (no commit). Neo4j stays clean.
- If Neo4j commit fails after Qdrant succeeds: issue a compensating delete of the Qdrant points that were just written. This is the "compensation" branch of the saga.

**Retry policy (from `retry_with_backoff` decorator):** max 3 retries, base delay 0.5s, max delay 30s, exponential backoff with jitter, retries on transient exceptions.

### Cross-Document Linking

**File:** `src/services/cross_doc_linking.py`

Runs post-commit, non-blocking (fire-and-forget from the worker). Creates `RELATED_TO` edges between semantically similar documents to enable the retrieval system to follow document relationships.

**Algorithm:**
1. Query Qdrant for chunks with similar `doc_title` dense vectors (threshold 0.50 for discovery)
2. Aggregate chunk-level similarity scores to document level (max-score wins per document)
3. Fuse dense and sparse signals via RRF with `k=60`
4. Optional: ColBERT rerank of top candidates (enabled via `cross_doc_linking.colbert_rerank: true`)
5. Filter by `rrf_threshold: 0.025` and create at most `max_edges_per_doc: 5` RELATED_TO edges

**Config location:** `development.yaml` under `ingestion.cross_doc_linking`

### Redis Queue System

**Key schema:**
- `ingest:jobs` — LIST, pending jobs (RPUSH to enqueue)
- `ingest:processing` — LIST, in-flight jobs (BRPOPLPUSH moves atomically)
- `ingest:status` — HASH, job status by job_id
- `ingest:dead` — LIST (dead-letter queue), jobs that exhausted retries
- `ingest:checksums:{tag}` — SET per document tag, stores content checksums for deduplication

**Job structure (`IngestJob` dataclass):** `job_id`, `kind="file"`, `path`, `source`, `enqueued_at`, `attempts`

**Lifecycle:**
1. `FileSystemWatcher` (`src/ingestion/auto/service.py`) detects file events on `/app/data/ingest` → RPUSH to `ingest:jobs`
2. Worker calls `brpoplpush("ingest:jobs", "ingest:processing", timeout=5)` to atomically dequeue and claim
3. Worker calls `AtomicIngestionCoordinator.ingest_document_atomic(job)`
4. On success: `ack(job)` removes from `ingest:processing`
5. On failure: increment `attempts`. If `attempts >= 5`, move to `ingest:dead`. Otherwise re-enqueue to `ingest:jobs`

**`JobReaper`:** Background async task that scans `ingest:processing` for jobs older than `queue_recovery.job_timeout_seconds` (600s) and re-enqueues them. Runs every `reaper_interval_seconds` (30s). Handles worker crashes that leave jobs stuck in the processing list.

---

## Part 2: Retrieval Pipeline

### MCP Entry Points

The MCP server is a FastAPI application (`src/mcp_server/main.py`) that exposes two MCP transports sharing the same tool implementations.

**Streamable HTTP (primary):**
- Mounted at `/_mcp` via `StreamableHTTPSessionManager` from the MCP SDK
- `MCP_HTTP_STREAMABLE_ENABLED` env var (default: `"true"`)
- JSON response mode available via `MCP_HTTP_STREAMABLE_JSON_RESPONSE` env var
- Stateless mode: `MCP_HTTP_STREAMABLE_STATELESS` env var

**STDIO (for local/subprocess clients):**
- Entry point: `src/mcp_server/stdio_server.py`, function `run_stdio_server()`
- Uses the same `build_mcp_server()` factory function as the HTTP path
- Designed for Claude Desktop integration

**ASGI health shortcut:** `/_mcp/health` returns `{"status": "ok"}` or 503 if the MCP session manager is not initialized. This is checked before the MCP handshake in load balancer health checks.

**Legacy REST (frozen):** The `/mcp/*` REST endpoints are a Phase 1 stub, intentionally frozen. `MCP_HTTP_LEGACY_REST_ENABLED` defaults to `"false"`. Do not modify these endpoints.

### Tool Surface

**Factory:** `build_mcp_server()` in `src/mcp_server/mcp_app.py`

**`_tool_specs()` returns a flat list of dicts**, each with keys: `name`, `handler`, `description`, `input_schema`, `output_schema`, `annotations`

**Total tools:** 17 canonical dot-notation tools + 17 underscore backward aliases = 34 entries in `full_tool_map`. The dot-notation names are the canonical names; underscore names are aliases that route to the same handlers.

**Tool profiles (controlled by `MCP_TOOL_PROFILE` env var):**

| Profile | Tools visible in `list_tools` | Aliases |
|---|---|---|
| `production` | `kb.retrieve_evidence`, `kb.read_excerpt`, `graph.expand` | Always callable via `call_tool` regardless of profile |
| `analyst` | All 17 dot-notation tools | — |
| `full` | Unfiltered (no profile restriction) | — |

**Default:** `MCP_TOOL_PROFILE=production` (set in `mcp_app.py:90`)

**Production tool instructions (shown to the AI client):**
> "Start with kb.retrieve_evidence... Use kb.read_excerpt for more context... Use graph.expand only for follow-up navigation"

**Naming contract drift (important):** The code registers tools with underscore names (e.g., `kb_search`). Contract tests in `tests/contracts/` expect dot notation (e.g., `kb.search`). The plan is to migrate to dot notation as canonical, with temporary underscore aliases. When writing tests or checking tool names, use dot notation.

### kb.retrieve_evidence — The Evidence Pack

**Handler location:** `src/mcp_server/mcp_app.py`, approximately lines 2171-2405.

**Parameters:**
- `question` (required) — natural language query
- `top_k` (default: 5) — number of quotes to return in evidence pack
- `max_quotes` (default: 6) — maximum quotes in the pack
- `max_quote_tokens` (default: 80) — token budget per quote
- `retrieval_depth` (default: `KB_EVIDENCE_INTERNAL_FETCH_K=60`) — how many candidates to retrieve internally
- `scope` — optional document scope filter
- `filters` — optional Qdrant payload filters
- `options` — additional retrieval options
- `session_id` — for ScratchStore correlation across tool calls

**Internal fetch depth calculation:**
```python
internal_fetch_k = max(max_quotes, min(retrieval_depth, KB_EVIDENCE_MAX_FETCH_K=150))
```

**Call sequence:**
1. Build `RetrievalTraceBuilder` for the session
2. Call `_kb_search_candidates(question, internal_fetch_k, ...)` → returns `(passages, search_metrics)`
3. Call `_expand_evidence_with_structure(seeds)` → adds graph neighbors to ScratchStore
4. Call `_extract_evidence_from_passages(session_id, passages, question)` → selects best quotes
5. Assemble coverage metadata and return structured evidence pack

#### Query Reformulation

**Called from:** `search_sections_light()` in `src/mcp_server/query_service.py` (lines 286-328)

**Two-stage reformulation:**
1. `_rewrite_keyword_query(query)` — strips question phrasing, extracts technical terms
2. `_llm_reformulate(query)` — sends to Qwen2.5-1.5B-Instruct via `/v1/chat/completions`. Currently falls back to heuristic because Qwen2.5-1.5B is not yet deployed.
3. `_heuristic_reformulate(query)` — fallback: keyword extraction, entity normalization

**Dual-query strategy:** `HybridRetriever.retrieve()` receives both the reformulated query (for vector/dense search) and the original keywords (for BM25 lexical search):
```python
retriever.retrieve(query=reformulated, query_original=original_keywords)
```

**Metrics recorded:** `query_rewrite_applied` (bool), `query_rewrite_original` (original text), `query_rewrite_result` (reformulated text)

#### _kb_search_candidates

**Location:** `src/mcp_server/mcp_app.py`, approximately lines 398-588

1. Calls `deps.query.search_sections_light(query, fetch_k, filters, expand)`
2. Post-retrieval deduplication: `_dedupe_by_doc()` with `max_per_doc=5` for evidence requests
3. Cursor pagination for multi-page retrieval
4. Per chunk: assigns a fresh `passage_id = uuid4()`, builds `scratch_payload` with 19 fields (see ScratchStore section), calls `ScratchStore.put(session_id, passage_id, payload)`
5. Returns `(list_of_passage_dicts, search_metrics_dict)`

#### Step 1: Parallel BM25 + Vector Search

**Called from:** `HybridRetriever.retrieve()` in `src/query/hybrid_retrieval.py` (lines 2996-3724)

**Candidate calculation:**
```python
candidate_k = min(top_k * 3 * entity_overfetch_multiplier, 200)
```

**BM25 search:** `bm25_retriever.search(lexical_query, candidate_k, bm25_filters)`
- Queries Neo4j fulltext index `chunk_text_index_v3_bge_m3`
- Uses `query_original` (the pre-reformulation keywords) as the lexical query
- Strips `embedding_version` from filters (BM25 index is version-agnostic)
- Returns top `bm25.top_k=50` candidates

**Vector search:** Multi-field Qdrant Query API
- Fields searched: `content` (dense), `title` (dense), `text-sparse`, `doc_title-sparse`, `title-sparse`, `entity-sparse`, `late-interaction` (if ColBERT enabled)
- `use_query_api: true` with `query_api_candidate_limit: 200`
- `query_strategy: "weighted"` — uses RRF internally via Qdrant's `FusionQuery`

**RRF field weights (from `development.yaml`):**
```yaml
rrf_field_weights:
  content: 2.0
  title: 1.5
  text-sparse: 0.5
  doc_title-sparse: 0.8
  title-sparse: 0.8
  entity-sparse: 0.8
```

#### Step 2: RRF Fusion

After BM25 and vector search return their candidate lists, Python-side RRF merges them:

```
fused_score = sum( weight_i / (k + rank_i) )
```

where `k=60` (`hybrid.rrf_k` in `development.yaml`), `rank_i` is the 0-based rank in each sub-list, and `weight_i` is the per-field weight from `rrf_field_weights`.

The standard RRF formula `1/(k+rank)` is multiplied by the field weight, so a chunk ranked #1 in `content` (weight 2.0) contributes `2.0/61 ≈ 0.033` to its fused score.

Feature flag `query_api_weighted_fusion: true` enables the `_search_via_query_api_weighted()` code path with per-field score tracking.

#### Step 3: Entity Boost + Structural Boost

**Entity boost:** GLiNER NER is run on the query text itself (not the documents). Extracted entities are matched against chunk entity metadata. Chunks that `MENTIONS` extracted entities receive a soft boost applied to their `fused_score`.

**Structural boost (from `development.yaml` `structural` section):**
- Query type detection (conceptual, cli, config, procedural, troubleshooting, reference)
- `cli_code_boost: 1.20` — 20% boost for `block_type=code` chunks on CLI-type queries
- `reference_table_boost: 1.20` — 20% boost for `block_type=table` chunks on reference queries
- `deep_nesting_penalty: 0.90` — 10% penalty for chunks at `parent_path_depth > max_depth_for_overview=2`

Feature flags that must be active: `graph_garbage_filter`, `graph_rel_types_wired`, `dedup_best_score`, `graph_score_normalized`, `graph_as_reranker`, `structure_aware_expansion`.

#### Step 4: Cross-Encoder Reranking

**Reranker config (from `development.yaml` `search.hybrid.reranker`):**
- `enabled: true`
- `provider: "local-reranker-service"`
- `model: "Qwen/Qwen3-Reranker-4B"`
- `top_n: 20` — final results returned after reranking
- `instruction` (domain-tuned, passed as system prefix to the cross-encoder):
  > "Judge whether the document is relevant to the search query about WEKA distributed file system. Consider configuration procedures, CLI commands, parameter references, and step-by-step instructions as highly relevant. Introductory overviews and general summaries are less relevant than specific technical procedures and settings. Answer only yes or no."

**Signal pool (`src/query/signal_pool.py`, class `build_signal_pool()`):** When `signal_diverse_rerank_pool` feature flag is enabled, instead of sending a flat "top N by fused score" list to the reranker, the signal pool builds a diversity-ensuring candidate set by filling slots in priority order:

1. **Consensus slots** — top by `fused_score` (multi-signal agreement)
2. **Per-signal unique slots** — chunks that ranked well on a single signal (BM25, dense, entity-sparse) but were buried by RRF fusion
3. **Structural graph expansion slots** — NEXT_CHUNK and sibling neighbors from the graph
4. **Per-document depth slots** — ensure top-ranked documents have deep representation
5. **Backfill** — remaining capacity filled from fused_score order

Note: `signal_diverse_rerank_pool` is not yet in `development.yaml` feature_flags — it defaults to `false` (class default). The feature is implemented and tested but not yet activated.

**ColBERT reranking:** `colbert_rerank_enabled: false` currently (disabled for testing cross-encoder alone).

#### Step 5: Expansion

**Bounded adjacency expansion** (`expansion.enabled: true`):
- Retrieves NEXT_CHUNK neighbors ±1 hop from each top result
- `max_neighbors: 1`
- Gating: `sparse_score_threshold: 0.15` — expanded neighbors are only kept if their sparse lexical score exceeds this threshold

**Structure-aware expansion** (`structure_aware_expansion: true`):
- Sibling chunks from same `parent_section_id`: up to `sibling_limit: 3`
- Parent section chunks: up to `parent_section_limit: 2`
- Entity-shared chunks (sharing MENTIONS targets with top results): up to `shared_entity_limit: 3`
- Hard timeout: `timeout_ms: 100`

**Context budget enforcement:**
- `MAX_TOKENS_PER_DOC = 7500` — maximum tokens summed across all chunks from one document
- `MAX_TOKENS_TOTAL = 8192` — maximum total tokens across all expanded results

#### ScratchStore

**File:** `src/mcp_server/scratch_store.py`, class `ScratchStore`

In-memory `OrderedDict` keyed by `(session_id, passage_id)`. Serves as a multi-tool scratch pad: `kb.retrieve_evidence` writes entries, and `kb.read_excerpt` reads them by `passage_id`. A single global session ID is used (MCP SDK session objects proved unreliable across calls).

**Configuration:**
- `SCRATCH_TTL_SECONDS = 1800` (30 minutes, via `MCP_SCRATCH_TTL_SECONDS` env)
- `SCRATCH_MAX_BYTES = 256 * 1024 * 1024` (256 MB, via `MCP_SCRATCH_MAX_BYTES` env)
- Eviction: LRU (OrderedDict, `popitem(last=False)` when budget exceeded)
- TTL eviction triggered on every `put()` and `get()` call

**URI format:** `wekadocs://scratch/{session_id}/{passage_id}` (constructed by `ScratchStore.build_uri()`)

**19 fields preserved per entry:**
`text`, `title`, `section_id`, `doc_tag`, `source_uri`, `parent_path`, `parent_path_norm`, `entity_names`, `block_type`, `has_code`, `has_table`, `rerank_score`, `fused_score`, `vector_score`, `bm25_score`, `entity_boost`, `structural_boost`, `size_bytes`, `source` (signal origin, e.g., `"retrieval"` or `"graph_expanded"`)

**Fallback lookup:** `find_by_section_id(session_id, section_id)` provides O(n) search for cases where the AI passes `section_id` instead of `passage_id`. Supports prefix/suffix matching for truncated IDs.

#### Evidence Extraction

**Function:** `_extract_evidence_from_passages()` in `src/mcp_server/mcp_app.py`, approximately lines 591-700.

**Stage 1 — Score-based sort:** Load all entries from ScratchStore for the session. Sort by `retrieval_score` using this precedence chain:
```python
retrieval_score = rerank_score or fused_score or vector_score or bm25_score or 0.0
```

**Stage 2 — Span selection:** For each passage (up to `max_quotes` limit), find the best text span using a blended score:
```
span_score = (0.7 × retrieval_score) + (0.3 × keyword_overlap)
```
where `keyword_overlap` is the fraction of query keywords present in the span.

**Output quote fields per result:** `quote`, `passage_id`, `section_id`, `doc_tag`, `title`, `parent_path`, `uri`, `confidence`, `source`, `rank`

#### Graph Enrichment

**Function:** `_expand_evidence_with_structure()` in `src/mcp_server/mcp_app.py`, approximately lines 703-791.

Runs after the initial retrieval to seed the evidence pack with additional context from the graph.

**Cypher query pattern:**
```cypher
UNWIND $seed_ids AS seed_id
MATCH (seed:Chunk {id: seed_id})
OPTIONAL MATCH (seed)-[:NEXT_CHUNK]-(neighbor)
OPTIONAL MATCH (sibling:Chunk {parent_section_id: seed.parent_section_id})
  WHERE sibling.id <> seed_id
WITH DISTINCT neighbor, sibling
WHERE NOT (neighbor.id IN $existing_ids OR sibling.id IN $existing_ids)
RETURN DISTINCT chunk
LIMIT 20
```

**Seeds:** Top 10 `section_id` values from the retrieval results (by `retrieval_score`).

**Synthetic scores:** Graph-expanded neighbors are stored in ScratchStore with `fused_score=0.3`, `source="graph_expanded"`. These scores are intentionally lower than retrieval scores to prevent graph expansion from dominating the evidence pack.

**Coverage metadata from graph enrichment:** `graph_expansion_applied = len(graph_passage_ids) > 0`

#### Coverage Metadata

Returned in the structured content of `kb.retrieve_evidence`:
- `documents_searched` — total documents touched by retrieval
- `documents_with_evidence` — documents contributing at least one quote
- `retrieval_depth` — the `internal_fetch_k` value used
- `reranker_applied` — `bool(search_metrics.get("reranker_applied"))`
- `signal_pool_active` — `bool(search_metrics.get("signal_pool_enabled"))`
- `graph_expansion_applied` — `len(graph_passage_ids) > 0`

### kb.read_excerpt

Given a `passage_id` from a prior `kb.retrieve_evidence` response, returns the full text of that passage.

**Three-layer fallback:**
1. Exact lookup by `(session_id, passage_id)` in ScratchStore
2. `find_by_section_id(session_id, section_id)` — O(n) fallback if AI passes section_id instead of passage_id
3. Live Qdrant point fetch by UUID5-encoded point ID (most expensive; avoids stale-session failures)

### graph.expand

Given a list of `node_ids` (Neo4j node IDs), performs a neighbor traversal and returns related nodes.

**Current state:** The input schema has `additionalProperties: true` with no formally defined parameters — this is a known technical debt item. The handler performs a Neo4j neighbor traversal using configured relationship types from `search.hybrid.query_type_relationships`.

**Note:** `graph_channel_enabled: false` and `graph_enrichment_enabled: false` in `development.yaml` mean that graph traversal in the primary retrieval path is currently disabled. `graph.expand` operates independently as a follow-up tool.

### Retrieval Traces

**File:** `src/mcp_server/retrieval_trace.py` (~450 lines), class `RetrievalTraceBuilder`

Every `kb.retrieve_evidence` call produces a complete trace file. Follow-up tool calls (`kb.read_excerpt`, `graph.expand`) within the same session are correlated and appended to the same trace.

**8 trace sections:**
1. **QUERY** — original and reformulated query, rewrite metrics
2. **CANDIDATES BY SIGNAL** — per-signal candidate breakdown (currently always "not available" — per-signal breakdown not populated)
3. **SIGNAL POOL** — signal pool slot fills if enabled
4. **RERANKER** — reranker input/output, score delta for each candidate
5. **GRAPH ENRICHMENT** — seeds used, neighbors found, synthetic scores assigned
6. **EVIDENCE PACK** — final quotes with scores, confidence, source attribution
7. **FOLLOW-UP CALLS** — correlated `kb.read_excerpt` / `graph.expand` calls appended here
8. **FULL TEXT APPENDIX** — complete text of each evidence chunk (full, untruncated)

**Output files:**
- `logs/retrieval_traces/<YYYYMMDD_HHMMSS>_<trace_id>.txt` — human-readable
- `logs/retrieval_traces/<YYYYMMDD_HHMMSS>_<trace_id>.json` — machine-readable sidecar

**MCP resources:** `wekadocs://traces/latest`, `wekadocs://traces/<trace_id>`

**Retention:** `MCP_RETRIEVAL_TRACE_RETENTION_HOURS` (default 72 hours)

**Active trace correlation:** `set_active_trace(trace_id)` stores the current trace in a request-scoped context. `_call_tool()` calls `append_followup_and_write()` for non-evidence tool calls, which appends a `TraceFollowup` entry to the active trace.

---

## Part 3: Configuration Reference

### Config Loading Chain

**Entry point:** `load_config()` in `src/shared/config.py`

1. Load `Settings()` from `pydantic_settings.BaseSettings` — reads environment variables
2. Determine config file path: `CONFIG_PATH` env var, or `config/<ENV>.yaml`
3. Parse YAML file into config dataclasses
4. Call `apply_embedding_profile(config)` — reads `config/embedding_profiles.yaml`, looks up the profile from `plan.dense` (e.g., `qwen3_0_6b`), overrides config fields with profile values
5. Validate and return

**Cached singletons:** `get_config()` and `get_settings()` use `@lru_cache` — calling them repeatedly returns the same object. `get_embedding_plan()` and `get_embedding_settings()` similarly cached.

**Config file selection:** `ENV` env var (default: `development`) determines which YAML file loads: `config/development.yaml`.

### Embedding Profiles YAML

**File:** `config/embedding_profiles.yaml`

**Top-level `plan:` section (active model assignments):**
```yaml
plan:
  dense: "qwen3_0_6b"
  sparse: "spladev3"
  colbert: "colbertv2"
  enable_sparse: true
  enable_colbert: true
```

**Production profiles (3 active):**

`qwen3_0_6b`:
- `provider: "embedding-service"`, `model_id: "Qwen/Qwen3-Embedding-0.6B"`, `dims: 1024`
- `query_instruction: "Instruct: Given a technical query about WEKA..."`
- Requires env: `EMBEDDING_BASE_URL`

`spladev3`:
- `provider: "embedding-service"`, `model_id: "naver/splade-v3"`, `dims: 30522`
- `supports_sparse: true`, no dense or ColBERT
- Requires env: `EMBEDDING_BASE_URL`

`colbertv2`:
- `provider: "embedding-service"`, `model_id: "colbert-ai/colbertv2.0"`, `dims: 128`
- `supports_colbert: true`, no dense or sparse
- Requires env: `EMBEDDING_BASE_URL`

**Legacy profiles (defined but not in active plan):** `bge_m3`, `jina_v3`, `voyage_context_3`, `snowflake_arctic_v2l`, `qwen3_4b` (deprecated), `voyage_3_large`, `snowflake_arctic_l_v2_0`, `st_minilm`

**Reranker config in YAML:**
```yaml
reranker:
  provider: "local-reranker-service"
  model_id: "Qwen/Qwen3-Reranker-4B"
  context_window: 8192
  base_url: "http://10.25.0.50:8080"
```

### development.yaml Key Settings

**Hybrid search mode:** `search.hybrid.mode: "legacy"` — RRF + BM25 path (not the bge_reranker vector-only path)

**Currently disabled (important for debugging):**
- `graph_channel_enabled: false` — graph retrieval channel disabled
- `graph_enrichment_enabled: false` — post-retrieval graph expansion disabled (only `_expand_evidence_with_structure` in mcp_app.py is active, not the retriever-level expansion)
- `colbert_rerank_enabled: false` — ColBERT MaxSim pre-reranker disabled

**Currently enabled:**
- `reranker.enabled: true` — cross-encoder enabled
- `expansion.enabled: true` — bounded adjacency expansion
- `graph_adaptive_enabled: true` — query-type specific relationship sets

**BM25 index name:** `chunk_text_index_v3_bge_m3` (the name is historical; the index serves the current model stack)

**Schema version:** `schema.version: "v4.0"` — Chunk-only schema, Section nodes deprecated

### Environment Variables

**Precedence (highest to lowest):**
1. `environment:` block in docker-compose.yml — explicitly set values and `${VAR:-default}` interpolations
2. `env_file: .env.docker` — values loaded into container environment
3. Shell/host environment when running `docker compose` — source for `${VAR}` interpolation in compose

**Key variables and their defaults:**

| Variable | Default in Compose | Purpose |
|---|---|---|
| `EMBEDDINGS_PROFILE` | `bge_m3` | Overridden by embedding_profiles.yaml plan; confusing mismatch (see Tech Debt) |
| `EMBEDDING_BASE_URL` | (must be set) | Base URL for unified GPU gateway |
| `RERANKER_BASE_URL` | `http://10.25.0.50:8080` | Reranker endpoint |
| `RERANKER_TIMEOUT_SECONDS` | `120` | Cross-encoder timeout |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j Bolt connection |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname (Docker service name) |
| `REDIS_URI` | `redis://:${REDIS_PASSWORD}@redis:6379/0` | Redis connection string |
| `MCP_TOOL_PROFILE` | (none — defaults to `"production"` in code) | Tool visibility profile |
| `MCP_SCRATCH_TTL_SECONDS` | `1800` | ScratchStore entry TTL |
| `MCP_RETRIEVAL_TRACE_RETENTION_HOURS` | `72` | Trace file retention |
| `HF_HUB_OFFLINE` | `1` | Prevents HuggingFace downloads in container |
| `BGE_M3_MAX_INPUT_TOKENS` | `8192` | Hard token limit for chunker |
| `BGE_M3_SAFE_INPUT_TOKENS` | `7500` | Practical token limit for chunker |
| `LOG_LEVEL` | `INFO` | Structured log level |

### Feature Flags

Defined in `development.yaml` under `feature_flags:`. All flags must have corresponding fields in `FeatureFlagsConfig` in `src/shared/config.py` (tested by `tests/test_config_schema_completeness.py`).

| Flag | Active Value | Effect |
|---|---|---|
| `query_api_weighted_fusion` | `true` | Enables `_search_via_query_api_weighted()` with per-field score tracking |
| `graph_garbage_filter` | `true` | Filters low-quality graph matches from retrieval |
| `graph_rel_types_wired` | `true` | Query-type specific relationship sets active |
| `dedup_best_score` | `true` | Keeps highest-scoring version of duplicate chunks |
| `graph_score_normalized` | `true` | Normalizes graph scores before fusion |
| `graph_as_reranker` | `true` | Graph-based candidate reordering step active |
| `structure_aware_expansion` | `true` | Structure-aware expansion (sibling, parent, entity) active |
| `signal_diverse_rerank_pool` | `false` (class default, not in YAML) | Signal-diverse pool for reranker input |
| `dual_write_1024d` | `false` | Dual-write to two Qdrant collections (migration) |
| `entity_embedding_fallback` | `false` | Entity embedding fallback path |

---

## Part 4: Deployment

### Docker Services

| Container | Image / Dockerfile | Ports | Role |
|---|---|---|---|
| `weka-mcp-server` | `docker/mcp-server.Dockerfile` | 8000 (via ts sidecar) | MCP server: retrieval tools, FastAPI, StreamableHTTP |
| `weka-ingestion-worker` | `docker/ingestion-worker.Dockerfile` | — | Processes ingestion jobs from Redis queue |
| `weka-ingestion-service` | `docker/ingestion-service.Dockerfile` | 8081 | FileSystemWatcher: enqueues files to Redis |
| `weka-neo4j` | `neo4j:2025.10.1-community` | 7474 (HTTP), 7687 (Bolt) | Graph database |
| `weka-qdrant` | `qdrant/qdrant:v1.16.0` | 6333 (HTTP), 6334 (gRPC) | Vector database |
| `weka-redis` | `redis:7.2-alpine` | 6379 | Queue + cache |
| `weka-ts-mcp-server` | `tailscale/tailscale:latest` | 8000 (host) | Tailscale sidecar for mcp-server; provides MagicDNS |
| `weka-ts-ingestion-worker` | `tailscale/tailscale:latest` | — | Tailscale sidecar for ingestion-worker |
| `weka-alloy` | `grafana/alloy:v1.4.0` | 4317, 4318, 12345 | OTLP telemetry collector; forwards to GCP LGTM |

**Start order (controlled by `depends_on`):**
- `ts-mcp-server` must be healthy before `mcp-server` starts
- `neo4j`, `qdrant`, `redis` must be healthy before `mcp-server` or `ingestion-worker` start
- `redis` must be healthy before `ingestion-service` starts

**Resource limits:**
- Neo4j: 2 CPU, 5 GB memory
- Qdrant: 2 CPU, 3 GB memory
- Redis: 1 CPU, 1 GB memory
- MCP server: 2 CPU, 4 GB memory
- Ingestion worker: 2 CPU, 3.5 GB memory
- Ingestion service: 1 CPU, 1 GB memory

### Health Checks

**`/health` (lightweight):** Always returns 200. Returns current config state:
```json
{
  "status": "ok",
  "embedding_profile": "qwen3_0_6b",
  "embedding_provider": "embedding-service",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "embedding_dims": 1024
}
```

**`/ready` (deep probe):** Probes Neo4j, Qdrant, and Redis connectivity. Returns `{"ready": true}` or `{"ready": false, "failures": [...]}` with HTTP 200/503.

**Startup fail-fast:** `monitoring.health_check_fail_fast: true` causes the server to exit if critical startup checks fail. The critical check is `REQUIRED_SCHEMA_VERSION="v4.0"` — verified against the `SchemaVersion` singleton node in Neo4j. If the schema version does not match, the server refuses to start.

**`/_mcp/health` (ASGI shortcut):** Returns `{"status": "ok"}` or 503 before MCP initialization completes.

**Health checker class:** `src/monitoring/health.py`, class `HealthChecker`. Checks: Neo4j constraints and indexes exist (v4.0 schema), vector indexes are 1024-D with cosine distance, Qdrant collection exists with 1024-D named vectors, `SchemaVersion` marker is `v4.0`, embedding configuration matches.

### GPU Gateway

Single unified endpoint at `http://10.25.0.50:8080`. All 6 models are served behind this gateway.

**Backends (5 distinct API paths):**
- `POST /v1/embeddings` — dense embeddings (Qwen3-Embedding-0.6B)
- `POST /v1/embeddings/sparse` — sparse embeddings (SPLADEv3)
- `POST /v1/embeddings/colbert` — late-interaction embeddings (ColBERTv2)
- `POST /v1/rerank` — cross-encoder reranking (Qwen3-Reranker-4B)
- `POST /v1/extract` — NER extraction (GLiNER)
- `POST /v1/chat/completions` — LLM inference (Qwen2.5-1.5B-Instruct — not yet deployed)

**Access from containers:** MCP server and ingestion worker access the gateway via Tailscale MagicDNS because they share the `ts-*` sidecar network namespace. The gateway is on the Tailnet at `10.25.0.50`.

### Activation Checklist

To bring up a fully functional pipeline from scratch:

1. Ensure GPU gateway is running at `10.25.0.50:8080` with all 5 model endpoints healthy
2. Set required secrets in `.env`: `NEO4J_PASSWORD`, `REDIS_PASSWORD`, `JWT_SECRET`, `TS_AUTHKEY`
3. Set `EMBEDDING_BASE_URL=http://10.25.0.50:8080` in `.env.docker`
4. Set `EMBEDDINGS_PROFILE=qwen3_0_6b` in `.env.docker` (or accept that YAML plan overrides the old default)
5. Run `docker compose up -d` — wait for all services healthy
6. Run `python bootstrap_schema.py` to initialize Neo4j schema v4.0 and the `SchemaVersion` singleton
7. Trigger re-ingestion: drop markdown files into `data/ingest/` and watch `weka-ingestion-service` enqueue them
8. Confirm Qdrant collection `chunks_multi_qwen3_0_6b` has points after ingestion
9. Test retrieval: send an MCP `tools/call` for `kb.retrieve_evidence` with a WEKA-domain question

**Note:** The Qdrant collection `chunks_multi_qwen3_0_6b` is currently empty. Full re-ingestion is required before retrieval produces results. The stale collection `chunks_multi_bge_m3` should not be used.

---

## Appendix

### Data Flow Diagram

```
User question: "How do I configure NFS mount options in WEKA?"
       |
       v
MCP call: kb.retrieve_evidence(question="...", top_k=5)
       |
       v
[query_service.py: search_sections_light()]
  |
  +-- _rewrite_keyword_query(): "configure NFS mount options WEKA"
  |
  +-- _llm_reformulate(): [FALLBACK: heuristic] "NFS mount configuration options"
  |
  +-- HybridRetriever.retrieve(query=reformulated, query_original=original)
       |
       +-- [PARALLEL]
       |    +-- BM25: Neo4j fulltext "configure NFS mount options WEKA" → 50 candidates
       |    +-- Qdrant QueryAPI (content+title+text-sparse+doc_title-sparse+title-sparse+entity-sparse)
       |         → up to 200 candidates per field, merged by Qdrant RRF
       |
       +-- Python RRF fusion (k=60, weighted by rrf_field_weights) → fused_score per chunk
       |
       +-- GLiNER NER on query → ["NFS", "WEKA", "mount"] → entity boost on matching chunks
       |
       +-- Structural boost: CLI query → +20% for code chunks
       |
       +-- [Reranker enabled] Qwen3-Reranker-4B cross-encoder
       |    Input: top ~50 fused candidates
       |    Instruction: "Judge whether the document is relevant to WEKA..."
       |    Output: rerank_score per candidate
       |
       +-- Bounded adjacency expansion (±1 NEXT_CHUNK hop)
       |
       +-- Context budget enforcement (7500 tokens/doc, 8192 total)
       |
       +-- Returns: List[ChunkResult] sorted by rerank_score
       |
[mcp_app.py: _kb_search_candidates()]
  |
  +-- Assign passage_id (uuid4) to each chunk
  +-- ScratchStore.put(session_id, passage_id, 19-field payload)
  |
[mcp_app.py: _expand_evidence_with_structure()]
  |
  +-- Cypher: NEXT_CHUNK + sibling neighbors of top 10 chunks
  +-- Store graph neighbors in ScratchStore (fused_score=0.3, source="graph_expanded")
  |
[mcp_app.py: _extract_evidence_from_passages()]
  |
  +-- Sort all scratch entries by: rerank_score > fused_score > vector_score > bm25_score
  +-- Per passage: find best text span
       span_score = (0.7 × retrieval_score) + (0.3 × keyword_overlap)
  |
  v
Evidence Pack returned to AI client:
{
  "quotes": [
    {
      "rank": 1,
      "quote": "To configure NFS mount options, use weka nfs permission add...",
      "passage_id": "...",
      "section_id": "...",
      "doc_tag": "weka-admin-guide",
      "title": "NFS Mount Configuration",
      "parent_path": "NFS Configuration > Mount Options",
      "uri": "wekadocs://scratch/{session_id}/{passage_id}",
      "confidence": 0.91,
      "source": "retrieval",
      "rank": 1
    },
    ...
  ],
  "coverage": {
    "documents_searched": 12,
    "documents_with_evidence": 3,
    "retrieval_depth": 60,
    "reranker_applied": true,
    "signal_pool_active": false,
    "graph_expansion_applied": true
  }
}
```

### Score Field Survival Table

This table tracks the lifecycle of each score field from `ChunkResult` through to the final evidence quote. This is the most critical reference for any work touching the retrieval → evidence path.

| Score Field | Set By | Stored In ScratchStore | Used In Evidence Sort | Survives to Quote Output |
|---|---|---|---|---|
| `rerank_score` | Qwen3-Reranker-4B cross-encoder | Yes | Yes (priority 1) | As `confidence` field |
| `fused_score` | Python RRF fusion | Yes | Yes (priority 2) | No (internal only) |
| `vector_score` | Qdrant multi-vector search | Yes | Yes (priority 3) | No (internal only) |
| `bm25_score` | Neo4j fulltext BM25 | Yes | Yes (priority 4) | No (internal only) |
| `entity_boost` | GLiNER entity match | Yes | No (applied pre-fusion) | No |
| `structural_boost` | Block-type query match | Yes | No (applied pre-fusion) | No |
| `parent_path_norm` | Structural edge builder | Yes | No | No (but in excerpt) |
| `source` | Set at write time | Yes | No | Yes (`"retrieval"` or `"graph_expanded"`) |

**Critical bug (known, unresolved):** Two score fields are lost in the current implementation:

1. **ScratchStore write** in `_kb_search_candidates()` (`mcp_app.py:464`): only 19 fields are stored. If a `ChunkResult` carries additional per-field scores (e.g., per-signal RRF contributions), they are not stored and cannot be recovered.

2. **Evidence extraction** in `_extract_evidence_from_passages()` (`mcp_app.py:578`): the keyword overlap blending uses `0.3 × keyword_overlap` which can dilute a high reranker score for chunks that happen not to contain the literal query keywords (e.g., synonyms, abbreviations).

### Known Technical Debt

- `EMBEDDINGS_PROFILE` env var in `.env.docker` is still set to `"bge_m3"`. The config YAML plan (`plan.dense: "qwen3_0_6b"`) overrides this at runtime, but the mismatch is confusing and may cause issues if `apply_embedding_profile()` is ever bypassed.

- `max_pairs` and `max_tokens_per_pair` under `search.hybrid.reranker` in `development.yaml` are dead config — they are defined but never referenced by any code in `src/providers/rerank/local_reranker_service.py` or elsewhere. Token limits in the actual reranker are handled internally by the service.

- `signal_diverse_rerank_pool` feature flag is implemented and tested (298-line `src/query/signal_pool.py`, 17 tests) but not activated. The flag is missing from `development.yaml` feature_flags section, defaulting to `false` via the class default in `FeatureFlagsConfig`.

- Qdrant collection `chunks_multi_qwen3_0_6b` is empty. Full re-ingestion with the Qwen3 model stack is required before production retrieval works.

- Per-signal candidate breakdown not populated in retrieval traces. Trace section 2 ("CANDIDATES BY SIGNAL") always shows "not available". The data structure exists but the fill logic in `hybrid_retrieval.py` does not write to it.

- 7 pre-existing test failures in `tests/query/` requiring live infrastructure (Neo4j + Qdrant + GPU gateway). These are environment failures, not code failures.

- Stale BGE-M3 references in docstrings of `src/providers/embeddings/chonkie_adapter.py` and `src/ingestion/semantic_chunker.py`. The code is correct but the docstrings reference the old model name.

- `graph.expand` input schema has `additionalProperties: true` with no formally defined parameters. The AI client has no schema to guide its arguments.

- Qwen2.5-1.5B-Instruct is not deployed to the GPU gateway. Query reformulation always falls back to the heuristic path. The LLM reformulation code path in `query_service.py` is dead until the model is deployed.

- `development.yaml` has an indentation error at line 190: `graph:` under `search:` is outdented, which may cause YAML parsing to treat it as a top-level key rather than a nested key under `search:`. Verify with `yaml.safe_load` before making config changes in that region.
