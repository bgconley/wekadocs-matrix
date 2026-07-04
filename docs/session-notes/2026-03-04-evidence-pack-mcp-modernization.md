# Session Context: Evidence Pack Architecture + MCP Modernization

**Date:** 2026-03-03 to 2026-03-04
**Branch:** `multi-embedder-reranker`
**Predecessor session:** `2026-03-03-signal-pool-provider-cleanup.md`
**Commits on branch before this session:** 8 (signal pool + provider cleanup)
**Author:** Architecture + retrieval quality + deployment session

---

## Session Overview

This session designed, planned, implemented, reviewed, and deployed a comprehensive overhaul of the MCP server's retrieval pipeline and tool surface. The work builds directly on the signal-diverse rerank pool and provider stack cleanup completed in the predecessor session. The central architectural shift: making `kb.retrieve_evidence` a first-class, server-orchestrated evidence path that fully uses the modern retrieval stack, rather than relying on LLM-driven multi-step tool planning.

### Key Outcomes

- Evidence pack now uses retrieval scores (rerank_score, fused_score) instead of keyword overlap for confidence
- Retrieval depth decoupled from output size (search 60-150 candidates, return 6 quotes)
- Always-on graph enrichment adds structural neighbors to every evidence pack
- Tool surface modernized: dot notation, 3-tool production profile, profile-aware instructions
- LLM-based query reformulation with dual-query strategy (reformulated for dense/reranker, original keywords for BM25/sparse)
- Domain-tuned instruction prefixes for both embedding model and reranker
- Full retrieval trace system with human-readable output and CLI chunk inspector
- All 44 tests passing (unit + MCP + live contract), MCP server container healthy and serving

---

## Part 1: Deep Codebase Research and Analysis

### The 8-Step Retrieval Pipeline (from prior session context)

The HybridRetriever in `src/query/hybrid_retrieval.py` implements an 8-step pipeline that the evidence pack now fully leverages:

1. **Embed Query** — Dense via Qwen3-Embedding-0.6B (1024-dim), sparse via SPLADEv3, ColBERT via ColBERTv2 (32 tokens × 128-dim). These are bundled into a `QueryEmbeddingBundle`.
2. **Multi-Vector Search (Qdrant)** — Single `query_points` call with nested Prefetch entries across 7 vector fields: content-dense (200 limit), title-dense (200), doc_title-dense (200), text-sparse (200), doc_title-sparse (50), title-sparse (50), entity-sparse (50). Server-side DBSF fusion.
3. **BM25 Retrieval (Neo4j)** — Parallel fulltext search against Lucene index on Chunk and CitationUnit nodes.
4. **Client-Side RRF Merge** — `score = 1/(k + bm25_rank) + 1/(k + vector_rank)` with k=60.
5. **Entity + Structural Boost** — GLiNER NER on query, EntityExtractor matches against payloads, structural boost by block type.
6. **Reranking** — Qwen3-Reranker-4B cross-encoder, 8K context, P(yes) logit scoring. The signal pool inserts here, feeding 200 signal-diverse candidates.
7. **Expansion** — Bounded adjacency expansion (±1 NEXT_CHUNK from top 5 seeds) + structure-aware expansion (siblings, parents, entity-shared).
8. **Context Assembly** — 14,000 token ceiling, citation hydration, dedup.

### The Data Flow Gap (Core Problem)

The pipeline produces rich `ChunkResult` objects with ~25 score fields. The MCP layer at `mcp_app.py` discards them at two points:

**Loss point 1 — ScratchStore write (`_kb_search_candidates`, line 464-474):**
The scratch payload stored only 6 fields: `section_id`, `doc_tag`, `title`, `text`, `source_uri`, `created_at`. All score fields were dropped. The `score` field in the result dict was a single collapsed value: `rerank_score or fused_score or vector_score or bm25_score or 0.0`.

**Loss point 2 — Evidence extraction (`_extract_evidence_from_passages`, line 563-637):**
After reading from scratch (which had no scores), the function re-scored spans using pure keyword overlap:
```python
hits = sum(1 for t in query_tokens if t in lowered)
score = hits / max(1, len(query_tokens))
```
The `confidence` field in evidence quotes was this keyword hit rate — a number between 0.0 and 1.0 that had zero correlation with the cross-encoder's `rerank_score`.

**Score field survival inventory (discovered via exploration):**

| ChunkResult field | In result dict (default) | In scratch | In evidence quote |
|---|---|---|---|
| `rerank_score` | Collapsed into `score` | NO | NO |
| `fused_score` | Collapsed into `score` (fallback) | NO | NO |
| `vector_score` | Collapsed into `score` (fallback) | NO | NO |
| `bm25_score` | Collapsed into `score` (fallback) | NO | NO |
| `title_vec_score` | NO | NO | NO |
| `entity_vec_score` | NO | NO | NO |
| `parent_path_norm` | NO | NO | NO |
| `rerank_rank` | NO | NO | NO |
| `fusion_method` | `source` label only | NO | NO |

The diagnostic path (`_emit_diagnostics()` via `diagnostic_context["chunks"]`) was the only place where all score fields survived — but that data was emitted out-of-band and never returned to MCP clients.

### MCP Tool Surface Analysis

**Tool registration (`_tool_specs()` at mcp_app.py:2800):**

24 tools registered in a flat list, with 7 exact-duplicate pairs where both `graph_*` prefixed and bare-name versions pointed to the same handler. No filtering mechanism, no profile support. The LLM received all 24 tools in `list_tools`, creating significant tool-selection entropy.

| Category | Tools | Notes |
|---|---|---|
| KB retrieval | `kb_search`, `kb_read_excerpt`, `kb_expand_excerpt`, `kb_extract_evidence`, `kb_retrieve_evidence` | 5 tools for what should be 1-2 operations |
| Graph (prefixed) | `graph_describe`, `graph_expand`, `graph_paths`, `graph_parents`, `graph_children`, `graph_entities_for_sections`, `graph_sections_for_entities` | 7 tools |
| Graph (bare name, duplicates) | `describe_nodes`, `expand_neighbors`, `get_paths_between`, `list_parents`, `list_children`, `get_entities_for_sections`, `get_sections_for_entities` | 7 exact duplicates of above |
| Legacy | `search_sections`, `get_section_text`, `traverse_relationships` | 3 tools |
| Synthesis | `summarize_neighborhood`, `compute_context_bundle` | 2 tools |

**Naming contract drift:**
- Code registered: underscore (`kb_search`)
- Contract tests expected: dot (`kb.search`) at `test_mcp_streamable_contracts.py:122`
- Instructions referenced: dot (`kb.search`) in `GRAPH_FIRST_INSTRUCTIONS`
- API docs (`api-contracts.md`): dot (`kb.*`)

**Dual MCP transports:**
- Streamable HTTP (`/_mcp`): OFF by default (`MCP_HTTP_STREAMABLE_ENABLED=false` at `main.py:66`). Uses `build_mcp_server()` with full 24-tool surface.
- Legacy REST (`/mcp/*`): ON by default (`MCP_HTTP_LEGACY_REST_ENABLED=true` at `main.py:74`). Frozen Phase 1 stub with only 2 broad tools: `search_documentation` and `traverse_relationships`.

**`kb_retrieve_evidence` was v1 quality:**
- Default `top_k=5` (`KB_SEARCH_DEFAULT_TOP_K` at `mcp_app.py:67`)
- Hard cap `KB_SEARCH_MAX_TOP_K=20`
- Keyword overlap scoring
- Thin metadata: quote, passage_id, section_id, title, uri, confidence (keyword hit rate)
- No doc_tag, parent_path, source provenance, retrieval rank, or coverage metadata

**Query rewriting:**
`_rewrite_keyword_query()` in `query_service.py:486` used a single hardcoded template: `f"Explain {query}. How does this work and what is the technical architecture?"`. This was sent unchanged to ALL signals (BM25, dense, ColBERT, reranker). No dual-query strategy, no intent detection, no LLM call.

The well-formed detection logic was reasonable (question words, function-word ratio > 15%, verb presence, word count ≤ 4) but the rewrite template was generic and counterproductive for configuration/troubleshooting queries.

**Qwen3-Embedding instruction prefix — missing entirely:**
`EmbeddingServiceProvider.embed_query()` hardcoded `"Represent this sentence for searching relevant passages: "` (BGE-M3 instruction). Qwen3-Embedding-0.6B supports and benefits from `Instruct:` prefixes but the code had zero support. `embed_query_all()` duplicated the same hardcoded instruction.

**Qwen3-Reranker instruction — not used:**
The reranker sent raw `{"query": query, "documents": [...]}` to `/v1/rerank`. Qwen3-Reranker-4B supports custom system instructions that guide relevance judgment, but this was not leveraged.

### ScratchStore Architecture

`ScratchStore` at `src/mcp_server/scratch_store.py`: in-memory `OrderedDict` keyed by `(session_id, passage_id)`. 30-min TTL (`MCP_SCRATCH_TTL_SECONDS`), 256MB byte budget (`MCP_SCRATCH_MAX_BYTES`). LRU eviction on capacity overflow.

Single global session (`_GLOBAL_SESSION_ID`) because "The MCP SDK's request_context.session object changes between calls, making it unreliable as a dict key." All concurrent users share one scratch namespace.

The `ScratchEntry` dataclass stored: `payload: Dict[str, Any]`, `size_bytes: int`, `created_at: float`, `last_access: float`. The payload was only 6 fields — the enrichment to 17 fields adds ~200 bytes per entry, negligible against the 256MB budget.

### Tool-Call Brittleness (existing code)

The codebase had extensive defensive code for LLM tool-call errors:
- `kb_read_excerpt`: 3-layer fallback for wrong IDs (exact passage_id → section_id search in scratch → direct Neo4j fetch)
- All graph tools: type coercion for MCP client string/int mismatches
- `_summary_for_tool()`: explicit passage_id hints in kb_search responses to guide the LLM

This brittleness motivated the architectural shift to server-orchestrated evidence packs — if the server handles retrieval, the LLM doesn't need to juggle IDs between tool calls.

### Configuration and Feature Flag State

The codebase has an extensive feature flag system in `FeatureFlagsConfig` (config.py:919-986). Flags active in `development.yaml`:

| Flag | Status | Purpose |
|---|---|---|
| `query_api_weighted_fusion` | `true` | Enables per-field score tracking via filtered Qdrant re-scoring queries |
| `graph_garbage_filter` | `true` | Filters low-quality graph results |
| `graph_rel_types_wired` | `true` | Enables typed relationship traversal |
| `dedup_best_score` | `true` | Dedup keeps best-scoring chunk per document |
| `graph_score_normalized` | `true` | Normalizes graph scores for fusion |
| `graph_as_reranker` | `true` | Graph-based candidate reordering |
| `structure_aware_expansion` | `true` | Structure-aware post-rerank expansion |
| `signal_diverse_rerank_pool` | `false` (not in YAML) | Signal pool — ready but not activated |
| `entity_focus_bias` | `false` (not in YAML) | Entity weighting bias |

The `signal_diverse_rerank_pool` flag is the gate for the signal pool implemented in the prior session. It requires both `signal_pool.enabled=True` in config AND `feature_flags.signal_diverse_rerank_pool=True`. The evidence pack improvements work regardless of the signal pool state — they improve the MCP layer's use of whatever the retriever produces.

### Embedding Plan System

The embedding plan in `config/embedding_profiles.yaml` drives provider selection:

```yaml
plan:
  dense: "qwen3_0_6b"      # Qwen3-Embedding-0.6B, 1024-dim
  sparse: "spladev3"        # SPLADE v3, learned sparse
  colbert: "colbertv2"      # ColBERTv2, 128-dim per-token
  enable_sparse: true
  enable_colbert: true
```

At startup, `get_embedding_plan()` resolves each role to a `EmbeddingRolePlan` containing the profile definition. `ProviderFactory.create_embedding_provider_for_role()` creates provider instances from profile settings. When profile names differ between roles, `QdrantMultiVectorRetriever.__init__` creates separate provider instances — the `is` identity check in `_build_query_bundle()` detects this for efficient single-vs-multi-provider embedding.

The Qdrant collection name is derived from the dense profile: `chunks_multi_{profile_name}` → `chunks_multi_qwen3_0_6b`. This is why the health check and retriever need a matching collection.

### Weighted Fusion Path

`query_api_weighted_fusion` (enabled in dev config) does per-field re-scoring of Qdrant candidates. It adds 5-7 filtered Qdrant queries (~40-70ms) but populates `title_vec_score`, `doc_title_vec_score`, `entity_vec_score`, `lexical_vec_score`, `doc_title_sparse_score` on every ChunkResult. Without this, the signal pool degrades to coarse BM25/vector provenance. The evidence pack benefits from this because richer score fields flow through to the trace system.

---

## Part 2: Design Decisions and Rationale

### The Signal Collapse Problem and How the Evidence Pack Solves It

The signal-diverse rerank pool (implemented in the prior session) addresses signal collapse at the retrieval level — ensuring the cross-encoder sees candidates from every signal source, not just the top-N by fused score. But the evidence pack was collapsing those signals again at the MCP layer by (a) using only 5 candidates and (b) re-scoring with keyword overlap.

The evidence pack v2 preserves the retriever's signal diversity through to the final output:

1. **Deep retrieval (60-150 candidates)** — gives the signal pool's diverse candidates a path through to evidence extraction instead of being truncated at 5
2. **Retrieval-score-based ranking** — the cross-encoder's judgment (`rerank_score`) becomes the primary quality signal, not keyword overlap
3. **Always-on graph enrichment** — structural neighbors enter the evidence candidate pool with their own scores, competing fairly in the blended ranking
4. **Coverage metadata** — the LLM (and operators) can see exactly what the retrieval pipeline did: how many documents searched, how many contributed evidence, whether the reranker and signal pool were active, whether graph expansion ran

The net effect: the evidence pack is no longer a black box. The LLM receives quotes with meaningful confidence scores, structural context (parent_path), signal provenance, and coverage information. This fundamentally changes the LLM's ability to assess whether the evidence is sufficient for answering the question or whether it should use `graph.expand` for follow-up navigation.

### Why Server-Orchestrated Evidence (Not LLM Tool Planning)

The LLM's job is answer synthesis, not retrieval orchestration. With 24 tools and graph-first instructions, the LLM spent tokens on multi-step retrieval planning (search → graph explore → get text → maybe search again). Timeouts were common. The evidence pack moves all retrieval logic server-side where it can use the full pipeline (signal pool, cross-encoder, graph expansion) in a single call.

### Why Always-On Graph Enrichment (Not Coverage-Gated)

Initial framing: "run graph expansion when coverage is low." User pushed back: graph structure is always valuable because it provides positional context (where something is in a document) that vector similarity can't. A chunk with `rerank_score=0.95` still benefits from knowing it's Step 3 of 7 in a procedure. Making expansion always-on aligns with how the retriever already works — `_bounded_expansion` always runs.

### Why 3 Production Tools (Not 1 or 4)

Three distinct intents in a QA session: (1) "Answer my question" → `kb.retrieve_evidence`, (2) "Show me more of that passage" → `kb.read_excerpt`, (3) "What else is near X?" → `graph.expand`. Each serves a different intent with minimal planning burden. `kb.search` was excluded from production because it serves the same intent as `kb.retrieve_evidence` with less value.

### Why Dual-Query Strategy

Dense embeddings and cross-encoders work best with natural language questions. BM25 and SPLADE work best with raw keywords. Sending one query form to all signals is suboptimal. The dual-query strategy sends the reformulated question to semantic signals and original keywords to lexical signals.

### Why Qwen2.5-1.5B-Instruct for Reformulation

The 3090 has ~11 GB headroom. 1.5B params ≈ 3 GB VRAM in FP16. The 0.5B model was rejected because it's inconsistent at following "rewrite only, don't answer" — tends to hallucinate extra content. 1.5B follows instructions reliably for this simple task. Same Qwen family for stack consistency.

### Why Domain-Tuned Reranker Instruction

The reranker instruction directly addresses the intro-dominance problem: "Introductory overviews and general summaries are less relevant than specific technical procedures and settings." This tells the 4B-parameter cross-encoder what "relevant" means for WEKA docs. Zero latency, zero VRAM — just a string prepended to the query.

---

## Part 3: Implementation Details (Batches 1-4)

### Batch 1: Core Pipeline Fixes (Phases 1, 3d-e, 4a-b)

**Phase 1a — ScratchStore payload enrichment:**
Added 11 fields to the scratch payload in `_kb_search_candidates()`: `rerank_score`, `fused_score`, `vector_score`, `bm25_score`, `graph_score`, `parent_path_norm`, `rerank_rank`, `fusion_method`, `is_expanded`, `expansion_source`, `source`. The `source` field was already computed (line 448-460) for the result dict but not stored in scratch.

**Phase 1b — Retrieval depth decoupling:**
New constants: `KB_EVIDENCE_INTERNAL_FETCH_K=60`, `KB_EVIDENCE_MAX_FETCH_K=150`. Added `retrieval_depth` parameter to `kb_retrieve_evidence`. The call to `_kb_search_candidates` uses `_fetch_k_override` to bypass the `KB_SEARCH_MAX_TOP_K=20` cap. Evidence options override `max_per_doc=5` (was 1) to allow depth within documents.

**Phase 1c — Two-stage evidence ranking:**
Stage 1: passages sorted by `rerank_score` (fallback chain: fused → vector → bm25 → 0.0). Stage 2: within each passage, best span selected by blended score: `(0.7 × retrieval_score) + (0.3 × keyword_overlap)`. The 30% lexical weight preserves keyword sensitivity for highlighting the specific terms the user asked about.

**Phase 1d — Enriched response:**
Quote schema: `doc_tag`, `parent_path` (from `parent_path_norm`), `source` (signal provenance), `rank` (retrieval position). Coverage block: `documents_searched`, `documents_with_evidence`, `retrieval_depth`, `reranker_applied`, `signal_pool_active`, `graph_expansion_applied`.

**Phase 3d — Embedding instruction:**
Added `query_instruction` field to `EmbeddingProfileDefinition` (Pydantic BaseModel in `config.py`). The YAML value flows through: `embedding_profiles.yaml` → config parse → `_build_settings_from_profile()` in factory (via `getattr(profile, "query_instruction", None)` → `settings.extra["query_instruction"]`) → `EmbeddingServiceProvider.embed_query()` reads from `self._settings.extra.get("query_instruction")`.

**Phase 3e — Reranker instruction:**
Added `instruction` field to `RerankerConfig`. Factory reads it via `getattr(reranker_cfg, "instruction", None)` and passes as kwargs. `LocalRerankerServiceProvider.__init__` stores `self._instruction`. In `rerank()`, prepended before token counting: `query = f"{self._instruction}\n\n{query}"`.

**Phase 4a-b — Tool naming:**
17 canonical dot-notation tools + 17 underscore backward aliases. Aliases generated by iterating canonical specs and appending copies with `[Deprecated: use {canonical_name}]` in descriptions. `_summary_for_tool()` updated to match both name forms.

**Code Review Round 1 (after Batch 1) — 3 findings, all fixed:**

1. (High) `query_instruction` was a no-op. The full chain was traced: YAML had the field → but `EmbeddingProfileDefinition` (a Pydantic `BaseModel` without `extra="allow"`) silently dropped it during parsing → factory's `getattr(profile, "query_instruction", None)` returned `None` → `settings.extra["query_instruction"]` was never populated → provider fell back to BGE-M3 default instruction. Fixed by adding `query_instruction: Optional[str] = Field(default=None)` to `EmbeddingProfileDefinition`. Verified end-to-end: YAML → config parse → factory → `settings.extra` → provider `embed_query()`.

2. (Medium) `top_k` backward compat claimed in schema but not implemented. The input schema described `top_k` as "backward compat alias for max_quotes" but the code never mapped it. Callers using `top_k=8` got `max_quotes=6` (the default). Fixed: `if top_k != KB_SEARCH_DEFAULT_TOP_K and max_quotes == 6: max_quotes = top_k`. Verified: top_k=9/max_quotes=default → 9 quotes; top_k=9/max_quotes=4 → 4 quotes; defaults → 6 quotes.

3. (Low) `traverse_relationships` touched despite plan saying "do not touch." The tool was renamed to `graph.traverse` with `traverse_relationships` as backward alias. Functionality unchanged, old name still works. Acknowledged as scope deviation.

### Batch 2: Enrichment + Reformulation (Phases 2, 3b-c, 4c)

**Phase 2 — Graph enrichment Cypher:**
```cypher
UNWIND $ids AS sid
MATCH (c:Chunk {id: sid})
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(nxt:Chunk)
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
OPTIONAL MATCH (sib:Chunk {parent_section_id: c.parent_section_id})
  WHERE sib.id <> c.id
WITH c, nxt, prev, collect(DISTINCT sib)[..3] AS sibs
UNWIND (...neighbors...) AS neighbor
WHERE neighbor.id NOT IN $ids
RETURN DISTINCT neighbor.id, neighbor.heading, neighbor.text,
       neighbor.doc_tag, neighbor.parent_path_norm, neighbor.parent_section_id
LIMIT 20
```
One round-trip, ~5-15ms for 10 seeds. Stores neighbors in scratch with `fused_score=0.3` (synthetic, below typical reranked scores), `source="graph_expanded"`.

**Phase 3b — LLM reformulation:**
`_llm_reformulate()` uses sync `httpx.Client` (consistent with existing sync Neo4j/Qdrant calls in the retrieval pipeline). System prompt: "You are a query reformulator for a technical documentation search system about WEKA... Output ONLY the rewritten question, nothing else." Timeout: 5s. Fallback to `_heuristic_reformulate()` on any exception.

**Phase 3c — Dual-query threading:**
`query_original` parameter added to: `HybridRetriever.retrieve()`, `QdrantMultiVectorRetriever.search()`, `_search_legacy()`, `_build_query_vectors()`, `_build_query_bundle()`. BM25 at line 3106 uses `lexical_query`. Sparse embedding in `_build_query_bundle` uses `sparse_query = lexical_query or query`. Legacy fallback path: `_build_query_vectors()` passes `sparse_q` to `_build_sparse_query()`.

**Code Review Round 2 (after Batch 2) — 2 caveats:**

1. (Caveat) Profile filtering is list-only, not execution-enforced. `call_tool` uses `full_tool_map` (all tools callable), while `list_tools` is profile-filtered. Assessed as intentional: the profile controls what the LLM discovers via `list_tools`, not what's callable. The LLM can't spontaneously call tools it doesn't know about. A strict enforcement mode (`MCP_TOOL_PROFILE_STRICT`) was discussed but deferred — it would break backward-compat clients using cached underscore names.

2. (Caveat) Sparse lexical routing only applied in Query API path, not legacy fallback. Initial assessment ("inconsequential because legacy is dense-only") was challenged and found to be wrong — the legacy `_search_legacy()` path does execute sparse embedding via `_build_sparse_query()`. When Query API fails and falls back to legacy, sparse embeddings would use the reformulated query instead of original keywords. Fixed by threading `lexical_query` through `_search_legacy()` → `_build_query_vectors()` → `_build_sparse_query()`.

**Code Review Round 3 (after Batch 3):** Confirmed all implementations. 39-test suite passing. CI guard clean.

### Batch 3: Instructions + Transport + Tests (Phases 4d, 5, 6)

`MCP_TOOL_PROFILE` definition moved to module-level constants (line ~83) before the instructions block to avoid `NameError` at import time. `PRODUCTION_INSTRUCTIONS` and `ANALYST_INSTRUCTIONS` replace `GRAPH_FIRST_INSTRUCTIONS` and `VECTOR_ONLY_INSTRUCTIONS`.

Test directory renamed from `tests/mcp/` to `tests/mcp_server_tests/` — the original name shadowed the `mcp` SDK package when pytest added `tests/` to `sys.path`.

### Batch 4: Retrieval Traces (Phase 7)

**Trace format (8 sections):**
1. QUERY — client query, reformulated, method, latency, dual-query status
2. CANDIDATES BY SIGNAL — top 5 per signal with 150-char previews
3. SIGNAL POOL — slot fill table, degraded flag
4. RERANKER — model, instruction, input/output counts, latency, top 10 with rank movement
5. GRAPH ENRICHMENT — seeds, neighbors added
6. EVIDENCE PACK — quotes, coverage metadata
7. FOLLOW-UP CALLS — appended as graph.expand/kb.read_excerpt are called
8. FULL TEXT APPENDIX — complete text of top 20 reranked candidates

Follow-up correlation: `_call_tool` dispatcher appends to active trace for any non-evidence tool call. Multi-source session resolution handles the edge case where `kb.retrieve_evidence` used an explicit `session_id` but the follow-up omits it.

---

## Part 4: Deployment and Integration Testing

### Infrastructure State

- **GPU gateway:** `10.25.0.50:8080` — all 5 endpoints verified (dense=1024-dim, sparse=25 terms, ColBERT=32×128, reranker scores, NER entities)
- **Neo4j:** 262 Documents, 3944 Chunks, SchemaVersion v4.0 — data intact
- **Qdrant:** `chunks_multi_qwen3_0_6b` created (1024 dense, 128 ColBERT, 4 sparse) — empty, awaiting ingestion
- **Redis:** healthy (PONG via docker exec)

### Deployment Config Changes

**`.env.docker` and `.env.local`:** Replaced all per-service URLs with unified gateway (`10.25.0.50:8080`). `RERANK_PROVIDER=local-reranker-service` (was `bge-reranker-service`). `RERANK_MODEL=Qwen/Qwen3-Reranker-4B` (was `Qwen3-Reranker-0.6B`). Added `MCP_TOOL_PROFILE=production`. Legacy `BGE_M3_API_URL` kept pointing to gateway for backward compat.

**`docker/mcp-server.Dockerfile`:** Tokenizer prefetch overhauled. Required models (Qwen3-0.6B, Qwen3-4B, bge-m3) fail the build if download fails. Optional models (SPLADEv3 — gated repo; ColBERTv2 — no HF tokenizer) fall back to `bert-base-uncased`. Build with `HF_HUB_OFFLINE=0`.

### Integration Bug Chain (5 bugs in dependency order)

**Bug 1 — Schema version mismatch (health check):**
`health.py:82` had `REQUIRED_SCHEMA_VERSION = "v2.2"`, Neo4j had `v4.0`. Container boot-looped with "Health check failed (fail_fast): schema_version". Fixed to `"v4.0"`.

**Bug 2 — Missing Qdrant collection:**
After schema version fix, health check failed on `qdrant_collection`: `chunks_multi_qwen3_0_6b` didn't exist. Created via Qdrant API with correct dims (1024 dense, 128 ColBERT).

**Bug 3 — Schema version mismatch (retriever init):**
`config/development.yaml:347` had `version: "v2.2"`. The retriever init at `schema.py:419` raises `SchemaVersion mismatch: expected v2.2, got v4.0`. A *different* check from bug 1. Fixed to `"v4.0"`.

**Bug 4 — Sparse embedding format mismatch:**
The unified gateway's `/v1/embeddings/sparse` returns `data[i]["embedding"]` as `[{"index": int, "value": float}, ...]`. The client at `embedding_client.py:89` expected `data[i]["indices"]` and `data[i]["values"]` (the old BGE-M3 format). KeyError: 'indices' on every sparse query. Fixed with dual-format parsing.

**Bug 5 — Jina Segmenter fallback chain:**
Root cause chain: `./hf-cache` host bind mount masks container's built-in `/opt/hf-cache` → Qwen3 tokenizer not found → HF tokenizer load fails → `allow_segmenter_fallback=True` for `qwen3_0_6b` (not in deny list) → falls back to `api.jina.ai/v1/segment` → Jina API returns HTTP 400 ("Invalid tokenizer(3) xlm-roberta-base"). Three fixes: (a) added `qwen3_0_6b`/`qwen3_4b` to segmenter deny list in `tokenizer_service.py`, (b) downloaded Qwen3 tokenizers to host `./hf-cache/`, (c) changed structlog-style logger kwargs to f-strings (the actual error was masked by a secondary `Logger._log() got an unexpected keyword argument 'error'`).

### The Docker Volume Masking Problem

A key lesson from this session: the docker-compose.yml bind-mounts `./hf-cache:/opt/hf-cache`, which means the host's `hf-cache` directory masks whatever the Dockerfile builds into `/opt/hf-cache`. The Dockerfile's tokenizer prefetch step downloads models into the image's filesystem, but at runtime the bind mount replaces that filesystem with the host's (possibly stale) cache.

This means: (a) the Dockerfile's prefetch is only useful for non-bind-mount deployments, and (b) for local Docker Compose development, the host's `./hf-cache` must contain all required tokenizers. We solved this by downloading `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-4B` directly to the host cache.

The SPLADEv3 and ColBERTv2 tokenizers are gated/missing on HuggingFace — they use BERT-based tokenizers. The `bert-base-uncased` tokenizer (already cached from other models) serves as a functional stand-in for token counting purposes, though it doesn't produce exact token counts matching the actual model vocabularies. This is acceptable for the token budget checks in the retrieval pipeline (approximate counts are sufficient for batch sizing and truncation decisions).

### The Structlog vs stdlib Logger Incompatibility

The MCP server codebase uses a mix of structlog-style and stdlib-style logging. Some modules import `get_logger(__name__)` from `src.shared.observability` which returns a structlog logger (accepts kwargs like `logger.info("event", key=value)`). Other modules use `logging.getLogger(__name__)` which returns a stdlib logger (only accepts `logger.info("message")` with no kwargs).

The `mcp_app.py` module uses stdlib `logging.getLogger`, but existing code in the file used both styles (some with kwargs, some with f-strings) because the structlog processor chain was configured to handle both in some environments but not others. The Docker container's logging configuration routes to stdlib, so kwargs-style calls like `logger.warning("msg", error=str(exc))` raise `TypeError: _log() got an unexpected keyword argument 'error'`.

Our new code used kwargs style (following the pattern of other log calls in the same file). The fix was to switch all new log calls to f-string format, which works reliably with both structlog and stdlib backends.

### Contract Test Modernization

The contract tests assumed a fixed tool surface. Modernized for profile-awareness:

`_infer_server_profile(tool_names)` — determines profile from the server's actual tool list (not local env vars). Production: small set without `kb.search`. Analyst: all `kb.*` and `graph.*`. Full: includes underscore aliases.

`test_streamable_kb_search_contract` — tests `kb.retrieve_evidence` under production (quotes + coverage), `kb.search` under analyst. Added `isError` guard that surfaces infra failures with clear error text instead of cryptic `payload.get("quotes") is None`.

`test_streamable_stdio_schema_parity` — infers HTTP profile, force-sets on STDIO subprocess via `env["MCP_TOOL_PROFILE"] = inferred_profile`. `_list_stdio_tools` now accepts `env` parameter.

---

## Part 5: Complete File Inventory

### New Files

| File | Lines | Purpose |
|---|---|---|
| `src/mcp_server/retrieval_trace.py` | ~450 | Trace builder + file writer + formatting |
| `src/tools/inspect_chunk.py` | ~100 | CLI Qdrant chunk inspector |
| `tests/mcp_server_tests/__init__.py` | 0 | Package init |
| `tests/mcp_server_tests/test_evidence_pack.py` | ~190 | 7 evidence pack tests |
| `tests/mcp_server_tests/test_tool_profiles.py` | ~100 | 12 tool profile tests |
| `docs/plans/2026-03-03-evidence-pack-architecture-mcp-modernization-plan.md` | ~660 | Implementation plan |

### Modified Files (17)

| File | Key Changes |
|---|---|
| `src/mcp_server/mcp_app.py` | Evidence extraction rewrite, scratch enrichment (11 fields), tool naming (dot + aliases), profiles, instructions, graph enrichment, trace integration, coverage metadata, follow-up correlation |
| `src/mcp_server/query_service.py` | `_llm_reformulate()`, `_heuristic_reformulate()`, dual-query in `search_sections_light()` |
| `src/query/hybrid_retrieval.py` | `query_original` in `retrieve()`, `lexical_query` routing in BM25/sparse/legacy paths |
| `src/providers/embeddings/embedding_service.py` | Profile-aware `query_instruction` in `embed_query()` and `embed_query_all()` |
| `src/providers/rerank/local_reranker_service.py` | `instruction` parameter, prepended to query |
| `src/providers/factory.py` | `query_instruction` + `instruction` passthrough |
| `src/shared/config.py` | `query_instruction` on `EmbeddingProfileDefinition`, `instruction` on `RerankerConfig` |
| `src/clients/embedding_client.py` | Dual-format sparse embedding parsing (legacy + gateway) |
| `src/providers/tokenizer_service.py` | Qwen3 profiles in segmenter fallback deny list |
| `src/monitoring/health.py` | `REQUIRED_SCHEMA_VERSION = "v4.0"`, docstrings |
| `src/mcp_server/main.py` | Transport defaults flipped, legacy deprecation warning |
| `config/embedding_profiles.yaml` | `query_instruction` on `qwen3_0_6b` profile |
| `config/development.yaml` | `schema.version: "v4.0"`, reranker instruction |
| `docker/mcp-server.Dockerfile` | Tokenizer prefetch overhaul (required/optional, BERT fallback) |
| `.env.docker` | Unified gateway URLs, tool profile, removed stale provider refs |
| `.env.local` | Unified gateway URLs, tool profile |
| `tests/contracts/test_mcp_streamable_contracts.py` | Profile inference, isError guards, STDIO env passthrough |

---

## Part 6: Model Stack Reference

| Model | Role | Context | VRAM | Endpoint | Instruction |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | Dense embedding | 32K | ~1.5 GB | `/v1/embeddings` | Domain-tuned `Instruct:` prefix |
| SPLADEv3 | Learned sparse | 512 | ~0.5 GB | `/v1/embeddings/sparse` | None |
| ColBERTv2 | Late interaction | 512 | ~0.5 GB | `/v1/embeddings/colbert` | None |
| Qwen3-Reranker-4B | Cross-encoder | 8K | ~8 GB | `/v1/rerank` | Domain-tuned relevance instruction |
| GLiNER Medium v2.1 | NER | — | ~1 GB | `/v1/extract` | None |
| Qwen2.5-1.5B-Instruct | Query reformulation | 32K | ~3 GB | `/v1/chat/completions` | System prompt |

Total projected VRAM: ~14.5 GB / 24 GB (60%). All served through unified gateway at `10.25.0.50:8080`.

---

## Part 7: Test Results

```
Signal pool:               14/14 passed
Reranker integration:       6/6 passed
Evidence pack:              7/7 passed
Tool profiles:             12/12 passed
Contract tests (live):      5/5 passed
────────────────────────────────────────
Total:                     44 passed, 0 failed, 0 skipped
CI dead import guard:      OK (1 allowlisted)
```

---

## Part 8: What's Next

### Ready for Production (feature-flagged or always-on)

- **Evidence pack v2** — always-on: retrieval-score-based confidence, enriched quotes, coverage metadata
- **parent_path_norm enrichment** — always-on: structural breadcrumbs in reranker input (from prior session)
- **Graph enrichment** — always-on: NEXT_CHUNK + sibling expansion in evidence pack
- **Tool profiles** — controlled by `MCP_TOOL_PROFILE` env var
- **Dot notation** — active, underscore aliases for backward compat
- **Retrieval traces** — always-on, files in `logs/retrieval_traces/`
- **Domain-tuned instructions** — embedding + reranker
- **Transport defaults** — streamable ON, legacy OFF

### Requires External Activation

- **Qwen2.5-1.5B-Instruct** — deploy to unified gateway for LLM query reformulation. Until then, keyword-soup queries fall back to intent-aware heuristic templates (config/error/procedure/default).
- **Signal pool** — `signal_pool.enabled=True` + `feature_flags.signal_diverse_rerank_pool=True` + `feature_flags.query_api_weighted_fusion=True`

### Requires Re-ingestion

The `chunks_multi_qwen3_0_6b` Qdrant collection is created with the correct schema but contains 0 points. A full re-ingestion is required because:

1. **SPLADEv3 vocabulary incompatibility** — SPLADE uses a BERT-based tokenizer (~30K vocabulary). Sparse vector indices are token IDs. BGE-M3 used a different vocabulary (XLM-RoBERTa, ~250K vocab). You cannot search a SPLADEv3 index with BGE-M3 sparse vectors — the token IDs map to completely different words.

2. **ColBERTv2 dimensionality change** — ColBERTv2 produces 128-dimensional per-token vectors (BERT-based). BGE-M3 produced 1024-dimensional ColBERT vectors. The Qdrant `late-interaction` field is configured with `size=128` in the new collection. Existing data from BGE-M3 (1024-dim) is dimensionally incompatible.

3. **Dense embedding space change** — Qwen3-Embedding-0.6B and BGE-M3 produce different embedding spaces despite both outputting 1024-dimensional vectors. Cosine similarity between vectors from different models is unreliable. All documents must be re-embedded in the same space.

4. **Query bundle decoupling** — The new `_build_query_bundle()` creates separate sparse embeddings (via SPLADEv3) and ColBERT embeddings (via ColBERTv2) from different providers. The ingestion pipeline must produce matching vectors using the same provider stack.

The existing `chunks_multi_bge_m3` collection (3944 chunks from 262 documents) remains intact and usable as a fallback — the `EMBEDDINGS_PROFILE` env var can point queries back to it if needed.

### Activation Checklist

To enable the full evidence pack + signal pool stack in production:

1. Deploy Qwen2.5-1.5B-Instruct to the unified gateway (adds `/v1/chat/completions` endpoint). Until deployed, query reformulation gracefully falls back to intent-aware heuristic templates.
2. Run full document ingestion against the `chunks_multi_qwen3_0_6b` collection using the new model stack (Qwen3-0.6B dense + SPLADEv3 sparse + ColBERTv2 ColBERT).
3. Enable signal pool: set `signal_pool.enabled=True` + `feature_flags.signal_diverse_rerank_pool=True` + `feature_flags.query_api_weighted_fusion=True` in config.
4. Verify via retrieval traces: check `logs/retrieval_traces/` for full pipeline activity.
5. Test reranker instruction impact: compare evidence quality with and without the domain-tuned instruction (A/B via config toggle).
6. Monitor: `coverage.documents_searched`, `coverage.reranker_applied`, `coverage.signal_pool_active` in evidence pack responses.

### Architecture Considerations for x86_64 Migration

The current deployment runs on Apple Silicon (arm64) for local development, with the GPU gateway on an x86_64 Linux box (RTX 3090). For production deployment:

- The MCP server Dockerfile uses `python:3.11-slim` which is multi-arch — builds for both arm64 and amd64 without changes.
- The unified gateway at `10.25.0.50:8080` is already x86_64 Linux. No migration needed for the model serving side.
- The Neo4j, Qdrant, and Redis containers all have official multi-arch images.
- The HuggingFace tokenizer cache (`./hf-cache`) is platform-independent (Python pickle files).
- The `httpx` client used for LLM reformulation and reranker calls is platform-independent.
- The only potential issue: any compiled Python extensions (e.g., `sentencepiece` for tokenizers) need matching architecture binaries. The `python:3.11-slim` base image handles this via pip's platform wheels.

### Remaining Technical Debt

- `EMBEDDINGS_PROFILE` env var in `.env.docker`/`.env.local` is still `bge_m3` — the config YAML plan overrides this, but it's confusing
- `max_pairs` and `max_tokens_per_pair` in `RerankerConfig` are dead config (never referenced)
- `compute_context_bundle`/`summarize_neighborhood` internal `_finalize_payload` calls still use old tool name strings
- Contract test `test_streamable_kb_search_contract` tests against empty Qdrant collection — passes but doesn't validate full pipeline with real data
- 7 pre-existing test failures in `tests/query/` remain (require live infrastructure)
- Stale BGE-M3 references in docstrings of `chonkie_adapter.py` and `semantic_chunker.py`

### Codebase Pruning (Phases C-F from prior session)

- Phase C: Pull forward active logic from `build_graph.py` → `embedding_context.py`
- Phase D: Delete 25 dead files + 3 dead directories
- Phase E: Remove ~3800 lines of dead methods from partially-active files
- Phase F: Retire 20 test files importing DEAD modules
- Plan doc: `docs/plans/2026-03-01-codebase-pruning-and-modernization-plan.md`
