# Session Context: Signal-Diverse Rerank Pool + Provider Stack Cleanup

**Date:** 2026-03-03
**Branch:** `multi-embedder-reranker`
**Commits pushed:** 8 total (5 Phase A/B pruning + 3 signal pool/provider)
**Author:** Architecture + retrieval quality session

---

## Session Overview

This session covered two major workstreams across a single extended conversation:

1. **Codebase pruning Phase A/B completion and review** — Annotating all modules with @status headers, cleaning phantom init.py chains, resolving code review findings, and preparing the codebase for future dead-code pruning.

2. **Retrieval pipeline improvement** — Analyzing the reranking architecture, identifying the "signal collapse" problem in the candidate pool, designing and implementing a signal-diverse rerank pool, and modernizing the provider configuration stack for the new model lineup.

---

## Part 1: Codebase Pruning (Phases A, B, B.6)

### Starting State

A prior architecture review had identified 21 dead modules, 14 phantom-loaded modules, and ~3800 lines of dead methods. Phase A (annotation) and Phase B (init.py cleanup) had been executed in a previous session. This session began by reviewing that work and incorporating external code review feedback.

### Code Review Findings (5 items validated against codebase)

An external review of the Phase A/B work identified 5 findings. We verified each against the actual codebase:

| # | Finding | Verdict |
|---|---|---|
| 1 | Plan document is stale (still says "PLANNING") | Valid — updated plan status, sequencing, counts |
| 2 | One phantom load remains (shadow_comparison.py via parsers/__init__.py:28) | Valid — resolved in B.6.1 |
| 3 | STANDALONE CLI depends on DEAD module (cli.py → progress.py) | Valid — user chose to keep CLI; progress.py reclassified to STANDALONE |
| 4 | 19 test files import from DEAD modules | Valid (corrected to 20 after B.6 reclassifications) — documented in plan Phase F |
| 5 | Migration script depends on dead symbols | Valid — script marked RETIRED after analysis confirmed all provisions are subsumed by active pipeline |

The reviewer's claim about "20 test files" was initially contested at 15, then re-examined and confirmed at 19 (later updated to 20 after reclassifications). A package-root import at `tests/p3_t4_integration_test.py:10` (`from src.ingestion import ingest_document`) was initially denied but verified to exist — a grep filter error had caused the miss.

### Phase B.6 Implementation

Five prerequisite items were resolved before declaring Phase A/B complete:

**B.6.1:** Removed the eager `ShadowModeError` import from `parsers/__init__.py:28`. The import was already present lazily inside `_parse_with_shadow_comparison()`, making the top-level import redundant. `shadow_comparison.py` was reclassified from PHANTOM to DORMANT (config-gated, analogous to `ner_gliner.py`).

**B.6.2:** User decided to keep the `ingestctl` CLI tool and develop it further. `progress.py` was reclassified from DEAD to STANDALONE (dependency of cli.py).

**B.6.3:** Migration script (`scripts/neo4j_structural_migration.py`) was analyzed against the active codebase. All provisions (parent_path_norm, parent_chunk_id, structural edges, entity normalization, contract checks) are now handled by `structural_edges.py` + `atomic.py` during regular ingestion. The 4 indexes the script created are monitored by `health.py`. Script was marked `@status: RETIRED` with a runtime guard that exits with an explanatory message unless `--force-retired-script` is passed.

**B.6.4:** Created `scripts/ci/check_dead_imports.py` — a CI guard that detects ACTIVE modules importing DEAD modules. Includes stale allowlist enforcement (fails if allowlist entries don't correspond to real violations). One entry allowlisted: `build_graph.py` → `reconcile.py` (dead import inside dead method, cleaned in Phase E).

**B.6.5:** Phase F test scope updated from the original 7 files to 20, with each file's dead import target and recommended action documented.

### Final Annotation Counts After B.6

| Status | Count | Description |
|---|---|---|
| ACTIVE | 110 | Module headers + entry point markers |
| DEAD | 92 | 34 module headers + 58 method/class-level in MIXED files |
| DORMANT | 3 | markdown.py, ner_gliner.py, shadow_comparison.py |
| STANDALONE | 3 | stdio_server.py, cli.py, progress.py |
| MIXED | 4 | build_graph.py, saga.py, contract_checks.py, connections.py |
| TEST_ONLY | 1 | templates/advanced/schemas.py |
| PHANTOM | 0 | All resolved |
| **Total** | **213** | Across all src/ Python files |

---

## Part 2: Retrieval Pipeline Analysis

### How Reranking Works: The 8-Step Pipeline

The system reranks **individual chunks**, not full documents. The HybridRetriever in `src/query/hybrid_retrieval.py` implements an 8-step pipeline:

**Step 1 — Embed Query:** The query is embedded via the embedding plan's dense provider (Qwen3-Embedding-0.6B, 1024-dim), sparse provider (SPLADEv3, learned term weights), and ColBERT provider (ColBERTv2, 32 tokens x 128-dim per token). These three embedding types are bundled into a `QueryEmbeddingBundle`.

**Step 2 — Multi-Vector Search (Qdrant):** A single Qdrant `query_points` call with nested Prefetch entries fans out to 7 vector fields simultaneously. Each field retrieves up to 200 candidates independently: content-dense (semantic body match), title-dense (heading semantic match), doc_title-dense (document title match), text-sparse (BM25-like lexical match on body), title-sparse (lexical match on headings), entity-sparse (entity name lexical match), and late-interaction (ColBERT token-level matching). Server-side Distribution-Based Score Fusion (DBSF) collapses these into a single ranked list.

**Step 3 — BM25 Retrieval (Neo4j):** A parallel fulltext search runs against Neo4j's Lucene index (`chunk_text_index_v3`) on Chunk and CitationUnit nodes. CitationUnit hits get a 1.25x boost. Results include `bm25_score` and `bm25_rank`.

**Step 4 — Client-Side RRF Merge:** The Qdrant results and BM25 results are merged via Reciprocal Rank Fusion: `score = 1/(k + bm25_rank) + 1/(k + vector_rank)` with k=60. Per-field scores from Step 2 survive on the ChunkResult objects (if weighted fusion was enabled).

**Step 5 — Entity + Structural Boost:** QueryDisambiguator runs GLiNER NER on the query to identify entities. EntityExtractor matches entities against Qdrant payloads. Structural boost (from `structural_retrieval.py`) adjusts scores based on block type (code blocks boosted for CLI queries, tables boosted for reference queries).

**Step 6 — Reranking:** [THIS IS WHERE OUR SIGNAL POOL INSERTS] The reranker receives candidates and scores each (query, chunk_text) pair using cross-attention. The Qwen3-Reranker-4B outputs a P(yes) logit indicating relevance. Results are re-sorted by rerank_score.

**Step 7 — Expansion:** Bounded adjacency expansion adds ±1 NEXT_CHUNK neighbors from the top 5 seeds. Structure-aware expansion (when enabled) adds siblings, parent sections, and entity-shared chunks. All expanded chunks get synthetic scores at 50% of their source.

**Step 8 — Context Assembly:** ContextBudgetManager enforces a 14,000 token ceiling. Citation labels are hydrated, results are deduplicated, and the final context is assembled for the LLM response.

The reranker was previously configured with `MAX_TOKENS_PER_DOC=2048` and `MAX_TOKENS_TOTAL=4096`, using only 25% of the Qwen3-Reranker-4B's actual 8K context window. Our changes raised these to 7500 and 8192 respectively.

### The Signal Collapse Problem

The pipeline collects rich, diverse signals — 7 Qdrant vector fields (content-dense, title-dense, doc_title-dense, text-sparse, title-sparse, entity-sparse, late-interaction), Neo4j BM25, entity extraction, graph structure — then immediately flattens them into a single ranked list via RRF. After RRF, you can't tell whether a chunk ranked well because of semantic similarity, lexical match, entity mention, or structural position. The reranker only sees the top 100 by fused score, which means:

- Introduction/summary chunks dominate (they score well across multiple embeddings)
- Deep procedural content never enters the rerank pool
- Per-document depth is not guaranteed
- Signal diversity is lost before the cross-encoder can evaluate it

This is analogous to how ensemble methods in machine learning work: the value of an ensemble comes from diversity among the base learners, not from running the same strong learner multiple times. The 7 vector fields + BM25 are the base learners. RRF treats them as a voting committee where rank-position is the only signal. Signal-aware selection treats them as a coverage map where each learner's unique discoveries are preserved for the cross-encoder to evaluate.

The fundamental insight: RRF is a rank-position function. It throws away the signal type and keeps only the rank ordinal. After RRF, a chunk ranked #5 because it was #2 on content-dense and nowhere else looks identical to a chunk ranked #5 because it was #15 on content, #12 on title, and #8 on entity-sparse. These represent fundamentally different retrieval evidence that should inform different decisions about the chunk's value, but RRF erases the distinction. The signal-diverse pool preserves that distinction through to the cross-encoder.

### Why 200 Candidates, Not 100

The Qwen3-Reranker-4B has an 8K context window. With batched HTTP requests (16 documents per batch), 200 candidates require approximately 13 HTTP batches. At ~30-40ms per batch on GPU, total reranking latency is ~250-500ms — within acceptable bounds for a RAG system that already does multi-vector Qdrant search + Neo4j BM25 + graph expansion.

With signal-diverse selection, candidates #100-200 aren't diminishing-return tail entries — they're from underrepresented signal sources that might represent completely different evidence. The marginal value of candidate #150 remains high because it might be the only chunk from the entity-sparse signal covering a specific entity the query mentions.

The 512-token ColBERT/SPLADE limit creates an asymmetry: chunks partially truncated during retrieval scoring (ColBERT only saw first 512 tokens) can be evaluated in full by the 8K reranker. This further strengthens the case for a larger pool — the reranker can rescue chunks that ColBERT underscored due to truncation.

### Structural Graph Candidates: The Depth Signal

The graph structural expansion is the key mechanism for surfacing "the meat" of documents. Vector similarity tells you "this chunk talks about similar topics." Graph structure tells you "this chunk is the next step in the procedure" or "this chunk is in the same configuration section." These are orthogonal signals that the cross-encoder can evaluate but cannot discover on its own.

The existing `_expand_with_structure` method provides three types of structural neighbors: (1) sibling chunks with the same `parent_section_id`, (2) parent section chunks via `CHILD_OF` traversal, and (3) chunks sharing entities via `MENTIONS` edges. When run pre-rerank with `force=True`, these structural neighbors enter the rerank pool and get proper cross-encoder scoring instead of the synthetic `source_score * 0.5` they would get post-rerank.

### The Dominance Gating Decision

The dead `_gate_to_primary_document` function (defined at line 3345 but never called) was designed to filter seeds to a single dominant document when one document dominated the top 8 results. For the signal-diverse pool, this is counterproductive — it narrows the pool to one document's top-scoring chunks rather than ensuring depth within that document. The signal pool's per-document depth slot accomplishes the same continuity goal (ensuring the dominant document has deep representation) without throwing away evidence from other documents.

### Model Stack Specifications

| Role | Model | Context | Output | Constraint |
|---|---|---|---|---|
| Dense | Qwen3-Embedding-0.6B | 32K tokens | 1024-dim (MRL) | No constraint for chunks |
| Sparse | SPLADEv3 (NAVER) | 512 tokens | ~25 sparse terms | BERT-based |
| ColBERT | ColBERTv2 (colbert-ai) | 512 tokens | 32 tokens x 128-dim | Late interaction |
| Reranker | Qwen3-Reranker-4B | 8K tokens | P(yes) logit | Cross-encoder |

All models are served by a **unified gateway** at `10.25.0.50:8080` using the same `/v1/embeddings/*` API pattern the existing `EmbeddingClient` already supports.

### Weighted Fusion Clarification

Despite its misleading name, `query_api_weighted_fusion` does NOT replace RRF with weighted-score fusion. It replaces server-side DBSF (which loses per-field scores) with client-side RRF (which preserves them). The `multi_vector_fusion_method` config (default: `"rrf"`) controls the actual fusion algorithm. Enabling `query_api_weighted_fusion` gives you: same RRF fusion + per-field score tracking + ~40-70ms latency from 5-7 filtered Qdrant re-scoring queries.

---

## Part 3: Signal-Diverse Rerank Pool Implementation

### Design

Instead of flat "top 100 by fused score," the signal pool builds a 200-candidate rerank pool from **signal-diverse slots**:

| Slot | Size | Source |
|---|---|---|
| Consensus | 50 | Top by fused_score (multi-signal agreement) |
| Content-dense unique | 20 | High on vector_score, not in consensus |
| Title-dense unique | 15 | High on title_vec_score, not in consensus |
| Doc-title-dense unique | 15 | High on doc_title_vec_score |
| Text-sparse unique | 20 | High on lexical_vec_score or bm25_score |
| Entity-sparse unique | 15 | High on entity_vec_score |
| Title-sparse unique | 15 | High on doc_title_sparse_score |
| Structural expansion | 20 | NEXT_CHUNK + sibling graph neighbors (pre-rerank) |
| Per-document depth | 30 | Top K documents, M signal-diverse chunks each |

Backfill fills remaining capacity from fused_score order. Graceful degradation: when per-field scores are unavailable (weighted fusion not enabled), falls back to BM25/vector provenance.

### Key Implementation Details

- `src/query/signal_pool.py` — Pure-function module, no IO, independently testable. 280 lines.
- Double-gated: `config.search.hybrid.signal_pool.enabled=True` AND `feature_flags.signal_diverse_rerank_pool=True`
- `_hydrate_parent_paths()` fetches `parent_path_norm` from Neo4j for all rerank candidates (one batch Cypher query). Runs for ALL reranker calls, not just signal pool — standalone improvement.
- Reranker text enriched: `"Configuration > S3 Backend > Buckets\n\nBucket Settings\n\nSet the bucket name..."` — prepends heading hierarchy path for structural context.
- `_expand_with_structure(force=True)` runs pre-rerank on top 10 fused results to feed the structural slot, bypassing the `structure_aware_expansion` feature flag.
- `_build_query_bundle` decoupled: uses `self.sparse_embedder is self.colbert_embedder` identity check. Same provider → efficient `embed_query_all`. Different providers → independent `embed_sparse` + `embed_colbert` calls.
- Dead `_gate_to_primary_document` removed (34 lines, defined but never called).
- Startup warning logged when signal pool enabled without weighted fusion.
- Token limits raised: `MAX_TOKENS_PER_DOC=7500`, `MAX_TOKENS_TOTAL=8192`, `service_max_batch_tokens=4096`.

---

## Part 4: Provider Stack Cleanup

### Problem: Naming Drift From Incremental Model Upgrades

The codebase had been through several model upgrades (MiniLM → BGE-M3 → Qwen3) but each upgrade added new code without cleaning the old naming. The result was a confusing layer cake of names:
- File `local_bge_service.py` / class `BGERerankerServiceProvider` running Qwen3-Reranker-4B
- File `bge_m3_service.py` / class `BGEM3ServiceProvider` serving as the generic embedding HTTP client
- Factory defaults pointing to `jina-ai` / `jina-reranker-v3` while actually using local reranker
- Embedding plan pointing to `qwen3_4b` and `bge_m3` while target stack is `qwen3_0_6b`, `spladev3`, `colbertv2`
- Hardcoded IP `10.25.0.50:8080` in 3 Python source files as fallback defaults

### Changes Made

**File renames (clean break):**
- `src/providers/rerank/local_bge_service.py` → `local_reranker_service.py`
- `BGERerankerServiceProvider` → `LocalRerankerServiceProvider`
- `src/providers/embeddings/bge_m3_service.py` → `embedding_service.py`
- `BGEM3ServiceProvider` → `EmbeddingServiceProvider`

**Factory routing:**
- Default reranker provider: `"jina-ai"` → `"local-reranker-service"`
- Default reranker model: `"jina-reranker-v3"` → `"Qwen/Qwen3-Reranker-4B"`
- Embedding provider: `"bge-m3-service"` → `"embedding-service"` (with legacy alias for backward compat)
- `_build_settings_from_profile` updated with `"embedding-service"` branch reading `EMBEDDING_BASE_URL`
- `EmbeddingServiceProvider.__init__` URL resolution chain: `EMBEDDING_BASE_URL` → `BGE_M3_API_URL` (legacy fallback) → error

**New profiles (`config/embedding_profiles.yaml`):**
```yaml
plan:
  dense: "qwen3_0_6b"      # Qwen3-Embedding-0.6B, 1024-dim
  sparse: "spladev3"        # SPLADE v3, learned sparse
  colbert: "colbertv2"      # ColBERTv2, 128-dim per-token
  enable_sparse: true
  enable_colbert: true
```

All three new profiles use `provider: "embedding-service"` and `requirements: ["EMBEDDING_BASE_URL"]`.

### Unified Gateway Architecture

A key simplification discovered during this session: all models (dense, sparse, ColBERT, reranker, NER) are served by a single unified gateway at `10.25.0.50:8080` using the same OpenAI-compatible API patterns the existing `EmbeddingClient` already speaks:

| Model | Endpoint | Request Format | Response Format |
|---|---|---|---|
| Qwen3-Embedding-0.6B | `POST /v1/embeddings` | `{"model": "...", "input": [...]}` | `{"data": [{"embedding": [...1024 floats...]}]}` |
| SPLADEv3 | `POST /v1/embeddings/sparse` | `{"model": "...", "input": [...]}` | `{"data": [{"indices": [...], "values": [...]}]}` |
| ColBERTv2 | `POST /v1/embeddings/colbert` | `{"model": "...", "input": [...]}` | `{"data": [{"vectors": [[...128 floats...], ...]}]}` |
| Qwen3-Reranker-4B | `POST /v1/rerank` | `{"query": "...", "documents": [...]}` | `{"results": [{"index": int, "score": float}]}` |
| GLiNER Medium v2.1 | `POST /v1/extract` | model-specific | model-specific |

This meant no new HTTP client classes were needed — the existing `EmbeddingClient` (supporting `/v1/embeddings`, `/v1/embeddings/sparse`, `/v1/embeddings/colbert`) and the reranker provider (supporting `/v1/rerank`) could be reused with only naming and default changes. The gateway handles model routing internally based on endpoint path and model parameter.

The previous architecture had separate service endpoints: BGE-M3 at `:9000` for sparse+ColBERT, Qwen3-4B Triton gateway at `:8101` for dense, and the reranker at `:9003`/`:9005`. The unified gateway consolidates all of these behind one URL, reducing configuration complexity and connection management overhead.

### Profile-Driven Provider Selection

The embedding plan system works as follows:

1. `config/embedding_profiles.yaml` defines named profiles with provider, model_id, dims, tokenizer, and capabilities
2. The `plan:` section assigns profiles to roles: `dense`, `sparse`, `colbert`
3. At startup, `get_embedding_plan()` resolves the plan into `EmbeddingRolePlan` objects
4. `ProviderFactory.create_embedding_provider_for_role(role_plan)` creates a provider instance from the profile settings
5. `QdrantMultiVectorRetriever.__init__` creates separate provider instances when profile names differ between roles

This system was already in place for embedders but had never been extended to the reranker. The reranker is still configured via ENV vars and config fields, not via the profile YAML. We added an informational `reranker:` section to the YAML as a stepping stone toward full profile-driven reranker configuration in a future session.

### Why SPLADEv3 and ColBERTv2 Require Re-ingestion

SPLADEv3 uses a different tokenizer vocabulary than BGE-M3. Sparse vector indices are token IDs — they reference positions in the model's vocabulary. SPLADEv3's vocabulary (BERT-based, ~30K tokens) is completely incompatible with BGE-M3's vocabulary. You cannot search an index built with BGE-M3 sparse vectors using a SPLADEv3 query — the token IDs map to different words. A full re-ingestion is required to rebuild all sparse vectors.

ColBERTv2 produces 128-dimensional per-token vectors versus BGE-M3's 1024-dimensional ColBERT vectors. The Qdrant `late-interaction` vector field is configured with a fixed `size` parameter. Changing from 1024 to 128 requires recreating the field. The `qdrant_schema.py` already supports this via `colbert_dims = embedding_plan.colbert.profile.dims`, so the schema update will happen automatically during collection creation, but existing documents need re-embedding.

The dense embedder switch (Qwen3-4B → Qwen3-0.6B) also changes the embedding space, though both produce 1024-dim vectors. Existing dense vectors would be in the 4B model's embedding space while new queries would be in the 0.6B space — cosine similarity between different model spaces is unreliable. A full re-ingestion ensures all vectors are in the same space.

**Query bundle decoupling — the critical architectural fix:**

The `_build_query_bundle` method (hybrid_retrieval.py:1247) previously assumed sparse and ColBERT embeddings came from the same provider:

```python
# OLD: assumes one provider produces both sparse and ColBERT
bundle_provider = (self.colbert_embedder if self.schema_supports_colbert
                   else self.sparse_embedder)
bundle = bundle_provider.embed_query_all(query)
sparse = bundle.sparse     # Gets sparse from colbert provider!
```

This was true for BGE-M3 (which has dense, sparse, and ColBERT heads in one model) but breaks when SPLADEv3 (sparse-only) and ColBERTv2 (ColBERT-only) are separate providers. The fix uses Python `is` identity to detect the configuration:

```python
# NEW: handles both same-provider and separate-provider cases
same_provider = self.sparse_embedder is self.colbert_embedder
if same_provider and hasattr(self.colbert_embedder, "embed_query_all"):
    bundle = self.colbert_embedder.embed_query_all(query)  # Efficient: one call
else:
    sparse = self.sparse_embedder.embed_sparse([query])     # Independent calls
    colbert = self.colbert_embedder.embed_colbert([query])
```

This is backward compatible: BGE-M3 setups where the plan assigns the same profile to both roles will use the efficient `embed_query_all` path. The separate-provider path only activates when the profiles differ.

The QdrantMultiVectorRetriever `__init__` (lines 737-763) already correctly creates separate provider instances when embedding plan profiles differ, so the identity check works reliably.

**Hardcoded IP removal:**
- `factory.py:178` — `os.getenv("EMBEDDING_BASE_URL")` with warning if unset
- `factory.py:399` — `os.getenv("RERANKER_BASE_URL")` with error if unset
- `embedding_client.py:33` — `os.getenv("EMBEDDING_BASE_URL")` with error if unset
- `local_reranker_service.py:92` — default `base_url=""` (factory always provides URL)

**Config updated:**
- `config/development.yaml:141` — `provider: "local-reranker-service"`, `model: "Qwen/Qwen3-Reranker-4B"`
- `RerankerConfig` defaults — `provider: "local-reranker-service"`, `model: "Qwen/Qwen3-Reranker-4B"`
- `.env.example` — Unified gateway documentation block, all legacy URLs commented out

---

## Part 5: Code Review Rounds — Process and Findings

Three formal review rounds were conducted during the session, each using the `superpowers:code-reviewer` agent dispatched against the working tree diff. The review process followed a strict pattern: implement → review against objectives → fix critical/important items → re-review. Each round produced categorized findings (Critical, Important, Suggestion) with specific file/line references.

### Review 1: Signal Pool Implementation

Reviewed the signal pool against 8 stated objectives. All objectives were implemented correctly, but the review identified 3 gaps requiring fixes:

- **(Critical) Missing weighted fusion warning:** The plan specified auto-enabling weighted fusion when signal pool is active. Instead of auto-enabling (which would add latency without opt-in), we added a startup-time warning log that clearly explains the degradation and how to fix it. This was a deliberate downgrade from the plan's "auto-enable" to a "warn-and-degrade" approach.

- **(Critical) Missing parent_path_norm reranker test:** The `_apply_reranker` text construction prepends parent_path_norm, but no test verified this behavior. Added `test_apply_reranker_prepends_parent_path_norm` and `test_apply_reranker_without_parent_path_norm` to the reranker integration test suite. This also required adding a `last_candidates` capture field to the `FakeRerankProvider` test double.

- **(Important) _hydrate_parent_paths gated inside signal pool branch:** The plan specified parent_path_norm enrichment as a "standalone improvement" independent of the signal pool. The initial implementation only called `_hydrate_parent_paths` inside the signal pool branch, meaning it wouldn't run when the signal pool was disabled. Moved the call outside the branch so it runs for ALL reranker calls. This was a design intent violation caught by the review.

Additional fixes: added 3 missing scores (`doc_title_vec_score`, `doc_title_sparse_score`, `lexical_vec_score`) to the per-document depth diversity sort key; fixed a pre-existing test bootstrap bug (`_reranker_available` missing from `object.__new__` bootstrap) that had been masking a wrong test assertion.

### Review 2: Provider Stack Cleanup

Reviewed the combined signal pool + provider rename changeset against 10 objectives. Found 3 critical issues that would have caused runtime failures:

- **(Critical) `_build_settings_from_profile` missing "embedding-service" branch:** The factory method that resolves profile settings to service URLs had branches for `"bge-m3-service"` and `"snowflake-arctic-service"` but not for the new `"embedding-service"` provider. This meant the new profiles (qwen3_0_6b, spladev3, colbertv2) would get `service_url=None`, and `EmbeddingServiceProvider.__init__` would fall through to checking `BGE_M3_API_URL` (legacy env var). If neither was set, the provider would raise a misleading `RuntimeError` about BGE_M3_API_URL. Fixed by adding the `"embedding-service"` branch and updating the URL resolution chain to check `EMBEDDING_BASE_URL` first.

- **(Critical) 3 test files with broken imports:** `test_bge_m3_service_provider.py`, `test_phase7c_provider_factory.py`, and `test_phase7e2_hybrid_retrieval.py` still imported from `src.providers.embeddings.bge_m3_service` which no longer existed. Fixed with aliased imports: `from src.providers.embeddings.embedding_service import EmbeddingServiceProvider as BGEM3ServiceProvider`.

- **(Critical) Stale BGE-M3 defaults in embedding_service.py:** The renamed class still had `self._provider_name = settings.provider or "bge-m3-service"` as fallback, and the URL resolution error message referenced `BGE_M3_API_URL`. Updated to `"embedding-service"` and `EMBEDDING_BASE_URL` respectively.

Additional fixes: `log_provider_config` stale defaults (jina-ai → local-reranker-service), `__init__.py` docstring update, fragile import alias in capture_baseline.py, stale log event name "bge_rerank_complete" → "rerank_complete".

### Review 3: Final Verification

External reviewer ran the full test suite and found:

- **(High) `development.yaml` still had `bge-reranker-service`:** The YAML config file that drives the actual deployed configuration hadn't been updated. The factory would have failed with `Unknown rerank provider: bge-reranker-service` since we did a clean break (no backward-compat alias). Fixed to `local-reranker-service`.

- **(High) 4 pre-existing query-path test failures:** Tests in `test_query_api_payload.py` and `test_qdrant_multivector_sparse.py` were failing. Confirmed pre-existing via git stash test (identical failures on the previous commit). Root cause: tests predated the capabilities system and created retrievers with `embedding_settings=None`, causing `supports_sparse` to evaluate as False. Fixed by providing `EmbeddingSettings` with proper capabilities in test constructors.

- **(Low) Hardcoded gateway IP in source:** `10.25.0.50:8080` appeared as fallback defaults in 3 Python source files. Removed — URLs are now exclusively env-var-driven. The IP only remains in `.env.example`, deployment configs, legacy Triton gateway providers, and one error message example string.

---

## Test Results

### Tests We Own (all passing)

| Test Suite | Count | Status |
|---|---|---|
| `tests/query/test_signal_pool.py` | 14 | All pass |
| `tests/query/test_reranker_integration.py` | 6 | All pass |
| `tests/query/test_query_api_payload.py` | 2 | Fixed (pre-existing failures) |
| `tests/query/test_qdrant_multivector_sparse.py` | 5 (2 fixed) | All pass |
| `tests/test_phase7c_provider_factory.py` | — | Pass (imports updated) |
| `tests/providers/test_bge_m3_service_provider.py` | — | Pass (imports updated) |
| **Total validated suite** | **47** | **All pass** |

### Improvement Over Baseline

| Metric | Before | After | Delta |
|---|---|---|---|
| `tests/query/` passing | 11 | 35 | **+24** |
| `tests/query/` failing | 16 | 7 | **-9 fixed** |

The 7 remaining failures are all pre-existing (confirmed by stash test) and require infrastructure (live Neo4j/Qdrant) or further capability-aware test bootstrapping.

---

## Files Changed (Full Inventory)

### New Files
| File | Lines | Purpose |
|---|---|---|
| `src/query/signal_pool.py` | 290 | Signal-diverse rerank pool builder |
| `tests/query/test_signal_pool.py` | 240 | 14 unit tests for pool builder |
| `scripts/ci/check_dead_imports.py` | 165 | CI guard for ACTIVE→DEAD import violations |

### Renamed Files
| Old | New |
|---|---|
| `src/providers/rerank/local_bge_service.py` | `local_reranker_service.py` |
| `src/providers/embeddings/bge_m3_service.py` | `embedding_service.py` |

### Modified Files (17)
`src/query/hybrid_retrieval.py`, `src/shared/config.py`, `src/providers/factory.py`, `src/clients/embedding_client.py`, `src/providers/rerank/base.py`, `src/providers/embeddings/__init__.py`, `config/embedding_profiles.yaml`, `config/development.yaml`, `.env.example`, `scripts/phase0/capture_baseline.py`, `tests/query/test_reranker_integration.py`, `tests/query/test_query_api_payload.py`, `tests/query/test_qdrant_multivector_sparse.py`, `tests/unit/test_phase1_reranker_batching.py`, `tests/integration/test_phase1_entity_edges.py`, `tests/providers/test_bge_m3_service_provider.py`, `tests/test_phase7c_provider_factory.py`, `tests/test_phase7e2_hybrid_retrieval.py`

---

## Commits Pushed (This Session)

```
fd55e3b fix: repair pre-existing test failures and update imports
836eca6 refactor: rename providers for unified gateway model stack
60e50cb feat: add signal-diverse rerank pool and parent_path enrichment
a96356d docs: fix stale 19→20 test file count in plan doc
cdd22ec docs: update plan with review feedback corrections
bcf98db fix: harden CI guard and resolve review feedback
1be7aa7 docs: mark Phase B.6 complete in pruning plan
f144ea3 fix: resolve Phase B.6 prerequisites before pruning
```

---

## Key Architectural Decisions and Rationale

**Decision 1: Signal pool as a pure-function module.** The pool builder has zero IO — no Qdrant, no Neo4j, no HTTP calls. Structural expansion results and parent_path_norm data are passed in as arguments. This makes it trivially unit-testable (14 tests run in <1s with synthetic ChunkResult objects) and keeps the IO boundary at the integration layer in `retrieve()`.

**Decision 2: Double-gating for the signal pool.** Both `config.signal_pool.enabled` AND `feature_flags.signal_diverse_rerank_pool` must be True. When either is False, the pipeline falls through to the exact legacy behavior with zero code path changes. This enables safe deployment: merge the code, enable in staging, verify metrics, then enable in production.

**Decision 3: parent_path_norm as standalone improvement.** The heading hierarchy enrichment (`parent_path_norm`) runs for ALL reranker calls regardless of the signal pool feature flag. This was initially incorrectly gated inside the signal pool branch; the code review caught this design intent violation. The enrichment adds ~20-30 tokens of structural context per candidate at a cost of one batch Neo4j query (~5-15ms).

**Decision 4: Clean break for provider rename.** No backward-compatible aliases for `"bge-reranker-service"` in the factory. The `development.yaml` config was updated simultaneously. Legacy BGE-M3 embedding aliases are kept (they route to `"embedding-service"`) because the embedding plan system uses profile names, not provider names, for routing.

**Decision 5: Env-var-driven URLs, not hardcoded defaults.** Hardcoded `10.25.0.50:8080` was removed from all Python source defaults. The URL now only appears in `.env.example` and deployment configs. This prevents silent connection failures when code runs on a different machine.

**Decision 6: Pre-existing test failures fixed opportunistically.** Four query-path test failures existed before our changes (confirmed by stash testing). We fixed them by providing `EmbeddingSettings` with capabilities to test constructors — the root cause was that tests predated the capabilities system and created retrievers with `embedding_settings=None`, which caused `supports_sparse` to evaluate as False regardless of the `schema_supports_sparse` constructor arg.

---

## What's Next

### Ready for Production (no further code changes needed)
- **parent_path_norm enrichment** — active on every reranker call, standalone improvement
- **Token limit increase** — Qwen3-4B now uses full 8K context window
- **Provider stack** — all naming, routing, profiles reflect actual deployed models

### Requires Activation (feature flags)
- **Signal pool**: Set `signal_pool.enabled=True` + `feature_flags.signal_diverse_rerank_pool=True` + `feature_flags.query_api_weighted_fusion=True` for full per-field signal diversity
- Without weighted fusion: pool degrades to BM25/vector provenance (startup warning logged)

### Requires Re-ingestion
- SPLADEv3 uses different vocabulary from BGE-M3 (sparse indices incompatible)
- ColBERTv2 uses 128-dim per-token vectors (vs BGE-M3's 1024-dim)
- Qdrant collection needs `late-interaction` field rebuilt with `size=128`

### Activation Checklist

To enable the signal-diverse rerank pool in production, these steps are required in order:

1. Ensure `EMBEDDING_BASE_URL` and `RERANKER_BASE_URL` are set in the deployment environment (`.env` or docker-compose)
2. Set `feature_flags.query_api_weighted_fusion: true` in config — this enables per-field score tracking (adds ~40-70ms of filtered Qdrant re-scoring queries per search)
3. Set `feature_flags.signal_diverse_rerank_pool: true` in config — this enables the signal pool feature flag gate
4. Set `search.hybrid.signal_pool.enabled: true` in config — this enables the signal pool config gate
5. Optionally tune slot allocations in `search.hybrid.signal_pool.*` (consensus_slots, content_dense_slots, etc.)
6. Monitor metrics: `signal_pool_enabled`, `signal_pool_size`, `signal_pool_slot_fills`, `signal_pool_degraded`
7. Compare retrieval quality with A/B testing against the flat top-N baseline

If step 2 is skipped, the signal pool will log a startup warning and degrade to BM25/vector provenance mode (still provides some signal diversity, but without per-field granularity).

### Remaining Technical Debt

- 7 pre-existing test failures in `tests/query/` remain (confirmed pre-existing via stash test). All require either live infrastructure (Neo4j/Qdrant) or capabilities-aware test bootstrapping.
- The `phase7e2` integration test suite (3 failures, 6 errors) requires a running embedding gateway and Neo4j with the correct schema version — these are environment-dependent, not code bugs.
- Stale BGE-M3 references remain in docstrings of `chonkie_adapter.py` (3 comments) and `semantic_chunker.py` (1 adapter name check). These are cosmetic and do not affect runtime behavior.
- The Triton gateway providers (`qwen3_triton.py`, `qwen3_chonkie_adapter.py`, `qwen3_embedding_client.py`) still have hardcoded `10.25.0.50:8101` as defaults — these are legacy providers superseded by the unified gateway but retained for backward compatibility.
- The reranker is not yet profile-driven like embedders — it still uses ENV vars + config fields. The informational `reranker:` section in `embedding_profiles.yaml` is a stepping stone toward full profile integration.
- `max_pairs` and `max_tokens_per_pair` fields in `RerankerConfig` are dead config (defined but never referenced by any code). They should be removed in the pruning plan's Phase E.

### Future Work (from pruning plan, Phases C-F)
- Phase C: Pull forward active logic from build_graph.py → embedding_context.py
- Phase D: Delete 25 dead files + 3 dead directories
- Phase E: Remove ~3800 lines of dead methods from partially-active files
- Phase F: Retire 20 test files importing DEAD modules
- Plan doc: `docs/plans/2026-03-01-codebase-pruning-and-modernization-plan.md`
