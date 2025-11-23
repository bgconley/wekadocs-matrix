# WekaDocs Matrix – BGE-M3 + Reranker Architecture Notes

_Session date: 2025-11-21 → 2025-11-22_

This document captures a comprehensive, end-to-end record of the architectural investigations, design
changes, and partial implementations we performed around the WekaDocs Matrix retrieval and ranking stack,
with a special focus on:

- Migrating to a **BGE-M3 + cross-encoder reranker** architecture
- Reducing dependence on **Neo4j BM25** while leveraging BGE-M3 sparse vectors
- Fixing structural issues in **Qdrant Query API usage** (dense + sparse + ColBERT)
- Understanding and correcting **ranking normalization and confidence calibration**
- Designing a new **cascading waterfall ranking architecture** that blends vector consensus, reranker
  scores, and graph context

The intent is for this document to serve as a durable reference for future maintainers as we continue the
migration from the legacy Phase 7E “BM25 + RRF + optional reranker” architecture to the new
“BGE-M3 + reranker + graph-aware ensemble” architecture.

---

## 1. High-Level Session Overview

During this session we:

1. **Re-loaded prior context** from `session-2025-11-21-ingestion-query-api-debug-notes.md`, which
   summarized earlier work on ingestion, Qdrant, and Query API payloads.
2. **Ran the database cleanup script** to reset Neo4j, Qdrant, and Redis to a clean state while
   preserving schema and metadata.
3. **Re-ran ingestion** for new documents and verified that BGE-M3 embeddings were being generated and
   written to Qdrant (`chunks_multi_bge_m3`) and Neo4j
   (`section_embeddings_v2_bge_m3`).
4. **Exercised the QueryService smoke test**, initially seeing Query API failures and Qdrant falling back
   to legacy search due to incorrect payload types.
5. Fixed the **Query API payload construction** for dense, sparse, and ColBERT, and added unit tests to
   validate payload shapes without requiring a live Qdrant instance.
6. Discovered that **per-evidence confidence values were pegged at 1.0**, traced this to the
   ranking normalization layer (`_normalize_score` in `src/query/ranking.py`), and iteratively corrected
   normalization behavior.
7. Identified that **late-interaction (ColBERT) scores** and BM25 scores were being treated as generic
   “similarity” values and thus clamped into `[0,1]` in ways that destroyed rank order and confidence.
8. Designed and then refined a new **architecture for ranking and reranking**, evaluating several
   alternatives and settling on an Architecture 2 “cascading waterfall” design as the primary target.
9. **Stood up and integrated** a local BGE reranker service (`bge-reranker-v2-m3`) via a new
   `bge-reranker-service` provider, wired through the existing rerank provider abstraction.
10. Introduced a **new hybrid search mode** (`mode: bge_reranker`) and gated BM25 usage accordingly in
    `HybridRetriever`.
11. Added a **new focused test** for reranker mode to validate that BM25 is skipped and that confidence is
    derived from `rerank_score`, using mocks to avoid coupling to real infrastructure.
12. Defined a **future architectural direction** where vector fusion, rerank scores, and graph neighbors
    are blended in a more principled way, instead of the current hard-coded heuristics.

The remainder of this document walks through these areas in depth.
---

## 2. Prior Context and Baseline Architecture

Before this session, we had already made several important changes and discoveries, which influenced the
current work.

### 2.1 Ingestion and Schema Baseline

From the prior session notes, the ingestion and schema baseline were:

- **Neo4j** running with a `SchemaVersion` node at `version: "v2.2"`, with
  `embedding_provider: "bge-m3-service"` and `embedding_model: "BAAI/bge-m3"`.
- **Qdrant** collection `chunks_multi_bge_m3` configured as a multi-vector collection with named vectors:
  - `content`, `title`, `entity`, and `late-interaction` (ColBERT)
  - plus a sparse vector field `text-sparse`.
- **Embedding profile** `bge_m3` configured via `.env` and `config/development.yaml` with:
  - `EMBEDDINGS_PROFILE=bge_m3`
  - `EMBEDDINGS_PROVIDER=bge-m3-service`
  - `EMBEDDINGS_MODEL=BAAI/bge-m3`
  - `EMBEDDINGS_DIM=1024`
  - `EMBEDDING_NAMESPACE_MODE=profile`

We also had ingestion workers writing chunks into Neo4j and Qdrant, updating embedding metadata, and
maintaining cache invalidation via Redis.

### 2.2 Namespace Suffix Unification

Previously, ingestion and retrieval had diverging logic for embedding namespace suffixes. The fix was to
introduce a unified helper:

- `src/shared/config.py`: `get_expected_namespace_suffix(settings, mode)`
  - It computes a suffix based on `EMBEDDING_NAMESPACE_MODE` and the embedding profile/version/model.

That helper is used by:

- Ingestion: `GraphBuilder._ensure_qdrant_collection` and `_upsert_to_qdrant` to validate and annotate the
  Qdrant collection and embedding metadata.
- Retrieval: `HybridRetriever.__init__` to validate BM25 index names and Qdrant collection names against
  the expected suffix.

For BGE-M3, with `EMBEDDING_NAMESPACE_MODE=profile`, the suffix is `_bge_m3`, aligning:

- Neo4j index: `section_embeddings_v2_bge_m3`
- Qdrant collection: `chunks_multi_bge_m3`

This foundation is critical because all subsequent retrieval and reranking assumes the vector collection
and Neo4j index are aligned on a single embedding profile.

### 2.3 Query API Payload Fixes (Dense, Sparse, ColBERT)

In the prior session, we discovered Qdrant’s Query API was rejecting our payloads due to type mismatches.
Specifically, we were passing Python `dict`s where Qdrant’s Pydantic models expected:

- Plain `List[float]` for dense vectors in `Prefetch.query`
- `SparseVector` models for sparse queries
- Named vector payloads with `vector` as a list (or list-of-lists) for ColBERT.

We had started to fix this by:

- Standardizing `QueryEmbeddingBundle` to hold `dense`, `sparse`, and `multivector`.
- Building `Prefetch` entries with:
  - Dense queries: `query: List[float]`
  - Sparse queries: `query: qdrant_client.http.models.SparseVector(...)`.
- Building the main `query` object for `query_points` using named vectors for ColBERT (late-interaction)
  and primary dense vectors.

However, one key bug remained: we were still passing a dict-shaped `query` object in
`_build_query_api_query`, leading to `Unsupported query type: <class 'dict'>` and fallback to the legacy
search path.

This session finished that fix and ensured Query API-based hybrid search is the default path.
---

## 3. Database Cleanup and Fresh Ingestion

Before attempting new ingestion and query tests, we ran the **intelligent cleanup script** to ensure that
we were working with a clean but schema-preserving environment.

### 3.1 Cleanup Script Behavior

File: `scripts/cleanup-databases.py`

Key properties of the script:

- **Neo4j:**
  - Preserves metadata and system labels such as `SchemaVersion`, `SystemMetadata`, `MigrationHistory`,
    and others.
  - Deletes only ingestion-generated data labels (e.g., `Document`, `Section`, `Chunk`, `Command`,
    `Configuration`, `Procedure`, `Step`, `CitationUnit`).
  - Runs batched `MATCH ... DETACH DELETE` queries to safely delete data without touching indexes or
    constraints.
  - Verifies that schema counts (constraints, indexes) are unchanged before vs. after.

- **Qdrant:**
  - Restricts deletions to allowed ingestion collections, currently:
    - `chunks`, `chunks_multi`, `chunks_multi_bge_m3`.
  - Deletes only points (vectors), not collection schemas.

- **Redis:**
  - Deletes data keys while preserving system/metadata keys (based on prefixes like `schema:*`, `system:*`,
    etc.).

In our run, we saw:

- Neo4j data nodes reduced from over a thousand down to the two preserved metadata nodes.
- Qdrant’s `chunks_multi_bge_m3` was emptied of vectors.
- Redis DB1 had no keys to delete.

### 3.2 Fresh Ingestion

After cleanup, we re-ran ingestion by dropping documents into `/app/data/ingest` inside the ingestion
worker container. The logs confirmed:

- Markdown parsing succeeded for a variety of documents.
- Entity extraction found commands, configurations, and procedures as expected.
- `GraphBuilder` initialized with the correct embedding profile and Qdrant collection names.
- Neo4j index `section_embeddings_v2_bge_m3` was ensured to exist at the correct dimensionality.
- Embedding provider `bge-m3-service` successfully produced 1024-dimensional vectors.
- Qdrant `chunks_multi_bge_m3` had new vectors written with correct dimensions and named vector fields.

In summary, the ingestion pipeline is functioning and aligned with the BGE-M3 profile. This gave us a
solid data foundation for debugging Query API and ranking.

---

## 4. Query API and Ranking: Problems Discovered

### 4.1 Query API Fallback and Payload Type Issues

Running the smoke test script from inside the `mcp-server` container:

- File: `scripts/smoke_test_query.py`
- Command: `docker compose exec mcp-server python scripts/smoke_test_query.py`

Initial behavior:

- Hybrid retrieval initialized correctly with BGE-M3 profile and Qdrant collection `chunks_multi_bge_m3`.
- BM25 search completed and returned some candidates.
- Query API attempted to run but logged:
  - `Query API search failed; falling back to legacy search error=Unsupported query type: <class 'dict'>`
- The pipeline then fell back to the legacy vector search path, and results were returned, but ColBERT and
  sparse vectors were not being exercised via Query API.

Root cause:

- `_build_query_api_query` in `src/query/hybrid_retrieval.py` was returning dicts of the form
  `{ "name": ..., "vector": ... }`, which do not match the type signature expected by
  `qdrant_client.QdrantClient.query_points`.
- The Qdrant client signature accepts:
  - `List[float]`, `List[List[float]]`, `SparseVector`, or Query objects (NearestQuery, FusionQuery, etc.),
    but not arbitrary dicts.

Fix:

- We updated `_build_query_api_query` to return:
  - For ColBERT: `([list(vec) for vec in bundle.multivector.vectors], "late-interaction")`
  - For dense: `(list(bundle.dense), self.primary_vector_name)`
- This aligns with `query_points` expectations and removed the `Unsupported query type` error.

After this change, the smoke test showed:

- `Multi-vector search completed via Query API` with a non-zero query_api duration.
- Dense, sparse, and ColBERT legs were used for recall via Qdrant Query API.

### 4.2 The “Confidence 1.0” Problem

Once Query API was functioning, a new issue surfaced in the smoke test output:

- The overall answer confidence might be something like `0.28` or `0.66`, but **each evidence item’s
  confidence was 1.0**.
- In the ranking diagnostics, we observed that for ColBERT/late-interaction scores, the normalization
  function `_normalize_score` was clamping values >1.0 to exactly 1.0.

Details:

- File: `src/query/ranking.py`
- Function: `_normalize_score(self, score: float, score_kind: str = "similarity")`
- For `score_kind` not recognized (e.g., `late-interaction`), the default branch was:
  - `return max(0.0, min(1.0, score))`

Since ColBERT scores are sums of per-token max similarities, they are typically **much larger than 1.0**.
This meant:

- Every ColBERT score was clamped to 1.0.
- When we used these normalized scores as evidence confidences, everything looked “perfect,” destroying
  the useful ranking signal.
### 4.3 Iterative Fixes to Normalization

We iteratively experimented with several strategies to fix normalization:

1. **Special-casing late-interaction:**
   - Added a branch in `_normalize_score` for `kind in {"late-interaction", "colbert"}` to avoid the
     default clamp.
   - Initially used a soft `tanh(score/10.0)` squash; this avoided clamping but had two problems:
     - It was query-length insensitive: the same numeric score meant different things depending on how
       many tokens were in the query.
     - It often produced top confidences like `0.66` for what were effectively near-perfect matches.

2. **Batch-relative normalization:**
   - Introduced per-batch max-score normalization for late-interaction scores:
     - Precomputed `max_scores_by_kind` inside `Ranker.rank`.
     - For `late-interaction`, used `safe / max_kind_score` as the normalized similarity when possible.
   - This correctly mapped the **top** ColBERT hit to ~1.0 and others proportionally, but the evidence
     confidences printed still showed unexpected 1.0s for non-top hits, indicating other paths were
     influencing the final numbers.

3. **Fallback semantics and default similarity handling:**
   - Adjusted the default similarity path to treat unknown similarities as cosine-like in `[-1, 1]` and
     map to `[0, 1]` via `(clamped + 1) / 2`, rather than a bare clamp.

4. **Reranker short-circuit:**
   - Once we decided to lean more heavily on a cross-encoder reranker, we added logic:
     - If `rerank_score` is present in metadata, treat it as the primary semantic signal and compute
       `semantic_score = sigmoid(rerank_score)` directly, bypassing much of the heuristic normalization.

These normalization corrections are partial and are meant to stabilize behavior **until** the new
reranker-augmented architecture is fully in place. They also highlighted a deeper architectural issue: we
were trying to make normalization do too much work instead of designing a cleaner ranking pipeline.

---

## 5. Architectural Issues Identified

From the above work, we identified several architectural issues in the existing retrieval and ranking
stack.

### 5.1 Over-Reliance on RRF and Rank-Based Fusion

- The hybrid retriever used **Reciprocal Rank Fusion (RRF)** in multiple layers:
  - First, internally inside Qdrant (when combining multiple fields/vectors).
  - Then, again inside the application (BM25 + vector RRF).
- While RRF is great for building a robust ordering without tuning, it **discards magnitude** and
  collapses large variations in scores (e.g., “very strong match” vs. “barely matches”).
- This made it difficult to calibrate confidence or to reason about scores across different algorithms.

### 5.2 Reranker Overriding Retrieval Scores

- Once we enabled a reranker (initially Jina, later a local BGE reranker), our pipeline treated the
  reranker’s score as an override:
  - Seeds were sorted almost purely by `rerank_score`.
  - The original vector similarity and BM25 scores were largely ignored at ranking time.
- This removed the **ensemble advantage** of having multiple retrieval signals (Dense, Sparse, ColBERT,
  BM25). The reranker became a “dictator” rather than one expert among many.

### 5.3 Graph Neighbors Penalized Too Strongly

- Graph neighbors (added by `_expand_microdoc_results` and `_apply_graph_enrichment`) were not reranked
  and often carried **no semantic score**.
- In the ranker, semantic score had a weight of 0.4, while graph distance and recency had smaller weights.
- As a result, neighbors could not compete with seeds unless they happened to have good vector scores
  themselves.

### 5.4 Conflation of Score Types in Normalization

- `_normalize_score` treated multiple fundamentally different score types under a single umbrella:
  - Cosine similarity in `[-1, 1]`.
  - Dot-product scores.
  - RRF fused scores.
  - BM25 scores.
  - ColBERT late-interaction sums.
- Without explicit `score_kind` metadata (`cosine`, `dot`, `late-interaction`, `bm25`, etc.), scores were
  often normalized incorrectly, leading to clamping and distorted confidences.

### 5.5 Lack of Clear Modes for Hybrid Search

- The code assumed a single hybrid configuration (BM25 + vectors + optional reranker) and used
  configuration flags in a somewhat ad-hoc way.
- There was no explicit high-level mode to say “run in vector-only recall + reranker mode” vs.
  “legacy BM25 + RRF + optional reranker.”

These issues motivated the introduction of a new **hybrid mode** and a new reranking architecture.
---

## 6. New Hybrid Mode and Reranker Service Integration

To prepare for a more modern architecture (BGE-M3 recall + cross-encoder reranker + graph context), we
implemented several structural changes.

### 6.1 Config: Hybrid Mode and Reranker Settings

File: `config/development.yaml`

We extended the `search.hybrid` section to include:

- `mode: "bge_reranker"` (in development config; default remains `"legacy"` in model)
  - `legacy` mode: BM25 + RRF + optional reranker (existing behavior)
  - `bge_reranker` mode: vector-only recall (Dense/Sparse/ColBERT) + cross-encoder reranker, no BM25
- Updated `reranker` block:
  - `enabled: true`
  - `provider: "bge-reranker-service"`
  - `model: "BAAI/bge-reranker-v2-m3"`
  - `top_n: 20`
  - `max_pairs: 50`
  - `max_tokens_per_pair: 800`
- Disabled BM25 for this mode:
  - `search.hybrid.bm25.enabled: false` in development.yaml

On the model side (Pydantic configuration models):

File: `src/shared/config.py`

- `class RerankerConfig(BaseModel)` gained:
  - `max_pairs: int = 50`
  - `max_tokens_per_pair: int = 800`
- `class HybridSearchConfig(BaseModel)` gained:
  - `mode: str = "legacy"`

This ensures:

- The config file and in-memory models are aligned.
- Code can safely read `config.search.hybrid.mode` without hacky monkeypatching.

### 6.2 New Reranker Provider: bge-reranker-service

We already had a standalone repo and service for BGE reranking:

- Repo: `/Users/brennanconley/vibecode/bge-reranker-v2-m3`
- Client: `src/clients/reranker_client.py`

To integrate this with the WekaDocs Matrix pipeline, we added a new provider implementation:

File: `src/providers/rerank/local_bge_service.py`

- `class BGERerankerServiceProvider(RerankProvider)`:
  - Wraps an `httpx.Client` pointing at a base URL (default `http://127.0.0.1:9001`).
  - Implements `rerank(query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]`.
  - Builds a request payload:
    - `{"query": query, "documents": [cand["text"] for cand in candidates], "model": model_id}`
  - Calls `/v1/rerank` on the service and reads a `results` list of `{index, score, document?}`.
  - Reconstructs candidate dicts with added fields:
    - `rerank_score`, `original_rank`, `reranker`.
  - Sorts by `rerank_score` in descending order and truncates to `top_k`.

We then wired this provider into the `ProviderFactory`:

File: `src/providers/factory.py`

- Extended `create_rerank_provider` to recognize:
  - `provider == "bge-reranker-service"` or `"bge-reranker"`.
  - Instantiates `BGERerankerServiceProvider` with:
    - `base_url` from `RERANKER_BASE_URL` or default.
    - `timeout` from `RERANKER_TIMEOUT_SECONDS` or default 60s.

Now, when `search.hybrid.reranker.provider` is set to `bge-reranker-service`, the reranker is backed by the
local BGE reranker HTTP service.

### 6.3 HybridRetriever: Mode-Aware BM25 Usage

File: `src/query/hybrid_retrieval.py`

In `HybridRetriever.__init__` we added:

- Reading `hybrid_mode = getattr(config.search.hybrid, "mode", "legacy")`.
- Conditional construction of `BM25Retriever`:
  - `self.bm25_retriever = None` by default.
  - Only instantiate BM25 retriever if `hybrid_mode != "bge_reranker"` and `bm25.enabled` is `True`.

In `HybridRetriever.retrieve` we refactored the retrieval phase to:

- Always perform vector search via `self.vector_retriever.search(...)`.
- Only perform BM25 search when `hybrid_mode != "bge_reranker"` and `self.bm25_retriever` is not None.
- For `bge_reranker` mode:
  - Skip BM25 entirely.
  - Use `vec_results` as `fused_results`.
  - Ensure each vector chunk has `fused_score` set (from `vector_score`) and `fusion_method = "vector"`.

This ensures:

- No BM25 index is consulted in the new mode.
- Vector scores are promoted to fused scores for seeds, ready for reranking and further processing.

### 6.4 Reranker Short-Circuit in Ranker

File: `src/query/ranking.py`

In `_extract_features`, we changed the order of semantic feature derivation:

- First, check for `rerank_score` in metadata:
  - If present, compute `features.semantic_score = sigmoid(rerank_score)`.
- Else, follow the legacy path:
  - Use `vector_score` and `vector_score_kind` (or `bm25_score`, fused score) with `_normalize_score`.

This reorients the ranking logic so that:

- In `bge_reranker` mode, when rerank scores exist, they become the primary semantic signal.
- Legacy mode continues to use vector/BM25-based normalization.

Note: In Architecture 2, we plan to **blend** vector_score and rerank_score; the current implementation
uses rerank_score as the main semantic signal when present. This is a stepping stone, not the final
behavior.
---

## 7. Focused Test for bge_reranker Mode

To avoid breaking the existing extensive test suite while evolving the architecture, we introduced a
focused, highly controlled test that validates the behavior of the new `bge_reranker` mode.

File: `tests/query/test_reranker_mode.py`

### 7.1 Test Goals

The test `test_bge_reranker_mode_skips_bm25` aims to ensure that:

1. When `search.hybrid.mode` is set to `"bge_reranker"`, **BM25 is not used** for retrieval.
2. A **dummy reranker** can drive the ordering and semantic confidence (via sigmoid) as expected.
3. The Ranker uses `rerank_score` to compute semantic confidence.

### 7.2 Test Strategy

Because `HybridRetriever` has many dependencies (Neo4j driver, Qdrant client, tokenizer, schema checks,
graph expansion, microdoc expansion), we heavily **stubbed and monkeypatched** within the test to isolate
just the behavior we care about.

Key points:

- We patched configuration:
  - `config.search.hybrid.mode = "bge_reranker"`
  - `config.search.hybrid.reranker.enabled = True`
  - `config.search.hybrid.reranker.top_n = 3`
  - `config.search.hybrid.bm25.enabled = False`
  - `config.search.bm25.enabled = False` (top-level BM25 config)
- We stubbed `QdrantMultiVectorRetriever` with a `SimpleNamespace` instance that:
  - Implements `search(query, top_k, filters)` returning a single fake `ChunkResult`-like object.
  - Exposes properties `supports_sparse`, `supports_colbert`, `schema_supports_sparse`,
    `schema_supports_colbert`, `sparse_field_name`, `field_weights`, and `use_query_api` to satisfy
    HybridRetriever’s feature checks.
- We stubbed `BM25Retriever` to a dummy object with `index_name=None, search=lambda: []` to avoid real
  full-text search.
- We patched `ensure_schema_version` to a no-op since we passed `neo4j_driver=None`.
- We patched several methods on `HybridRetriever` that require real Neo4j or graph data:
  - `_hydrate_missing_citations` → no-op
  - `_expand_microdoc_results` → returns `( [], 0 )`
  - `_apply_graph_enrichment` → returns `( [], {"graph_time_ms": 0, "graph_count": 0} )`
  - `_annotate_coverage` → no-op

We then injected a dummy reranker:

- `DummyRerankProvider` with `model_id = "dummy-reranker"`.
- Instead of using the normal `_get_reranker` implementation, we monkeypatched `hr._get_reranker` to
  return `DummyRerankProvider()`.
- We monkeypatched `_apply_reranker` on `HybridRetriever` to assign deterministic `rerank_score` values to
  the candidate seeds and flag `metrics["reranker_applied"] = True`.

Finally, we called:

- `results, metrics = hr.retrieve("q", top_k=1, expand=False)`

and then validated:

- `metrics["bm25_count"] == 0` (no BM25 usage)
- `len(results) == 1` and `results[0].rerank_score` was assigned as expected.
- `Ranker._extract_features` correctly computed `semantic_score = sigmoid(rerank_score)` when given a
  `SearchResult` built from the result chunk.

This test gives us confidence that the new path behaves conceptually as intended, while remaining decoupled
from the real Qdrant and Neo4j infrastructure.

---

## 8. Architecture 2 – “Cascading Waterfall” Concept

Having stabilized the core plumbing, we stepped back to evaluate and redesign the ranking and reranking
workflow more holistically. The key design we converged on is **Architecture 2: Cascading Waterfall**.

### 8.1 Problems With the Current (Reranker-Override) Flow

In the current hybrid + reranker implementation (even with normalization fixes), the pipeline effectively
behaves as:

1. **Qdrant recall**: Dense + sparse + ColBERT are combined (currently via RRF) to produce a candidate
   list.
2. **Reranker override**: The reranker is applied to the top N candidates, and seeds are re-ordered almost
   purely by `rerank_score`.
3. **Ranking**: The Ranker uses `rerank_score` as the semantic score and adds smaller contributions from
   graph distance and recency.
4. **Graph neighbors**: Neighbors added after reranking often have little or no semantic signal and are
   heavily penalized compared to seeds.

This causes several problems:

- **Signal Loss:** Dense, sparse, and ColBERT consensus is largely thrown away once reranker scores are
  available.
- **Reranker Overreach:** The reranker becomes the single source of truth for semantics, even though it
  only sees `(query, text)` and lacks knowledge of vector consensus or graph structure.
- **Graph Weakness:** Neighbors are treated as second-class results; their relevance is not appropriately
  propagated from the seeds they are attached to.

### 8.2 Cascading Waterfall Goals

Architecture 2 aims to:

1. Preserve the **vector consensus** (Dense + Sparse + ColBERT) as a first-class signal.
2. Use the **Cross-Encoder reranker** to correct or refine this consensus, not to overwrite it.
3. Let **graph neighbors inherit semantic strength** from their seeds via score propagation, rather than
   starting at zero.
4. Provide a transparent, tunable **weighted ensemble** that combines all signals without requiring a
   full-blown learned model.

### 8.3 Proposed Formula

We defined a conceptual formula:

- Let `S_recall` = vector-based recall score (e.g., weighted fusion of Dense, Sparse, ColBERT).
- Let `S_rerank` = reranker logit (Cross-Encoder output).
- Let `S_graph` = graph-based score derived from distance (1.0 for seed, decaying for neighbors).

Then:

- `Recall_Norm = S_recall / max(S_recall_batch)` (per-batch normalization).
- `Rerank_Prob = sigmoid(S_rerank)`.
- Final combined semantic+graph score might be:

  ```
  FinalScore = (W_recall * Recall_Norm) + (W_rerank * Rerank_Prob) + (W_graph * S_graph)
  ```

with typical weights like:

- `W_recall = 0.4`
- `W_rerank = 0.4`
- `W_graph = 0.2`

These weights are tunable and could be folded into configuration.
### 8.4 Recall Fusion – Dense, Sparse, ColBERT

We considered two main options for recall fusion:

1. **Client-Side Weighted Fusion:**
   - Retrieve per-field scores from Qdrant (if supported) and compute:
     - `S_recall = W_dense * S_dense + W_sparse * S_sparse + W_colbert * S_colbert`
   - Normalize scores for stability, using per-batch max.
   - Pros:
     - Full control over weights.
     - Easy to experiment with different scalings.
   - Cons:
     - Requires either multiple queries or access to internal per-field scores.

2. **Qdrant Fusion Modes:**
   - Use Query API’s native fusion modes (e.g., RRF, DBSF) with field-specific weights.
   - Current configuration uses RRF; we would likely move to a magnitude-preserving mode (e.g., DBSF) or a
     simpler scheme (max or sum) to avoid purely rank-based scores.

We agreed that **ColBERT should not be underweighted**. A proposed weight set is:

- Dense: 1.0
- ColBERT (late-interaction): 1.0
- Sparse: 0.5

This gives equal importance to dense semantics and fine-grained token-level matching, with sparse vectors
as an additional but slightly noisier signal.

### 8.5 Reranker as a Feature, Not Dictator

Instead of letting `rerank_score` completely override vector scores, Architecture 2 treats rerank as a
**refinement feature**:

- Compute `Rerank_Prob = sigmoid(S_rerank)`.
- Combine with `Recall_Norm` via a blend like:
  - `Semantic = 0.5 * Recall_Norm + 0.5 * Rerank_Prob`.
- Add a **veto mechanism**:
  - If `Rerank_Prob` < some threshold (e.g., `0.2`), treat the candidate as invalid (final score ~0), even
    if the recall score is high.
  - This protects against keyword-stuffed but semantically wrong matches (e.g., document contains many
    query terms but states the opposite of the desired fact).

### 8.6 Graph Score Propagation

The final part of Architecture 2 is to more fairly treat graph neighbors:

- When a seed chunk has `Semantic_Score`, neighbors discovered via adjacency expansion should inherit a
  portion of that score.
- For example:

  ```
  neighbor.semantic_score = seed.semantic_score * 0.85
  neighbor.graph_score = exp(-0.5 * distance)
  ```

- This prevents neighbors from starting at zero semantic confidence.
- It respects the intuition that “chunks near a very good match are themselves likely to be relevant,”
  especially in structured docs like best-practice guides or step-by-step procedures.

We have **not yet implemented** this graph propagation logic, but it is a key part of the target design.

---

## 9. Current Implementation Status vs Architecture 2

At the end of this session, we have implemented some, but not all, of Architecture 2.

### 9.1 Implemented

1. **Hybrid Mode and BM25 Gating**
   - `HybridSearchConfig.mode` added to config models.
   - `HybridRetriever` uses `mode` and per-mode BM25 enabling to decide whether to construct and use
     `BM25Retriever`.
   - In `bge_reranker` mode, BM25 search is skipped, and vector recall is used alone.

2. **Local BGE Reranker Integration**
   - `BGERerankerServiceProvider` implemented, which talks to a local BGE reranker service at
     `/v1/rerank`.
   - `ProviderFactory.create_rerank_provider` extended to support `bge-reranker-service`.
   - Config updated to use this provider in development.

3. **Reranker-Driven Semantic Score**
   - In `Ranker._extract_features`, `rerank_score` now takes precedence:
     - If present, `semantic_score = sigmoid(rerank_score)`.
   - This ensures evidence confidence and final ranking are directly influenced by the cross-encoder.

4. **Vector-Only Fusion in bge_reranker Mode**
   - In HybridRetriever’s `retrieve`, when mode is `bge_reranker`:
     - Only vector results are used as seeds (`fused_results = vec_results`).
     - `fused_score` is set from `vector_score` where needed.

5. **Focused Test for Reranker Behavior**
   - `tests/query/test_reranker_mode.py` confirms that:
     - BM25 is skipped when mode is set to `bge_reranker`.
     - Reranker and ranker cooperate to produce semantic scores via sigmoid(rerank_score).

### 9.2 Not Yet Implemented

1. **True Weighted Fusion of Dense/Sparse/ColBERT**
   - We have not yet replaced RRF inside Qdrant with a magnitude-preserving fusion mode.
   - We also have not yet exposed per-field scores from Qdrant and combined them client-side.

2. **Blending Recall and Rerank Scores**
   - Current code uses rerank_score as the main semantic signal when present.
   - Architecture 2 requires combining `Recall_Norm` (vector-based) and `Rerank_Prob` (reranker-based) into
     a single semantic score.

3. **Graph Score Propagation**
   - We have not yet implemented the inheritance of seed semantic scores by neighbors.
   - Neighbors are still treated as primarily graph-driven with minimal semantic signal.

4. **Reranker Veto**
   - No explicit veto threshold is implemented yet for low rerank probabilities.
   - Bad matches with high recall scores are not yet explicitly suppressed based on rerank scores.

5. **Refined Weight Tuning and Config-Driven Weights**
   - The weights for recall vs rerank vs graph (e.g., 0.4/0.4/0.2) are not yet surfaced as config knobs.
   - Dense/sparse/ColBERT weights are still at their legacy settings.

---

## 10. Files Touched and Summary of Modifications

This section lists the key files that were touched during this session, and summarizes the changes.

### 10.1 Configuration and Models

- `config/development.yaml`
  - Added `search.hybrid.mode: "bge_reranker"` for development.
  - Updated `search.hybrid.reranker` to point to `bge-reranker-service` and use `BAAI/bge-reranker-v2-m3`.
  - Set `search.hybrid.bm25.enabled: false` in development to disable BM25 for the new mode.

- `src/shared/config.py`
  - `RerankerConfig` extended with `max_pairs` and `max_tokens_per_pair`.
  - `HybridSearchConfig` extended with `mode: str = "legacy"`.

### 10.2 Providers and Rerank Integration

- `src/providers/factory.py`
  - `create_rerank_provider` extended to:
    - Normalize provider names (e.g., `bge-reranker-service`, `bge-reranker`).
    - Instantiate `BGERerankerServiceProvider` when configured.
  - Ensured that the module-level `create_rerank_provider` wrapper delegates properly.

- `src/providers/rerank/base.py`
  - Already defined `RerankProvider` protocol; no changes needed, but this guided how we implemented the
    BGE reranker provider.

- `src/providers/rerank/local_bge_service.py` (new)
  - Implements `BGERerankerServiceProvider` wrapping an HTTP client to the local reranker service.
  - Maps service responses to the internal candidate format and returns reranked candidates.
### 10.3 Hybrid Retrieval and Ranking

- `src/query/hybrid_retrieval.py`
  - Added `self.hybrid_mode` and conditional BM25 retriever construction.
  - Refactored `retrieve` to:
    - Skip BM25 in `bge_reranker` mode.
    - Use vector-only fused results with `fusion_method = "vector"` when appropriate.
  - Hardened namespace checks and logging to handle `self.bm25_retriever` being `None`.
  - Ensured Query API payload construction uses correct types (lists and SparseVector models) and no
    longer passes dicts to `query_points`.
  - Used reranker provider via `_get_reranker` and `_apply_reranker`, which is now backed by
    `BGERerankerServiceProvider` when configured.

- `src/query/ranking.py`
  - Adjusted `_extract_features` to:
    - Prefer `rerank_score` when present, using `semantic_score = sigmoid(rerank_score)`.
    - Fall back to vector/BM25-based normalization otherwise.
  - Introduced batch-relative normalization for late-interaction scores (partial and subject to future
    refinement as Architecture 2 is implemented).

### 10.4 MCP Server Query Service

- `src/mcp_server/query_service.py`
  - In `_wrap_chunks_as_ranked`, ensured metadata includes relevant fields:
    - `rerank_score`, `rerank_rank`, `reranker`, etc.
  - Delegates ranking to `Ranker.rank`, allowing the new semantic computation logic to be applied.

### 10.5 Tests

- `tests/query/test_reranker_mode.py` (new)
  - Introduces a targeted test for `bge_reranker` mode.
  - Uses monkeypatching and stubs to isolate logic from infrastructure and Graph/Neo4j effects.

---

## 11. What Works vs. What is Still Broken

### 11.1 Working

- **Ingestion with BGE-M3**:
  - Ingestion pipeline successfully parses markdown documents, extracts entities, builds graph chunks, and
    writes embeddings to Qdrant and Neo4j using BGE-M3.

- **Qdrant Query API with Dense/Sparse/ColBERT**:
  - Correct payload types are passed to `query_points`:
    - Dense vectors as `List[float]`.
    - Sparse queries as `SparseVector` models.
    - ColBERT multi-vector queries as lists of lists under `late-interaction`.
  - The smoke test confirms `Multi-vector search completed via Query API` with non-zero durations.

- **HybridRetriever Mode Separation**:
  - `bge_reranker` mode correctly skips BM25 and uses vector-only recall.
  - `legacy` mode retains BM25 + RRF + optional reranker behavior.

- **Local BGE Reranker Integration**:
  - A local BGE reranker service is available and reachable via HTTP.
  - `BGERerankerServiceProvider` can call `/v1/rerank` and transform the results into internal candidate
    dictionaries.

- **Reranker-Driven Semantic Confidence**:
  - When rerank scores are present, `Ranker` uses `semantic_score = sigmoid(rerank_score)`, which produces
    more meaningful confidence values than previous heuristics.

- **Test Coverage for Reranker Mode**:
  - The new unit test passes, giving confidence that mode gating and reranker short-circuiting in the
    Ranker are wired correctly.

### 11.2 Still Broken or Incomplete

- **Confidence Calibration is Still Heuristic**:
  - Although using sigmoid(rerank_score) is an improvement, we are not yet blending vector-based recall
    scores, nor have we implemented reranker veto logic.

- **Graph Neighbors are Still Underweighted**:
  - Neighbors still effectively start with little or no semantic signal; score propagation from seeds has
    not been implemented.

- **Dense/Sparse/ColBERT Fusion is Not Yet Reworked**:
  - We still rely on RRF-like or legacy mechanisms for combining vector signals inside Qdrant.
  - A magnitude-preserving fusion approach (weighted sum, DBSF, or client-side combination) has not been
    implemented.

- **BM25 Removal is Partial**:
  - In development config we disable BM25, but legacy mode still depends on it.
  - Future work is needed to fully decommission BM25 if we rely entirely on BGE-M3 sparse vectors, or keep
    it only for specific use cases.

- **Reranker Service is Not Yet Exercised in Integration Tests**:
  - Most testing so far relies on mocks or unit-level checks.
  - We have not yet added a full integration test that spins up the BGE reranker service, runs a query
    end-to-end, and validates ranking and confidence.

---

## 12. Next Steps by Area

This section outlines concrete next steps for each major area of the architecture.

### 12.1 Retrieval Fusion (Dense/Sparse/ColBERT)

- Evaluate Qdrant Query API support for returning per-field scores or scoring details.
- If feasible:
  - Retrieve per-field scores and compute a client-side weighted sum with tuned weights.
- If not feasible:
  - Use Qdrant’s DBSF or simple max/sum fusion modes with carefully chosen weights.
- Tune weights to:
  - Dense: 1.0
  - ColBERT: 1.0
  - Sparse: 0.5
- Ensure the resulting `vector_score` reflects both strength and consensus among the three modes.

### 12.2 Blending Rerank and Recall Scores

- Modify `Ranker._extract_features` to compute semantic similarity as a blend:

  ```python
  recall_norm = vector_score / max_batch_vector_score
  rerank_prob = sigmoid(rerank_score)
  semantic = 0.5 * recall_norm + 0.5 * rerank_prob
  ```

- Add config-driven weights (e.g., `hybrid.semantic_recall_weight`, `hybrid.semantic_rerank_weight`).
- Preserve the reranker short-circuit only when vector scores are completely absent.

### 12.3 Reranker Veto

- Add a configurable threshold (e.g., `hybrid.reranker_veto_threshold = 0.2`).
- In Ranker, if `rerank_prob < threshold`:
  - Set `semantic_score = 0.0` or drastically downweight the candidate.
- This will allow the pipeline to filter out high-recall but semantically wrong candidates.
### 12.4 Graph Propagation

- In `_apply_graph_enrichment`, implement semantic propagation from seeds to neighbors:
  - For each neighbor, compute:

    ```python
    neighbor.semantic_score = seed.semantic_score * graph_decay
    ```

    where `graph_decay` might be 0.8–0.9 depending on distance.

- Ensure Ranker sees both:
  - `semantic_score` (possibly inherited).
  - `graph_score` (distance-based decay), as separate features.

- Adjust weights so that neighbors can meaningfully compete with weaker seeds but do not overtake strong
  seeds unless graph context is compelling.

### 12.5 Observability and Metrics

- Add reranker-specific metrics:
  - `reranker_latency_ms`, `reranker_calls_total`, `reranker_errors_total`.
- Add metrics for:
  - Distribution of `vector_score` before and after normalization.
  - Distribution of `rerank_prob` and final semantic scores.

These metrics will be crucial for tuning weights and thresholds.

### 12.6 Testing and Validation

- Add integration tests that:
  - Spin up the BGE reranker service alongside the MCP server and other containers.
  - Run the smoke test (`scripts/smoke_test_query.py`) end-to-end with `mode=bge_reranker`.
  - Assert that:
    - Sparse and ColBERT are being used.
    - Reranker is called and its scores influence ordering.
    - Graph neighbors appear as expected after the seed chunk.

- Expand unit tests to cover:
  - Reranker veto behavior.
  - Graph propagation semantics.
  - Blended semantic scores for various combinations of recall and rerank.

---

## 13. Closing Notes

This session represented a major step away from the older Phase 7E “BM25 + RRF + optional reranker” stack
and toward a more modern architecture centered on:

- BGE-M3 embeddings (dense, sparse, ColBERT) for high recall.
- A BGE-based cross-encoder reranker for high precision and calibrated confidence.
- A more principled, ensemble-style ranking layer that can combine vector consensus, rerank scores, and
  graph context.

We implemented several foundational pieces of this architecture:

- A dedicated `bge_reranker` hybrid mode.
- A local BGE reranker service provider integrated through the existing provider factory.
- Changes to `HybridRetriever` to skip BM25 in the new mode and rely on Qdrant Query API.
- A ranking modification that uses rerank scores when present.
- A focused test that validates the new mode’s basic behavior.

At the same time, we identified important remaining work, particularly around blending vector-based and
reranker-based signals and propagating semantic strength through graph neighbors.

The design for **Architecture 2: Cascading Waterfall** provides a clear, incremental path forward that does
not require training additional models, but significantly improves the robustness and transparency of the
ranking pipeline.

Future sessions should focus on implementing the blending logic, reranker veto, and graph propagation; on
replacing RRF with a more information-preserving fusion scheme; and on validating the end-to-end system
with realistic queries and documents. Once those steps are complete, the WekaDocs Matrix retrieval stack
will be well-positioned as a production-ready, BGE-M3 + reranker + graph-augmented RAG system.
