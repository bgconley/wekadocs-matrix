# Line-by-line Evaluation & Integration Check — Phase‑7E (BM25 + Vector + RRF + Jina Reranker)

**Repo root:** `/mnt/data/repo`  
**Generated:** 2025-10-28T21:57:22.942738Z

This review compares your **current code** with the **proposed architecture** you shared. It flags exact insertion points, confirms what already exists, and provides a ready-to-apply patch.

---

## What you already have (confirmed in code)

1. **Jina Reranker provider is present and production-grade**
   - File: `src/providers/rerank/jina.py` — calls `https://api.jina.ai/v1/rerank`, handles retries & rate limiting.
   - Factory wiring: `src/providers/factory.py` (`RERANK_PROVIDER=jina-ai`, `RERANK_MODEL=jina-reranker-v2-base-multilingual`).

2. **Reranking is already integrated into the pipeline**
   - File: `src/query/hybrid_search.py`
     - Reranking applied in `_apply_reranking(...)` and used in `HybridSearchEngine.search(...)` right **after vector candidates** and **before graph expansion**.
     - Lines ~260–305: vector seeds, then `results = self._apply_reranking(...)`.

3. **Graph expansion & ranking layers exist**
   - `HybridSearchEngine._expand_from_seeds(...)` and `_find_connecting_paths(...)`.
   - `src/query/ranking.py` blends semantic, graph distance, recency, and entity priority.

4. **Config system present (YAML + Pydantic)**
   - `src/shared/config.py` and `config/development.yaml`.

---

## What’s missing vs. the proposal

1. **No BM25/keyword candidate generator** (Neo4j fulltext)
   - No files named BM25 or fulltext; no `CALL db.index.fulltext.queryNodes` usage.

2. **No rank fusion (RRF) stage**
   - Only vector results are reranked; sparse leg not used.

3. **No knobs for `vector_top_k`, `bm25_top_k`, `rrf_k`, `rerank_candidates`**
   - Current `HybridSearchConfig` exposes only `top_k` and weights.

4. **No fulltext index DDL in the repo** for chunk text.

---

## Minimal patch set (adds BM25 + RRF; keeps your reranker exactly where we want it)

> **Exactly as recommended:** Dense (Qdrant) + Sparse (BM25 via Neo4j fulltext) → **RRF** → **Jina Reranker** → Graph expansion → Final ranking.

### Files changed / added

- **NEW** `src/query/bm25_search.py` — Neo4j fulltext BM25 retriever  
- **NEW** `src/query/rank_fusion.py` — RRF implementation  
- **MOD** `src/query/hybrid_search.py` — Wire hybrid recall + RRF, then rerank  
- **MOD** `src/query/ranking.py` — Recognize `bm25`/`fused` score kinds  
- **MOD** `src/shared/config.py` — Add knobs (`vector_top_k`, `bm25_top_k`, `rrf_k`, `rerank_candidates`, `rerank_top_k`)  
- **MOD** `config/development.yaml` — Provide sane defaults  
- **NEW** `scripts/neo4j/create_fulltext_index.cypher` — Create `chunk_text_index`

### Unified patch

See: `patches/phase7e_hybrid_bm25_rrf_rerank.patch`

> Contains all diffs, including the two new Python modules and the Neo4j DDL.

---

## Key insertion points (file & line references)

- `src/query/hybrid_search.py`
  - **Constructor** (≈200–216): adds knobs + `BM25Searcher(...)` initialization.
  - **Search() step 1** (replaces vector-only block around lines **261–305**): now runs **vector** and **BM25** in parallel, **fuses with RRF**, truncates to `rerank_candidates`, and then enters the **existing** rerank step.
  - **Reranking**: left intact; still called **before** graph expansion — this is the desired position.

- `src/query/ranking.py`
  - `_normalize_score(...)`: minor extension so `bm25` / `fused` / `rrf` go through unchanged (pass‑through), avoiding incorrect normalization.

- `src/shared/config.py`
  - `HybridSearchConfig`: adds the extra knobs; defaults chosen to match your plan.
    - `vector_top_k=50`, `bm25_top_k=50`, `rrf_k=60`, `rerank_candidates=100`, `rerank_top_k=20`

- `config/development.yaml`
  - Mirrors the new knobs with the same defaults.

- `scripts/neo4j/create_fulltext_index.cypher`
  - One‑liner to create the **Chunk** fulltext index on `(text, heading)`.

---

## Edge‑case checks (done against current code)

- **Provider wiring:** `ProviderFactory.create_rerank_provider(...)` already supports Jina + Noop; no changes required.
- **Reranker API:** Jina reranker is called with `Authorization: Bearer $JINA_API_KEY` — consistent with your env expectations.
- **SearchResult flow:** RRF keeps IDs and we de‑duplicate while preserving RRF order; your downstream ranking consumes the same `SearchResult` type, so no changes beyond the score_kind normalization were needed.
- **Focus bias:** `_apply_reranking(...)` already includes optional entity‑focus boost; left untouched.

---

## Risks & mitigations

- **No existing fulltext index** → Include `scripts/neo4j/create_fulltext_index.cypher`; run it once per environment.
- **Score scale mismatch** → RRF is rank‑based, robust to BM25 vs vector scale; downstream ranking uses pass‑through for `bm25` / `fused`.
- **Latency** → Use `rerank_candidates=100` cap. BM25 query is typically sub‑50ms on indexed text; RRF cost is negligible.
- **Fallbacks** → If the fulltext index call fails, BM25 returns `[]` and the pipeline gracefully behaves as vector‑only (your current behavior).

---

## Exactly what to review

1. Apply the patch and verify type checks + unit tests compile.
2. Ensure `scripts/neo4j/create_fulltext_index.cypher` is run (once).
3. Set (if not already):
   - `RERANK_PROVIDER=jina-ai`
   - `RERANK_MODEL=jina-reranker-v2-base-multilingual`
   - `JINA_API_KEY=...`
4. Tune knobs in `config/development.yaml` if needed.

---

## Appendix — Quick sanity diffs (snippets)

> See the full patch for all details; the lines below show the highest‑leverage changes where reviewers should focus.

**`src/query/hybrid_search.py` (search step)**

```diff
- # Step 1: Vector search for seed nodes
- query_vector = self.embedder.embed_query(query_text)
- vector_seeds = self.vector_store.search(query_vector, k=vector_k, filters=filters)
+ # Step 1: Hybrid recall (Vector + BM25) with RRF fusion (Phase 7E)
+ vector_k = self.rerank_candidates if self.rerank_enabled else self.vector_top_k
+ bm25_k   = self.rerank_candidates if self.rerank_enabled else self.bm25_top_k
+ query_vector = self.embedder.embed_query(query_text)
+ vector_hits  = self.vector_store.search(query_vector, k=vector_k, filters=filters)
+ bm25_hits    = self.bm25.search(query_text, top_k=bm25_k)
+ # RRF fuse → results
```

**`src/shared/config.py` (new knobs)**

```diff
 class HybridSearchConfig(BaseModel):
     vector_weight: float = 0.7
     graph_weight: float = 0.3
     top_k: int = 20
+    vector_top_k: int = 50
+    bm25_top_k: int = 50
+    rrf_k: int = 60
+    rerank_candidates: int = 100
+    rerank_top_k: int = 20
```

---

**Done.** This aligns the codebase with the architecture you described, while preserving your existing reranker integration and keeping the change set minimal and reviewable.
