# Integration Guide — Phase‑7E Hybrid Retrieval with Jina Reranker

**Goal:** Enable **BM25 + Vector + RRF → Jina Reranker → Context Expansion** with minimal changes to your current code.

**Repo root:** `/mnt/data/repo`  
**Generated:** 2025-10-28T21:57:22.942738Z

---

## 1) Apply the patch

- File: `/mnt/data/patches/phase7e_hybrid_bm25_rrf_rerank.patch`
- Apply with git from repo root:

```bash
git apply /mnt/data/patches/phase7e_hybrid_bm25_rrf_rerank.patch
```

Creates/updates:
- `src/query/bm25_search.py` (NEW)
- `src/query/rank_fusion.py` (NEW)
- `src/query/hybrid_search.py` (MOD)
- `src/query/ranking.py` (MOD)
- `src/shared/config.py` (MOD)
- `config/development.yaml` (MOD)
- `scripts/neo4j/create_fulltext_index.cypher` (NEW)

---

## 2) Create the Neo4j fulltext index (one‑time)

```cypher
// In Neo4j Browser or cypher-shell
CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.heading];
```

> **Tip:** If your text lives on `:Section` instead of `:Chunk`, adjust the label accordingly.

---

## 3) Config & env

**YAML knobs** (now available under `search.hybrid`):
```yaml
search:
  hybrid:
    top_k: 20
    vector_top_k: 50
    bm25_top_k: 50
    rrf_k: 60
    rerank_candidates: 100
    rerank_top_k: 20
```

**Reranker env (if not already set):**
```bash
export RERANK_PROVIDER=jina-ai
export RERANK_MODEL=jina-reranker-v2-base-multilingual
export JINA_API_KEY=...  # required
```

---

## 4) Pipeline order (exactly how it runs now)

1. **Vector recall** (`vector_top_k` or `rerank_candidates` if rerank enabled)
2. **BM25 recall** (same K as above)
3. **RRF fusion** (`rrf_k=60`) → fused shortlist
4. **Jina Reranker** on the fused shortlist (cap by `rerank_candidates`) → keep `rerank_top_k`
5. **Graph expansion** (your existing `_expand_from_seeds` / `_find_connecting_paths`)
6. **Final ranking** (`src/query/ranking.py`)

> This matches your intended “recall → precision → context” layout.

---

## 5) Observability & fallbacks

- If fulltext index is absent or Neo4j rejects the call, BM25 returns `[]` and the pipeline behaves **vector‑only**.
- Reranker time is recorded in `HybridSearchResults.rerank_time_ms` (already present).
- Keep your existing timeouts/circuit breakers; the BM25 call is a single Cypher query.

---

## 6) Tuning cheatsheet

- **Latency budget tight?** Lower `rerank_candidates` to 60–80.
- **Recall too shallow?** Increase `vector_top_k` & `bm25_top_k` to 80 each.
- **Reranker very strong?** Use `rerank_top_k: 15` if answers get long.
- **Too keyword‑heavy corpus?** Lower `alpha` in your ranker or boost `bm25_top_k` (RRF will balance).

---

## 7) Quick validation steps

1. **Smoke test** a few queries — expect similar or better top‑3 relevance.
2. **Check rerank timing** in logs (should be stable; increases with `rerank_candidates`).
3. **Relevance spot‑check** short, ambiguous queries — Jina reranker should fix lexical false positives.
4. **No index?** Run the Cypher above and retry — BM25 results will appear.

---

## 8) Rollback plan

- Revert the patch or set `bm25_top_k: 0` in YAML to effectively disable BM25.
- Set rerank provider to `noop` to disable reranking while keeping the fusion in place.

```bash
export RERANK_PROVIDER=noop
```

---

## 9) Future optional improvements

- Weighted fusion fallback (α‑blend) alongside RRF for A/B.
- Per‑doc filters for BM25 (e.g., restrict by `document_id` or domain).
- Telemetry: log pre‑ vs post‑rerank MRR/Hit@k for learning.

---

**You’re done.** The agent can ship this change set safely and measure uplift immediately.
