# Sparse Vector Coverage Policy (B.3)

**Date:** 2025-11-26
**Status:** Adopted
**Decision:** Policy C (Hybrid)

---

## Context

Analysis revealed ~17% of chunks in `chunks_multi_bge_m3` lack sparse vectors. Investigation identified two root causes:

1. **Capability Gating:** If `supports_sparse` is False, no sparse embeddings are generated
2. **Batch-Level Error Poisoning:** In `_process_embeddings`, a single `embed_sparse` failure set `sparse_embeddings = None` for ALL remaining chunks in the document

The second issue was particularly problematic: one transient API error would cause an entire document's worth of chunks to lose sparse vectors.

---

## Policy Options Evaluated

### Policy A – Intentionally Sparse-less Stubs
- **Approach:** Accept that stubs/zero-token chunks don't get sparse vectors
- **Pros:** Simple, aligns with stub semantics
- **Cons:** Doesn't address non-stub chunks that lost sparse due to errors

### Policy B – Sparse for All Non-Empty Chunks
- **Approach:** Guarantee sparse for any chunk with `token_count > 5`
- **Pros:** Maximum sparse coverage
- **Cons:** May fail on edge cases (very short API descriptions, etc.)

### Policy C – Hybrid (Selected)
- **Approach:** Stubs remain sparse-less by design; all other chunks require sparse with per-batch error isolation
- **Pros:** Balances coverage with pragmatic error handling
- **Cons:** Slightly more complex tracking

---

## Decision: Policy C (Hybrid)

### Rules

1. **Microdoc stubs** (`is_microdoc_stub: true`) are intentionally sparse-less
2. **All other chunks** with `token_count >= 5` should have sparse vectors
3. **Error isolation** ensures one failed batch doesn't affect others

### Implementation

The batch-level error poisoning has been fixed in B.2:

```python
# OLD (batch-level error poisoning):
except Exception as exc:
    sparse_embeddings = None  # Kills ALL remaining sparse

# NEW (per-batch isolation):
except Exception as exc:
    # Insert None placeholders for failed batch only
    sparse_embeddings.extend([None] * len(batch_content))
```

### Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Sparse coverage | ~83% | ≥95% |
| Stubs without sparse | Expected | Expected |
| Non-stub without sparse | 14%+ | <2% |

---

## Metrics & Guardrails

### Ingestion Metrics
- `sparse_coverage_ratio`: % of non-stub chunks with sparse vectors
- `sparse_failures_per_batch`: Count of batches that failed sparse generation

### Retrieval Metrics
- `sparse_scored_ratio`: % of returned chunks with sparse scores
- `sparse_topk_ratio`: % of top-K results with sparse vectors

### Alerts
- **Warning:** If `sparse_coverage_ratio < 90%` after re-ingestion
- **Critical:** If `sparse_failures_per_batch > 5` for a single document

---

## Validation Checklist (Post Re-Ingestion)

- [ ] Run Qdrant scan: count points with non-empty `text-sparse`
- [ ] Verify `sparse_coverage_ratio >= 95%`
- [ ] Confirm microdoc stubs are intentionally sparse-less
- [ ] Run eval harness on lexical-heavy queries to confirm sparse impact

---

## Appendix: Root Cause Analysis

### Code Location
`src/ingestion/build_graph.py` → `_process_embeddings()` (lines 1541-1561)

### Error Pattern
The BGE-M3 or Jina sparse embedding API occasionally fails on:
- Very long texts exceeding token limits
- Texts with unusual Unicode characters
- Transient API timeouts

With per-batch isolation, these failures now affect only the specific chunks in the failed batch rather than the entire document.

---

*Document Author: Claude Code (automated)*
*Review Status: Implementation complete, awaiting re-ingestion validation*
