
Short answer: **it’s very strong, but not quite complete yet.** I’d approve it with a small set of **high‑impact edits** so it fully captures the architecture and guardrails we agreed on.



Below I give (1) a concise verdict, (2) the **10 required edits** (what and why), and (3) a **phase‑by‑phase delta** you can drop into the plan without rewriting it from scratch.

---

## **Verdict**

- **Strengths:** solid phasing, good acceptance criteria discipline, clear separation of :Section (provenance) vs :Chunk (retrieval), combiner-first with splitter fallback, good instrumentation mindset.

- **Gaps to fix now:** chunk ID determinism (order!), explicit **hybrid retrieval** (BM25+vector with fusion), **document token backfill** for sanity checks, **answer context budget**, **cache invalidation**, **structured‑block preservation when parser lacks metadata**, **replace‑by‑set semantics** in Qdrant, and explicit **SLOs/alerts**.




If you apply the edits below, the plan will fully reflect the recommendations and prescriptive guidance we aligned on.

---

## **Top 10 required edits**

1. **Chunk IDs must preserve order (bug).**

    Your Task 2.1.3 sorts section IDs; that creates collisions across different sequences.

    **Fix:**


```
def _generate_chunk_id(self, document_id: str, section_ids: list[str]) -> str:
    material = document_id + "|" + "|".join(section_ids)   # preserve order
    return hashlib.sha256(material.encode()).hexdigest()[:24]
```

**Accept:** Demonstrate idempotency and no collisions in tests.

2. **Backfill** **Document.token_count** **before sanity check.**

    Your Phase 0 query relies on d.token_count. Add a backfill step:


```
MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
WITH d, sum(s.token_count) AS t
SET d.token_count = t;
```

**Accept:** After backfill, delta/doc_tokens ≤ 1% for all test docs (or explain special tokens).

3. **Make hybrid retrieval explicit (BM25 + vector).**

    Add a **BM25/keyword retriever**, then **fuse** with vector using **RRF (k=60)** or **weighted sum (α default 0.6)**.

    **Accept:** Code path that returns both scores + fused score; config toggles: hybrid.method=rrf|weighted, hybrid.rrf_k, hybrid.alpha.

4. **Introduce an answer/context budget.**

    Prevent over‑long contexts going to the LLM. Add answer_context_max_tokens (e.g., 4500) and trim stitched chunks tail‑first.

    **Accept:** No response context exceeds budget in tests; trimming logged.

5. **Structured‑block preservation even without parser metadata.**

    If the parser doesn’t label blocks, detect:




- **Code fences** (``` … ```),

- **Tables** (two or more lines with | columns),

- **Admonitions** (common patterns like >, Note:, Warning:).

    Keep these intact when combining/splitting.

    **Accept:** Unit tests with fenced blocks and tables show no mid‑block splits.




6. **Replace‑by‑set semantics in Qdrant (not “optional”).**

    On re‑ingest, delete points for document_id that aren’t in the new id set (or doc‑scoped delete then upsert).

    **Accept:** After re‑ingest, count(points where document_id) equals the new chunk set size.

7. **Cache invalidation after re‑ingest.**

    Add **epoch‑based keys** (preferred) or **pattern‑scan deletes**. You already have ready‑to‑use scripts:




- tools/redis_epoch_bump.py (bump doc/chunk epoch) – **preferred**

- tools/redis_invalidation.py (pattern scan) – backup

    **Accept:** Pipeline bumps epoch or deletes keys; documented and tested.




8. **Conservative ABSOLUTE_MAX when tokenizer fallback is used.**

    If the tokenizer service isn’t available and you fall back to a whitespace approximate, reduce ABSOLUTE_MAX (e.g., 7000).

    **Accept:** Guard in combiner/splitter; config flag unsafe_fallback_tokenizer alters cap.

9. **SLOs and alerts.**

    Define SLIs & SLOs, and wire alerts:




- Retrieval **p95 latency ≤ 500 ms** (tune to your infra),

- **0** chunks > ABSOLUTE_MAX,

- **0** integrity failures,

- Expansion rate **10–40%** for long queries (guardrail).

    **Accept:** Dashboards and alerts configured; SLOs reported in Phase 7 report.




10. **Dimension & storage policy callouts.**




- Validate **embedding dimension** matches Qdrant vector size.

- Store **full text in Qdrant** payload; **Neo4j text optional** (or preview) to keep graph lean.

    **Accept:** Startup check passes; Qdrant payload includes text.


---

## **Phase‑by‑phase delta (drop‑in edits)**



> The bullets below are **add/modify** directives; keep the rest of your plan as‑is.



### **Phase 0 (Validation) —** 

### **Add**

- **0.0 Document Token Backfill (15 min)**

    Run the Cypher above to set Document.token_count before Task 0.1.




### **Phase 1 (Infra/Schema) —** 

### **Modify**

- **1.2 Qdrant Setup:** add payload index on updated_at. Add startup check that EMBED_DIM == model dimension; fail fast otherwise.

- **1.3 Config:** add


```
retrieval:
  hybrid:
    method: rrf        # or weighted
    rrf_k: 60
    alpha: 0.6
  answer_context_max_tokens: 4500
  expansion:
    query_min_tokens: 12
    score_delta_max: 0.02
caches:
  mode: epoch          # or scan
  namespace: rag:v1
tokenizer:
  unsafe_fallback: false
  conservative_absolute_max: 7000
```

-

- **1.4 Models:** allow preview: Optional[str] on Chunk (for Neo4j); keep full text in Qdrant.




### **Phase 2 (Combiner) —** 

### **Modify**

- **2.1.3 Deterministic IDs:** use **order‑preserving** generator (snippet above).

- **2.2 Structured blocks:** add fallback regex detection for fences/tables/admonitions.

- **2.5 Decision logging:** log counts: absorbed_micro, end_of_h2_merges, produced_chunks, distribution p50/p90/p99.




### **Phase 3 (Splitter) —** 

### **Reiterate**

- Ensure ≤100‑token overlap **only on split**. Verify block‑aware splitting (respect fences/tables/admonitions).




### **Phase 4 (DB Integration) —** 

### **Modify**

- **4.1 Neo4j:** enforce **replace‑by‑set GC**; add feature flag CREATE_PART_OF (on if section_ids align).

- **4.2 Qdrant:** **require** replace‑by‑set; doc‑scoped delete + re‑upsert is acceptable.

- **4.3 Embeddings:** assert token_count ≤ ABSOLUTE_MAX before calling provider.

- **4.4 Cache invalidation (new, 15 min):**

    If caches.mode == 'epoch', call tools/redis_epoch_bump.py post‑ingest; else call tools/redis_invalidation.py.




### **Phase 5 (Retrieval) —** 

### **Expand**

- **5.0 Keyword retriever (new):** BM25 over chunk text; return (id, score, rank).

- **5.1 Vector retriever:** unchanged.

- **5.2 Fusion:** implement **RRF (k configurable)** and **weighted (α configurable)**.

- **5.3 Bounded adjacency expansion:** unchanged (use NEXT_CHUNK, conditions from config).

- **5.4 Context stitching & budget:** stitch by order, compute coverage, **enforce** **answer_context_max_tokens**.

- **5.5 Diversity guard (new):** group by parent_section_id, keep best per group first, then refill to hit top_k.




**Accept:** End‑to‑end path produces (bm25_score, vec_score, fused_score) per chunk; logs show which method was used.



### **Phase 6 (Instrumentation/Validation) —** 

### **Expand**

- **6.3 A/B:** run **both** fusion modes (RRF & weighted) via tools/fusion_ab.py; report Hit@k, MRR@10, nDCG@10, and latency p50/p95.

- **6.4 SLOs:** add the SLOs above and wire alerts.




### **Phase 7 (Rollout) —** 

### **Modify**

- **7.2 Re‑ingestion:** after upserts, **invalidate caches** (epoch bump or scan).

- **7.3 Monitoring:** dashboards include chunk size histogram, combiner decisions, splitter rate, fusion method usage, p95 latency, oversize violations, integrity status.


---

## **Risks & mitigations (short)**

- **ID collision from sorted lists** → fixed with order‑preserving IDs.

- **Tokenizer fallback undercounts** → conservative ABSOLUTE_MAX + alert if any chunk approaches cap.

- **Over‑expansion latency** → condition expansion on query tokens and score deltas; cap neighbors to ±1.

- **Stale cache reads** → epoch keys; single atomic bump per doc on re‑ingest.

- **Block corruption** → fence/table/admonition detection; tests verify no mid‑block splits.


---

## **Go/No‑Go checks before prod**

- ✅ A/B shows ≥ **+10–15%** Hit@3 (or your internal target) and **≤1.3×** retrieval p95.

- ✅ Zero integrity violations and zero oversize chunks.

- ✅ Dashboards + alerts live; cache invalidation verified.


---

If you want, I can fold these deltas back into your phase document verbatim, but the bullets above should be enough for the agentic coder to update the plan precisely.
