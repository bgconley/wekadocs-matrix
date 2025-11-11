Of course. I have integrated the requested changes into the pseudocode document. The updated version is below.

---

# Pseudocode – Deterministic Retrieval Upgrade

> **Legend**
> `::` = module boundary | `#` = comment | ⚙ = config

---

## 1. Ingest Path (Phase 1 & 2)
```python
module src/ingestion/build_graph.py

# === extract_doc_tag =====================================================
TAG_RE = re.compile(r"\b([A-Z]+-\d+)\b", re.I)

def extract_doc_tag(source_uri: str, content: str) -> str | None:
    m = TAG_RE.search(content[:8192] or "")
    if not m:
        m = TAG_RE.search(source_uri or "")
    return m.group(1).upper() if m else None

# === derive_doc_key ======================================================

def derive_doc_key(source_uri: str, title: str, top_headings: list[str]) -> str:
    slug = slugify(source_uri)              # e.g., "docs_api_users"
    basis = title + "|" + "|".join(top_headings[:3])
    return f"{slug}_{xxhash64(basis)[:8]}"

# === ingest_document =====================================================

def ingest_document(source_uri, content, format="markdown"):
    doc_tag = extract_doc_tag(source_uri, content)
    doc_key = derive_doc_key(source_uri, title, top_heads)
    doc_id  = uuid4()

    create_document_node(id=doc_id, doc_key=doc_key, doc_tag=doc_tag, ...)

    for chunk in chunker(content):
        chunk_id = uuid4()
        create_chunk_node(id=chunk_id, doc_id=doc_id, doc_tag=doc_tag, ...)
        qdrant.upsert(vector=embed(chunk.text),
                      payload={"doc_id": doc_id, "doc_tag": doc_tag, ...})
```

---

## 2. Query Entry Point (`QueryService.search`)
```python
module src/mcp_server/query_service.py

def search(query: str, *, verbosity="graph", filters=None):
    filters = filters or {}
    ambiguous_router = False
    ambiguous_reranker = False
    reranker_ok = False

    # Phase 1 — detect legacy tag
    if m := TAG_RE.search(query):
        filters["doc_tag"] = m.group(1).upper()
        logger.info("Detected doc_tag=%s", filters["doc_tag"])

    # Phase 2 — parent-level routing
    router_cfg = cfg.router
    cand_docs = doc_router.hybrid_search(query, k=router_cfg.k_docs * 24)

    # After router.hybrid_search():
    scores = normalize_minmax([d.score for d in cand_docs])  # per-query normalization
    margin = scores[0] - scores[1] if len(scores) > 1 else 1.0
    if margin < cfg.router.thresholds.margin_mu:
        ambiguous_router = True

    # If reranker enabled:
    if cfg.reranker.enabled:
        r = reranker.rerank(query, [d.summary for d in cand_docs], timeout_ms=cfg.reranker.timeouts_ms)
        if r.timed_out:
            log.warn("reranker_timeout", doc_stage=True)
            reranker_ok = False
        else:
            reranker_ok = True
            reranker_margin = r.top1.score - r.top2.score if len(r) > 1 else 1.0
            cand_docs = order_by_rr(r, cand_docs) # Reorder candidates based on reranker scores

        if reranker_ok and reranker_margin < cfg.reranker.thresholds.doc_margin_mu:
            ambiguous_reranker = True

    chosen_docs, ambiguous_decision = choose_docs(cand_docs, router_cfg) # Original threshold logic

    if ambiguous_router or (reranker_ok and ambiguous_reranker) or ambiguous_decision:
        return AMBIGUOUS_RESPONSE(cand_docs[:2], diagnostics=...)

    # Child-level retrieval scoped to doc_ids
    chunks = hybrid_retriever.retrieve(query,
                                       filters={"doc_id": chosen_docs})

    ctx = context_assembler.assemble(chunks)
    return StructuredResponse(answer=ctx.to_markdown())
```

---

## 3. Hybrid Retriever (`HybridRetriever.retrieve`)
```python
module src/query/hybrid_retrieval.py

# This is the second-stage, chunk-level retrieval, strictly scoped to chosen doc_ids
def retrieve(query, filters=None):
    chosen_doc_ids = filters.get("doc_id")

    # ensure strict scoping even on fallback:
    chunks = chunk_index.search(query, k=cfg.child.n_chunks, filter={"doc_id": {"$in": chosen_doc_ids}})

    if cfg.reranker.enabled:
        rr = reranker.rerank(query, [c.text for c in chunks], timeout_ms=cfg.reranker.timeouts_ms)
        if rr.timed_out:
            log.warn("reranker_timeout", chunk_stage=True)
            ranked = chunks  # fallback: hybrid order, still strictly scoped
        else:
            ranked = order_by_rr(rr, chunks)[:cfg.child.m_chunks]
    else:
        ranked = chunks[:cfg.child.m_chunks]

    # Expansion would happen here on 'ranked' if enabled, still respecting doc_id scope
    return ranked
```

---

## 4. Citation Assembly (ordering fix)
```python
module src/query/context_assembly.py

for chunk in stitched:
    labels = sorted({(int(o or 0), t.strip().lower())
                     for o, t in (chunk.citation_labels or []) if t})
    if not labels:
        labels = [(0, chunk.heading or "Section")]
    for order, title in labels:
        parts.append(f"[{counter}] {title}")
```

---

## 5. Jina Reranker Client Wrapper
```python
module src/shared/jina_client.py

class JinaReranker:
    def __init__(self, api_key, model="jina-reranker-v2-base-multilingual"):
        self._client = JinaClient(api_key)
        self._model = model
        self._cache = RedisCache(ttl=cfg.reranker.cache_ttl_sec)

    def rerank(self, query: str, docs: list[DocLike]) -> list[DocLike]:
        payload = [d.text for d in docs]
        cache_key = sha256(query + "|" + sha256_array(payload))
        if scores := self._cache.get(cache_key):
            return sort_by(scores, docs)

        scores = self._client.rerank(query, payload, model=self._model)
        self._cache.set(cache_key, scores)
        return sort_by(scores, docs)
```

---

## 6. Threshold Decision Helper```python
# choose_docs ------------------------------------------------------------

def choose_docs(cand_docs: list[ScoredDoc], cfg) -> tuple[list[str], bool]:
    top1, top2 = cand_docs[0], cand_docs[1] if len(cand_docs) > 1 else None
    if top1.score >= cfg.delta and (not top2 or top1.score - top2.score >= cfg.margin):
        return [top1.id], False
    if cfg.k_docs == 2 and top2:
        return [top1.id, top2.id], False
    return [top1.id, top2.id] if top2 else [top1.id], True  # ambiguous
```

---

## 7. Migration Script Skeleton
```python
# scripts/migrations/backfill_doc_tag.py
with neo4j_session() as s:
    docs = s.run("MATCH (d:Document) RETURN d.id AS id, d.source_uri AS uri")
    for doc in docs:
        tag = extract_doc_tag(doc['uri'], None)
        if tag:
            s.run("MATCH (d:Document {id:$id}) SET d.doc_tag=$tag", id=doc['id'], tag=tag)```

---

**End of Pseudocode**
