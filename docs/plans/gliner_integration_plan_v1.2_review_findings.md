# GLiNER Integration Plan Review Findings

**Reviewed:** December 2025
**Plan Version:** 1.2 (Integrated with Codebase Architecture)
**Reviewer:** Claude Code (Opus 4.5)

---

## Executive Summary

This document contains a critical review of the GLiNER integration plan against the existing wekadocs-matrix codebase architecture and session context.

---

## What the Plan Gets Right

### 1. Leveraging Existing `_mentions` Architecture
The plan correctly identifies that `atomic.py` already generates `entity-sparse` vectors from `_mentions`. This is the right integration point—additive enrichment rather than parallel infrastructure.

### 2. Soft Filtering (Boosting) Over Hard Filtering
```python
# The plan correctly avoids this:
filter={"must": [{"entity_values": "weka"}]}  # Zero recall if no match

# In favor of this:
should_clauses.append(rest.FieldCondition(...))  # Boost, don't exclude
```
This prevents the classic "filter kills recall" problem.

### 3. Singleton Pattern for Model Loading
GLiNER models are ~200-340MB. Loading once and reusing is correct. The pattern avoids the catastrophic mistake of loading per-chunk.

### 4. Batch Processing with M1 Max Optimization
Batch size of 32 for MPS is a reasonable starting point. The plan acknowledges hardware constraints.

### 5. Domain-Specific Labels via Config
Making labels configurable rather than hardcoded is the right abstraction.

---

## Critical Issues Requiring Revision

### Issue 1: Text Mutation Pollutes Stored Content

**The Problem:**
```python
# From the plan - Section 5.1
chunk["text"] = f"{chunk['text']}\n\n[Context: {entity_str}]"
```

This modifies the canonical chunk text. That modified text will:
- Be stored in Neo4j (`Chunk.text` property)
- Be stored in Qdrant payload (`text` field)
- Be returned to users in search results
- Appear in the MCP server responses

**Impact:** Users will see `[Context: software: WEKA; command: mount]` appended to every chunk. This breaks the "clean content" contract.

**Recommended Fix:**
```python
# Create enriched text for embedding ONLY - don't persist it
embedding_text = f"{chunk['text']}\n\n[Context: {entity_str}]"
chunk["_embedding_text"] = embedding_text  # Transient field for embedding generation
# chunk["text"] stays clean
```

Then modify `atomic.py:_generate_embeddings()` to check for `_embedding_text` before falling back to `text`.

---

### Issue 2: Graph Inconsistency - Missing Neo4j Entity Nodes

**The Problem:**
The current pipeline does this:
```
Document → Section → Chunk ←MENTIONS→ Entity (Neo4j node)
```

The plan creates `_mentions` entries with synthetic IDs:
```python
"entity_id": f"gliner:{e.label}:{e.text}"  # No corresponding Neo4j node!
```

**Impact:**
- Graph traversal queries will find MENTIONS edges pointing to non-existent entities
- `find_referencing_symbols` patterns break
- `neo4j_disabled: true` masks this now, but future graph re-enablement will expose it

**Recommended Fix Options:**

**Option A (Recommended):** Create Entity nodes for GLiNER extractions
```python
# In build_graph.py, add GLiNER entities to the entity creation batch
for gliner_entity in chunk.get("_gliner_entities", []):
    entities.append({
        "id": f"gliner:{label}:{text_hash}",
        "name": gliner_entity["text"],
        "type": gliner_entity["label"],
        "source": "gliner"
    })
```

**Option B:** Keep GLiNER entities sparse-only (don't add to `_mentions`)
- Create a separate `_gliner_mentions` field
- Modify `atomic.py` to merge both for sparse vector generation
- Never create MENTIONS edges for GLiNER entities

---

### Issue 3: `_mentions` Structure Incompatibility

**Current `_mentions` structure** (from `build_graph.py:444-475`):
```python
{
    "section_id": "sha256-...",
    "entity_id": "sha256-...",
    "name": "WEKA",
    "type": "software"
    # ... other fields from entity extraction
}
```

**Plan's proposed structure:**
```python
{
    "name": e.text,
    "label": e.label,  # Different key! Current uses "type"
    "entity_id": f"gliner:{e.label}:{e.text}",
    "source": "gliner"
}
```

**Impact:** The field name mismatch (`label` vs `type`) may cause downstream processing to fail silently.

**Fix:** Match existing schema:
```python
{
    "name": e.text,
    "type": e.label,  # Use "type" to match existing
    "entity_id": f"gliner:{e.label}:{hashlib.sha256(e.text.encode()).hexdigest()[:16]}",
    "source": "gliner",
    "confidence": e.score
}
```

---

### Issue 4: Boosting Implementation is Incomplete

**The Problem:**
The plan shows:
```python
should_clauses.append(rest.FieldCondition(...))
# 3. Pass to vector retriever (Requires updating search signature)
# self.vector_retriever.search(..., should=should_clauses)
```

But `QdrantMultiVectorRetriever` uses prefetch queries that don't natively support `should` boosting. The Qdrant Query API structure is:

```python
client.query_points(
    prefetch=[...],  # Each prefetch is independent
    query=fusion_query,
    query_filter=Filter(must=[...], should=[...])  # Filter, not boost
)
```

**Impact:** `should` in a filter still *filters* (returns docs matching ANY should clause OR all must clauses). It doesn't *boost* scores.

**Recommended Fix:**
Implement boosting as **post-retrieval re-scoring** in the RRF fusion stage:

```python
# In hybrid_retrieval.py, after RRF fusion
for result in fused_results:
    entity_boost = 0.0
    for entity in query_entities:
        if entity.text.lower() in result.payload.get("entity_values_normalized", []):
            entity_boost += 0.1  # Configurable boost factor
    result.score += entity_boost
```

---

### Issue 5: Device Detection Should Auto-Detect

**The Problem:**
```yaml
device: "mps"  # Hardcoded
```

**Impact:** Fails on non-Apple Silicon (CI runners, Linux servers, Intel Macs).

**Fix:**
```python
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

---

### Issue 6: Query-Time Latency Concern

**The Problem:**
Every query runs GLiNER inference:
```python
def retrieve(self, query: str, ...):
    processed = self.disambiguator.process(query)  # GLiNER inference here
```

GLiNER inference is ~10-50ms per query. This adds to every search request.

**Recommended Mitigations:**
1. Make query-time NER optional via config flag
2. Cache query→entities mappings (LRU cache with TTL)
3. Only run for queries above a length threshold

```python
@lru_cache(maxsize=1000)
def cached_extract(query_hash: str, query: str) -> List[Entity]:
    return self.service.extract_entities(query, self.labels)
```

---

### Issue 7: Integration Point Clarity

**The Problem:**
The plan says:
> Modify `_prepare_ingestion` in `src/ingestion/atomic.py`

But looking at the current flow:
1. `build_graph.py` orchestrates ingestion
2. `chunk_assembler.py` creates chunks
3. `atomic.py` handles embedding generation
4. Entity extraction already happens in `build_graph.py` (existing pipeline)

**Recommended Integration Point:**
Hook into `build_graph.py` **after** existing entity extraction but **before** `_process_embeddings`:

```python
# build_graph.py:~444 (after existing entity attachment)

# NEW: Augment with GLiNER entities
if config.ner.enabled:
    from src.ingestion.extract.ner_gliner import augment_with_gliner_entities
    augment_with_gliner_entities(sections)  # Adds to _mentions

# Step 5: Compute embeddings (existing)
embedding_stats = self._process_embeddings(document, sections, entities)
```

---

## Missing Elements

### 1. No Graceful Degradation
What happens if:
- GLiNER model fails to download?
- MPS device initialization fails?
- HuggingFace is rate-limited?

**Add:** Circuit breaker pattern with fallback to no-op:
```python
try:
    enrich_chunks_with_entities(sections)
except GLiNERUnavailableError:
    logger.warning("GLiNER unavailable, continuing without entity enrichment")
```

### 2. No Observability Integration
The plan doesn't integrate with existing `src/shared/observability.py` patterns:
- No `GLINER_EXTRACTION_DURATION` histogram
- No entity count metrics
- No span tracing for debugging

### 3. No Migration Strategy for Existing Data
If we enable GLiNER on a populated Qdrant collection:
- Existing chunks won't have `entity_metadata` payload fields
- Queries with entity boosting will fail or behave inconsistently

**Add:** Migration script or "backfill on read" pattern.

### 4. Labels Tuning Guidance
The plan lists labels but doesn't explain how to tune them. GLiNER is zero-shot, so label wording matters:
- `"error code"` vs `"error_code"` vs `"error message"` produce different results
- Need A/B testing framework for label optimization

---

## Architectural Recommendations

### Recommendation 1: Separate Enrichment from Storage

```
┌────────────────┐     ┌─────────────────┐     ┌────────────────┐
│  chunk["text"] │────▶│ GLiNER Extract  │────▶│ _mentions      │ (for sparse)
│  (unchanged)   │     │                 │     │ _gliner_meta   │ (for payload)
└────────────────┘     │                 │     │ _enriched_text │ (for dense embedding only)
                       └─────────────────┘     └────────────────┘
                                                      │
                                                      ▼ (transient, not stored)
                                               ┌────────────────┐
                                               │ BGE-M3 Dense   │
                                               │ Embedding      │
                                               └────────────────┘
```

### Recommendation 2: Config Hierarchy Awareness

From our session context, we learned that `CHUNK_ASSEMBLER=structured` causes YAML config to override env vars. The plan should explicitly document:

```yaml
# config/development.yaml
ner:
  enabled: true  # THIS takes precedence in structured mode
```

And add a startup log:
```python
logger.info(f"NER config: enabled={config.ner.enabled}, model={config.ner.model_name}")
```

### Recommendation 3: Feature Flag with Gradual Rollout

```yaml
ner:
  enabled: false  # Default OFF for safety
  ingestion_enabled: true  # Separate flag for ingestion vs query
  query_enabled: false  # Start with ingestion only
```

---

## Summary Verdict

| Aspect | Assessment |
|--------|------------|
| Overall architecture | ✅ Sound approach |
| `_mentions` integration | ⚠️ Needs schema alignment |
| Text mutation | ❌ Must fix - breaks content contract |
| Graph consistency | ⚠️ Needs Entity node creation or isolation |
| Boosting implementation | ❌ Incomplete - needs post-retrieval scoring |
| Device handling | ⚠️ Needs auto-detection |
| Error handling | ⚠️ Needs circuit breaker |
| Observability | ❌ Missing metrics/tracing |

**Recommendation:** Address the ❌ items before implementation. The ⚠️ items can be handled during implementation with awareness.

---

*Review completed: December 2025*
