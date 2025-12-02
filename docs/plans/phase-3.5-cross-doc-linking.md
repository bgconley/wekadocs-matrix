# Phase 3.5: Vector-Based Cross-Document Linking

**Status**: PLANNED (Reviewed & Corrected)
**Branch**: `dense-graph-enhance`
**Prerequisite**: Phase 3 code complete (REFERENCES infrastructure exists)
**Created**: 2025-12-01
**Last Reviewed**: 2025-12-02 (GPT-5.1-codex & Gemini-3-Pro feedback incorporated)

---

## Problem Statement

Phase 3 implemented explicit cross-document reference extraction (hyperlinks, "see also" phrases), but the WEKA corpus contains **zero explicit cross-references**. However, implicit relationships clearly exist:

| Document A | Document B | Implicit Relationship |
|------------|------------|----------------------|
| Configure audit webhook using CLI | Configure audit webhook using GUI | Same topic, different method |
| Deploy Local WEKA Home v2.x | Deploy Local WEKA Home v3.0+ | Version variants |
| Set up WEKAmon | Explore performance statistics in Grafana | WEKAmon integrates Grafana |

**Goal**: Leverage existing BGE-M3 vectors (dense, sparse, ColBERT) to discover and materialize these implicit relationships.

---

## Architecture Overview

> **CRITICAL**: Qdrant stores **chunks**, not documents. All discovery queries return chunks,
> which must be aggregated to document-level before RRF fusion.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CORRECTED PHASE 3.5 PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │     Stage 1      │    │     Stage 2      │    │     Stage 3      │       │
│  │ CHUNK DISCOVERY  │───▶│  DOC AGGREGATION │───▶│    RRF FUSION    │       │
│  │                  │    │                  │    │                  │       │
│  │ Dense: limit=100 │    │ Group by doc_id  │    │ Fuse doc-level   │       │
│  │ Sparse: limit=100│    │ Max-score wins   │    │ ranks from each  │       │
│  │ Title FT: limit=20│   │ → N unique docs  │    │ discovery method │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│           │                      │                       │                   │
│           ▼                      ▼                       ▼                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 4: COLBERT RERANK                          │   │
│  │              First-chunk only (order=0) for each candidate           │   │
│  │                    ~500×500 ops per pair = fast                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 5: PERSIST EDGES                           │   │
│  │         MERGE (source)-[:RELATED_TO {phase:'3.5'}]->(target)         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Stages

### Stage 1: Multi-Signal Chunk Discovery

For each source document, gather candidate **chunks** using three methods.
All chunks from the same document have **identical** `doc_title` vectors, so we're
effectively searching by document similarity at the vector level.

#### 1A. Dense Vector Similarity (doc_title vectors)
```python
# Query Qdrant for CHUNKS with similar doc_title embeddings
# Note: limit=100 to capture diverse documents before aggregation
dense_chunk_hits = qdrant.search(
    collection_name="chunks_multi_bge_m3",
    query_vector=models.NamedVector(
        name="doc_title",
        vector=source_doc_title_embedding
    ),
    query_filter=models.Filter(
        must_not=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=source_doc_id)
            )
        ]
    ),
    limit=100,  # Fetch many chunks to ensure document diversity
    score_threshold=0.60  # Relaxed threshold since we aggregate later
)
```

#### 1B. Sparse Vector Similarity (doc_title-sparse)
```python
# Query for chunks sharing lexical terms in document titles
# Verified: doc_title-sparse vectors exist and are populated in live system
sparse_chunk_hits = qdrant.search(
    collection_name="chunks_multi_bge_m3",
    query_vector=models.NamedSparseVector(
        name="doc_title-sparse",
        vector=models.SparseVector(
            indices=source_title_sparse_indices,
            values=source_title_sparse_values
        )
    ),
    query_filter=exclude_self_filter,
    limit=100
)
```

#### 1C. Full-Text Title Matching (Neo4j)
```cypher
-- Find chunks that explicitly mention this document's title
CALL db.index.fulltext.queryNodes('chunk_text_index', $title, {limit: 50})
YIELD node, score
WHERE node.document_id <> $source_doc_id AND score > 2.0
RETURN node.document_id as target_doc_id, score
```

---

### Stage 2: Chunk-to-Document Aggregation (CRITICAL)

> **Why this stage exists**: Qdrant returns chunks, but we need document-level similarity.
> Without aggregation, `limit=100` might return 50 chunks from 2 documents, missing diversity.

```python
from collections import defaultdict
from typing import List, Tuple
from qdrant_client.models import ScoredPoint

def aggregate_chunks_to_documents(
    chunk_hits: List[ScoredPoint],
    exclude_doc_id: str
) -> List[Tuple[str, float]]:
    """
    Aggregate chunk-level search results to document-level scores.

    Strategy: Max-Score Wins
    - Group chunks by document_id
    - Take the highest chunk score as the document's score
    - This captures the "best match" semantic from any chunk

    Args:
        chunk_hits: Raw Qdrant search results (chunks)
        exclude_doc_id: Source document ID to exclude from results

    Returns:
        List of (document_id, max_score) tuples, sorted by score descending
    """
    doc_scores: dict[str, float] = defaultdict(float)

    for hit in chunk_hits:
        doc_id = hit.payload.get('document_id')
        if doc_id == exclude_doc_id:
            continue
        # Max-score aggregation: best chunk represents document
        doc_scores[doc_id] = max(doc_scores[doc_id], hit.score)

    return sorted(doc_scores.items(), key=lambda x: -x[1])


# Apply aggregation to each discovery method's results BEFORE fusion
dense_docs = aggregate_chunks_to_documents(dense_chunk_hits, source_doc_id)
sparse_docs = aggregate_chunks_to_documents(sparse_chunk_hits, source_doc_id)
# title_docs already document-level from Cypher DISTINCT
```

---

### Stage 3: RRF Fusion & Candidate Selection

Merge **document-level** results from all three discovery methods using Reciprocal Rank Fusion:

```python
def reciprocal_rank_fusion(
    dense_docs: List[Tuple[str, float]],
    sparse_docs: List[Tuple[str, float]],
    title_docs: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine document rankings from multiple retrieval methods.

    RRF formula: score = Σ 1/(k + rank_i)

    IMPORTANT: Inputs must be document-level (post-aggregation), not chunk-level.
    RRF operates on ranks, so chunk-level fusion would over-weight multi-chunk docs.

    Benefits:
    - Rank-based, not score-based (handles different score scales)
    - Documents appearing in multiple lists get boosted
    - Robust to outliers in any single method
    """
    scores = defaultdict(float)

    for rank, (doc_id, _) in enumerate(dense_docs):
        scores[doc_id] += 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(sparse_docs):
        scores[doc_id] += 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(title_docs):
        scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by fused score, take top candidates for reranking
    return sorted(scores.items(), key=lambda x: -x[1])[:10]
```

**Threshold**: Keep candidates with fused_score > 0.02 (appears in at least 2 methods or high-ranked in 1)

---

### Stage 4: ColBERT Reranking (First-Chunk Strategy)

> **Performance Fix**: Full document ColBERT comparison is O(N×M) where N,M can be 5000+ vectors.
> Use first-chunk (order=0) only, reducing to ~500×500 = 250,000 ops (feasible in milliseconds).

```python
def get_first_chunk_colbert_vectors(
    doc_id: str,
    qdrant_client: QdrantClient
) -> List[List[float]]:
    """
    Get ColBERT vectors from the FIRST chunk (order=0) of a document only.

    Why first chunk?
    - Usually contains title/overview with document's key concepts
    - Reduces 7000+ vectors to ~300-500 vectors
    - 240x faster than full document comparison

    Alternative: Could use chunk with 'is_microdoc=True' if available
    """
    result = qdrant_client.scroll(
        collection_name="chunks_multi_bge_m3",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=doc_id)
                ),
                models.FieldCondition(
                    key="order",
                    match=models.MatchValue(value=0)  # First chunk only
                )
            ]
        ),
        limit=1,
        with_vectors=["late-interaction"]
    )

    if result[0]:  # result is (points, next_page_offset)
        return result[0][0].vector.get("late-interaction", [])
    return []


def colbert_rerank(
    source_doc_id: str,
    candidates: List[Tuple[str, float]],
    qdrant_client: QdrantClient
) -> List[Tuple[str, float]]:
    """
    Rerank candidates using ColBERT MaxSim scoring on first chunks only.
    """
    reranked = []

    # Get source document's first-chunk ColBERT vectors
    source_colbert = get_first_chunk_colbert_vectors(source_doc_id, qdrant_client)

    if not source_colbert:
        # Fallback: return candidates as-is if no ColBERT vectors
        return candidates

    for candidate_id, rrf_score in candidates:
        target_colbert = get_first_chunk_colbert_vectors(candidate_id, qdrant_client)

        if not target_colbert:
            # Keep RRF score if no ColBERT vectors for target
            reranked.append((candidate_id, rrf_score * 0.8))
            continue

        # Compute MaxSim score
        maxsim_score = compute_maxsim(source_colbert, target_colbert)
        # Blend ColBERT with RRF: ColBERT is precision signal
        final_score = 0.7 * maxsim_score + 0.3 * rrf_score
        reranked.append((candidate_id, final_score))

    return sorted(reranked, key=lambda x: -x[1])


def compute_maxsim(
    source_tokens: List[List[float]],  # Shape: [N, 1024]
    target_tokens: List[List[float]]   # Shape: [M, 1024]
) -> float:
    """
    ColBERT MaxSim: For each source token, find the max cosine similarity
    to any target token. Return the average.

    Computational cost: O(N × M × D) where D=1024
    With first-chunk strategy: ~500 × 500 × 1024 ≈ 250M ops (sub-second)
    """
    import numpy as np

    source = np.array(source_tokens)
    target = np.array(target_tokens)

    # Normalize for cosine similarity
    source_norm = source / np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = target / np.linalg.norm(target, axis=1, keepdims=True)

    # Compute all pairwise similarities: [N, M]
    similarities = source_norm @ target_norm.T

    # MaxSim: max over target tokens for each source token
    max_sims = similarities.max(axis=1)

    return float(max_sims.mean())
```

**Why ColBERT as reranker (not discovery)**:
- Discovery: Need to compare 1 doc against 50+ docs → expensive
- Reranking: Compare 1 doc against 10 candidates → feasible
- ColBERT catches nuanced term overlap dense vectors miss

---

### Stage 5: Edge Persistence

Create edges in Neo4j with relationship metadata:

```python
def persist_cross_doc_edges(
    source_doc_id: str,
    ranked_targets: List[Tuple[str, float]],
    neo4j_driver,
    min_score: float = 0.5,
    max_edges_per_doc: int = 5
):
    """
    Create RELATED_TO edges for top-scoring document pairs.
    """
    edges_created = 0

    for target_doc_id, score in ranked_targets:
        if score < min_score:
            break
        if edges_created >= max_edges_per_doc:
            break

        # Use MERGE for idempotency
        neo4j_driver.execute_query("""
            MATCH (source:Document {id: $source_id})
            MATCH (target:Document {id: $target_id})
            MERGE (source)-[r:RELATED_TO]->(target)
            SET r.score = $score,
                r.method = 'vector_similarity',
                r.created_at = datetime(),
                r.phase = '3.5'
        """, source_id=source_doc_id, target_id=target_doc_id, score=score)

        edges_created += 1

    return edges_created
```

---

## Relationship Types

| Relationship | Source | Description |
|--------------|--------|-------------|
| `RELATED_TO` | Vector similarity | Documents with similar content/topic |
| `REFERENCES` | Phase 3 explicit | Hyperlinks, "see also" (when present) |
| `SHARES_CONCEPT` | Future: ColBERT tokens | Documents sharing specific technical terms |

---

## Configuration

```yaml
cross_doc_linking:
  enabled: true

  discovery:
    chunk_limit: 100           # Chunks to fetch per method (before aggregation)
    dense_threshold: 0.60      # Relaxed since we aggregate
    sparse_threshold: 0.30
    title_match_threshold: 2.0

  aggregation:
    strategy: "max_score"      # "max_score" | "mean_score" | "first_chunk_score"
    min_docs_per_method: 5     # Ensure diversity before RRF

  fusion:
    method: "rrf"
    rrf_k: 60
    candidate_limit: 10        # Top documents to rerank

  reranking:
    enabled: true
    method: "colbert_first_chunk"
    colbert_weight: 0.7        # Blend weight with RRF score
    min_score: 0.50            # Minimum final score to create edge

  persistence:
    max_edges_per_doc: 5
    relationship_type: "RELATED_TO"
    bidirectional: false       # A→B doesn't imply B→A
```

---

## Implementation Phases

### Phase 3.5a: Dense Similarity Only (MVP)
- **Effort**: 2 hours
- **Scope**: Query doc_title vectors (limit=100), aggregate to docs, create edges for >0.70 similarity
- **Key deliverable**: `aggregate_chunks_to_documents()` function
- **Expected edges**: ~100-200

### Phase 3.5b: Add Sparse + RRF Fusion
- **Effort**: 2 hours
- **Scope**: Add sparse vector search, implement document-level RRF fusion
- **Key deliverable**: `reciprocal_rank_fusion()` on aggregated results
- **Expected edges**: ~150-250 (higher recall)

### Phase 3.5c: Add Full-Text Title Matching
- **Effort**: 1 hour
- **Scope**: Create full-text index on chunk text, scan for title mentions
- **Expected edges**: +20-50 explicit references

### Phase 3.5d: ColBERT Reranking
- **Effort**: 3 hours
- **Scope**: First-chunk ColBERT retrieval, MaxSim scoring, blend with RRF
- **Key deliverable**: `get_first_chunk_colbert_vectors()` + `compute_maxsim()`
- **Expected improvement**: Higher precision, fewer false positives

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Cross-document edges | 0 | >200 |
| Documents with ≥1 cross-link | 0 | >40 (75%) |
| Average edges per document | 0 | 3-5 |
| Query recall improvement | Baseline | +10-15% |

---

## Rollback Strategy

All edges created in Phase 3.5 are tagged with `phase: '3.5'`:

```cypher
-- Remove all Phase 3.5 edges if needed
MATCH ()-[r:RELATED_TO {phase: '3.5'}]->()
DELETE r
```

---

## Dependencies

- [x] Phase 3 code complete (REFERENCES infrastructure)
- [x] doc_title dense vectors populated in Qdrant (verified 2025-12-02)
- [x] doc_title-sparse vectors populated in Qdrant (verified 2025-12-02)
- [x] ColBERT (late-interaction) vectors populated in Qdrant (verified 2025-12-02)
- [ ] Full-text index on chunk text (create if missing)

---

## Review Notes (2025-12-02)

### Issues Identified by External Review

| Issue | Status | Resolution |
|-------|--------|------------|
| "No doc_title-sparse vector" | **FALSE** | Reviewer used stale inventory file; live schema verified correct |
| Chunk-vs-document abstraction | **VALID** | Added Stage 2 (aggregation) before RRF fusion |
| ColBERT 60M operations per pair | **VALID** | Changed to first-chunk strategy (~250K ops) |
| Query limit too low | **VALID** | Increased from 20 to 100 chunks |

### Verification Commands

```bash
# Verify doc_title-sparse exists in live schema
curl -s http://localhost:6333/collections/chunks_multi_bge_m3 | jq '.result.config.params.sparse_vectors'
# Expected: {"doc_title-sparse": {...}, "text-sparse": {...}}

# Verify vectors are populated (not just schema)
curl -s "http://localhost:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "with_vectors": ["doc_title-sparse"]}' | jq '.result.points[0].vector["doc_title-sparse"]'
# Expected: {"indices": [...], "values": [...]}

# Check chunk distribution for limit sizing
curl -s "http://localhost:6333/collections/chunks_multi_bge_m3/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 100}' | python3 -c "
import sys, json
from collections import Counter
data = json.load(sys.stdin)
docs = Counter(p['payload']['document_id'] for p in data['result']['points'])
print(f'100 chunks → {len(docs)} unique documents')
"
# Expected: ~30-40 unique documents
```

---

## References

- [Cross-Document Coreference Resolution in Knowledge Graphs](https://arxiv.org/abs/2504.05767)
- [Qdrant Hybrid Search Documentation](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)
- [BGE-M3: Multi-Functionality Embeddings](https://arxiv.org/abs/2402.03216)
- [Neo4j GraphRAG Python Library](https://neo4j.com/docs/neo4j-graphrag-python/)
