# Graph Channel Rehabilitation Plan

**Author:** Claude (Investigation & Planning)
**Date:** 2025-11-26
**Branch:** `dense-graph-enhance`
**Status:** Proposed

---

## Executive Summary

This plan addresses six critical defects discovered in the graph channel that render it ineffective for conceptual documentation retrieval. The graph channel currently extracts CLI commands and config parameters instead of semantic concepts, has 11% entity coverage, produces garbage matches via broken Cypher logic, and scores results in ways incompatible with vector retrieval.

The rehabilitation strategy prioritizes:
1. **Immediate harm reduction** (Phase 0)
2. **Leveraging existing document structure** (Phase 1)
3. **Proper query-time integration** (Phase 2)
4. **Semantic entity fallback** (Phase 3)
5. **Context expansion capabilities** (Phase 4)
6. **Validation and hardening** (Phase 5)

---

## Defects Addressed

| ID | Defect | Severity | Root Cause |
|----|--------|----------|------------|
| #1 | Wrong entity model | Critical | Extraction focuses on CLI/config patterns, not concepts |
| #2 | Dead code (rel_pattern unused) | High | Cypher ignores the computed relationship pattern |
| #3 | Bi-directional CONTAINS garbage | High | "at" matches "metadata" because metadata contains "at" |
| #4 | Sparse entity coverage (11%) | Critical | Only 252 of 2,233 chunks have entity links |
| #5 | Score incompatibility | Medium | Graph scores ~1.0 vs vector scores 0.3-0.7 |
| #6 | Entity trie noise | High | 2,321 NULL-named entities, generic terms like "help" |

---

## Starting Principles

Before diving into phases, the plan establishes what the graph channel *should* do:

**The Graph Channel's Purpose:**
1. Provide **structure-aware retrieval** that vectors can't do (document hierarchy, section relationships)
2. Enable **concept-anchored lookup** for domain terminology
3. Support **cross-document discovery** via shared entities
4. Add **precision signals** to complement vector recall

**Key Insight:**
The documentation already has rich structure - section headings like "Metadata units calculation" ARE the concepts. We don't need sophisticated NLP to extract what authors already labeled. The failure is that this structure isn't being leveraged.

---

## Phase 0: Surgical Bug Fixes

**Objective:** Stop active harm without architectural changes.
**Timeline:** 1-2 days
**Addresses:** Defects #2, #3, #5, #6 (partial)

### 0.1 Remove Garbage Matching

**File:** `src/query/hybrid_retrieval.py:3115-3116`

The bi-directional CONTAINS is fundamentally broken:

```cypher
-- Current (broken):
WHERE toLower(e.name) CONTAINS toLower(name)
   OR toLower(name) CONTAINS toLower(e.name)  -- CAUSES GARBAGE

-- Fixed:
WHERE toLower(e.name) CONTAINS toLower(name)
```

**Why it's broken:** The second clause allows entity names to match if they're substrings of the query term. So entity "at" matches query "metadata" because "metadata" contains "at".

**Additional Python-side filter:**

```python
# Filter out short/generic terms before graph query
STOPWORDS = {"at", "in", "id", "on", "to", "of", "for", "the", "and", "or", "is", "it"}

def filter_entity_candidates(entities: List[str]) -> List[str]:
    return [
        e for e in entities
        if len(e) >= 4 and e.lower() not in STOPWORDS
    ]
```

### 0.2 Wire Up the Unused rel_pattern

**File:** `src/query/hybrid_retrieval.py:3109-3148`

**Current state:** `rel_pattern` is computed at line 3110 and passed at line 3147, but the Cypher query hardcodes `[:MENTIONED_IN]` and ignores it.

**Fix the Cypher:**

```cypher
-- Before (broken):
MATCH (e)-[:MENTIONED_IN]->(c:Chunk)

-- After (uses the parameter):
MATCH (e)-[r]->(c:Chunk)
WHERE type(r) IN $rel_types
```

**Fix the Python:**

```python
# Change from pipe-delimited string to list
rels, max_related = self._relationships_for_query(query)
# rel_pattern = "|".join(rels)  # OLD - delete this

result = session.run(
    cypher,
    entities=list(set(entities)),
    doc_tag=doc_tag,
    limit=limit_per_entity,
    rel_types=rels,  # Pass as list, not pattern string
)
```

### 0.3 Fix Dedup to Preserve Best Score

**File:** `src/query/hybrid_retrieval.py:2662-2671`

**Current behavior:** Takes first occurrence, discarding graph boost if vector result came first.

**Fixed implementation:**

```python
def _dedup_results(self, results: List[ChunkResult]) -> List[ChunkResult]:
    """Remove duplicate chunks, keeping highest score and merging signals."""
    by_chunk: Dict[str, ChunkResult] = {}

    for r in results:
        key = self._result_id(r)
        existing = by_chunk.get(key)

        if existing is None:
            by_chunk[key] = r
            continue

        # Determine winner by fused_score
        if (r.fused_score or 0) > (existing.fused_score or 0):
            winner, loser = r, existing
        else:
            winner, loser = existing, r

        # Merge signals from both sources
        winner.graph_score = max(winner.graph_score or 0, loser.graph_score or 0)
        winner.vector_score = max(winner.vector_score or 0, loser.vector_score or 0)

        # Recompute fused score after merging
        winner.fused_score = self._compute_fused_score(
            winner.vector_score,
            winner.graph_score
        )

        by_chunk[key] = winner

    return list(by_chunk.values())
```

### 0.4 Normalize Graph Scores to [0,1]

**File:** `src/query/hybrid_retrieval.py:3152-3155`

**Current code:**
```python
chunk.graph_score = max(1.0, float(record.get("match_count") or 1))
chunk.fused_score = chunk.graph_score
chunk.vector_score = chunk.graph_score  # Overwrites vector score!
```

**Fixed code:**
```python
import math

raw_match_count = float(record.get("match_count") or 1)

# Normalize using saturating exponential: 1 match → ~0.28, 3 matches → ~0.63, 5+ → ~0.81
chunk.graph_score = 1.0 - math.exp(-raw_match_count / 3.0)

# Don't overwrite vector_score - it should come from vector retrieval
# chunk.vector_score stays as-is or defaults to 0 if not set

# Fused score will be computed properly during fusion
chunk.fused_score = chunk.graph_score  # Temporary until Phase 2 fusion
chunk.vector_score_kind = "graph_entity"
```

### 0.5 Add Observability

Add structured logging for every graph channel invocation:

```python
logger.info(
    "graph_channel_executed",
    query=query[:100],
    query_type=classify_query(query),
    entities_extracted=len(entities),
    entities_used=entities[:5],
    chunks_returned=len(chunks),
    top_chunk_scores=[c.graph_score for c in chunks[:3]],
    top_chunk_ids=[c.chunk_id for c in chunks[:3]],
    execution_time_ms=elapsed_ms,
)
```

### Phase 0 Acceptance Criteria

- [ ] Query "metadata" no longer matches entity "at"
- [ ] `rel_types` parameter is actually used in Cypher
- [ ] Dedup preserves highest score, not first occurrence
- [ ] Graph scores are in [0,1] range
- [ ] Dedup recomputes fused_score after merging signals
- [ ] Every graph query is logged with entity/result counts

---

## Phase 1: Leverage Existing Structure

**Objective:** The documentation has structure. Use it.
**Timeline:** 1 week
**Addresses:** Defects #1, #4, #6

### 1.1 Promote Section Headings to Entities

**Key Insight:** Section headings are author-defined concepts. "Metadata units calculation" is literally a `####` heading in the source. We don't need NLP - we need to index what's already there.

**New extraction function:**

```python
def extract_heading_entities(sections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract concept entities from section headings.

    Returns:
        Tuple of (entities, mentions)
    """
    entities = []
    mentions = []

    # Generic headings to skip
    SKIP_HEADINGS = {
        "overview", "introduction", "summary", "notes", "description",
        "prerequisites", "requirements", "see also", "related topics",
        "example", "examples", "syntax", "parameters", "returns",
    }

    for section in sections:
        heading = section.get("heading", "").strip()

        # Skip empty or too short
        if not heading or len(heading) < 5:
            continue

        # Skip generic headings
        if heading.lower() in SKIP_HEADINGS:
            continue

        # Skip headings that are just numbers or bullets
        if re.match(r'^[\d\.\-\*\#]+$', heading):
            continue

        canonical = canonicalize_entity_name(heading)
        entity_id = hashlib.sha256(f"heading:{canonical}".encode()).hexdigest()

        entity = {
            "id": entity_id,
            "label": "Concept",
            "name": heading,
            "canonical_name": canonical,
            "entity_type": "heading_concept",
            "source_section_id": section["id"],
            "source_doc_tag": section.get("doc_tag"),
        }
        entities.append(entity)

        # Create mention linking entity to section's chunk
        chunk_id = section.get("chunk_id") or section.get("id")
        mention = {
            "entity_id": entity_id,
            "section_id": chunk_id,
            "confidence": 1.0,  # High confidence - it's the actual heading
            "source": "heading_extraction",
        }
        mentions.append(mention)

    return entities, mentions


def canonicalize_entity_name(name: str) -> str:
    """Normalize entity name for matching."""
    # Lowercase
    name = name.lower()
    # Normalize whitespace and separators
    name = re.sub(r'[-_\s]+', ' ', name).strip()
    # Remove trailing punctuation
    name = name.rstrip(':;,.')
    # Remove common prefixes
    name = re.sub(r'^(the|a|an)\s+', '', name)
    return name
```

**Cypher for creating heading entities:**

```cypher
UNWIND $entities AS ent
MERGE (e:Entity:Concept {id: ent.id})
SET e.name = ent.name,
    e.canonical_name = ent.canonical_name,
    e.entity_type = ent.entity_type,
    e.source_section_id = ent.source_section_id,
    e.source_doc_tag = ent.source_doc_tag,
    e.updated_at = datetime()

WITH e, ent
MATCH (c:Chunk {chunk_id: ent.source_section_id})
MERGE (e)-[r:DEFINES]->(c)
SET r.confidence = 1.0, r.source = 'heading_extraction'

MERGE (e)-[m:MENTIONED_IN]->(c)
SET m.confidence = 1.0, m.source = 'heading_extraction'
```

**Expected outcome:** ~500-800 meaningful heading entities for 2,233 chunks (one per logical section).

### 1.2 Create Section-to-Chunk Linkages

Currently, chunks have `parent_section_id` but this isn't used for retrieval. Add explicit relationships.

**Migration Cypher:**

```cypher
// Create IN_SECTION relationships
MATCH (c:Chunk)
WHERE c.parent_section_id IS NOT NULL
MATCH (s:Section {section_id: c.parent_section_id})
MERGE (c)-[:IN_SECTION]->(s);

// Create section hierarchy
MATCH (child:Section)
WHERE child.parent_section_id IS NOT NULL
MATCH (parent:Section {section_id: child.parent_section_id})
MERGE (child)-[:CHILD_OF]->(parent);

// Create document membership
MATCH (c:Chunk)
WHERE c.document_id IS NOT NULL
MATCH (d:Document {document_id: c.document_id})
MERGE (c)-[:IN_DOCUMENT]->(d);
```

**This enables structure-aware queries:**
- "Find all chunks in sections about X"
- "Find sibling chunks to this result"
- "Find parent section context"

### 1.3 Clean Up Entity Label Pollution

Currently, Step/Configuration/Procedure nodes all have the Entity label, polluting queries.

**Migration Cypher:**

```cypher
// Remove Entity label from Step nodes (they have NULL names anyway)
MATCH (n:Step:Entity)
REMOVE n:Entity;

// Remove Entity label from nodes with NULL or empty names
MATCH (n:Entity)
WHERE n.name IS NULL OR trim(n.name) = ''
REMOVE n:Entity;

// Optionally: Remove Entity label from generic config params
MATCH (n:Configuration:Entity)
WHERE n.name IN ['help', 'color', 'profile', 'json', 'output', 'format',
                  'filter', 'verbose', 'id', 'name', 'type', 'value', 'data']
REMOVE n:Entity;
```

**New Policy:** Entity label is reserved for nodes that:
1. Have a non-null, non-empty `name`
2. Have `canonical_name` set
3. Are intended for query-time entity matching

### 1.4 Rebuild Entity Trie from Clean Data

After cleanup, rebuild the trie with proper filtering:

```python
GENERIC_BLACKLIST = {
    # Common words
    "help", "color", "profile", "json", "output", "format", "filter",
    "verbose", "config", "data", "info", "test", "name", "type", "value",
    # Stopwords that somehow became entities
    "at", "in", "id", "on", "to", "of", "for", "the", "and", "or", "is", "it",
    # Generic technical terms
    "true", "false", "null", "none", "yes", "no", "default", "example",
}


def build_clean_entity_trie(neo4j_driver) -> Trie:
    """Build trie from clean, validated entities only."""
    trie = Trie()

    cypher = """
    MATCH (e:Entity)
    WHERE e.canonical_name IS NOT NULL
      AND size(e.canonical_name) >= 4
      AND e.entity_type IN ['heading_concept', 'cli_command', 'config_param', 'concept']
    RETURN e.id, e.canonical_name, e.name, e.entity_type
    """

    with neo4j_driver.session() as session:
        for record in session.run(cypher):
            canonical = record["canonical_name"].lower().strip()

            # Skip blacklisted terms
            if canonical in GENERIC_BLACKLIST:
                continue

            # Skip if too short after normalization
            if len(canonical) < 4:
                continue

            entity_id = record["id"]

            # Insert canonical name
            trie.insert(canonical, entity_id)

            # Also insert display name if different
            name = (record["name"] or "").lower().strip()
            if name and name != canonical and len(name) >= 4 and name not in GENERIC_BLACKLIST:
                trie.insert(name, entity_id)

    return trie
```

### 1.5 Integrate Heading Extraction into Ingestion

Add heading entity extraction to the ingestion pipeline:

**File:** `src/ingestion/build_graph.py`

```python
# After section parsing, before entity creation
from src.ingestion.extract.headings import extract_heading_entities

def build_graph(document, sections, ...):
    # ... existing code ...

    # Extract structural entities (existing)
    entities, mentions = extract_entities(sections)

    # Extract heading concepts (NEW)
    heading_entities, heading_mentions = extract_heading_entities(sections)

    # Merge, preferring heading entities for conflicts
    for he in heading_entities:
        entities[he["id"]] = he
    mentions.extend(heading_mentions)

    # ... continue with entity creation ...
```

### Phase 1 Acceptance Criteria

- [ ] Every meaningful section heading becomes a Concept entity
- [ ] Chunks have explicit IN_SECTION relationships
- [ ] Section hierarchy is navigable via CHILD_OF
- [ ] Entity label only on nodes with valid, non-generic names
- [ ] Trie contains heading concepts, excludes NULL names and stopwords
- [ ] New documents automatically get heading entities during ingestion
- [ ] Entity coverage increases from 11% to >50%

---

## Phase 2: Query-Time Graph Integration

**Objective:** Make graph signals contribute meaningfully at query time.
**Timeline:** 1 week
**Addresses:** Defects #2, #3, #5

### 2.1 Two-Path Entity Resolution

Instead of just trie matching, use two complementary approaches:

**Path A: Trie Exact/Prefix Match (Fast, High Precision)**

```python
def resolve_entities_trie(query: str, trie: Trie) -> List[str]:
    """Fast exact/prefix matching for known entities."""
    tokens = query.lower().split()
    entity_ids = []

    # Try multi-word matches first (greedy, longest match)
    i = 0
    while i < len(tokens):
        matched = False
        # Try decreasing lengths
        for length in range(min(5, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i:i+length])
            match = trie.get(phrase)
            if match:
                entity_ids.append(match)
                i += length
                matched = True
                break
        if not matched:
            i += 1

    return entity_ids
```

**Path B: Embedding Similarity (For Unknown Concepts)**

```python
def resolve_entities_embedding(
    query: str,
    embedder,
    qdrant_client,
    top_k: int = 5,
    score_threshold: float = 0.7
) -> List[str]:
    """Find similar entities via vector search when trie fails."""
    query_vec = embedder.embed([query])[0]

    results = qdrant_client.search(
        collection_name="entity_embeddings",
        query_vector=query_vec,
        limit=top_k,
        score_threshold=score_threshold,
    )

    return [r.payload["entity_id"] for r in results]
```

**Combined Resolution Strategy:**

```python
def resolve_entities(self, query: str) -> List[str]:
    """
    Resolve query to entity IDs using trie + embedding fallback.

    Strategy:
    1. Try trie first (fast, high precision)
    2. If trie found matches, use them
    3. Otherwise, fall back to embedding similarity
    """
    # Try trie first
    trie_matches = resolve_entities_trie(query, self.entity_trie)

    if trie_matches:
        logger.debug("Entity resolution via trie", count=len(trie_matches))
        return trie_matches

    # Fall back to embedding similarity
    embedding_matches = resolve_entities_embedding(
        query,
        self.embedder,
        self.qdrant_client
    )

    if embedding_matches:
        logger.debug("Entity resolution via embedding", count=len(embedding_matches))
    else:
        logger.debug("No entities resolved for query")

    return embedding_matches
```

### 2.2 Rewrite Graph Retrieval Cypher

New Cypher that uses entity IDs (not string matching) and respects relationship types:

```cypher
// Entity-anchored chunk retrieval
UNWIND $entity_ids AS eid
MATCH (e:Entity {id: eid})

// Follow appropriate relationships based on query type
MATCH (e)-[r]->(target)
WHERE type(r) IN $rel_types
  AND (target:Chunk OR target:Section)

// If target is Section, get its chunks
OPTIONAL MATCH (target)<-[:IN_SECTION]-(c:Chunk)
WITH coalesce(c, target) AS chunk, e
WHERE chunk:Chunk
  AND ($doc_tag IS NULL OR chunk.doc_tag = $doc_tag)

// Aggregate results
WITH chunk,
     count(DISTINCT e) AS entity_match_count,
     collect(DISTINCT e.name)[..5] AS matched_entity_names

RETURN chunk {
    .chunk_id,
    .document_id,
    .heading,
    .text,
    .doc_tag,
    .tokens,
    .source_path
} AS props,
entity_match_count,
matched_entity_names

ORDER BY entity_match_count DESC
LIMIT $limit
```

### 2.3 Query Type Classification

Implement query type classification to select appropriate relationships:

```python
def classify_query(query: str) -> str:
    """
    Classify query type to determine graph traversal strategy.

    Returns one of: 'conceptual', 'cli', 'config', 'procedural'
    """
    query_lower = query.lower()

    # CLI indicators (need 2+ signals to classify as CLI)
    cli_patterns = [
        r'\bweka\s+\w+',      # weka subcommand
        r'--[a-z][\w-]+',     # --flag-name
        r'\s-[a-z]\b',        # -f flag
        r'\bcli\b',           # explicit "cli"
        r'\bcommand\b',       # explicit "command"
        r'\brun\b.*\bcommand', # "run command"
    ]
    cli_score = sum(1 for p in cli_patterns if re.search(p, query_lower))
    if cli_score >= 2:
        return "cli"

    # Config indicators
    config_patterns = [
        r'\w+\s*=\s*\w+',     # KEY=value
        r'\.yaml\b',          # .yaml file
        r'\.json\b',          # .json file
        r'\.conf\b',          # .conf file
        r'\bconfig(ure|uration)?\b',  # config/configure/configuration
        r'\bsetting\b',       # setting
        r'\bparameter\b',     # parameter
        r'\benvironment\s+variable', # environment variable
    ]
    if any(re.search(p, query_lower) for p in config_patterns):
        return "config"

    # Procedural indicators
    proc_patterns = [
        r'\bhow\s+(do|to|can)\b',  # how to/how do/how can
        r'\bsteps?\s+(to|for)\b',  # steps to/for
        r'\bprocedure\b',          # procedure
        r'\binstall(ation)?\b',    # install/installation
        r'\bsetup\b',              # setup
        r'\bguide\b',              # guide
    ]
    if any(re.search(p, query_lower) for p in proc_patterns):
        return "procedural"

    # Default: conceptual (documentation lookup)
    return "conceptual"


def get_relationships_for_query_type(query_type: str) -> List[str]:
    """Map query type to relationship types for graph traversal."""
    mapping = {
        "conceptual": ["MENTIONED_IN", "DEFINES", "IN_SECTION"],
        "cli": ["MENTIONED_IN", "CONTAINS_STEP", "HAS_PARAMETER"],
        "config": ["MENTIONED_IN", "HAS_PARAMETER", "DEFINES"],
        "procedural": ["MENTIONED_IN", "CONTAINS_STEP", "NEXT_CHUNK", "IN_SECTION"],
    }
    return mapping.get(query_type, ["MENTIONED_IN"])
```

### 2.4 Graph as Reranker (Not Independent Channel)

**Critical Design Decision:** Instead of extending results with graph candidates, use graph to **rerank** vector candidates.

**Why this matters:**
1. **Safer** - Graph can only improve ranking, not harm recall
2. **Bounded latency** - Graph only processes vector's top-K, not entire corpus
3. **Graceful degradation** - If graph fails, you still have vector results
4. **Prevents garbage surfacing** - Graph can't introduce completely unrelated content

**New search flow:**

```python
def search(self, query: str, top_k: int, doc_tag: Optional[str] = None) -> List[ChunkResult]:
    """
    Hybrid search with graph as reranker.

    Flow:
    1. Vector retrieval (broad recall)
    2. Entity resolution
    3. Graph signal computation on vector candidates
    4. Score fusion
    5. Re-sort by fused score
    """
    # Step 1: Vector retrieval (get more candidates than needed)
    vector_results = self.vector_retriever.search(
        query,
        top_k=top_k * 3,  # Over-fetch for reranking headroom
        doc_tag=doc_tag
    )

    if not vector_results:
        return []

    # Step 2: Entity resolution
    entity_ids = self.resolve_entities(query)
    query_type = classify_query(query)

    # Step 3: Graph signal computation
    graph_signals = {}
    if entity_ids and self.graph_channel_enabled:
        graph_signals = self._compute_graph_signals(
            entity_ids=entity_ids,
            candidate_chunk_ids=[r.chunk_id for r in vector_results],
            query_type=query_type,
            doc_tag=doc_tag,
        )

    # Step 4: Score fusion
    for result in vector_results:
        signal = graph_signals.get(result.chunk_id, {})
        result.graph_score = signal.get("score", 0.0)
        result.matched_entities = signal.get("entities", [])
        result.fused_score = self._fuse_scores(
            vector_score=result.vector_score or 0,
            graph_score=result.graph_score,
            query_type=query_type,
        )

    # Step 5: Re-sort by fused score
    vector_results.sort(key=lambda r: r.fused_score or 0, reverse=True)

    # Log for observability
    logger.info(
        "hybrid_search_complete",
        query=query[:100],
        query_type=query_type,
        entities_resolved=len(entity_ids),
        chunks_with_graph_signal=len(graph_signals),
        top_fused_scores=[r.fused_score for r in vector_results[:5]],
    )

    return vector_results[:top_k]
```

**Graph signal computation:**

```python
def _compute_graph_signals(
    self,
    entity_ids: List[str],
    candidate_chunk_ids: List[str],
    query_type: str,
    doc_tag: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Compute graph signals for candidate chunks.

    Returns:
        Dict mapping chunk_id to {score, entities}
    """
    if not entity_ids or not candidate_chunk_ids:
        return {}

    rel_types = get_relationships_for_query_type(query_type)

    cypher = """
    UNWIND $entity_ids AS eid
    MATCH (e:Entity {id: eid})-[r]->(c:Chunk)
    WHERE type(r) IN $rel_types
      AND c.chunk_id IN $chunk_ids
      AND ($doc_tag IS NULL OR c.doc_tag = $doc_tag)
    RETURN c.chunk_id AS chunk_id,
           count(DISTINCT e) AS entity_count,
           collect(DISTINCT e.name)[..5] AS entity_names
    """

    signals = {}
    try:
        with self.neo4j_driver.session() as session:
            results = session.run(
                cypher,
                entity_ids=entity_ids,
                rel_types=rel_types,
                chunk_ids=candidate_chunk_ids,
                doc_tag=doc_tag,
            )
            for record in results:
                chunk_id = record["chunk_id"]
                entity_count = record["entity_count"]

                # Normalized score: 1 match → ~0.28, 3 → ~0.63, 5+ → ~0.81
                score = 1.0 - math.exp(-entity_count / 3.0)

                signals[chunk_id] = {
                    "score": score,
                    "entities": record["entity_names"],
                    "entity_count": entity_count,
                }
    except Exception as exc:
        logger.warning("Graph signal computation failed", error=str(exc))

    return signals
```

### 2.5 Score Fusion Strategy

```python
def _fuse_scores(
    self,
    vector_score: float,
    graph_score: float,
    query_type: str,
) -> float:
    """
    Combine vector and graph scores based on query type.

    Strategy: Linear combination with query-type-specific weights.
    Graph acts as a boost/penalty on vector score.
    """
    # Query-type-specific weights
    weights = {
        "conceptual": {"vector": 0.7, "graph": 0.3},
        "cli": {"vector": 0.5, "graph": 0.5},       # Graph more important for CLI
        "config": {"vector": 0.5, "graph": 0.5},    # Graph more important for config
        "procedural": {"vector": 0.6, "graph": 0.4},
    }

    w = weights.get(query_type, {"vector": 0.7, "graph": 0.3})

    fused = w["vector"] * vector_score + w["graph"] * graph_score

    return fused
```

### Phase 2 Acceptance Criteria

- [ ] Entity resolution works via trie + embedding fallback
- [ ] Graph Cypher uses entity IDs, not string CONTAINS
- [ ] Relationship types are selected based on query type
- [ ] Graph acts as reranker on vector candidates (not independent channel)
- [ ] Fusion weights are query-type-aware
- [ ] Fused scores are in reasonable range (0-1)
- [ ] Graph failure doesn't crash search (graceful degradation)

---

## Phase 3: Entity Embedding Index

**Objective:** Enable semantic entity matching for queries with novel terminology.
**Timeline:** 3-5 days
**Addresses:** Defects #1, #6 (enhancement)

### 3.1 Create Entity Embedding Collection

```python
from qdrant_client.models import VectorParams, Distance

def create_entity_embedding_collection(qdrant_client, vector_size: int = 1024):
    """Create Qdrant collection for entity embeddings."""
    qdrant_client.recreate_collection(
        collection_name="entity_embeddings",
        vectors_config=VectorParams(
            size=vector_size,  # BGE-M3 dimension
            distance=Distance.COSINE,
        ),
    )
```

### 3.2 Embed All Entities

```python
def embed_all_entities(neo4j_driver, embedder, qdrant_client):
    """
    Create embeddings for all valid entities.

    Embedding text format: "{name} ({entity_type})"
    This helps distinguish "weka user" (command) from "user" (concept).
    """
    # Fetch all valid entities
    cypher = """
    MATCH (e:Entity)
    WHERE e.canonical_name IS NOT NULL
      AND size(e.canonical_name) >= 4
      AND e.entity_type IS NOT NULL
    RETURN e.id, e.canonical_name, e.name, e.entity_type
    """

    entities = []
    with neo4j_driver.session() as session:
        entities = [dict(r) for r in session.run(cypher)]

    if not entities:
        logger.warning("No entities found to embed")
        return

    logger.info("Embedding entities", count=len(entities))

    # Create embedding texts
    texts = [
        f"{e['name']} ({e['entity_type']})"
        for e in entities
    ]

    # Embed in batches
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embedder.embed(batch)
        all_embeddings.extend(embeddings)

    # Create Qdrant points
    points = [
        PointStruct(
            id=i,
            vector=all_embeddings[i],
            payload={
                "entity_id": entities[i]["id"],
                "name": entities[i]["name"],
                "canonical_name": entities[i]["canonical_name"],
                "entity_type": entities[i]["entity_type"],
            }
        )
        for i in range(len(entities))
    ]

    # Upsert to Qdrant
    qdrant_client.upsert(
        collection_name="entity_embeddings",
        points=points,
    )

    logger.info("Entity embeddings created", count=len(points))
```

### 3.3 Integrate into Ingestion Pipeline

When new entities are created during ingestion, also create their embeddings:

```python
def create_entity_with_embedding(
    entity: Dict,
    neo4j_driver,
    embedder,
    qdrant_client,
):
    """Create entity in Neo4j and add embedding to Qdrant."""
    # Create in Neo4j
    create_entity_neo4j(entity, neo4j_driver)

    # Skip embedding for entities without proper names
    if not entity.get("name") or len(entity["name"]) < 4:
        return

    # Create embedding
    text = f"{entity['name']} ({entity.get('entity_type', 'concept')})"
    embedding = embedder.embed([text])[0]

    # Generate stable numeric ID from entity ID
    numeric_id = int(hashlib.md5(entity["id"].encode()).hexdigest()[:8], 16)

    # Upsert to Qdrant
    qdrant_client.upsert(
        collection_name="entity_embeddings",
        points=[PointStruct(
            id=numeric_id,
            vector=embedding,
            payload={
                "entity_id": entity["id"],
                "name": entity["name"],
                "canonical_name": entity.get("canonical_name"),
                "entity_type": entity.get("entity_type"),
            }
        )],
    )
```

### Phase 3 Acceptance Criteria

- [ ] Entity embedding collection exists in Qdrant
- [ ] All existing entities have embeddings
- [ ] New entities get embeddings during ingestion
- [ ] Embedding fallback returns relevant entities for unknown queries
- [ ] Embedding search has reasonable latency (<50ms)

---

## Phase 4: Structure-Aware Context Expansion

**Objective:** Use graph for context window enhancement, not just ranking.
**Timeline:** 1 week (optional enhancement)
**Addresses:** Enhancement (no specific defect)

### 4.1 Sibling Chunk Retrieval

When a chunk is highly ranked, its sibling chunks (same section) are often relevant:

```python
def get_sibling_chunks(chunk_id: str, neo4j_driver, limit: int = 3) -> List[Dict]:
    """
    Get chunks from the same section as the given chunk.

    Useful for expanding context around a highly-ranked result.
    """
    cypher = """
    MATCH (c:Chunk {chunk_id: $chunk_id})-[:IN_SECTION]->(s:Section)
    MATCH (sibling:Chunk)-[:IN_SECTION]->(s)
    WHERE sibling.chunk_id <> $chunk_id
    RETURN sibling {
        .chunk_id,
        .heading,
        .order,
        .tokens
    } AS sibling
    ORDER BY sibling.order
    LIMIT $limit
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher, chunk_id=chunk_id, limit=limit)
        return [dict(r["sibling"]) for r in results]
```

### 4.2 Parent Section Context

Retrieve the parent section's title and other chunks for context:

```python
def get_section_context(chunk_id: str, neo4j_driver) -> Dict:
    """
    Get hierarchical context for a chunk.

    Returns section title, parent section info, and sibling chunk IDs.
    """
    cypher = """
    MATCH (c:Chunk {chunk_id: $chunk_id})-[:IN_SECTION]->(s:Section)
    OPTIONAL MATCH (s)-[:CHILD_OF]->(parent:Section)
    OPTIONAL MATCH (parent)<-[:IN_SECTION]-(parent_chunk:Chunk)
    RETURN s.title AS section_title,
           s.level AS section_level,
           parent.title AS parent_title,
           parent.section_id AS parent_section_id,
           collect(DISTINCT parent_chunk.chunk_id)[..3] AS parent_chunks
    """

    with neo4j_driver.session() as session:
        result = session.run(cypher, chunk_id=chunk_id).single()
        return dict(result) if result else {}
```

### 4.3 Entity-Connected Chunks

Find other chunks that share entities with the result:

```python
def get_entity_related_chunks(
    chunk_id: str,
    entity_ids: List[str],
    neo4j_driver,
    limit: int = 3,
) -> List[Dict]:
    """
    Find chunks that share entities with the given chunk.

    Useful for cross-document discovery.
    """
    if not entity_ids:
        return []

    cypher = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {chunk_id: $chunk_id})
    WHERE e.id IN $entity_ids
    MATCH (e)-[:MENTIONED_IN]->(related:Chunk)
    WHERE related.chunk_id <> $chunk_id
    RETURN related.chunk_id AS chunk_id,
           related.doc_tag AS doc_tag,
           related.heading AS heading,
           count(DISTINCT e) AS shared_entity_count,
           collect(DISTINCT e.name)[..3] AS shared_entities
    ORDER BY shared_entity_count DESC
    LIMIT $limit
    """

    with neo4j_driver.session() as session:
        results = session.run(
            cypher,
            chunk_id=chunk_id,
            entity_ids=entity_ids,
            limit=limit,
        )
        return [dict(r) for r in results]
```

### 4.4 Context Expansion API

Expose context expansion as an optional feature:

```python
def expand_context(
    self,
    results: List[ChunkResult],
    entity_ids: List[str],
    expansion_mode: str = "siblings",  # siblings, hierarchy, entities, all
) -> List[ChunkResult]:
    """
    Expand context around top results.

    Modes:
    - siblings: Add chunks from same section
    - hierarchy: Add parent section chunks
    - entities: Add chunks sharing entities
    - all: All of the above
    """
    if not results:
        return results

    expanded = list(results)
    seen_ids = {r.chunk_id for r in results}

    # Only expand top N results to limit latency
    top_n = min(3, len(results))

    for result in results[:top_n]:
        if expansion_mode in ("siblings", "all"):
            siblings = self.get_sibling_chunks(result.chunk_id)
            for s in siblings:
                if s["chunk_id"] not in seen_ids:
                    # Create lightweight result for context
                    ctx = ChunkResult(
                        chunk_id=s["chunk_id"],
                        heading=s.get("heading"),
                        context_source="sibling",
                        context_parent=result.chunk_id,
                    )
                    expanded.append(ctx)
                    seen_ids.add(s["chunk_id"])

        if expansion_mode in ("hierarchy", "all"):
            context = self.get_section_context(result.chunk_id)
            for pid in context.get("parent_chunks", []):
                if pid not in seen_ids:
                    ctx = ChunkResult(
                        chunk_id=pid,
                        context_source="parent_section",
                        context_parent=result.chunk_id,
                    )
                    expanded.append(ctx)
                    seen_ids.add(pid)

        if expansion_mode in ("entities", "all") and entity_ids:
            related = self.get_entity_related_chunks(result.chunk_id, entity_ids)
            for r in related:
                if r["chunk_id"] not in seen_ids:
                    ctx = ChunkResult(
                        chunk_id=r["chunk_id"],
                        doc_tag=r.get("doc_tag"),
                        heading=r.get("heading"),
                        context_source="shared_entities",
                        context_parent=result.chunk_id,
                        matched_entities=r.get("shared_entities", []),
                    )
                    expanded.append(ctx)
                    seen_ids.add(r["chunk_id"])

    return expanded
```

### Phase 4 Acceptance Criteria

- [ ] Can retrieve sibling chunks for any chunk
- [ ] Can retrieve parent section context
- [ ] Can find entity-related chunks across documents
- [ ] Context expansion is optional and configurable
- [ ] Context expansion doesn't add >100ms latency
- [ ] Expanded chunks are marked with context_source

---

## Phase 5: Validation and Hardening

**Objective:** Ensure correctness, prevent regressions, enable monitoring.
**Timeline:** Ongoing
**Addresses:** All defects (verification)

### 5.1 Unit Tests

```python
# tests/test_graph_channel.py

import pytest
import math

class TestEntityTrie:
    def test_no_null_names(self, neo4j_driver):
        """Trie should not contain entries from NULL-named entities."""
        trie = build_clean_entity_trie(neo4j_driver)
        # Verify no None keys
        assert None not in trie.all_keys()

    def test_no_stopwords(self, neo4j_driver):
        """Trie should not contain stopwords."""
        trie = build_clean_entity_trie(neo4j_driver)
        for word in GENERIC_BLACKLIST:
            assert trie.get(word) is None, f"Stopword '{word}' found in trie"

    def test_no_short_entries(self, neo4j_driver):
        """Trie entries should be at least 4 characters."""
        trie = build_clean_entity_trie(neo4j_driver)
        for key in trie.all_keys():
            assert len(key) >= 4, f"Short key '{key}' found in trie"

    def test_heading_concepts_present(self, neo4j_driver):
        """Trie should contain heading-derived concepts."""
        trie = build_clean_entity_trie(neo4j_driver)
        # These should exist based on document headings
        assert trie.get("metadata units calculation") is not None
        assert trie.get("encrypted filesystems") is not None


class TestDedup:
    def test_preserves_highest_score(self):
        """Dedup should keep the result with highest fused_score."""
        results = [
            ChunkResult(chunk_id="a", fused_score=0.5, vector_score=0.5, graph_score=0),
            ChunkResult(chunk_id="a", fused_score=0.8, vector_score=0, graph_score=0.8),
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 1
        assert deduped[0].fused_score >= 0.8

    def test_merges_signals(self):
        """Dedup should preserve signals from both sources."""
        results = [
            ChunkResult(chunk_id="a", fused_score=0.5, vector_score=0.5, graph_score=0),
            ChunkResult(chunk_id="a", fused_score=0.3, vector_score=0, graph_score=0.8),
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 1
        # Should have best of both
        assert deduped[0].vector_score == 0.5
        assert deduped[0].graph_score == 0.8

    def test_recomputes_fused_score(self):
        """Dedup should recompute fused_score after merging."""
        results = [
            ChunkResult(chunk_id="a", fused_score=0.5, vector_score=0.5, graph_score=0),
            ChunkResult(chunk_id="a", fused_score=0.3, vector_score=0, graph_score=0.5),
        ]
        deduped = _dedup_results(results)
        # Fused should be recomputed from merged signals
        # With default weights (0.7 vector, 0.3 graph): 0.7*0.5 + 0.3*0.5 = 0.5
        assert 0.4 < deduped[0].fused_score < 0.6


class TestGraphScoreNormalization:
    def test_score_in_range(self):
        """Graph scores should be normalized to [0, 1]."""
        for match_count in [0, 1, 2, 5, 10, 100]:
            score = 1.0 - math.exp(-match_count / 3.0)
            assert 0 <= score <= 1, f"Score {score} out of range for count {match_count}"

    def test_monotonic_increase(self):
        """More matches should give higher scores."""
        scores = [1.0 - math.exp(-i / 3.0) for i in range(10)]
        for i in range(1, len(scores)):
            assert scores[i] > scores[i-1], "Scores should increase monotonically"

    def test_reasonable_values(self):
        """Verify expected score values."""
        assert 0.25 < (1.0 - math.exp(-1 / 3.0)) < 0.35  # 1 match → ~0.28
        assert 0.60 < (1.0 - math.exp(-3 / 3.0)) < 0.70  # 3 matches → ~0.63
        assert 0.75 < (1.0 - math.exp(-5 / 3.0)) < 0.85  # 5 matches → ~0.81


class TestQueryClassification:
    def test_cli_queries(self):
        assert classify_query("weka fs list") == "cli"
        assert classify_query("how to run weka cluster status") == "cli"
        assert classify_query("weka user login --username admin") == "cli"

    def test_config_queries(self):
        assert classify_query("WEKA_PORT=14000 configuration") == "config"
        assert classify_query("how to configure settings.yaml") == "config"

    def test_procedural_queries(self):
        assert classify_query("how to install weka") == "procedural"
        assert classify_query("steps for setting up cluster") == "procedural"

    def test_conceptual_queries(self):
        assert classify_query("metadata units calculation") == "conceptual"
        assert classify_query("encrypted filesystems in weka") == "conceptual"
        assert classify_query("SSD resource planning") == "conceptual"
```

### 5.2 Integration Tests

```python
# tests/integration/test_graph_retrieval.py

class TestGraphRetrievalIntegration:
    def test_metadata_units_query(self, retriever):
        """
        The canonical failure case from investigation.

        Query about 'metadata units calculation' should find the
        correct chunk from weka-system-overview_filesystems.
        """
        results = retriever.search("metadata units calculation", top_k=5)

        # The correct chunk should be in top 3
        correct_doc_tag = "weka-system-overview_filesystems"
        top_3_tags = [r.doc_tag for r in results[:3]]
        assert correct_doc_tag in top_3_tags, (
            f"Expected {correct_doc_tag} in top 3, got {top_3_tags}"
        )

        # Verify graph contributed
        correct_result = next(
            (r for r in results if r.doc_tag == correct_doc_tag),
            None
        )
        assert correct_result is not None
        assert correct_result.graph_score > 0, "Graph should have contributed"

    def test_no_garbage_entity_matches(self, retriever):
        """
        Entities like 'at', 'in' should never be matched.
        """
        # This query previously matched 'at' entity
        entities = retriever.resolve_entities("metadata calculation")

        # Get entity names for debugging
        entity_names = []
        for eid in entities:
            name = get_entity_name(eid, retriever.neo4j_driver)
            entity_names.append(name)

        garbage = {"at", "in", "id", "to", "of", "for", "the", "and"}
        matched_garbage = [n for n in entity_names if n.lower() in garbage]

        assert not matched_garbage, f"Garbage entities matched: {matched_garbage}"

    def test_graph_as_reranker(self, retriever):
        """
        Graph should rerank vector results, not add new ones.
        """
        # Get vector-only results
        vector_results = retriever.vector_retriever.search("weka user", top_k=10)
        vector_ids = {r.chunk_id for r in vector_results}

        # Get hybrid results
        hybrid_results = retriever.search("weka user", top_k=10)
        hybrid_ids = {r.chunk_id for r in hybrid_results}

        # All hybrid results should be from vector candidates
        new_ids = hybrid_ids - vector_ids
        assert not new_ids, f"Graph introduced new chunks: {new_ids}"

    def test_cli_query_uses_correct_relationships(self, retriever, mocker):
        """
        CLI queries should use CLI-appropriate relationship types.
        """
        spy = mocker.spy(retriever, '_compute_graph_signals')

        retriever.search("weka fs list command", top_k=5)

        # Check that CLI relationships were used
        call_args = spy.call_args
        assert "CONTAINS_STEP" in call_args.kwargs.get("rel_types", []) or \
               "CONTAINS_STEP" in call_args[1].get("rel_types", [])
```

### 5.3 Metrics and Monitoring

**Metrics to track:**

```python
# In hybrid_retrieval.py search method

metrics = {
    # Entity resolution
    "entities_resolved": len(entity_ids),
    "entity_resolution_method": "trie" if trie_matches else "embedding",

    # Graph contribution
    "chunks_with_graph_signal": len(graph_signals),
    "avg_graph_score": sum(s["score"] for s in graph_signals.values()) / len(graph_signals) if graph_signals else 0,

    # Ranking impact
    "top_1_graph_score": results[0].graph_score if results else 0,
    "top_1_has_graph_signal": results[0].graph_score > 0 if results else False,

    # Query classification
    "query_type": query_type,

    # Timing
    "entity_resolution_ms": entity_resolution_time,
    "graph_signal_ms": graph_signal_time,
    "total_ms": total_time,
}

logger.info("graph_channel_metrics", **metrics)
```

**Dashboard queries:**

```sql
-- Entity coverage over time
SELECT
    date_trunc('day', timestamp) as day,
    avg(entities_resolved) as avg_entities,
    count(*) filter (where entities_resolved > 0) * 100.0 / count(*) as hit_rate_pct
FROM graph_metrics
GROUP BY 1
ORDER BY 1;

-- Graph contribution to top results
SELECT
    query_type,
    avg(top_1_graph_score) as avg_top1_graph_score,
    count(*) filter (where top_1_has_graph_signal) * 100.0 / count(*) as top1_graph_pct
FROM graph_metrics
GROUP BY 1;

-- Latency breakdown
SELECT
    percentile_cont(0.5) within group (order by entity_resolution_ms) as p50_entity,
    percentile_cont(0.5) within group (order by graph_signal_ms) as p50_graph,
    percentile_cont(0.95) within group (order by total_ms) as p95_total
FROM graph_metrics;
```

### 5.4 Alerting

Set up alerts for:

```yaml
alerts:
  - name: graph_entity_coverage_low
    condition: avg(entities_resolved) < 0.5 over 1h
    severity: warning
    message: "Entity resolution hit rate dropped below 50%"

  - name: graph_channel_errors
    condition: count(graph_channel_error) > 10 in 5m
    severity: error
    message: "Graph channel experiencing errors"

  - name: graph_latency_high
    condition: p95(graph_signal_ms) > 200 over 15m
    severity: warning
    message: "Graph signal computation latency elevated"

  - name: graph_zero_contribution
    condition: avg(top_1_has_graph_signal) < 0.1 over 1h
    severity: warning
    message: "Graph contributing to less than 10% of top results"
```

### Phase 5 Acceptance Criteria

- [ ] All unit tests pass
- [ ] Integration tests cover canonical failure cases
- [ ] Metrics are logged for every graph query
- [ ] Dashboard shows entity coverage, graph contribution, latency
- [ ] Alerts configured for degradation scenarios
- [ ] No regressions on existing test suite

---

## Implementation Priority

| Phase | Priority | Effort | Impact | Dependencies |
|-------|----------|--------|--------|--------------|
| Phase 0 | P0 (Critical) | 1-2 days | High (stops harm) | None |
| Phase 1 | P0 (Critical) | 1 week | High (fixes root cause) | Phase 0 |
| Phase 2 | P1 (High) | 1 week | High (proper integration) | Phase 1 |
| Phase 3 | P2 (Medium) | 3-5 days | Medium (cold start) | Phase 1 |
| Phase 4 | P3 (Low) | 1 week | Low (enhancement) | Phase 2 |
| Phase 5 | Ongoing | Continuous | High (quality) | All phases |

**Recommended order:** Phase 0 → Phase 1 → Phase 2 → Phase 5 (tests) → Phase 3 → Phase 4

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Heading extraction produces too many entities | Medium | Medium | Filter generic headings, set minimum length |
| Entity embeddings don't match well | Medium | Low | Use same embedder as chunks, tune threshold |
| Performance regression | Low | High | Benchmark before/after, set latency budgets |
| Graph makes results worse | Medium | High | A/B test, feature flag, Option B (reranker) design |
| Migration corrupts existing data | Low | High | Backup before migration, test on staging |

---

## Success Metrics

After full implementation, we should see:

1. **Entity coverage:** >50% of chunks have ≥1 entity link (up from 11%)
2. **Query entity hit rate:** >70% of queries resolve ≥1 entity
3. **Graph contribution:** >30% of top-5 results have graph_score > 0
4. **Precision improvement:** "metadata units calculation" query returns correct doc in top 3
5. **No garbage matches:** Zero queries match entities like "at", "in", "id"
6. **Latency budget:** Graph adds <100ms p95 to query time

---

## Appendix A: Migration Scripts

### A.1 Clean Up Entity Labels

```cypher
// Run in Neo4j Browser or via driver

// Step 1: Count current state
MATCH (n:Entity) RETURN count(n) as total_entities;
MATCH (n:Entity) WHERE n.name IS NULL RETURN count(n) as null_name_entities;

// Step 2: Remove Entity label from Steps with NULL names
MATCH (n:Step:Entity)
WHERE n.name IS NULL
REMOVE n:Entity;

// Step 3: Remove Entity label from generic config params
MATCH (n:Configuration:Entity)
WHERE n.name IN ['help', 'color', 'profile', 'json', 'output', 'format',
                  'filter', 'verbose', 'id', 'name', 'type', 'value', 'data',
                  'at', 'in', 'to', 'of', 'for', 'the', 'and', 'or']
REMOVE n:Entity;

// Step 4: Verify cleanup
MATCH (n:Entity) RETURN count(n) as remaining_entities;
MATCH (n:Entity) WHERE n.name IS NULL RETURN count(n) as null_after_cleanup;
```

### A.2 Create Structure Relationships

```cypher
// Create IN_SECTION relationships
MATCH (c:Chunk)
WHERE c.parent_section_id IS NOT NULL
MATCH (s:Section {section_id: c.parent_section_id})
MERGE (c)-[:IN_SECTION]->(s);

// Create section hierarchy
MATCH (child:Section)
WHERE child.parent_section_id IS NOT NULL
MATCH (parent:Section {section_id: child.parent_section_id})
MERGE (child)-[:CHILD_OF]->(parent);

// Verify
MATCH ()-[r:IN_SECTION]->() RETURN count(r);
MATCH ()-[r:CHILD_OF]->() RETURN count(r);
```

### A.3 Create Heading Entities (Python)

```python
# scripts/create_heading_entities.py

import hashlib
import re
from neo4j import GraphDatabase

def canonicalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[-_\s]+', ' ', name).strip()
    name = name.rstrip(':;,.')
    name = re.sub(r'^(the|a|an)\s+', '', name)
    return name

SKIP_HEADINGS = {
    "overview", "introduction", "summary", "notes", "description",
    "prerequisites", "requirements", "see also", "related topics",
    "example", "examples", "syntax", "parameters", "returns",
}

def create_heading_entities(driver):
    # Fetch all sections with headings
    fetch_cypher = """
    MATCH (s:Section)
    WHERE s.title IS NOT NULL AND size(s.title) >= 5
    RETURN s.section_id, s.title, s.doc_tag, s.chunk_id
    """

    entities = []
    with driver.session() as session:
        for record in session.run(fetch_cypher):
            heading = record["title"].strip()

            if heading.lower() in SKIP_HEADINGS:
                continue
            if re.match(r'^[\d\.\-\*\#]+$', heading):
                continue

            canonical = canonicalize(heading)
            if len(canonical) < 5:
                continue

            entity_id = hashlib.sha256(f"heading:{canonical}".encode()).hexdigest()

            entities.append({
                "id": entity_id,
                "name": heading,
                "canonical_name": canonical,
                "entity_type": "heading_concept",
                "source_section_id": record["section_id"],
                "chunk_id": record["chunk_id"] or record["section_id"],
                "doc_tag": record["doc_tag"],
            })

    print(f"Creating {len(entities)} heading entities...")

    # Create entities and relationships
    create_cypher = """
    UNWIND $entities AS ent
    MERGE (e:Entity:Concept {id: ent.id})
    SET e.name = ent.name,
        e.canonical_name = ent.canonical_name,
        e.entity_type = ent.entity_type,
        e.source_section_id = ent.source_section_id,
        e.doc_tag = ent.doc_tag,
        e.updated_at = datetime()

    WITH e, ent
    MATCH (c:Chunk {chunk_id: ent.chunk_id})
    MERGE (e)-[r:DEFINES]->(c)
    SET r.confidence = 1.0
    MERGE (e)-[m:MENTIONED_IN]->(c)
    SET m.confidence = 1.0
    """

    with driver.session() as session:
        session.run(create_cypher, entities=entities)

    print("Done!")

if __name__ == "__main__":
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))
    create_heading_entities(driver)
    driver.close()
```

---

## Appendix B: Configuration Changes

### B.1 development.yaml Updates

```yaml
hybrid:
  mode: "bge_reranker"
  enabled: true

  # Graph channel configuration (UPDATED)
  graph_channel_enabled: true
  graph_channel_mode: "reranker"  # NEW: reranker | independent

  # Entity resolution (NEW)
  entity_resolution:
    trie_enabled: true
    embedding_fallback_enabled: true
    embedding_threshold: 0.7
    max_entities_per_query: 10

  # Query type weights (NEW)
  query_type_weights:
    conceptual:
      vector: 0.7
      graph: 0.3
    cli:
      vector: 0.5
      graph: 0.5
    config:
      vector: 0.5
      graph: 0.5
    procedural:
      vector: 0.6
      graph: 0.4

  # Graph traversal (UPDATED)
  graph_relationships:
    conceptual: ["MENTIONED_IN", "DEFINES", "IN_SECTION"]
    cli: ["MENTIONED_IN", "CONTAINS_STEP", "HAS_PARAMETER"]
    config: ["MENTIONED_IN", "HAS_PARAMETER", "DEFINES"]
    procedural: ["MENTIONED_IN", "CONTAINS_STEP", "NEXT_CHUNK"]
```

---

*End of Plan Document*
