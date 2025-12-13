Below is a single, end‑to‑end plan that unifies both versions into **one coherent “GraphRAG‑ready lexical + hierarchy + entity bridge” implementation**—while also giving you a clean path from the older `:Section`/`HAS_SECTION` world into the newer **chunk‑only** `:Chunk`/`HAS_CHUNK` world.

The plan is written so you can:

* **Fix the database you already have** (one‑time backfill)
* **Make it impossible to regress** (per‑document builder + contract checks)
* **Unlock retrieval upgrades immediately** (adjacency, hierarchy, entity anchoring, bounded expansion, diffusion rerank)

---

## 0) The end state you’re standardizing on

### 0.1 Canonical node model

**Documents**

* `(:Document {id, ...})`

**Content units (everything retrievable)**

* `(:Chunk {chunk_id, document_id, order, level, parent_path, heading, text, ...})`
* Optional subtype labels are fine:

  * `(:CodeChunk:Chunk)`
  * `(:TableChunk:Chunk)`

> If you still physically have `:Section` nodes (or dual labels), the plan below migrates you safely to `:Chunk` without requiring you to redesign your content extraction.

### 0.2 Canonical relationships

**Lexical “must‑haves”**

* `(:Document)-[:HAS_CHUNK]->(:Chunk)`
* `(:Chunk)-[:NEXT_CHUNK]->(:Chunk)` (document sequence adjacency)

**Hierarchy (heading structure without separate Section nodes)**

* `(:Chunk)-[:PARENT_HEADING]->(:Chunk)` (child → parent heading chunk)
* `(:Chunk)-[:CHILD_OF]->(:Chunk)` (alias of `PARENT_HEADING` direction for compatibility)
* `(:Chunk)-[:PARENT_OF]->(:Chunk)` (reverse direction, optional but useful)

**Sibling adjacency (optional but valuable)**

* `(:Chunk)-[:NEXT]->(:Chunk)` within the same parent scope (and usually same level)

**Entity bridge**

* `(:Chunk)-[:MENTIONS {confidence,...}]->(:Entity)`
* `(:Entity {entity_type, name, normalized_name, ...})`

### 0.3 Canonical parent pointer property

Use **one correct pointer**, optionally keep a legacy alias temporarily:

* **Primary:** `Chunk.parent_chunk_id` (points to the parent’s `chunk_id`)
* **Temporary compatibility alias:** `Chunk.parent_section_id` (same value)
  Keep for 1–2 releases if older code expects it, then delete it.

---

## 1) First: detect which world you’re in (and what’s missing)

Run these quick checks to avoid “migrations that succeed but don’t fix anything.”

### 1.1 Do any `:Section` nodes still exist?

```cypher
MATCH (n:Section) RETURN count(n) AS section_nodes;
```

### 1.2 Do you have `HAS_SECTION` and/or `HAS_CHUNK`?

```cypher
MATCH (:Document)-[r:HAS_SECTION]->() RETURN count(r) AS has_section;
MATCH (:Document)-[r:HAS_CHUNK]->()   RETURN count(r) AS has_chunk;
```

### 1.3 Preconditions for building adjacency + hierarchy

```cypher
MATCH (c:Chunk) WHERE c.chunk_id IS NULL RETURN count(c) AS missing_chunk_id;
MATCH (c:Chunk) WHERE c.document_id IS NULL RETURN count(c) AS missing_document_id;
MATCH (c:Chunk) WHERE c.order IS NULL RETURN count(c) AS missing_order;
```

If `missing_order` or `missing_document_id` is non‑zero, fix ingestion first (or backfill those properties) before you build `NEXT_CHUNK`.

### 1.4 Check `chunk_id` uniqueness before creating a constraint

```cypher
MATCH (c:Chunk)
WITH c.chunk_id AS id, count(*) AS n
WHERE id IS NOT NULL AND n > 1
RETURN id, n
ORDER BY n DESC
LIMIT 50;
```

If that returns rows, don’t create a uniqueness constraint yet—resolve duplicates first.

---

## 2) One-time migration/backfill (works for both schemas)

This single migration does five critical things:

1. **Unifies legacy content nodes into `:Chunk`**
2. **Unifies doc→chunk edges into `HAS_CHUNK`** (without breaking `HAS_SECTION`)
3. Builds stable **`parent_path_norm`**
4. Computes **`parent_chunk_id`** (and optionally `parent_section_id`)
5. Rebuilds **hierarchy + adjacency** edges (`PARENT_HEADING`, `NEXT_CHUNK`, `NEXT`, …)

### 2.1 Compatibility normalization (safe even if you’re already chunk‑only)

```cypher
// 1) If any content nodes still carry :Section, also label them :Chunk.
// (No-op if :Section is already gone)
MATCH (s:Section)
SET s:Chunk;

// 2) If doc->content edges exist as HAS_SECTION, mirror them as HAS_CHUNK.
// (No-op if you already have HAS_CHUNK everywhere)
MATCH (d:Document)-[:HAS_SECTION]->(c)
MERGE (d)-[:HAS_CHUNK]->(c);
```

> Don’t remove `:Section` labels or `HAS_SECTION` edges yet unless you’re certain nothing references them. This plan is designed to be additive first, then you delete legacy structures after you’re stable.

---

### 2.2 Indexes/constraints (add only after prechecks)

**Recommended indexes (generally safe):**

```cypher
CREATE RANGE INDEX chunk_doc_order IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id, c.order);

CREATE RANGE INDEX chunk_doc_parent_path_norm IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id, c.parent_path_norm);

CREATE RANGE INDEX chunk_parent_chunk_id IF NOT EXISTS
FOR (c:Chunk) ON (c.parent_chunk_id);
```

**Uniqueness constraint (only if your precheck showed no duplicates):**

```cypher
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;
```

---

### 2.3 Normalize `parent_path` into `parent_path_norm`

This makes matching stable across delimiter and whitespace variation.

```cypher
MATCH (c:Chunk)
WHERE c.parent_path IS NOT NULL
WITH c,
     [p IN split(replace(c.parent_path, ' > ', '>'), '>') | trim(p)] AS parts
WITH c,
     reduce(path = '', p IN parts |
       path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
     ) AS norm
SET c.parent_path_norm = norm;
```

---

### 2.4 Compute `parent_chunk_id` from `parent_path_norm`

This assumes your `parent_path_norm` is a full breadcrumb ending in the current heading. The parent’s path is the breadcrumb without the last segment.

A robust choice is “nearest earlier chunk in the same document with the matching parent breadcrumb,” optionally consistent with `level`.

```cypher
MATCH (child:Chunk)
WHERE child.parent_path_norm IS NOT NULL
  AND child.parent_path_norm CONTAINS ' > '
WITH child, split(child.parent_path_norm, ' > ') AS parts
WITH child, parts[0..size(parts)-1] AS parentParts
WITH child,
     reduce(path = '', p IN parentParts |
       path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
     ) AS parentPathNorm
MATCH (parent:Chunk {document_id: child.document_id, parent_path_norm: parentPathNorm})
WHERE parent.order < child.order
  AND (child.level IS NULL OR parent.level IS NULL OR parent.level < child.level)
WITH child, parent
ORDER BY parent.order DESC
WITH child, head(collect(parent)) AS parent
SET child.parent_chunk_id = parent.chunk_id,
    // optional temporary alias:
    child.parent_section_id = parent.chunk_id;
```

---

### 2.5 Rebuild structural relationships from scratch (recommended)

This is the part that makes the migration **repeatable and correct** even if you run it more than once.

#### 2.5.1 Clear existing structural edges between chunks

```cypher
MATCH (a:Chunk)-[r:NEXT_CHUNK|NEXT|PARENT_HEADING|CHILD_OF|PARENT_OF]-(b:Chunk)
DELETE r;
```

#### 2.5.2 Recreate hierarchy edges

```cypher
MATCH (child:Chunk)
WHERE child.parent_chunk_id IS NOT NULL
MATCH (parent:Chunk {chunk_id: child.parent_chunk_id})
MERGE (child)-[ph:PARENT_HEADING]->(parent)
ON CREATE SET ph.level_delta = coalesce(child.level, 0) - coalesce(parent.level, 0)
MERGE (child)-[:CHILD_OF]->(parent)
MERGE (parent)-[:PARENT_OF]->(child);
```

#### 2.5.3 Recreate `NEXT_CHUNK` (document-wide sequence)

```cypher
MATCH (c:Chunk)
WHERE c.document_id IS NOT NULL AND c.order IS NOT NULL
WITH c.document_id AS doc_id, collect(c ORDER BY c.order) AS chs
UNWIND range(0, size(chs)-2) AS i
WITH chs[i] AS c1, chs[i+1] AS c2
MERGE (c1)-[:NEXT_CHUNK]->(c2);
```

#### 2.5.4 Recreate sibling `NEXT` within the same parent scope

```cypher
MATCH (c:Chunk)
WHERE c.document_id IS NOT NULL AND c.order IS NOT NULL
WITH c.document_id AS doc_id, c.parent_chunk_id AS pid, c.level AS lvl, c
WITH doc_id, pid, lvl, collect(c ORDER BY c.order) AS chs
UNWIND range(0, size(chs)-2) AS i
WITH chs[i] AS c1, chs[i+1] AS c2
MERGE (c1)-[:NEXT]->(c2);
```

---

## 3) Validate immediately after migration

### 3.1 Lexical relationships exist

```cypher
MATCH (:Document)-[r:HAS_CHUNK]->(:Chunk)
RETURN count(r) AS has_chunk_edges;

MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) AS next_chunk_edges;
```

### 3.2 Hierarchy exists

```cypher
MATCH ()-[r:PARENT_HEADING]->() RETURN count(r) AS parent_heading_edges;
MATCH ()-[r:CHILD_OF]->()       RETURN count(r) AS child_of_edges;
MATCH ()-[r:PARENT_OF]->()      RETURN count(r) AS parent_of_edges;
```

### 3.3 Parent coverage for chunks that should have a parent

```cypher
MATCH (c:Chunk)
WHERE c.parent_path_norm CONTAINS ' > '
WITH count(c) AS should_have_parent
MATCH (c:Chunk)
WHERE c.parent_path_norm CONTAINS ' > ' AND c.parent_chunk_id IS NOT NULL
WITH should_have_parent, count(c) AS has_parent
RETURN
  should_have_parent,
  has_parent,
  CASE
    WHEN should_have_parent = 0 THEN 1.0
    ELSE (has_parent * 1.0 / should_have_parent)
  END AS coverage;
```

### 3.4 Inspect unresolved parents (top samples)

```cypher
MATCH (c:Chunk)
WHERE c.parent_path_norm CONTAINS ' > '
  AND c.parent_chunk_id IS NULL
RETURN c.chunk_id, c.document_id, c.order, c.level, c.parent_path_norm
LIMIT 50;
```

If unresolved rows are meaningful, the typical causes are:

* breadcrumb segments don’t match actual headings 1:1
* parent headings are not emitted as chunks (you’d need “heading-only chunks” or adjust parser)
* `order` isn’t aligned with document order

---

## 4) Make it permanent: per-document structural builder in ingestion

The migration fixes history. The ingestion step prevents drift.

### 4.1 Required ingestion invariants

For every chunk you write:

* `chunk_id` present and stable
* `document_id` present
* `order` present (monotonic within doc)
* `parent_path` present when hierarchical structure exists
* `level` present if you can compute it (strongly recommended)

### 4.2 Run a scoped builder once per document

Create a function in your ingestion pipeline like:

* `build_structural_edges_for_document(doc_id)`

This does:

1. normalize parent path
2. recompute parent pointers
3. delete and rebuild structural edges **for that one document**
4. re-create `NEXT_CHUNK` and hierarchy for that doc

#### Per-document Cypher (scoped rebuild)

```cypher
WITH $doc_id AS doc_id

// --- Normalize parent_path -> parent_path_norm
MATCH (c:Chunk {document_id: doc_id})
WHERE c.parent_path IS NOT NULL
WITH c,
     [p IN split(replace(c.parent_path, ' > ', '>'), '>') | trim(p)] AS parts
WITH c,
     reduce(path = '', p IN parts |
       path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
     ) AS norm
SET c.parent_path_norm = norm;

// --- Recompute parent_chunk_id (overwrite for correctness)
MATCH (child:Chunk {document_id: doc_id})
WHERE child.parent_path_norm IS NOT NULL
  AND child.parent_path_norm CONTAINS ' > '
WITH child, split(child.parent_path_norm, ' > ') AS parts
WITH child, parts[0..size(parts)-1] AS parentParts
WITH child,
     reduce(path = '', p IN parentParts |
       path + CASE WHEN path = '' THEN '' ELSE ' > ' END + p
     ) AS parentPathNorm
MATCH (parent:Chunk {document_id: doc_id, parent_path_norm: parentPathNorm})
WHERE parent.order < child.order
  AND (child.level IS NULL OR parent.level IS NULL OR parent.level < child.level)
WITH child, parent
ORDER BY parent.order DESC
WITH child, head(collect(parent)) AS parent
SET child.parent_chunk_id = parent.chunk_id,
    child.parent_section_id = parent.chunk_id; // optional alias

// --- Clear structural edges among chunks in this document
MATCH (a:Chunk {document_id: doc_id})-[r:NEXT_CHUNK|NEXT|PARENT_HEADING|CHILD_OF|PARENT_OF]-(b:Chunk {document_id: doc_id})
DELETE r;

// --- Recreate hierarchy
MATCH (child:Chunk {document_id: doc_id})
WHERE child.parent_chunk_id IS NOT NULL
MATCH (parent:Chunk {chunk_id: child.parent_chunk_id})
MERGE (child)-[ph:PARENT_HEADING]->(parent)
ON CREATE SET ph.level_delta = coalesce(child.level, 0) - coalesce(parent.level, 0)
MERGE (child)-[:CHILD_OF]->(parent)
MERGE (parent)-[:PARENT_OF]->(child);

// --- Recreate NEXT_CHUNK (doc-wide)
MATCH (c:Chunk {document_id: doc_id})
WHERE c.order IS NOT NULL
WITH collect(c ORDER BY c.order) AS chs
UNWIND range(0, size(chs)-2) AS i
WITH chs[i] AS c1, chs[i+1] AS c2
MERGE (c1)-[:NEXT_CHUNK]->(c2);

// --- Recreate NEXT (siblings)
MATCH (c:Chunk {document_id: doc_id})
WHERE c.order IS NOT NULL
WITH c.parent_chunk_id AS pid, c.level AS lvl, c
WITH pid, lvl, collect(c ORDER BY c.order) AS chs
UNWIND range(0, size(chs)-2) AS i
WITH chs[i] AS c1, chs[i+1] AS c2
MERGE (c1)-[:NEXT]->(c2);
```

### 4.3 Fail fast if the document is malformed

After running the builder, run doc-scoped sanity checks in code and raise/log if they fail:

* If doc has `N > 1` chunks, expect `N-1` `NEXT_CHUNK` edges.
* If doc has any `parent_path_norm` containing `>`, expect a high parent coverage.

---

## 5) Graph contract checks (CI gate + runtime monitoring)

These are the “you cannot merge unless the graph contract is intact” checks.

### 5.1 Exactly one doc membership per chunk (recommended)

```cypher
MATCH (c:Chunk)
OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
WITH c, count(d) AS doc_count
WHERE doc_count <> 1
RETURN c.chunk_id, c.document_id, doc_count
LIMIT 50;
```

### 5.2 `NEXT_CHUNK` correctness signals

```cypher
MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) AS next_chunk_edges;

MATCH (c:Chunk)-[r:NEXT_CHUNK]->()
WITH c, count(r) AS out_deg
WHERE out_deg > 1
RETURN c.chunk_id, out_deg
LIMIT 25;

// cycle smoke test
MATCH p=(c:Chunk)-[:NEXT_CHUNK*1..50]->(c)
RETURN c.chunk_id
LIMIT 10;
```

### 5.3 Hierarchy coverage threshold

```cypher
MATCH (c:Chunk)
WHERE c.parent_path_norm CONTAINS ' > '
WITH count(c) AS should_have_parent
MATCH (c:Chunk)
WHERE c.parent_path_norm CONTAINS ' > ' AND c.parent_chunk_id IS NOT NULL
WITH should_have_parent, count(c) AS has_parent
RETURN
  should_have_parent,
  has_parent,
  CASE WHEN should_have_parent = 0 THEN 1.0 ELSE (has_parent*1.0/should_have_parent) END AS coverage;
```

Set a threshold appropriate to your corpus (often 0.95+ once parsing is correct).

---

## 6) Entity hygiene (improves linking and traversal quality immediately)

If entities aren’t normalized, everything downstream becomes noisy: candidate expansion, overlap scoring, diffusion edges, de-dup.

### 6.1 One-time backfill

```cypher
CREATE RANGE INDEX entity_type_normalized_name IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type, e.normalized_name);

MATCH (e:Entity)
WHERE e.name IS NOT NULL AND e.normalized_name IS NULL
SET e.normalized_name = toLower(trim(e.name));
```

### 6.2 In ingestion

Always set:

* `normalized_name = lower(trim(name))`

### 6.3 Query-time entity linking policy (deterministic)

1. Try `(entity_type, normalized_name)`
2. If type is missing, try `normalized_name`
3. If multiple hits:

   * prefer entity with most `(:Chunk)-[:MENTIONS]->(:Entity)` support
   * optionally prefer entities whose mentions cluster in the same document(s) as the query hits

---

## 7) Retrieval behavior you unlock once the graph contract is stable

This merges the “safe expansion” idea with the “middle path” rerank/diffuse idea.

### 7.1 Baseline flow

1. **Vector search** returns top‑K candidate `chunk_id`s with similarity scores.
2. You fetch **graph features + local edges** among those candidates.
3. You apply:

   * **bounded rerank** (graph-aware scoring)
   * optional **diffusion** within candidate subgraph
4. You do **context expansion** (add adjacency + parent heading) *without letting expansion outrank seeds*.

### 7.2 Candidate subgraph extraction query (no APOC / no GDS)

This is essentially the second plan’s query, kept chunk-only, and designed to return:

* node feature rows for the candidates
* edges among candidates (`shared_entity` + `next_chunk`)

```cypher
WITH $candidates AS candidates,
     $query_entities AS query_entities,
     $mention_conf_min AS confMin,
     $max_entities_per_chunk AS maxE,
     $max_candidate_chunks_per_entity AS maxPerEnt,
     $return_edges AS returnEdges

UNWIND candidates AS c
MATCH (ch:Chunk {chunk_id: c.chunk_id})
WITH collect({ch: ch, base_score: coalesce(c.score, 0.0)}) AS cand,
     [q IN query_entities | toLower(q)] AS qents,
     confMin, maxE, maxPerEnt, returnEdges

WITH cand, [x IN cand | x.ch.chunk_id] AS candIds, qents, confMin, maxE, maxPerEnt, returnEdges

// ---- Node features
CALL {
  WITH cand, qents, confMin, maxE
  UNWIND cand AS row
  WITH row.ch AS ch, row.base_score AS base_score, qents, confMin, maxE

  OPTIONAL MATCH (ch)-[m:MENTIONS]->(e:Entity)
  WHERE coalesce(m.confidence, 1.0) >= confMin
  WITH ch, base_score, qents,
       collect({e:e, conf:coalesce(m.confidence,1.0)})[0..maxE] AS mentions
  WITH ch, base_score, qents,
       [x IN mentions | x.e] AS ents,
       [x IN mentions | x.conf] AS confs,
       [x IN mentions | toLower(coalesce(x.e.normalized_name, x.e.name))] AS entNamesLower

  WITH ch, base_score, ents, confs,
       [i IN range(0,size(entNamesLower)-1) WHERE entNamesLower[i] IN qents | i] AS overlapIdx

  CALL {
    WITH ents
    UNWIND ents AS e
    OPTIONAL MATCH (e)<-[:MENTIONS]-(:Chunk)
    WITH count(*) AS deg
    RETURN max(deg) AS max_deg, avg(deg) AS avg_deg
  }

  RETURN collect({
    chunk_id: ch.chunk_id,
    document_id: ch.document_id,
    base_score: base_score,
    features: {
      mention_entity_count: size(ents),
      mention_conf_sum: reduce(s=0.0, x IN confs | s + x),
      mention_conf_mean: CASE WHEN size(confs)=0 THEN 0.0 ELSE reduce(s=0.0, x IN confs | s + x) / size(confs) END,

      query_entity_overlap_count: size(overlapIdx),
      query_entity_overlap_conf_sum: reduce(s=0.0, i IN overlapIdx | s + confs[i]),

      max_entity_mention_degree: coalesce(max_deg, 0),
      avg_entity_mention_degree: coalesce(avg_deg, 0)
    }
  }) AS node_rows
}

// ---- Shared-entity edges among candidates
CALL {
  WITH cand, confMin, maxPerEnt, returnEdges
  WHERE returnEdges = true

  UNWIND cand AS row
  WITH row.ch AS ch, confMin
  MATCH (ch)-[m:MENTIONS]->(e:Entity)
  WHERE coalesce(m.confidence,1.0) >= confMin
  WITH e, collect(DISTINCT ch.chunk_id) AS chIds
  WHERE size(chIds) > 1 AND size(chIds) <= maxPerEnt

  UNWIND range(0, size(chIds)-2) AS i
  UNWIND range(i+1, size(chIds)-1) AS j
  WITH chIds[i] AS a, chIds[j] AS b
  WITH a, b, count(*) AS w
  RETURN collect({src:a, dst:b, type:"shared_entity", weight:toFloat(w)}) AS shared_edges
}

// ---- NEXT_CHUNK edges inside candidate pool
CALL {
  WITH cand, returnEdges
  WHERE returnEdges = true

  UNWIND cand AS row
  WITH row.ch AS a
  MATCH (a)-[:NEXT_CHUNK]->(b:Chunk)
  WHERE b.chunk_id IN [x IN cand | x.ch.chunk_id]
  RETURN collect({src:a.chunk_id, dst:b.chunk_id, type:"next_chunk", weight:1.0}) AS next_edges
}

RETURN
  node_rows AS nodes,
  coalesce(shared_edges, []) + coalesce(next_edges, []) AS edges;
```

### 7.3 How to use this result safely

* **Reranking:** adjust ordering of the candidate list based on node features + graph signals.
* **Diffusion:** run a lightweight PPR-like diffusion in application code using returned edges.
* **Context expansion:** add neighbors (parent + ±1 adjacency) as *context only*, not as new top answers.

A conservative expansion policy that’s hard to mess up:

* For each final top chunk:

  * include `PARENT_HEADING` (1 hop) if present
  * include `NEXT_CHUNK` previous + next (bounded to ±1)
* Do **not** add expansions back into the ranking pool; treat them as attached context blocks.

---

## 8) Cleanup and deprecation (after you’ve stabilized)

Once `HAS_CHUNK`, `NEXT_CHUNK`, and hierarchy are proven stable in CI and production:

1. Stop writing `HAS_SECTION`
2. Remove `:Section` labels (if still present)
3. Delete the compatibility alias property `parent_section_id`
4. Update any query templates to use only:

   * `:Chunk`
   * `HAS_CHUNK`
   * `chunk_id`
   * `parent_chunk_id`

---

## 9) Minimal “do these first” checklist (compressed but complete)

If you want the fastest path to “works and stays working”:

1. **Run the one-time migration**:

   * normalize to `:Chunk`
   * mirror `HAS_SECTION` → `HAS_CHUNK`
   * compute `parent_chunk_id`
   * rebuild `PARENT_HEADING` + `NEXT_CHUNK`
2. **Add the per-document builder** to ingestion (scoped rebuild)
3. **Add CI contract checks**:

   * doc membership per chunk
   * `NEXT_CHUNK` existence and non-branching
   * hierarchy coverage threshold
4. **Normalize entities** (`normalized_name`) and enforce in ingestion
5. **Turn on bounded context expansion** (parent + adjacency)
6. **Then** add diffusion reranking as an incremental enhancement

---

If you share the output of these four counts after the migration:

```cypher
MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*) AS has_chunk;
MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) AS next_chunk;
MATCH ()-[r:PARENT_HEADING]->() RETURN count(r) AS parent_heading;
MATCH (c:Chunk) WHERE c.parent_path_norm CONTAINS ' > ' AND c.parent_chunk_id IS NULL RETURN count(c) AS unresolved;
```

…I can tell you very precisely whether the remaining work is parser semantics (breadcrumb mismatch) vs. ordering/level consistency vs. missing heading-chunks.
