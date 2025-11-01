For **documentation RAG** (multi‑turn, high precision/recall, and strict provenance), the winning pattern is:

*   **A compact, doc‑centric subgraph** (≈ 6–10 labels) that captures _only_ what improves retrieval and grounding (documents, chunks, entities, topics).
*   **Strong provenance**: answers are built **only** from retrieved chunks/excerpts; no chunk → no claim (“no‑cite, no‑claim”).
*   **Session context** to bias follow‑up turns to the same entities/topics—without storing everything in the long‑term graph.
*   **Separate the runtime/infra graph** (your detailed v2) into its own DB or at least a distinct subgraph; bridge with a small set of `MENTIONS`/`REFERS_TO` edges when needed.

Below is the **lean, purpose‑built DocRAG schema (v3)** that I recommend. It keeps the _full package_ of documentation content (down to the paragraph/snippet level) without the operational noise.

* * *

DocRAG v3 — lean schema (concept)
---------------------------------

**Nodes**

*   `Document {doc_id, title, doc_type, product, version, uri, language, last_indexed_at, text, embedding}`
    One per source file/page. `doc_type` e.g., `guide`, `runbook`, `faq`, `api`, `cli`, `release_notes`.
*   `Chunk {chunk_id, hash, text, order, heading_path, page, tokens, embedding}`
    Atomic retrieval unit (300–800 tokens). `heading_path` preserves hierarchy.
*   `Entity {entity_id, name, etype, aliases, version}`
    Canonical concepts surfaced in docs (e.g., Feature, CLI command, API endpoint, Alert type, Config key). `etype` is an enum: `['Feature','CLI','API','Alert','Config','Component','Concept']`.
*   `Topic {name}`
    Lightweight taxonomy tags (curated or auto‑assigned).
*   `Session {session_id, started_at}` _(optional but helpful for multi‑turn)_
*   `Query {query_id, text, turn, asked_at}` _(optional)_
*   `Answer {answer_id, created_at}` _(optional)_

**Relationships**

*   `(:Document)-[:HAS_CHUNK]->(:Chunk)`
*   `(:Chunk)-[:MENTIONS]->(:Entity)` (many‑to‑many; built by EL/NER + alias map)
*   `(:Document)-[:HAS_TOPIC]->(:Topic)` and/or `(:Chunk)-[:HAS_TOPIC]->(:Topic)`
*   Optional knowledge edges (curated or learned) to improve disambiguation:
    *   `(:Entity)-[:RELATED_TO {rel_type, confidence}]->(:Entity)`
        `rel_type` e.g., `synonym`, `requires`, `incompatible_with`, `updates`, `replaces`.
*   **Multi‑turn memory** (short‑lived, can be periodically pruned):
    *   `(:Session)-[:HAS_QUERY]->(:Query)`
    *   `(:Query)-[:FOCUSED_ON {score}]->(:Entity)` (entities inferred from the turn)
    *   `(:Query)-[:RETRIEVED {rank, score_text, score_vec, score_graph}]->(:Chunk)`
    *   `(:Query)-[:ANSWERED_AS]->(:Answer)`
    *   `(:Answer)-[:SUPPORTED_BY {rank}]->(:Chunk)`

> Keep “ops graph” (Servers, NICs, Filesystems, etc.) in a **separate database** or at least a different label namespace. If you really need runtime tie‑ins, bridge via `(:Entity {etype:'Component'})-[:REFERS_TO]->(:Server|:Feature|:APIEndpoint)` in that other DB. Do **not** attach runtime nodes directly to `Chunk`/`Document` in the DocRAG DB.

* * *

Why this is more effective (and reduces hallucination)
------------------------------------------------------

1.  **Fewer labels, tighter neighborhoods** → less spurious fan‑out, cleaner expansions.
2.  **Everything flows through `Chunk`** → every claim has line‑of‑text provenance.
3.  **Entity focus** in multi‑turn: each turn updates the set of entities in focus; retrieval is biased to chunks that `MENTIONS` those entities.
4.  **Evidence gating**: if top‑k chunk scores (or coverage of the asked entities) are below thresholds, answer with uncertainty or a targeted follow‑up—**don’t** synthesize.

* * *

DocRAG v3 — machine‑readable sketch (JSON)
------------------------------------------

```json
{
  "version": "3.0",
  "nodes": [
    {"label": "Document", "identity": ["doc_id"],
     "properties": {"doc_id":"String","title":"String","doc_type":"String","product":"String","version":"String","uri":"String","language":"String","last_indexed_at":"DateTime","text":"String","embedding":"List<Float>"}},
    {"label": "Chunk", "identity": ["chunk_id"],
     "properties": {"chunk_id":"String","hash":"String","text":"String","order":"Integer","heading_path":"String","page":"Integer","tokens":"Integer","embedding":"List<Float>"}},
    {"label": "Entity", "identity": ["entity_id"],
     "properties": {"entity_id":"String","name":"String","etype":"String","aliases":"List<String>","version":"String","embedding":"List<Float>"}},
    {"label": "Topic", "identity": ["name"], "properties": {"name":"String"}},
    {"label": "Session", "identity": ["session_id"], "properties": {"session_id":"String","started_at":"DateTime"}},
    {"label": "Query", "identity": ["query_id"], "properties": {"query_id":"String","text":"String","turn":"Integer","asked_at":"DateTime"}},
    {"label": "Answer", "identity": ["answer_id"], "properties": {"answer_id":"String","created_at":"DateTime"}}
  ],
  "relationships": [
    {"type":"HAS_CHUNK","start":"Document","end":"Chunk","properties":{}},
    {"type":"MENTIONS","start":"Chunk","end":"Entity","properties":{"confidence":"Float"}},
    {"type":"HAS_TOPIC","start":"Document","end":"Topic","properties":{}},
    {"type":"HAS_TOPIC","start":"Chunk","end":"Topic","properties":{}},
    {"type":"RELATED_TO","start":"Entity","end":"Entity","properties":{"rel_type":"String","confidence":"Float"}},
    {"type":"HAS_QUERY","start":"Session","end":"Query","properties":{}},
    {"type":"FOCUSED_ON","start":"Query","end":"Entity","properties":{"score":"Float"}},
    {"type":"RETRIEVED","start":"Query","end":"Chunk","properties":{"rank":"Integer","score_text":"Float","score_vec":"Float","score_graph":"Float"}},
    {"type":"ANSWERED_AS","start":"Query","end":"Answer","properties":{}},
    {"type":"SUPPORTED_BY","start":"Answer","end":"Chunk","properties":{"rank":"Integer"}}
  ]
}
```

* * *

Neo4j 5 DDL (constraints & indexes)
-----------------------------------

```cypher
// === Uniques / Node Keys ===
CREATE CONSTRAINT doc_id_uniq IF NOT EXISTS
FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_uniq IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT chunk_hash_uniq IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.hash IS UNIQUE;

CREATE CONSTRAINT entity_id_uniq IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE CONSTRAINT topic_name_uniq IF NOT EXISTS
FOR (t:Topic) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT session_id_uniq IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT query_id_uniq IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;

CREATE CONSTRAINT answer_id_uniq IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

// === Existence (hygiene) ===
CREATE CONSTRAINT doc_title_exists IF NOT EXISTS
FOR (d:Document) REQUIRE d.title IS NOT NULL;

CREATE CONSTRAINT chunk_text_exists IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.text IS NOT NULL;

CREATE CONSTRAINT entity_name_exists IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS NOT NULL;

// === Full-text ===
CREATE FULLTEXT INDEX doc_search IF NOT EXISTS
FOR (d:Document) ON EACH [d.title, d.text];

CREATE FULLTEXT INDEX chunk_search IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text];

CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases];

// === Property Indexes ===
CREATE INDEX doc_type_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_type);
CREATE INDEX doc_product_idx IF NOT EXISTS FOR (d:Document) ON (d.product);
CREATE INDEX chunk_order_idx IF NOT EXISTS FOR (c:Chunk) ON (c.order);
CREATE INDEX entity_etype_idx IF NOT EXISTS FOR (e:Entity) ON (e.etype);

// === Optional vector indexes (uncomment and set your embedding dimension) ===
// CREATE INDEX chunk_embedding_vec IF NOT EXISTS
// FOR (c:Chunk) ON (c.embedding)
// OPTIONS {indexProvider:'vector-1.0', indexConfig:{dimension:1536, similarityFunction:'cosine'}};
//
// CREATE INDEX doc_embedding_vec IF NOT EXISTS
// FOR (d:Document) ON (d.embedding)
// OPTIONS {indexProvider:'vector-1.0', indexConfig:{dimension:1536, similarityFunction:'cosine'}};
//
// CREATE INDEX entity_embedding_vec IF NOT EXISTS
// FOR (e:Entity) ON (e.embedding)
// OPTIONS {indexProvider:'vector-1.0', indexConfig:{dimension:768, similarityFunction:'cosine'}};
```

* * *

Ingestion playbook (kept simple & robust)
-----------------------------------------

1.  **Chunking**
    *   Split by headings, keep 300–800 tokens per chunk.
    *   Store `heading_path` (e.g., `"Install > Requirements > NIC Drivers"`), `order`, and optional `page`.
    *   Compute `hash` (e.g., SHA‑256 of `doc_id + order + text`).
2.  **Embeddings**
    *   Embed `Chunk.text`; optionally embed `Document.text` for broader recall.
    *   If you embed Entities, use shorter descriptions (name + one‑line definition).
3.  **Entity Linking**
    *   Build a curated alias map for high‑value entities (features, commands, APIs, alert codes).
    *   Run NER/EL pass; `MERGE` `(:Entity {entity_id})` and `(:Chunk)-[:MENTIONS {confidence}]->(:Entity)`.
    *   Keep `etype` consistent—this drives multi‑turn focus.
4.  **Topics/Taxonomy**
    *   Curate a small set (10s to low 100s). Tag at **Document** first; spill to **Chunk** only for outliers.
5.  **Optional knowledge edges**
    *   A few `RELATED_TO` edges with `rel_type` in a fixed enum can improve routing (e.g., `Feature A requires Feature B`).
6.  **Session (multi‑turn)**
    *   On each user turn, `MERGE` a `(:Query)` inside the active `(:Session)`.
    *   Persist top‑K entities as `:FOCUSED_ON` (score‑weighted), and top‑K chunks as `:RETRIEVED`.

* * *

Retrieval & answer assembly (hallucination‑resistant)
-----------------------------------------------------

**Retriever policy (pseudocode):**

1.  **Initial recall** = `fulltext(chunk_search)` + (optional) `vector` on `Chunk.embedding`.
2.  **Entity focus** = from the question + prior `(:Query)-[:FOCUSED_ON]->(:Entity)` in the same `Session`.
3.  **Graph boost** = bump chunks that `MENTIONS` focused entities/topics.
4.  **Rank** = `0.5*text_score + 0.3*vec_score + 0.2*graph_boost`
5.  **Gate**: if top‑1 score < τ₁ or total coverage of focused entities < τ₂ → return clarification/not‑found.
6.  **Assemble**: cite the top snippets verbatim; no snippet → no claim.

**Example Cypher (simple booster without vectors):**

```cypher
CALL db.index.fulltext.queryNodes('chunk_search', $q) YIELD node AS c, score AS ft
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
WITH c, ft, collect(DISTINCT e.etype) AS etypes, count(e) AS ecnt
WITH c, ft + 0.1*ecnt + CASE WHEN 'CLI' IN etypes THEN 0.2 ELSE 0 END AS rank
RETURN c.chunk_id, c.text, c.heading_path, rank
ORDER BY rank DESC
LIMIT 8;
```

**Multi‑turn bias:**

```cypher
// Entities in focus from the active session
MATCH (s:Session {session_id:$sid})-[:HAS_QUERY]->(q)
WITH apoc.coll.tail(collect(q), $lastK) AS recentQs
UNWIND recentQs AS rq
MATCH (rq)-[f:FOCUSED_ON]->(e:Entity)
WITH collect(DISTINCT e) AS focus

CALL db.index.fulltext.queryNodes('chunk_search', $qText) YIELD node AS c, score AS ft
OPTIONAL MATCH (c)-[:MENTIONS]->(e) WHERE e IN focus
WITH c, ft, count(e) AS focusHits
RETURN c.chunk_id, c.text, ft + 0.3*focusHits AS rank
ORDER BY rank DESC LIMIT 8;
```

**Answer building guardrails**

*   Always show references: `(doc_id, chunk_id, heading_path, uri#anchor)`.
*   If **no** chunk passes thresholds → return “I can’t find an authoritative answer; try one of these related entities …”.
*   Keep a per‑product/per‑version filter when applicable to avoid mixing versions.

* * *

So… is simpler better?
----------------------

For **Doc RAG**: **Yes**—up to the point where you still retain **chunk‑level provenance** and **entity grounding**. The v3 schema above is intentionally compact. It preserves the _full documentation content_ (down to retrievable chunks), adds just enough entity/ topic structure for precision, and includes optional session memory to handle multi‑turn without drifting.
