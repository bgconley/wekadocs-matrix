
Awesome — here’s a concrete, copy‑pasteable spec for **Neo4j** (nodes/edges, schema, and idempotent upserts) and the **Qdrant** upsert shape for the _combined_ chunks and NEXT_CHUNK edges.



> **Modeling choice (clear separation):**

> Keep your parser’s micro‑sections as :Section (provenance & rendering), and create a new label **:Chunk** for the _combined_ (and, when necessary, split) units that you will **embed** and **retrieve**. This avoids accidentally indexing the tiny sections and keeps retrieval logic clean.

---

## **1) Neo4j data model**



### **Labels**

- **:Document**

    - document_id : STRING (unique)

    - Other metadata: title, source_url, path, updated_at…


- **:Section** _(original parser output; not embedded)_

    - id : STRING (unique, deterministic)

    - document_id : STRING (for quick filtering)

    - heading : STRING

    - level : INTEGER (2 for H2, 3 for H3, etc.)

    - text : STRING

    - token_count : INTEGER

    - is_original : BOOLEAN = true


- **:Chunk** _(combined or split units;_ **_this is what you embed_**_)_

    - id : STRING (unique, deterministic; e.g., sha256(document_id + join(original_section_ids))[:24])

    - document_id : STRING

    - parent_section_id : STRING  _(H2 anchor or “logical parent” id)_

    - order : INTEGER _(0‑based within the parent)_

    - total_chunks : INTEGER

    - is_combined : BOOLEAN _(true if produced by combiner)_

    - is_split : BOOLEAN _(true if a single logical unit was split due to hard cap)_

    - heading : STRING _(combined heading or roll‑up title)_

    - text : STRING _(full chunk text; optional if you prefer to store only in Qdrant)_

    - token_count : INTEGER

    - embedding_provider : STRING = "jina-ai"

    - embedding_version : STRING = "jina-embeddings-v3"

    - original_section_ids : ARRAY<STRING> _(provenance)_

    - boundaries_json : STRING _(serialize any offsets / boundary metadata as JSON string)_

    - updated_at : INTEGER _(Unix ms)_





### **Relationships**

- (:Document)-[:HAS_SECTION]->(:Section) _(existing)_

- (:Document)-[:HAS_CHUNK]->(:Chunk) _(new)_

- (:Section)-[:PART_OF]->(:Chunk) _(provenance; many :Section → one :Chunk)_

- (:Chunk)-[:NEXT_CHUNK {parent_section_id}]->(:Chunk) _(ordered adjacency within the same parent)_




> You _could_ also keep (:Section)-[:NEXT_SECTION]->(:Section) if you already have it; the :NEXT_CHUNK is the retrieval‑level adjacency.

---

## **2) Neo4j schema (constraints & indexes)**

```
// Documents
CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.document_id IS UNIQUE;

// Original Sections
CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

CREATE INDEX section_doc_idx IF NOT EXISTS
FOR (s:Section) ON (s.document_id);

// Combined/Split Chunks
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE INDEX chunk_doc_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

CREATE INDEX chunk_parent_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.parent_section_id, c.order);
```

> Neo4j properties cannot store nested objects; use boundaries_json : STRING for any structured boundary metadata. original_section_ids can be an array.

---

## **3) Idempotent upsert Cypher (Chunks + edges)**



### **Parameters you pass from your combiner**

```
{
  "document_id": "doc_123",
  "chunks": [
    {
      "id": "a1b2c3d4e5f6a7b8c9d0aa11",          // 24-char sha256 prefix
      "document_id": "doc_123",
      "parent_section_id": "sec_h2_42",
      "order": 0,
      "total_chunks": 3,
      "is_combined": true,
      "is_split": false,
      "heading": "Networking – IP Configuration",
      "text": "…full chunk text…",
      "token_count": 1134,
      "embedding_provider": "jina-ai",
      "embedding_version": "jina-embeddings-v3",
      "original_section_ids": ["sec_h3_4201", "sec_h3_4202", "sec_h3_4203"],
      "boundaries_json": "{\"first_h3\":\"IP Address\",\"last_h3\":\"MTU\"}",
      "updated_at": 1730040000000
    }
    // …more chunks in order…
  ]
}
```

### **Upsert (create/update) the** 

### **:Chunk**

###  **nodes and link to Document & Sections**

```
// 1) Ensure Document exists
MERGE (d:Document {document_id: $document_id});

// 2) Upsert Chunk nodes with properties and HAS_CHUNK relations
UNWIND $chunks AS row
MERGE (c:Chunk {id: row.id})
  ON CREATE SET
    c.document_id = row.document_id,
    c.parent_section_id = row.parent_section_id,
    c.order = row.order,
    c.total_chunks = row.total_chunks,
    c.is_combined = row.is_combined,
    c.is_split = row.is_split,
    c.heading = row.heading,
    c.text = row.text,
    c.token_count = row.token_count,
    c.embedding_provider = row.embedding_provider,
    c.embedding_version = row.embedding_version,
    c.original_section_ids = row.original_section_ids,
    c.boundaries_json = row.boundaries_json,
    c.updated_at = row.updated_at
  ON MATCH SET
    c.document_id = row.document_id,
    c.parent_section_id = row.parent_section_id,
    c.order = row.order,
    c.total_chunks = row.total_chunks,
    c.is_combined = row.is_combined,
    c.is_split = row.is_split,
    c.heading = row.heading,
    c.text = row.text,
    c.token_count = row.token_count,
    c.embedding_provider = row.embedding_provider,
    c.embedding_version = row.embedding_version,
    c.original_section_ids = row.original_section_ids,
    c.boundaries_json = row.boundaries_json,
    c.updated_at = row.updated_at
MERGE (d)-[:HAS_CHUNK]->(c);

// 3) Provenance: connect original Sections to their Chunk
UNWIND $chunks AS row
UNWIND row.original_section_ids AS sid
MATCH (s:Section {id: sid})
MATCH (c:Chunk {id: row.id})
MERGE (s)-[:PART_OF]->(c);

// 4) NEXT_CHUNK adjacency within each parent_section_id
//    Build ordered pairs (i -> i+1) per parent_section_id
WITH $chunks AS chs
UNWIND chs AS row
WITH row.parent_section_id AS pid, row.order AS idx, row.id AS cid
ORDER BY pid, idx
WITH pid, collect({idx: idx, cid: cid}) AS ordered
UNWIND range(0, size(ordered)-2) AS i
WITH pid, ordered[i] AS a, ordered[i+1] AS b
MATCH (c1:Chunk {id: a.cid})
MATCH (c2:Chunk {id: b.cid})
MERGE (c1)-[:NEXT_CHUNK {parent_section_id: pid}]->(c2);
```

### **(Optional) Garbage‑collect stale chunks for this document**



If you’re doing **replace‑by‑set** (recommended for idempotency):

```
// $valid_chunk_ids is the set of chunk_ids you just (re)upserted for this doc
MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
WHERE NOT c.id IN $valid_chunk_ids
DETACH DELETE c;
```

_(This removes old chunks and their_ _NEXT_CHUNK__/__PART_OF_ _edges for the doc.)_

---

## **4) Qdrant collection & upsert shape**



> **Use named vectors** so you can add more modalities (e.g., title or code) later without schema churn.



### **Collection creation (one‑time)**

```
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

EMBED_DIM = YOUR_EMBEDDING_DIM  # set from your embedding service
COLLECTION = "chunks"

if COLLECTION not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "content": qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        },
        optimizers_config=qm.OptimizersConfigDiff(memmap_threshold=20000),
        # Optional: quantization/compression can be added later
    )

# Payload indexes (fast filters & group-by)
client.create_payload_index(COLLECTION, field_name="document_id", field_schema=qm.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION, field_name="parent_section_id", field_schema=qm.PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION, field_name="order", field_schema=qm.PayloadSchemaType.INTEGER)
```

### **Point (vector) upsert shape**



> Use a **stable UUID** for the Qdrant point id; keep id in payload as well.

```
import uuid, hashlib

def stable_uuid(id: str) -> str:
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    return str(uuid.uuid5(ns, id))  # stable across runs

point = {
  "id": stable_uuid(chunk["id"]),          # UUID string
  "vector": {
    "content": embedding_vector                  # list[float], len == EMBED_DIM
  },
  "payload": {
    # Identity & grouping
    "id": chunk["id"],
    "document_id": chunk["document_id"],
    "parent_section_id": chunk["parent_section_id"],
    "order": chunk["order"],
    "total_chunks": chunk["total_chunks"],

    # Flags
    "is_combined": chunk["is_combined"],
    "is_split": chunk["is_split"],

    # Text & metadata
    "heading": chunk["heading"],
    "text": chunk["text"],                       # store full text here for fast read
    "token_count": chunk["token_count"],
    "embedding_provider": "jina-ai",
    "embedding_version": "jina-embeddings-v3",
    "original_section_ids": chunk["original_section_ids"],  # array
    "boundaries": chunk.get("boundaries_json"),             # JSON string or dict

    # Timestamps
    "updated_at": chunk["updated_at"],
  }
}
```

### **Batch upsert**

```
from qdrant_client.conversions.common_types import Record

points: list[Record] = []
for chunk, embedding_vector in batch:  # embedding_vector computed from chunk["text"]
    points.append({
        "id": stable_uuid(chunk["id"]),
        "vector": {"content": embedding_vector},
        "payload": {/* as above */}
    })

client.upsert(collection_name=COLLECTION, wait=True, points=points)
```

### **(Optional) Remove stale points for a doc**

```
client.delete(
    collection_name=COLLECTION,
    points_selector=qm.FilterSelector(
        filter=qm.Filter(
            must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
        )
    )
)
# …then re-upsert the new set
```

> Alternative “replace‑by‑set”: fetch existing chunk_ids for the doc and delete only the ones not in your new set.

---

## **5) End‑to‑end ingest (Neo4j + Qdrant) — skeleton**

```
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

def upsert_chunks(document_id: str, chunks: list[dict], embeddings: dict[str, list[float]]):
    # 1) Neo4j upsert
    neo = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with neo.session() as s:
        s.run("""
            MERGE (d:Document {document_id: $document_id})
        """, document_id=document_id)

        s.run("""
            UNWIND $chunks AS row
            MERGE (c:Chunk {id: row.id})
              ON CREATE SET
                c.document_id = row.document_id,
                c.parent_section_id = row.parent_section_id,
                c.order = row.order,
                c.total_chunks = row.total_chunks,
                c.is_combined = row.is_combined,
                c.is_split = row.is_split,
                c.heading = row.heading,
                c.text = row.text,
                c.token_count = row.token_count,
                c.embedding_provider = row.embedding_provider,
                c.embedding_version = row.embedding_version,
                c.original_section_ids = row.original_section_ids,
                c.boundaries_json = row.boundaries_json,
                c.updated_at = row.updated_at
              ON MATCH SET
                c.document_id = row.document_id,
                c.parent_section_id = row.parent_section_id,
                c.order = row.order,
                c.total_chunks = row.total_chunks,
                c.is_combined = row.is_combined,
                c.is_split = row.is_split,
                c.heading = row.heading,
                c.text = row.text,
                c.token_count = row.token_count,
                c.embedding_provider = row.embedding_provider,
                c.embedding_version = row.embedding_version,
                c.original_section_ids = row.original_section_ids,
                c.boundaries_json = row.boundaries_json,
                c.updated_at = row.updated_at
            WITH row
            MATCH (d:Document {document_id: row.document_id})
            MATCH (c:Chunk {id: row.id})
            MERGE (d)-[:HAS_CHUNK]->(c)
        """, chunks=chunks)

        s.run("""
            UNWIND $chunks AS row
            UNWIND row.original_section_ids AS sid
            MATCH (s:Section {id: sid})
            MATCH (c:Chunk {id: row.id})
            MERGE (s)-[:PART_OF]->(c)
        """, chunks=chunks)

        s.run("""
            WITH $chunks AS chs
            UNWIND chs AS row
            WITH row.parent_section_id AS pid, row.order AS idx, row.id AS cid
            ORDER BY pid, idx
            WITH pid, collect({idx: idx, cid: cid}) AS ordered
            UNWIND range(0, size(ordered)-2) AS i
            WITH pid, ordered[i] AS a, ordered[i+1] AS b
            MATCH (c1:Chunk {id: a.cid})
            MATCH (c2:Chunk {id: b.cid})
            MERGE (c1)-[:NEXT_CHUNK {parent_section_id: pid}]->(c2)
        """, chunks=chunks)

    # 2) Qdrant upsert
    qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    points = []
    for ch in chunks:
        vec = embeddings[ch["id"]]  # ensure computed with same text
        points.append({
            "id": stable_uuid(ch["id"]),
            "vector": {"content": vec},
            "payload": {
                "id": ch["id"],
                "document_id": ch["document_id"],
                "parent_section_id": ch["parent_section_id"],
                "order": ch["order"],
                "total_chunks": ch["total_chunks"],
                "is_combined": ch["is_combined"],
                "is_split": ch["is_split"],
                "heading": ch["heading"],
                "text": ch["text"],
                "token_count": ch["token_count"],
                "embedding_provider": ch["embedding_provider"],
                "embedding_version": ch["embedding_version"],
                "original_section_ids": ch["original_section_ids"],
                "boundaries": ch.get("boundaries_json"),
                "updated_at": ch["updated_at"],
            }
        })
    qdr.upsert(collection_name="chunks", points=points, wait=True)
```

---

## **6) Quick retrieval note (so** 

## **NEXT_CHUNK**

##  **gets used)**



At query time:

1. Vector search top‑k on chunks with an optional filter on document_id/parent_section_id.

2. **Group by** **parent_section_id** and consider pulling **±1 neighbor by** **NEXT_CHUNK** for any hit that looks borderline (close scores, long query).

3. Stitch the selected chunks (by order) to form the final context.




Pseudocode for neighbor expansion:

```
# after initial top-k
for hit in hits:
    pid = hit.payload["parent_section_id"]
    idx = hit.payload["order"]
    # pull neighbors (idx-1, idx+1) with the same pid
    # You can do this via a second filtered vector search or a payload filter query.
```

---

## **What to configure**

- TARGET_MIN, TARGET_MAX, and ABSOLUTE_MAX in the combiner (suggested: 800 / 1500 / 7900).

- Whether to store full text in both Neo4j and Qdrant (I recommend **yes** for Qdrant; optional for Neo4j if you want to cut graph storage).

- Whether to GC stale chunks per‑doc on re‑ingest (recommended).


---

If you’d like, I can turn this into a small **migration script** that:

1. creates the Neo4j indexes/constraints,

2. creates the Qdrant collection/indexes, and

3. re‑ingests your 5 test docs using the combiner so you can compare metrics A/B.
