# API Contracts & Frozen Schema - Phase 7-9

**Status:** FROZEN as of 2025-10-20
**Scope:** Phase 7a through Phase 9c
**Version:** 1.0

## Overview

This document defines the **frozen schema and API contracts** for Phase 7-9 implementation. No breaking changes to these contracts are permitted without explicit approval and re-baselining.

## Purpose

By freezing the schema and API contracts:
1. **Stability:** Prevents schema drift during multi-week implementation
2. **Testability:** Golden set queries remain valid across all phases
3. **Compatibility:** Ensures Phase 7-9 changes are backward compatible
4. **Provenance:** Guarantees section IDs remain stable for citation

---

## Graph Schema (Neo4j)

### Node Labels

#### Documentation Nodes

**`:Document`**
```cypher
// Represents a source documentation file
{
  id: String (SHA-256, UNIQUE),
  source_uri: String (UNIQUE),
  source_type: String,
  title: String,
  version: String,
  last_edited: DateTime,
  created_at: DateTime,
  updated_at: DateTime
}
```

**`:Section`** (also labeled as `:Chunk` for DocRAG compatibility)
```cypher
// Represents a documentation section with content
{
  id: String (SHA-256, UNIQUE),
  document_id: String,
  title: String,
  text: String,
  level: Integer,
  order: Integer,
  anchor: String,
  vector_embedding: Float[384],  // OpenAI text-embedding-3-small
  created_at: DateTime,
  updated_at: DateTime
}
```

#### Domain Entity Nodes

**`:Command`**
```cypher
{
  id: String (UNIQUE),
  name: String (INDEXED),
  syntax: String,
  description: String,
  vector_embedding: Float[384]
}
```

**`:Configuration`**
```cypher
{
  id: String (UNIQUE),
  name: String (INDEXED),
  type: String,
  default_value: String,
  description: String,
  vector_embedding: Float[384]
}
```

**`:Procedure`**
```cypher
{
  id: String (UNIQUE),
  title: String (INDEXED),
  description: String,
  vector_embedding: Float[384]
}
```

**`:Error`**
```cypher
{
  id: String (UNIQUE),
  code: String (INDEXED),
  message: String,
  description: String,
  vector_embedding: Float[384]
}
```

**`:Concept`**
```cypher
{
  id: String (UNIQUE),
  term: String (INDEXED),
  definition: String,
  vector_embedding: Float[384]
}
```

**`:Example`**
```cypher
{
  id: String (UNIQUE),
  title: String,
  code: String,
  description: String
}
```

**`:Step`**
```cypher
{
  id: String (UNIQUE),
  order: Integer,
  instruction: String
}
```

**`:Parameter`**
```cypher
{
  id: String (UNIQUE),
  name: String,
  type: String,
  description: String
}
```

**`:Component`**
```cypher
{
  id: String (UNIQUE),
  name: String (INDEXED),
  type: String,
  description: String
}
```

---

### Relationship Types (Frozen for Phase 7-9)

#### Phase 7a Whitelist (Documentation Focus)

**`MENTIONS`**
```cypher
(:Section)-[:MENTIONS]->(:Entity)
// Section references an entity (Command, Configuration, Error, etc.)
// No properties required
```

**`HAS_SECTION`**
```cypher
(:Document)-[:HAS_SECTION]->(:Section)
// Document contains section
// No properties required
```

#### Additional Relationships (Present but not used in Phase 7a traversal)

**`CONTAINS_STEP`**
```cypher
(:Procedure)-[:CONTAINS_STEP {order: Integer}]->(:Step)
```

**`HAS_PARAMETER`**
```cypher
(:Command|:Configuration)-[:HAS_PARAMETER {required: Boolean}]->(:Parameter)
```

**`REQUIRES`**
```cypher
(:Entity)-[:REQUIRES]->(:Entity)
```

**`AFFECTS`**
```cypher
(:Configuration)-[:AFFECTS]->(:Component)
```

**`RESOLVES`**
```cypher
(:Procedure)-[:RESOLVES]->(:Error)
```

**`RELATED_TO`**
```cypher
(:Entity)-[:RELATED_TO {strength: Float}]->(:Entity)
```

**`EXECUTES`**
```cypher
(:Step)-[:EXECUTES]->(:Command)
```

---

## Constraints (13 Total)

```cypher
// Document
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT document_source_uri_unique FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;

// Section
CREATE CONSTRAINT section_id_unique FOR (s:Section) REQUIRE s.id IS UNIQUE;

// Domain Entities
CREATE CONSTRAINT command_id_unique FOR (c:Command) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT configuration_id_unique FOR (c:Configuration) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT procedure_id_unique FOR (p:Procedure) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT error_id_unique FOR (e:Error) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT concept_id_unique FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT example_id_unique FOR (e:Example) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT step_id_unique FOR (s:Step) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT parameter_id_unique FOR (p:Parameter) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT component_id_unique FOR (c:Component) REQUIRE c.id IS UNIQUE;
```

---

## Indexes (35 Total)

### Regular Indexes (29)

```cypher
// Document indexes (3)
CREATE INDEX document_source_type FOR (d:Document) ON (d.source_type);
CREATE INDEX document_version FOR (d:Document) ON (d.version);
CREATE INDEX document_last_edited FOR (d:Document) ON (d.last_edited);

// Section indexes (3)
CREATE INDEX section_document_id FOR (s:Section) ON (s.document_id);
CREATE INDEX section_level FOR (s:Section) ON (s.level);
CREATE INDEX section_order FOR (s:Section) ON (s.order);

// Domain entity indexes (6)
CREATE INDEX command_name FOR (c:Command) ON (c.name);
CREATE INDEX configuration_name FOR (c:Configuration) ON (c.name);
CREATE INDEX procedure_title FOR (p:Procedure) ON (p.title);
CREATE INDEX error_code FOR (e:Error) ON (e.code);
CREATE INDEX concept_term FOR (c:Concept) ON (c.term);
CREATE INDEX component_name FOR (c:Component) ON (c.name);
```

### Vector Indexes (6)

All vector indexes use:
- **Dimensions:** 384 (OpenAI text-embedding-3-small)
- **Similarity:** cosine
- **Property:** `vector_embedding`

```cypher
CREATE VECTOR INDEX section_embeddings FOR (n:Section) ON n.vector_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX command_embeddings FOR (n:Command) ON n.vector_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX configuration_embeddings FOR (n:Configuration) ON n.vector_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX procedure_embeddings FOR (n:Procedure) ON n.vector_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX error_embeddings FOR (n:Error) ON n.vector_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX concept_embeddings FOR (n:Concept) ON n.vector_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
```

---

## Vector Store Schema (Qdrant)

### Collection: `weka_docs`

**Dimensions:** 384
**Distance:** Cosine
**Indexed Vectors:** ~560 (as of baseline)

**Vector Payload:**
```json
{
  "section_id": "string (SHA-256)",
  "document_id": "string (SHA-256)",
  "title": "string",
  "text": "string (full content)",
  "level": "integer",
  "order": "integer",
  "anchor": "string",
  "metadata": {
    "source_uri": "string",
    "version": "string",
    "created_at": "timestamp"
  }
}
```

---

## API Contracts

### MCP Tool: `search_documentation` (Extended in Phase 7a)

**Request Schema:**
```json
{
  "query": "string (required)",
  "top_k": "integer (default: 8, max: 20)",
  "verbosity": "string (default: 'snippet', enum: ['snippet', 'full', 'graph'])",
  "providers": {
    "embedding": "string (optional, enum: ['jina', 'openai', 'hf', 'auto'])",
    "reranker": "string (optional, enum: ['jina', 'cohere', 'none'])"
  },
  "limits": {
    "graph": {
      "max_depth_soft": "integer (default: 3, max: 5)",
      "max_depth_hard": "integer (default: 5, max: 5)",
      "max_nodes_hard": "integer (default: 200, max: 200)",
      "rel_whitelist": "string[] (default: ['MENTIONS', 'HAS_SECTION'])"
    }
  }
}
```

**Response Schema (Snippet Mode):**
```json
{
  "answer": "string (optional, LLM-generated)",
  "evidence": [
    {
      "section_id": "string (SHA-256)",
      "title": "string",
      "snippet": "string (max 200 chars)",
      "metadata": {
        "doc_uri": "string",
        "anchor": "string",
        "score": "float (0.0-1.0)"
      }
    }
  ],
  "diagnostics": {
    "retrieval": {
      "provider": "string",
      "reranker": "string",
      "lat_ms": "integer"
    }
  }
}
```

**Response Schema (Full Mode):**
```json
{
  "answer": "string (optional)",
  "evidence": [
    {
      "section_id": "string",
      "title": "string",
      "snippet": "string (max 200 chars)",
      "full_text": "string (max 16KB)",  // NEW in Phase 7a
      "metadata": {
        "doc_uri": "string",
        "anchor": "string",
        "score": "float"
      }
    }
  ],
  "diagnostics": {
    "retrieval": {
      "provider": "string",
      "reranker": "string",
      "lat_ms": "integer"
    }
  }
}
```

**Response Schema (Graph Mode):**
```json
{
  "answer": "string (optional)",
  "evidence": [ /* same as full mode */ ],
  "graph": {  // NEW in Phase 7a
    "nodes": [
      {
        "id": "string",
        "label": "string (Chunk|Entity|Topic|Document)",
        "title": "string",
        "score": "float (0.0-1.0)"
      }
    ],
    "edges": [
      {
        "src": "string (node id)",
        "dst": "string (node id)",
        "type": "string (MENTIONS|HAS_SECTION)",
        "path": ["string (node ids in path)"],
        "depth": "integer (1-5)"
      }
    ],
    "budget": {
      "expanded": "integer (nodes expanded)",
      "skipped": "integer (nodes pruned)",
      "depth_reached": "integer (max depth reached)"
    }
  },
  "diagnostics": {
    "retrieval": { /* ... */ },
    "graph": {  // NEW in Phase 7a
      "expanded": "integer",
      "pruned": "integer",
      "reason": "string (delta<τ | budget_exhausted | depth_limit)"
    },
    "plan": {  // NEW in Phase 7a
      "explain_ok": "boolean",
      "timeout_ms": "integer"
    }
  }
}
```

---

### MCP Tool: `traverse_relationships` (NEW in Phase 7b)

**Request Schema:**
```json
{
  "seed_ids": ["string (node IDs)"],
  "direction": "string (enum: ['out', 'in', 'both'], default: 'out')",
  "max_depth": "integer (default: 3, max: 5)",
  "max_nodes": "integer (default: 150, max: 200)",
  "rel_whitelist": ["string (default: ['MENTIONS', 'HAS_SECTION'])"],
  "label_whitelist": ["string (default: ['Chunk', 'Entity', 'Topic', 'Document'])"],
  "frontier": {
    "top_k": "integer (default: 20)",
    "delta_threshold": "float (default: 0.05)",
    "novelty_penalty": "float (default: 0.2)"
  },
  "filters": {
    "topic": ["string (optional)"],
    "entity": ["string (optional)"]
  }
}
```

**Response Schema:**
```json
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "title": "string",
      "score": "float (0.0-1.0)"
    }
  ],
  "edges": [
    {
      "src": "string",
      "dst": "string",
      "type": "string",
      "depth": "integer"
    }
  ],
  "explain": {
    "scoring": "string (formula description)",
    "caps": {
      "max_depth": "integer",
      "max_nodes": "integer"
    }
  }
}
```

---

## Performance Contracts (SLOs)

### Latency Targets

| Mode | P50 | P95 | P99 |
|------|-----|-----|-----|
| `snippet` | <150ms | <200ms | <300ms |
| `full` | <250ms | <350ms | <500ms |
| `graph` | <350ms | <450ms | <600ms |

### Safety Limits (Hard Caps)

| Parameter | Limit | Enforcement |
|-----------|-------|-------------|
| Max depth (traversal) | 5 hops | Server-enforced |
| Max nodes (traversal) | 200 nodes | Server-enforced |
| Max estimated rows (Cypher) | 10,000 rows | EXPLAIN guard |
| Max db hits (Cypher) | 100,000 hits | EXPLAIN guard |
| Connection timeout | 1500ms | Driver-level |
| Query timeout | 2000ms | Session-level |

### Relationship Whitelist (Phase 7a)

**Allowed:** `MENTIONS`, `HAS_SECTION`
**Rejected:** All others (enforced server-side)

### Label Whitelist (Phase 7a)

**Allowed:** `Section`, `Chunk`, `Entity`, `Document`, `Topic`
**Rejected:** Unlabeled scans, other labels

---

## MCP Streamable HTTP (Addendum)

- Streamable HTTP MCP is mounted at `/_mcp` and exposes the same tool surface as STDIO.
- Migration: point MCP clients at `http://<host>:8000/_mcp`. JSON-only clients should enable `MCP_HTTP_STREAMABLE_JSON_RESPONSE=true` or send `Accept: application/json`.
- Primary tools are `kb.*` and `graph.*`; legacy `/mcp/*` REST endpoints remain behind `MCP_HTTP_LEGACY_REST_ENABLED`.
- Legacy `/mcp/*` responses include `Deprecation` + `Warning` headers to prompt migration.
- Tool results return a short text summary plus structured content; large text is accessed via scratch resources.
- `kb.search` and `kb.retrieve_evidence` include optional `diagnostic_id` pointers (no debug payload bloat).
- Diagnostics summaries are available via `wekadocs://diagnostics/{date}/{diagnostic_id}` when `MCP_DIAGNOSTICS_RESOURCES_ENABLED=true` (Markdown only). CLI fallback: `python scripts/retrieval_diagnostics/show.py --id <diagnostic_id>`.
- `search_documentation` remains disabled by default; enable only behind the legacy flag.

Example migration snippet:
```bash
export MCP_BASE_URL=http://localhost:8000/_mcp
export MCP_HTTP_STREAMABLE_JSON_RESPONSE=true
```

---

## Change Management

### Frozen Elements (No Changes Allowed)

1. ✋ **Node property names** in `:Section`, `:Document`
2. ✋ **Section ID format** (SHA-256 hex strings)
3. ✋ **Vector dimensions** (384 for OpenAI baseline)
4. ✋ **Relationship types** `MENTIONS` and `HAS_SECTION`
5. ✋ **Constraint names** (all 13 constraints)
6. ✋ **Vector index names** (all 6 indexes)
7. ✋ **MCP response schema** for existing fields

### Additive Changes (Allowed)

1. ✅ **New node labels** (e.g., `:Topic`, `:Session`, `:Query`, `:Answer` for DocRAG)
2. ✅ **New relationship types** (e.g., `HAS_TOPIC`, `RETRIEVED`, `SUPPORTED_BY`)
3. ✅ **New optional fields** in MCP responses (must not break existing parsers)
4. ✅ **New indexes** on existing properties
5. ✅ **New vector indexes** for new node labels

### Breaking Changes (Require Re-Baselining)

1. 🚫 Changing `:Section` property names
2. 🚫 Changing section ID generation algorithm
3. 🚫 Changing vector dimensions
4. 🚫 Removing or renaming `MENTIONS` or `HAS_SECTION`
5. 🚫 Changing MCP response schema for existing modes

**Process for Breaking Changes:**
1. Written proposal with justification
2. Technical lead approval
3. Re-run golden set baseline (all 20 queries)
4. Update version number in this document
5. Update all dependent documentation

---

## Version History

| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 1.0 | 2025-10-20 | Initial freeze for Phase 7-9 | System |

---

## References

- **Schema Implementation:** `/src/shared/schema.py`
- **Cypher Script:** `/scripts/neo4j/create_schema.cypher`
- **Golden Set:** `/docs/golden-set-queries.md`
- **Integration Plan:** `/docs/phase-7-integration-plan.md`
- **Task List:** `/docs/phase7-target-phase-tasklist.md`
