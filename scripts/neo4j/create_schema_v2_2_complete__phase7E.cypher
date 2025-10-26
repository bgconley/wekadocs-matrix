
// ============================================================================
// WekaDocs GraphRAG Schema v2.2 — COMPLETE Standalone DDL (Phase‑7E Ready)
// ============================================================================
// Generated: 2025-10-29T00:41:14.368931Z
// Target: Neo4j 5+ (Community or Enterprise)
// Apply to: EMPTY databases (no data migration included).
//
// Key changes vs v2.1:
// - Adds FULLTEXT index `chunk_text_index` on (:Chunk).text + .heading for BM25‑style recall
// - Expands property indexes for retrieval stitching (parent_section_id, order)
// - Retains 1024‑D VECTOR indexes (Jina v3) on both :Section and :Chunk
// - Canonical embedding field names (embedding_version, provider, dimensions, timestamp)
// - Idempotent (IF NOT EXISTS) for safe re‑runs
//
// Notes:
// * Neo4j Community does not support property‑existence constraints; required fields
//   are enforced in the application layer.
// * All property names are canonicalized for GraphRAG v2.2 + Phase‑7E.
// * For BM25/keyword retrieval we use Neo4j FULLTEXT; application computes BM25 if needed.
//
// ============================================================================

// ============================================================================
// PART 1: CORE NODE CONSTRAINTS (Document + Section/Chunk + Domain Entities)
// ============================================================================

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;

// Sections (also dual‑labeled as :Chunk; both share id uniqueness)
CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

// Domain entity uniqueness (carry‑over from v2.1)
CREATE CONSTRAINT command_id_unique IF NOT EXISTS
FOR (c:Command) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT configuration_id_unique IF NOT EXISTS
FOR (c:Configuration) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS
FOR (p:Procedure) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT error_id_unique IF NOT EXISTS
FOR (e:Error) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT example_id_unique IF NOT EXISTS
FOR (e:Example) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT step_id_unique IF NOT EXISTS
FOR (s:Step) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT parameter_id_unique IF NOT EXISTS
FOR (p:Parameter) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT component_id_unique IF NOT EXISTS
FOR (c:Component) REQUIRE c.id IS UNIQUE;

// ============================================================================
// PART 2: SESSION/QUERY/ANSWER CONSTRAINTS (Multi‑turn Tracking)
// ============================================================================

CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT query_id_unique IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;

CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

// ============================================================================
// PART 3: PROPERTY INDEXES (Document, Section, Chunk, Domain)
// ============================================================================

// Document indexes
CREATE INDEX document_source_type IF NOT EXISTS
FOR (d:Document) ON (d.source_type);

CREATE INDEX document_version IF NOT EXISTS
FOR (d:Document) ON (d.version);

CREATE INDEX document_last_edited IF NOT EXISTS
FOR (d:Document) ON (d.last_edited);

// Section indexes (primary retrieval metadata)
CREATE INDEX section_document_id IF NOT EXISTS
FOR (s:Section) ON (s.document_id);

CREATE INDEX section_level IF NOT EXISTS
FOR (s:Section) ON (s.level);

CREATE INDEX section_order IF NOT EXISTS
FOR (s:Section) ON (s.order);

// Chunk indexes (v2.2: emphasize retrieval stitching & embedding metadata)
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE INDEX chunk_document_id IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

CREATE INDEX chunk_parent_order IF NOT EXISTS
FOR (c:Chunk) ON (c.parent_section_id, c.order);

CREATE INDEX chunk_embedding_version IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_version);

CREATE INDEX chunk_updated_at IF NOT EXISTS
FOR (c:Chunk) ON (c.updated_at);

// Domain property indexes (unchanged)
CREATE INDEX command_name IF NOT EXISTS
FOR (c:Command) ON (c.name);

CREATE INDEX configuration_name IF NOT EXISTS
FOR (c:Configuration) ON (c.name);

CREATE INDEX procedure_title IF NOT EXISTS
FOR (p:Procedure) ON (p.title);

CREATE INDEX error_code IF NOT EXISTS
FOR (e:Error) ON (e.code);

CREATE INDEX concept_term IF NOT EXISTS
FOR (c:Concept) ON (c.term);

CREATE INDEX component_name IF NOT EXISTS
FOR (c:Component) ON (c.name);

// ============================================================================
// PART 4: FULLTEXT INDEXES (BM25/Keyword Candidate Recall)
// ============================================================================
// Neo4j 5+ syntax. Analyzer can be adjusted based on corpus; 'standard' is robust.
// The application layer may compute BM25 over results; Neo4j returns TF‑IDF style score.

CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text, c.heading]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard' } };

// Optional (keep commented unless you need it):
// CREATE FULLTEXT INDEX document_title_index IF NOT EXISTS
// FOR (d:Document) ON EACH [d.title]
// OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard' } };

// ============================================================================
// PART 5: VECTOR INDEXES — 1024‑D for Jina v3 (Phase‑7E Default)
// ============================================================================
// Both labels point to identical physical nodes when dual‑labeled.
// Similarity: cosine. Dimensions: 1024.

CREATE VECTOR INDEX section_embeddings_v2_2 IF NOT EXISTS
FOR (s:Section)
ON s.vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX chunk_embeddings_v2_2 IF NOT EXISTS
FOR (c:Chunk)
ON c.vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

// ============================================================================
// PART 6: DUAL‑LABELING (Migration‑safe, No‑op on Empty DB)
// ============================================================================
// On fresh empty databases this does nothing; retained for idempotence.
MATCH (s:Section)
WHERE NOT s:Chunk
SET s:Chunk;

// ============================================================================
// PART 7: SCHEMA VERSION MARKER (v2.2 — Phase‑7E Ready)
// ============================================================================
// Tracks schema version and core embedding defaults used by ingestion.

MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v2.2',
    sv.edition = 'community',
    sv.vector_dimensions = 1024,
    sv.embedding_provider = 'jina-ai',
    sv.embedding_model = 'jina-embeddings-v3',
    sv.updated_at = datetime(),
    sv.description = 'Phase‑7E readiness: adds full‑text index for BM25 recall, strengthens Chunk indexes, retains 1024‑D vector indexes.',
    sv.validation_note = 'Property existence constraints enforced in application (Community Edition).',
    sv.migration_path = 'Fresh installation — complete v2.2 schema';

// ============================================================================
// PART 8: (Commented) VERIFICATION QUERIES
// ============================================================================

// -- 1) Vector indexes present and 1024‑D
// SHOW INDEXES YIELD name, type, labelsOrTypes, properties
// WHERE type = 'VECTOR'
// RETURN name, labelsOrTypes, properties
// ORDER BY name;

// -- 2) FULLTEXT index present
// SHOW INDEXES YIELD name, type, labelsOrTypes, properties
// WHERE type = 'FULLTEXT'
// RETURN name, labelsOrTypes, properties
// ORDER BY name;

// -- 3) Dual‑label sanity (should match once data exists)
// MATCH (s:Section) WITH count(s) AS sections
// MATCH (c:Chunk)   WITH sections, count(c) AS chunks
// RETURN sections, chunks;

// -- 4) Schema version
// MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv;

// ============================================================================
// END — GraphRAG v2.2 COMPLETE DDL (Phase‑7E Ready)
// ============================================================================
