// === Phase 7E-4 / GraphRAG v2.1: Canonical Neo4j Schema (idempotent) ===
// Safety: create essential metadata constraint and node first
CREATE CONSTRAINT schema_version_singleton IF NOT EXISTS FOR (sv:SchemaVersion) REQUIRE sv.id IS UNIQUE;
MERGE (sv:SchemaVersion {id: 'singleton'})
  ON CREATE SET sv.version = 'v2.1', sv.created_at = datetime(), sv.description = 'Phase 7E-4 canonical schema'
  ON MATCH SET  sv.version = 'v2.1', sv.updated_at = datetime();

//
// IDEMPOTENCY:
//   This script is idempotent (IF NOT EXISTS on all CREATE statements).
//   Safe to re-run multiple times without errors or duplicates.
//
// COMPANION STORES:
//   Neo4j: This file (graph structure + constraints + indexes)
//   Qdrant: See QDRANT_CONFIGURATION section below for vector store setup
//
// DATA STATE AT BACKUP:
//   Neo4j:  0 nodes (schema-only backup after wipe)
//   Qdrant: 0 points (schema-only backup after wipe)
//
// CRITICAL SPECIFICATIONS:
//   Vector dimensions: 1024-D (Jina v3, not 384-D or 768-D)
//   Embedding provider: jina-ai
//   Model: jina-embeddings-v3
//   Similarity: cosine
//   Community Edition: No property existence constraints (enforced in app layer)
//
// ============================================================================

// ============================================================================
// PART 1: CORE NODE CONSTRAINTS (Document + Section + Entities)
// ============================================================================
// Purpose: Unique identifiers for all node types
// Impact: Prevents duplicate nodes, enables fast lookups
// Captured: 15 unique constraints from deployed schema

// Document constraints (2)
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;
// Section constraint (1)
// NOTE: Sections are dual-labeled as :Chunk for v3 compatibility
CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;
// Domain entity constraints (9)
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
// Session/Query/Answer constraints (3)
CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;
CREATE CONSTRAINT query_id_unique IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;
CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;
// ============================================================================
// PART 2: PROPERTY INDEXES (Document, Section, Entities)
// ============================================================================
// Purpose: Fast filtering and sorting in queries
// Impact: Query performance optimization
// Captured: 43 property indexes from deployed schema

// Document indexes (3)
CREATE INDEX document_source_type IF NOT EXISTS
FOR (d:Document) ON (d.source_type);
CREATE INDEX document_version IF NOT EXISTS
FOR (d:Document) ON (d.version);
CREATE INDEX document_last_edited IF NOT EXISTS
FOR (d:Document) ON (d.last_edited);
// Section indexes (4)
CREATE INDEX section_document_id IF NOT EXISTS
FOR (s:Section) ON (s.document_id);
CREATE INDEX section_level IF NOT EXISTS
FOR (s:Section) ON (s.level);
CREATE INDEX section_order IF NOT EXISTS
FOR (s:Section) ON (s.order);
// Chunk indexes (3) - for dual-labeled nodes
CREATE INDEX chunk_document_id IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);
CREATE INDEX chunk_level IF NOT EXISTS
FOR (c:Chunk) ON (c.level);
CREATE INDEX chunk_embedding_version IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_version);
// Domain entity indexes (6)
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
// Session indexes (4)
CREATE INDEX session_started_at IF NOT EXISTS
FOR (s:Session) ON (s.started_at);
CREATE INDEX session_expires_at IF NOT EXISTS
FOR (s:Session) ON (s.expires_at);
CREATE INDEX session_active IF NOT EXISTS
FOR (s:Session) ON (s.active);
CREATE INDEX session_user_id IF NOT EXISTS
FOR (s:Session) ON (s.user_id);
// Query indexes (2)
CREATE INDEX query_turn IF NOT EXISTS
FOR (q:Query) ON (q.turn);
CREATE INDEX query_asked_at IF NOT EXISTS
FOR (q:Query) ON (q.asked_at);
// Answer indexes (2)
CREATE INDEX answer_created_at IF NOT EXISTS
FOR (a:Answer) ON (a.created_at);
CREATE INDEX answer_user_feedback IF NOT EXISTS
FOR (a:Answer) ON (a.user_feedback);

// Optional fulltext indexes used by search (safe, idempotent). Comment out if not needed.
CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE FULLTEXT INDEX document_title_fulltext IF NOT EXISTS FOR (d:Document) ON EACH [d.title];

// Wait for indexes to come online
CALL db.awaitIndexes(300);

// Vector indexes (from backup, idempotent)
CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
FOR (s:Section)
ON s.vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
FOR (c:Chunk)
ON c.vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};
