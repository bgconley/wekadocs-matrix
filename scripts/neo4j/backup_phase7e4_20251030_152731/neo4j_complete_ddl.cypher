// ============================================================================
// Neo4j Complete DDL Backup - Phase 7E-4 Production Ready
// Generated: 2025-10-30 15:27:31
// Database: Neo4j Enterprise 5.15
// Schema Version: v2.1
// ============================================================================
//
// This DDL file provides COMPLETE recovery of Neo4j schema including:
// - 16 uniqueness constraints
// - 46 total indexes (including vector and fulltext)
// - SchemaVersion singleton node
// - All performance and search optimizations
//
// Usage:
//   docker exec weka-neo4j cypher-shell -u neo4j -p $PASSWORD -f /path/to/this/file.cypher
//
// ============================================================================

// ----------------------------------------------------------------------------
// SECTION 1: UNIQUENESS CONSTRAINTS (16 total)
// These MUST be created first as they auto-create backing indexes
// ----------------------------------------------------------------------------

// Core entity constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;
CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE;

// Component constraints
CREATE CONSTRAINT command_id_unique IF NOT EXISTS FOR (c:Command) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT configuration_id_unique IF NOT EXISTS FOR (c:Configuration) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT error_id_unique IF NOT EXISTS FOR (e:Error) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT example_id_unique IF NOT EXISTS FOR (e:Example) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT step_id_unique IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT parameter_id_unique IF NOT EXISTS FOR (p:Parameter) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT component_id_unique IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE;

// Query tracking constraints
CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE;
CREATE CONSTRAINT query_id_unique IF NOT EXISTS FOR (q:Query) REQUIRE q.query_id IS UNIQUE;
CREATE CONSTRAINT answer_id_unique IF NOT EXISTS FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

// SchemaVersion singleton constraint (CRITICAL for health checks)
CREATE CONSTRAINT schema_version_singleton IF NOT EXISTS FOR (sv:SchemaVersion) REQUIRE sv.id IS UNIQUE;

// ----------------------------------------------------------------------------
// SECTION 2: B-TREE INDEXES (27 total)
// Performance indexes for common query patterns
// ----------------------------------------------------------------------------

// Document indexes
CREATE INDEX document_created_at IF NOT EXISTS FOR (d:Document) ON (d.created_at);
CREATE INDEX document_document_type IF NOT EXISTS FOR (d:Document) ON (d.document_type);
CREATE INDEX document_last_modified IF NOT EXISTS FOR (d:Document) ON (d.last_modified);
CREATE INDEX document_source_type IF NOT EXISTS FOR (d:Document) ON (d.source_type);
CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title);

// Section indexes (CRITICAL - health check expects these exact names with _idx suffix)
CREATE INDEX section_document_id_idx IF NOT EXISTS FOR (s:Section) ON (s.document_id);
CREATE INDEX section_level_idx IF NOT EXISTS FOR (s:Section) ON (s.level);
CREATE INDEX section_order_idx IF NOT EXISTS FOR (s:Section) ON (s.order);

// Section indexes (additional, without _idx suffix)
CREATE INDEX section_document_id IF NOT EXISTS FOR (s:Section) ON (s.document_id);
CREATE INDEX section_level IF NOT EXISTS FOR (s:Section) ON (s.level);
CREATE INDEX section_order IF NOT EXISTS FOR (s:Section) ON (s.order);

// Chunk indexes
CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.document_id);
CREATE INDEX chunk_section_id IF NOT EXISTS FOR (c:Chunk) ON (c.section_id);
CREATE INDEX chunk_position IF NOT EXISTS FOR (c:Chunk) ON (c.position);
CREATE INDEX chunk_order IF NOT EXISTS FOR (c:Chunk) ON (c.order);

// Entity and relationship indexes
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_normalized_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name);

// Query tracking indexes
CREATE INDEX session_started_at IF NOT EXISTS FOR (s:Session) ON (s.started_at);
CREATE INDEX session_ended_at IF NOT EXISTS FOR (s:Session) ON (s.ended_at);
CREATE INDEX query_asked_at IF NOT EXISTS FOR (q:Query) ON (q.asked_at);
CREATE INDEX query_session_id IF NOT EXISTS FOR (q:Query) ON (q.session_id);
CREATE INDEX answer_created_at IF NOT EXISTS FOR (a:Answer) ON (a.created_at);
CREATE INDEX answer_query_id IF NOT EXISTS FOR (a:Answer) ON (a.query_id);

// Component indexes
CREATE INDEX component_type IF NOT EXISTS FOR (c:Component) ON (c.type);
CREATE INDEX procedure_category IF NOT EXISTS FOR (p:Procedure) ON (p.category);

// ----------------------------------------------------------------------------
// SECTION 3: VECTOR INDEXES (2 total)
// For semantic similarity search with Jina embeddings (1024-D)
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// SECTION 4: FULLTEXT INDEXES (3 total)
// For text search functionality
// ----------------------------------------------------------------------------

// Individual field fulltext indexes
CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.text];

CREATE FULLTEXT INDEX document_title_fulltext IF NOT EXISTS
FOR (d:Document)
ON EACH [d.title];

// Composite fulltext index for advanced search
CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.text, c.heading];

// ----------------------------------------------------------------------------
// SECTION 5: WAIT FOR ALL INDEXES TO COME ONLINE
// ----------------------------------------------------------------------------

CALL db.awaitIndexes(300);

// ----------------------------------------------------------------------------
// SECTION 6: CREATE SCHEMA VERSION NODE
// This singleton node is REQUIRED for MCP server health checks
// ----------------------------------------------------------------------------

MERGE (sv:SchemaVersion {id: 'singleton'})
ON CREATE SET
  sv.version = 'v2.1',
  sv.created_at = datetime(),
  sv.description = 'Phase 7E-4 schema with 1024-D Jina vectors, chunk model, integrity tracking',
  sv.phase = '7E-4',
  sv.embedding_provider = 'jina-ai',
  sv.embedding_model = 'jina-embeddings-v3',
  sv.embedding_dimensions = 1024,
  sv.vector_distance = 'cosine'
ON MATCH SET
  sv.updated_at = datetime(),
  sv.last_verified = datetime();

// ----------------------------------------------------------------------------
// SECTION 7: VERIFICATION QUERIES (Optional - for manual verification)
// ----------------------------------------------------------------------------

// To verify after running this DDL:
// MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv;
// SHOW CONSTRAINTS;
// SHOW INDEXES;
// CALL db.indexes() YIELD name, type, state WHERE state <> 'ONLINE' RETURN name, type, state;

// ----------------------------------------------------------------------------
// END OF DDL - System ready for Phase 7E-4 operations
// ----------------------------------------------------------------------------
