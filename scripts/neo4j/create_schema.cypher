// Implements Phase 1, Task 1.3 (Database schema initialization)
// See: /docs/spec.md §3 (Data model)
// See: /docs/implementation-plan.md → Task 1.3 DoD & Tests
// Neo4j schema creation - idempotent, can be run multiple times

// ============================================================================
// CONSTRAINTS (unique identifiers)
// ============================================================================

// Document constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;

// Section constraints
CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

// Domain entity constraints
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
// INDEXES (for query performance)
// ============================================================================

// Document indexes
CREATE INDEX document_source_type IF NOT EXISTS
FOR (d:Document) ON (d.source_type);

CREATE INDEX document_version IF NOT EXISTS
FOR (d:Document) ON (d.version);

CREATE INDEX document_last_edited IF NOT EXISTS
FOR (d:Document) ON (d.last_edited);

// Section indexes
CREATE INDEX section_document_id IF NOT EXISTS
FOR (s:Section) ON (s.document_id);

CREATE INDEX section_level IF NOT EXISTS
FOR (s:Section) ON (s.level);

CREATE INDEX section_order IF NOT EXISTS
FOR (s:Section) ON (s.order);

// Domain entity indexes (common properties)
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
// VECTOR INDEXES
// ============================================================================
// NOTE: Vector indexes will be created programmatically via schema.py
// because they require config-driven dimensions and similarity metrics.
// This ensures the embedding configuration is the single source of truth.

// The following indexes will be created by src/shared/schema.py:
// - section_embeddings (on Section.vector_embedding)
// - command_embeddings (on Command.vector_embedding)
// - configuration_embeddings (on Configuration.vector_embedding)
// - procedure_embeddings (on Procedure.vector_embedding)
// - error_embeddings (on Error.vector_embedding)
// - concept_embeddings (on Concept.vector_embedding)

// ============================================================================
// SCHEMA VERSION NODE
// ============================================================================
// Create or update schema version singleton
MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v1',
    sv.updated_at = datetime(),
    sv.description = 'WekaDocs GraphRAG schema v1'
RETURN sv;

// ============================================================================
// VERIFICATION
// ============================================================================
// List all constraints and indexes to verify creation
CALL db.constraints();
CALL db.indexes();
