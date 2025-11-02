// ============================================================================
// WekaDocs GraphRAG Schema v2.1 - Complete Standalone DDL
// ============================================================================
// Community Edition Compatible
// Phase 7C: ENV-selectable providers, 1024-D vectors, multi-turn tracking
//
// PURPOSE:
// This is the COMPLETE schema for fresh installations starting with v2.1.
// It merges the base v1 schema + v2.1 enhancements into a single file.
//
// For fresh installations: Run this file only (not v1 + v2.1 separately)
// For migrations from v1: Run create_schema_v2_1.cypher (diff file)
//
// CRITICAL SPECIFICATIONS:
// - Vector dimensions: 1024-D (Jina v3)
// - Embedding provider: jina-ai (default)
// - Community Edition: No property existence constraints
// - Application-layer validation: Required embedding fields enforced in code
//
// DESIGN PRINCIPLES:
// - Idempotent: Safe to re-run multiple times
// - Additive: No breaking changes to existing queries
// - Provenance-first: Schema supports full citation chains
// - Multi-turn aware: Session/Query/Answer tracking built-in
//
// COMPATIBILITY:
// - Section queries work unchanged
// - Dual-labeled as :Chunk for v3 tool compatibility
// - All 12 domain entity types preserved
// - Backward compatible with v2.0 queries
//
// ============================================================================

// ============================================================================
// PART 1: CORE NODE CONSTRAINTS (Document + Section + Entities)
// ============================================================================
// Purpose: Unique identifiers for all node types
// Impact: Prevents duplicate nodes, enables fast lookups
// Idempotent: IF NOT EXISTS

// Document constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;

// Section constraints
// NOTE: Sections are dual-labeled as :Chunk for v3 compatibility
// Both labels share the same uniqueness constraint
CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

// Domain entity constraints (12 types preserved from v2.0)
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
// PART 2: SESSION/QUERY/ANSWER CONSTRAINTS (NEW in v2.1)
// ============================================================================
// Purpose: Multi-turn conversation tracking
// Impact: Enables conversational documentation assistance
// Phase: 7C.8 (session tracking implementation)
// Idempotent: IF NOT EXISTS

CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT query_id_unique IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;

CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

// ============================================================================
// PROPERTY EXISTENCE CONSTRAINTS (Community Edition Limitation)
// ============================================================================
// COMMUNITY EDITION NOTE:
// Property existence constraints (REQUIRE property IS NOT NULL) are not
// supported in Neo4j Community Edition. They require Enterprise Edition.
//
// VALIDATION STRATEGY:
// All required field validation is enforced in the application layer:
// - Section embedding fields: Validated in ingestion pipeline
// - Query.text / Answer.text: Validated before node creation
// - See: src/ingestion/build_graph.py for validation logic
//
// REQUIRED FIELDS (enforced in application code):
// Section nodes:
//   - vector_embedding (List<Float>) - CRITICAL for hybrid search
//   - embedding_version (String) - Provenance tracking
//   - embedding_provider (String) - Provider identification
//   - embedding_dimensions (Integer) - Dimension validation
// --------------------------------------------------------------------------
// CANONICAL PROPERTY NAMES (Authoritative — used across Neo4j and Qdrant)
// --------------------------------------------------------------------------
// Section/Chunk nodes (dual-labeled :Section :Chunk):
//   - id                   (String, UNIQUE key)
//   - document_id          (String, FK to :Document.id)
//   - level                (Integer, heading depth)
//   - order                (Integer, position within parent)
//   - parent_section_id    (String, optional, logical parent anchor)
//   - heading              (String, optional)
//   - text                 (String, chunk text)
//   - is_combined          (Boolean)
//   - is_split             (Boolean)
//   - original_section_ids (List<String>)
//   - boundaries_json      (String, JSON-serialized bounds)
//   - token_count          (Integer)
//   - updated_at           (DateTime)
//   - vector_embedding     (List<Float>)  // 1024-D for jina-embeddings-v3
//   - embedding_version    (String)       // e.g., 'jina-embeddings-v3'
//   - embedding_provider   (String)       // e.g., 'jina-ai'
//   - embedding_dimensions (Integer)      // e.g., 1024
//   - embedding_timestamp  (DateTime)
//
// Document nodes:
//   - id                   (String, UNIQUE key)
//   - source_type, version, last_edited, title, source_url, path (optional)
//
// NOTE: Property existence constraints for Community Edition are enforced in the
//       application layer; this list is provided so *all* upstream docs/tools
//       adopt the exact same names.
// --------------------------------------------------------------------------
//   - embedding_timestamp (DateTime) - Freshness tracking
//
// Query nodes:
//   - text (String) - User's query text
//
// Answer nodes:
//   - text (String) - Generated answer text
//
// ============================================================================

// ============================================================================
// PART 3: PROPERTY INDEXES (Document, Section, Entities)
// ============================================================================
// Purpose: Fast filtering and sorting in queries
// Impact: Query performance optimization
// Idempotent: IF NOT EXISTS

// Document indexes
CREATE INDEX document_source_type IF NOT EXISTS
FOR (d:Document) ON (d.source_type);

CREATE INDEX document_version IF NOT EXISTS
FOR (d:Document) ON (d.version);

CREATE INDEX document_last_edited IF NOT EXISTS
FOR (d:Document) ON (d.last_edited);

// Section indexes (primary retrieval path)
CREATE INDEX section_document_id IF NOT EXISTS
FOR (s:Section) ON (s.document_id);

CREATE INDEX section_level IF NOT EXISTS
FOR (s:Section) ON (s.level);

CREATE INDEX section_order IF NOT EXISTS
FOR (s:Section) ON (s.order);

// Domain entity indexes (for entity-specific queries)
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
// PART 4: SESSION/QUERY/ANSWER PROPERTY INDEXES (NEW in v2.1)
// ============================================================================
// Purpose: Fast lookups for multi-turn tracking
// Impact: Efficient session history queries
// Idempotent: IF NOT EXISTS

// Session indexes
CREATE INDEX session_started_at IF NOT EXISTS
FOR (s:Session) ON (s.started_at);

CREATE INDEX session_expires_at IF NOT EXISTS
FOR (s:Session) ON (s.expires_at);

CREATE INDEX session_active IF NOT EXISTS
FOR (s:Session) ON (s.active);

CREATE INDEX session_user_id IF NOT EXISTS
FOR (s:Session) ON (s.user_id);

// Query indexes
CREATE INDEX query_turn IF NOT EXISTS
FOR (q:Query) ON (q.turn);

CREATE INDEX query_asked_at IF NOT EXISTS
FOR (q:Query) ON (q.asked_at);

// Answer indexes
CREATE INDEX answer_created_at IF NOT EXISTS
FOR (a:Answer) ON (a.created_at);

CREATE INDEX answer_user_feedback IF NOT EXISTS
FOR (a:Answer) ON (a.user_feedback);

// ============================================================================
// PART 5: CHUNK PROPERTY INDEXES (NEW in v2.1 - Dual-label Support)
// ============================================================================
// Purpose: v3 tool compatibility - mirrors Section indexes for :Chunk label
// Impact: Queries using :Chunk label perform equally well
// Note: Same physical nodes as Section, just accessible via :Chunk label
// Idempotent: IF NOT EXISTS

CREATE INDEX chunk_document_id IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

CREATE INDEX chunk_level IF NOT EXISTS
FOR (c:Chunk) ON (c.level);

CREATE INDEX chunk_embedding_version IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_version);

// ============================================================================
// PART 6: VECTOR INDEXES - 1024-D (NEW in v2.1)
// ============================================================================
// Purpose: Semantic search via vector similarity
// CRITICAL SPECIFICATION: 1024 dimensions (Jina v3, not 384-D or 768-D)
// Provider: jina-ai (default), bge-m3 (fallback)
// Model: jina-embeddings-v3
// Similarity: cosine
//
// ARCHITECTURE NOTE:
// In v1, vector indexes were created programmatically via src/shared/schema.py
// to allow dimension flexibility. In v2.1, we pin to 1024-D for:
// - Consistency (all installations use same dimensions)
// - Quality (Jina v3 @ 1024-D validated in benchmarks)
// - Simplicity (single source of truth: this file)
//
// DUAL INDEXES:
// We create two vector indexes on the same data (dual-labeled nodes):
// 1. section_embeddings_v2 - Primary retrieval path (Section label)
// 2. chunk_embeddings_v2 - v3 compatibility path (Chunk label)
//
// Both indexes point to the same vector_embedding property on the same nodes.
// This enables backward compatibility and v3 tool support without data duplication.
//
// Idempotent: IF NOT EXISTS

// Section vector index (primary retrieval path)
CREATE VECTOR INDEX section_embeddings_v2 IF NOT EXISTS
FOR (s:Section)
ON s.vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

// Chunk vector index (v3 compatibility - same data, dual-labeled)
CREATE VECTOR INDEX chunk_embeddings_v2 IF NOT EXISTS
FOR (c:Chunk)
ON c.vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

// ============================================================================
// PART 7: DUAL-LABELING EXISTING SECTIONS (Migration-safe)
// ============================================================================
// Purpose: Add :Chunk label to existing :Section nodes for v3 compatibility
// Impact: Enables v3 tools to access sections via :Chunk label
// Pattern: Dual-labeling (not renaming) - both labels coexist
// Idempotent: WHERE NOT s:Chunk prevents redundant labeling
//
// IMPORTANT:
// For FRESH INSTALLATIONS, this statement is a no-op (no Sections exist yet).
// For MIGRATIONS, this adds :Chunk label to existing :Section nodes.
//
// Section queries continue working unchanged:
//   MATCH (s:Section) WHERE ... - works as before
// Chunk queries now also work:
//   MATCH (c:Chunk) WHERE ... - accesses same nodes
//
// This is safe to run even on fresh installations (no harm, just no effect).

MATCH (s:Section)
WHERE NOT s:Chunk
SET s:Chunk;

// ============================================================================
// PART 8: SCHEMA VERSION MARKER (v2.1)
// ============================================================================
// Purpose: Track schema version for migrations and validation
// Pattern: Singleton node with version metadata
// Idempotent: MERGE ensures only one version node exists
//
// METADATA FIELDS:
// - version: Schema version identifier (v2.1)
// - edition: Neo4j edition (community or enterprise)
// - vector_dimensions: Embedding dimensionality (1024)
// - embedding_provider: Default provider (jina-ai)
// - embedding_model: Model identifier (jina-embeddings-v3)
// - updated_at: Timestamp of schema creation/update
// - description: Human-readable summary
// - validation_note: Documentation of validation strategy

MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v2.1',
    sv.edition = 'community',
    sv.vector_dimensions = 1024,
    sv.embedding_provider = 'jina-ai',
    sv.embedding_model = 'jina-embeddings-v3',
    sv.updated_at = datetime(),
    sv.description = 'Phase 7C: Complete v2.1 schema with 1024-D vectors, dual-labeling, session tracking',
    sv.validation_note = 'Property existence constraints enforced in application layer (Community Edition)',
    sv.migration_path = 'Fresh installation - complete schema from v2.1';

// ============================================================================
// PART 9: VERIFICATION QUERIES (Commented - for manual testing)
// ============================================================================
// These queries can be uncommented and run separately to verify schema state.
// DO NOT uncomment for automated schema creation (they're for diagnostics).

// -- 1. Verify dual-labeling (Section and Chunk counts should match)
// MATCH (s:Section)
// WITH count(s) as section_count
// MATCH (c:Chunk)
// RETURN section_count, count(c) as chunk_count,
//        CASE WHEN section_count = count(c) THEN 'PASS ✓' ELSE 'FAIL ✗' END as dual_label_test;

// -- 2. List all constraints
// SHOW CONSTRAINTS
// YIELD name, type, labelsOrTypes
// RETURN name, type, labelsOrTypes
// ORDER BY labelsOrTypes, name;

// -- 3. List all property indexes
// SHOW INDEXES
// YIELD name, type, labelsOrTypes, properties
// WHERE type IN ['RANGE', 'BTREE', 'LOOKUP']
// RETURN name, type, labelsOrTypes, properties
// ORDER BY labelsOrTypes, name;

// -- 4. List vector indexes with dimensions
// SHOW INDEXES
// YIELD name, type, labelsOrTypes, properties
// WHERE type = 'VECTOR'
// RETURN name, labelsOrTypes as label, properties,
//        'Dimensions: 1024, Similarity: cosine' as config;

// -- 5. Verify schema version
// MATCH (sv:SchemaVersion {id: 'singleton'})
// RETURN sv.version as version,
//        sv.edition as edition,
//        sv.vector_dimensions as dimensions,
//        sv.embedding_provider as provider,
//        sv.embedding_model as model,
//        sv.updated_at as updated,
//        sv.description as description;

// -- 6. Count nodes by label (diagnostic)
// CALL db.labels() YIELD label
// CALL {
//   WITH label
//   MATCH (n)
//   WHERE label IN labels(n)
//   RETURN count(n) as count
// }
// RETURN label, count
// ORDER BY count DESC;

// -- 7. Validate Section embedding completeness (should return 0)
// MATCH (s:Section)
// WHERE s.vector_embedding IS NULL
//    OR s.embedding_version IS NULL
//    OR s.embedding_provider IS NULL
//    OR s.embedding_timestamp IS NULL
//    OR s.embedding_dimensions IS NULL
// RETURN count(s) as sections_missing_required_embedding_fields,
//        CASE WHEN count(s) = 0 THEN 'PASS ✓' ELSE 'FAIL ✗' END as validation_test;

// ============================================================================
// END OF SCHEMA v2.1 COMPLETE DDL
// ============================================================================
// Schema creation complete!
//
// Next steps:
// 1. Verify constraints: SHOW CONSTRAINTS;
// 2. Verify indexes: SHOW INDEXES;
// 3. Check schema version: MATCH (sv:SchemaVersion) RETURN sv;
// 4. Begin ingestion: Start with dual-labeled Section creation
//
// For troubleshooting schema issues, see:
// - /docs/schema-v2.1-specification.md (complete schema spec)
// - /docs/phase7C-execution-plan.md (implementation guide)
// - /reports/phase-7C/schema-validation.md (validation checklist)
//
// ============================================================================
