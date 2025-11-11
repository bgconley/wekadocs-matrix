// WekaDocs Schema v2.1 - Complete DDL (Community Edition Compatible)
// Phase 7C, Task 7C.3: Schema v2.1 activation with 1024-D vector indices
//
// COMMUNITY EDITION ADAPTATION:
// - Removed property existence constraints (require Enterprise Edition)
// - Validation moved to application layer (already implemented in ingestion)
// - All other features preserved: dual-labeling, vector indices, session tracking
//
// CRITICAL UPDATES from v2.0:
// 1. Vector dimensions: 1024-D (not 384-D or 768-D)
// 2. Dual vector indices: Section AND Chunk (same data, two labels)
// 3. Session/Query/Answer multi-turn tracking nodes
// 4. Schema version marker updated to v2.1
//
// IMPORTANT: All changes are ADDITIVE and IDEMPOTENT
// - Safe to re-run multiple times
// - No breaking changes to existing queries
// - Dual-labeling preserves all Section queries

// ============================================================================
// Part 1: Dual-label existing Sections as :Chunk
// ============================================================================
// Purpose: v3 tool compatibility while preserving v2 queries
// Impact: Existing Section queries continue working unchanged
// Idempotent: SET is idempotent (no-op if label already exists)

MATCH (s:Section)
WHERE NOT s:Chunk
SET s:Chunk;

// ============================================================================
// Part 2: Session/Query/Answer Constraints (NEW)
// ============================================================================
// Purpose: Multi-turn conversation tracking (Phase 7C.8)
// Impact: None until Phase 7C.8 creates these nodes
// Idempotent: IF NOT EXISTS prevents constraint duplication

// Session constraints
CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

// Query constraints
CREATE CONSTRAINT query_id_unique IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;

// NOTE: Property existence constraints removed for Community Edition compatibility
// Application code must validate:
// - q.text IS NOT NULL (before creating Query nodes)
// - a.text IS NOT NULL (before creating Answer nodes)
// - All Section embedding fields (already validated in ingestion pipeline)

// Answer constraints
CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

// ============================================================================
// Part 3: Vector Indices - 1024-D (UPDATED from stub)
// ============================================================================
// Purpose: Semantic search on Section/Chunk nodes
// CRITICAL: Dimensions set to 1024-D for Jina v4 (not 384-D or 768-D)
// Idempotent: IF NOT EXISTS

// Section vector index (primary retrieval)
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
// Part 4: Property Indices for Session/Query/Answer
// ============================================================================
// Purpose: Fast lookups for multi-turn tracking
// Idempotent: IF NOT EXISTS

// Session indices
CREATE INDEX session_started_at IF NOT EXISTS
FOR (s:Session) ON (s.started_at);

CREATE INDEX session_expires_at IF NOT EXISTS
FOR (s:Session) ON (s.expires_at);

CREATE INDEX session_active IF NOT EXISTS
FOR (s:Session) ON (s.active);

CREATE INDEX session_user_id IF NOT EXISTS
FOR (s:Session) ON (s.user_id);

// Query indices
CREATE INDEX query_turn IF NOT EXISTS
FOR (q:Query) ON (q.turn);

CREATE INDEX query_asked_at IF NOT EXISTS
FOR (q:Query) ON (q.asked_at);

// Answer indices
CREATE INDEX answer_created_at IF NOT EXISTS
FOR (a:Answer) ON (a.created_at);

CREATE INDEX answer_user_feedback IF NOT EXISTS
FOR (a:Answer) ON (a.user_feedback);

// ============================================================================
// Part 5: Additional Section/Chunk Indices
// ============================================================================
// Purpose: Optimize dual-label queries
// Idempotent: IF NOT EXISTS

// Chunk-specific indices (mirror Section indices for v3 compatibility)
CREATE INDEX chunk_document_id IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

CREATE INDEX chunk_level IF NOT EXISTS
FOR (c:Chunk) ON (c.level);

CREATE INDEX chunk_embedding_version IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_version);

// ============================================================================
// Part 6: Schema Version Marker
// ============================================================================
// Purpose: Track schema version for migrations and validation
// Pattern: Singleton node with version property
// Idempotent: MERGE ensures only one version node exists

MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v2.1',
    sv.edition = 'community',
    sv.updated_at = datetime(),
    sv.description = 'Phase 7C: 1024-D vectors, dual-labeling, session tracking (Community Edition)',
    sv.vector_dimensions = 1024,
    sv.embedding_provider = 'jina-ai',
    sv.embedding_model = 'jina-embeddings-v4',
    sv.validation_note = 'Property existence constraints enforced in application layer'
;

// ============================================================================
// Verification Queries (commented out - for manual testing)
// ============================================================================

// -- 1. Verify dual-labeling (Section and Chunk counts should match)
// MATCH (s:Section)
// WITH count(s) as section_count
// MATCH (c:Chunk)
// RETURN section_count, count(c) as chunk_count,
//        CASE WHEN section_count = count(c) THEN 'PASS' ELSE 'FAIL' END as test;

// -- 2. Verify constraints exist
// SHOW CONSTRAINTS
// YIELD name, type
// WHERE name CONTAINS 'session' OR name CONTAINS 'query' OR name CONTAINS 'answer'
// RETURN name, type
// ORDER BY name;

// -- 3. Verify vector indices exist with correct dimensions
// SHOW INDEXES
// YIELD name, type, labelsOrTypes, properties
// WHERE type = 'VECTOR'
// RETURN name, labelsOrTypes, properties;

// -- 4. Verify schema version
// MATCH (sv:SchemaVersion {id: 'singleton'})
// RETURN sv.version as version,
//        sv.edition as edition,
//        sv.vector_dimensions as dims,
//        sv.embedding_provider as provider,
//        sv.embedding_model as model,
//        sv.updated_at as updated,
//        sv.description as description,
//        sv.validation_note as validation_note;

// -- 5. Count Sections with missing embedding fields (should be 0)
// MATCH (s:Section)
// WHERE s.vector_embedding IS NULL
//    OR s.embedding_version IS NULL
//    OR s.embedding_provider IS NULL
//    OR s.embedding_timestamp IS NULL
//    OR s.embedding_dimensions IS NULL
// RETURN count(s) as sections_missing_embeddings;

// ============================================================================
// End of Schema v2.1 DDL (Community Edition)
// ============================================================================
