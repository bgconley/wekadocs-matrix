// WekaDocs Schema v2.1 - Additive DDL
// Pre-Phase 7: Foundation preparation for Phase 7 features
// Status: Safe to run in dev, idempotent, non-disruptive
//
// Changes from v2.0:
// 1. Dual-label Sections as :Chunk (v3 compatibility)
// 2. Add Session/Query/Answer constraints (multi-turn tracking)
// 3. Add schema version marker
//
// IMPORTANT: All changes are ADDITIVE and IDEMPOTENT
// - Safe to re-run multiple times
// - No data modification beyond dual-labeling
// - No breaking changes to existing queries

// ============================================================================
// Part 1: Dual-label existing Sections as :Chunk
// ============================================================================
// Purpose: v3 tool compatibility while preserving v2 queries
// Impact: Existing Section queries continue working unchanged
// Idempotent: SET is idempotent (no-op if label already exists)

MATCH (s:Section)
SET s:Chunk;

// ============================================================================
// Part 2: Session/Query/Answer Constraints (NEW)
// ============================================================================
// Purpose: Multi-turn conversation tracking (Phase 7)
// Impact: None until Phase 7 creates these nodes
// Idempotent: IF NOT EXISTS prevents constraint duplication

// Session constraints
CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

// Query constraints
CREATE CONSTRAINT query_id_unique IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;

CREATE CONSTRAINT query_text_exists IF NOT EXISTS
FOR (q:Query) REQUIRE q.text IS NOT NULL;

// Answer constraints
CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

CREATE CONSTRAINT answer_text_exists IF NOT EXISTS
FOR (a:Answer) REQUIRE a.text IS NOT NULL;

// ============================================================================
// Part 3: Schema Version Marker
// ============================================================================
// Purpose: Track schema version for migrations
// Pattern: Singleton node with version property
// Idempotent: MERGE ensures only one version node exists

MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v2.1',
    sv.updated_at = datetime(),
    sv.description = 'Pre-Phase 7 foundation: dual-labeling + session tracking prep';

// ============================================================================
// Verification Queries (for manual testing)
// ============================================================================
// Uncomment to verify schema changes after running script

// -- Verify dual-labeling
// MATCH (s:Section)
// WITH count(s) as section_count
// MATCH (c:Chunk)
// RETURN section_count, count(c) as chunk_count;
// -- Should return equal counts

// -- Verify constraints exist
// SHOW CONSTRAINTS;

// -- Verify schema version
// MATCH (sv:SchemaVersion {id: 'singleton'})
// RETURN sv.version, sv.updated_at, sv.description;
