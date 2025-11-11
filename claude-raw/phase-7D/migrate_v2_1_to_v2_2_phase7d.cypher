

// ============================================================================
// PART X: Phase 7D - Hierarchical Chunking Additions (v2.2)
// ============================================================================
// Purpose:
// - Add hierarchical tiered chunking support (L1/L2/L3) without breaking v2.1.
// - Preserve existing :Section and dual label :Chunk usage.
// - Introduce `tier` (1=L1 Summary, 2=L2 Concept Group, 3=L3 Structural)
// - Introduce `parent_id` on Section for convenience (graph edges remain source of truth).
// - Add new relationships: HAS_SUMMARY, HAS_CHILD.
// - Add indexes to support fast fan-outs and filtering by tier/order/document.
//
// Notes:
// - All CREATE statements are idempotent via IF NOT EXISTS.
// - We DO NOT enforce property existence constraints to maintain v2.1 compatibility.
// - Existing queries using `level` (markdown heading depth) keep working unchanged.
// - New queries should use `tier` for hierarchy traversal.


// ---- Node Key Constraints (existing constraints preserved above) ----
// (No new uniqueness constraints required; Section.id remains the stable key)


// ---- Section Indexes (new) ----
CREATE INDEX section_tier IF NOT EXISTS
FOR (s:Section) ON (s.tier);

CREATE INDEX section_parent_id IF NOT EXISTS
FOR (s:Section) ON (s.parent_id);

CREATE INDEX section_anchor IF NOT EXISTS
FOR (s:Section) ON (s.anchor);

CREATE INDEX section_checksum IF NOT EXISTS
FOR (s:Section) ON (s.checksum);

CREATE INDEX section_embedding_version IF NOT EXISTS
FOR (s:Section) ON (s.embedding_version);

// Composite index to accelerate doc-tier ordering scans:
CREATE INDEX section_doc_tier_order IF NOT EXISTS
FOR (s:Section) ON (s.document_id, s.tier, s.order);


// ---- Dual-label :Chunk Indexes (mirror of Section) ----
CREATE INDEX chunk_tier IF NOT EXISTS
FOR (c:Chunk) ON (c.tier);

CREATE INDEX chunk_parent_id IF NOT EXISTS
FOR (c:Chunk) ON (c.parent_id);

CREATE INDEX chunk_anchor IF NOT EXISTS
FOR (c:Chunk) ON (c.anchor);

CREATE INDEX chunk_checksum IF NOT EXISTS
FOR (c:Chunk) ON (c.checksum);

// Composite for doc-tier-order on :Chunk:
CREATE INDEX chunk_doc_tier_order IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id, c.tier, c.order);


// ---- Relationship Types (no constraints required, but documenting new types)
// New edges created by the ingestion pipeline:
// (:Document)-[:HAS_SUMMARY]->(:Section {tier:1})
// (:Section {tier:1})-[:HAS_CHILD]->(:Section {tier:2})
// (:Section {tier:2})-[:HAS_CHILD]->(:Section {tier:3})
// Existing (:Document)-[:HAS_SECTION]->(:Section) remains intact for compatibility.


// ---- Schema Version Bump to v2.2 ----
MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v2.2',
    sv.edition = coalesce(sv.edition, 'community'),
    sv.vector_dimensions = coalesce(sv.vector_dimensions, 1024),
    sv.embedding_provider = coalesce(sv.embedding_provider, 'jina-ai'),
    sv.embedding_model = coalesce(sv.embedding_model, 'jina-embeddings-v4'),
    sv.updated_at = datetime('2025-10-26T05:51:38.696094Z'),
    sv.description = 'WekaDocs GraphRAG Schema v2.2 (Phase 7D: hierarchical chunking)',
    sv.features = ['phase-7C-1024d', 'phase-7D-tiered-chunking'],
    sv.compatible_with = ['v2.1'],
    sv.validation_note = 'All v2.1 indexes/constraints preserved; new tier indexes added';

// Optional verification:
// 1) SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties WHERE name CONTAINS 'tier' OR name CONTAINS 'parent_id';
// 2) MATCH (sv:SchemaVersion {id:'singleton'}) RETURN sv.version, sv.features, sv.updated_at;
