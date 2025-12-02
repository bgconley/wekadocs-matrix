// ============================================================================
// WekaDocs GraphRAG Schema v2.2 - BACKWARD COMPATIBLE + HYBRID RAG
// ============================================================================
// Created:     2025-11-05
// Base:        v2.1 clean snapshot (kept intact)
// Edition:     Neo4j Community Edition (no property-existence constraints)
// Purpose:     Enable Strategy 1 (smaller chunks) & Strategy 3 (hybrid retrieval)
//              while remaining 100% backward compatible.
// Canonical:   document_id (canonical) ; doc_id (legacy alias)
// Idempotent:  Yes (IF NOT EXISTS / MERGE everywhere).
// ============================================================================


// ---------------------------------------------------------------------------
// PART 0: CORE NODE CONSTRAINTS (same as v2.1)
// ---------------------------------------------------------------------------

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;

CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

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

CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT query_id_unique IF NOT EXISTS
FOR (q:Query) REQUIRE q.query_id IS UNIQUE;

CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (a:Answer) REQUIRE a.answer_id IS UNIQUE;

CREATE CONSTRAINT schema_version_singleton IF NOT EXISTS
FOR (sv:SchemaVersion) REQUIRE sv.id IS UNIQUE;

// Phase 3: GhostDocument for forward reference resolution
// Ghost Documents are placeholders created when a reference target doesn't exist yet
CREATE CONSTRAINT ghost_document_id_unique IF NOT EXISTS
FOR (g:GhostDocument) REQUIRE g.id IS UNIQUE;

// Phase 3: PendingReference for fuzzy title forward references
// Created when a non-hyperlink reference can't be resolved at ingestion time
CREATE CONSTRAINT pending_reference_hint_unique IF NOT EXISTS
FOR (p:PendingReference) REQUIRE p.hint IS UNIQUE;


// ---------------------------------------------------------------------------
// PART 1: PROPERTY INDEXES (v2.1 set + v2.2 additions)
// ---------------------------------------------------------------------------

// Document indexes (v2.1)
CREATE INDEX document_source_type IF NOT EXISTS
FOR (d:Document) ON (d.source_type);

CREATE INDEX document_version IF NOT EXISTS
FOR (d:Document) ON (d.version);

CREATE INDEX document_last_edited IF NOT EXISTS
FOR (d:Document) ON (d.last_edited);

// Fulltext index to accelerate fuzzy REFERENCES resolution
CALL db.index.fulltext.createNodeIndex(
  'document_title_ft',
  ['Document'],
  ['title'],
  {analyzer: 'standard-folding'}
) IF NOT EXISTS;

// Section indexes (v2.1) - names match running DB with _idx suffix
CREATE INDEX section_document_id_idx IF NOT EXISTS
FOR (s:Section) ON (s.document_id);

CREATE INDEX section_level_idx IF NOT EXISTS
FOR (s:Section) ON (s.level);

CREATE INDEX section_order_idx IF NOT EXISTS
FOR (s:Section) ON (s.order);

CREATE INDEX section_parent_section_id IF NOT EXISTS
FOR (s:Section) ON (s.parent_section_id);

// Chunk indexes (v2.1)
CREATE INDEX chunk_document_id IF NOT EXISTS
FOR (c:Chunk) ON (c.document_id);

CREATE INDEX chunk_level IF NOT EXISTS
FOR (c:Chunk) ON (c.level);

CREATE INDEX chunk_order IF NOT EXISTS
FOR (c:Chunk) ON (c.order);

CREATE INDEX chunk_embedding_version IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_version);

// CitationUnit indexes (v2.1)
CREATE INDEX citation_unit_parent_chunk_id IF NOT EXISTS
FOR (cu:CitationUnit) ON (cu.parent_chunk_id);

CREATE INDEX citation_unit_document_id IF NOT EXISTS
FOR (cu:CitationUnit) ON (cu.document_id);

CREATE INDEX citation_unit_order IF NOT EXISTS
FOR (cu:CitationUnit) ON (cu.order);

// Domain entity indexes (v2.1)
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

// Entity indexes (C.1.1 / C.1.3 - heading concepts and entity hygiene)
CREATE INDEX entity_canonical_name IF NOT EXISTS
FOR (e:Entity) ON (e.canonical_name);

CREATE INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

CREATE INDEX entity_document_id IF NOT EXISTS
FOR (e:Entity) ON (e.document_id);

// Session indexes (v2.1)
CREATE INDEX session_started_at IF NOT EXISTS
FOR (s:Session) ON (s.started_at);

CREATE INDEX session_expires_at IF NOT EXISTS
FOR (s:Session) ON (s.expires_at);

CREATE INDEX session_active IF NOT EXISTS
FOR (s:Session) ON (s.active);

CREATE INDEX session_user_id IF NOT EXISTS
FOR (s:Session) ON (s.user_id);

// Query indexes (v2.1)
CREATE INDEX query_turn IF NOT EXISTS
FOR (q:Query) ON (q.turn);

CREATE INDEX query_asked_at IF NOT EXISTS
FOR (q:Query) ON (q.asked_at);

// Answer indexes (v2.1)
CREATE INDEX answer_created_at IF NOT EXISTS
FOR (a:Answer) ON (a.created_at);

CREATE INDEX answer_user_feedback IF NOT EXISTS
FOR (a:Answer) ON (a.user_feedback);


// ---- v2.2 additions for hybrid retrieval, safety & filters -----------------

// Chunk properties commonly used in scoring or filters
CREATE INDEX chunk_heading IF NOT EXISTS
FOR (c:Chunk) ON (c.heading);

CREATE INDEX chunk_token_count IF NOT EXISTS
FOR (c:Chunk) ON (c.token_count);

CREATE INDEX chunk_doc_tag IF NOT EXISTS
FOR (c:Chunk) ON (c.doc_tag);

CREATE INDEX chunk_is_microdoc IF NOT EXISTS
FOR (c:Chunk) ON (c.is_microdoc);

CREATE INDEX chunk_source_path IF NOT EXISTS
FOR (c:Chunk) ON (c.source_path);

CREATE INDEX chunk_tenant IF NOT EXISTS
FOR (c:Chunk) ON (c.tenant);

CREATE INDEX chunk_updated_at IF NOT EXISTS
FOR (c:Chunk) ON (c.updated_at);

CREATE INDEX chunk_embedding_provider IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_provider);

CREATE INDEX chunk_embedding_dimensions IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding_dimensions);

// Optional language/version/text hashes (enable if you populate them)
CREATE INDEX chunk_lang IF NOT EXISTS
FOR (c:Chunk) ON (c.lang);

CREATE INDEX chunk_version IF NOT EXISTS
FOR (c:Chunk) ON (c.version);

CREATE INDEX chunk_text_hash IF NOT EXISTS
FOR (c:Chunk) ON (c.text_hash);

CREATE INDEX chunk_shingle_hash IF NOT EXISTS
FOR (c:Chunk) ON (c.shingle_hash);

// Document helpers
CREATE INDEX document_doc_tag IF NOT EXISTS
FOR (d:Document) ON (d.doc_tag);

CREATE INDEX document_title IF NOT EXISTS
FOR (d:Document) ON (d.title);

// NEW: Legacy alias property for doc scoping (indexed to support old callers)
CREATE INDEX chunk_doc_id IF NOT EXISTS
FOR (c:Chunk) ON (c.doc_id);

CREATE INDEX section_doc_id IF NOT EXISTS
FOR (s:Section) ON (s.doc_id);


// ---------------------------------------------------------------------------
// PART 2: FULL-TEXT INDEXES (lexical tier for hybrid retrieval)
// ---------------------------------------------------------------------------

// Legacy single-field fulltext index (kept for backward compatibility)
CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
FOR (n:Chunk) ON EACH [n.text];

// Primary fulltext index for hybrid retrieval (text + heading)
CREATE FULLTEXT INDEX chunk_text_index_v3_bge_m3 IF NOT EXISTS
FOR (n:Chunk|CitationUnit) ON EACH [n.text, n.heading];

CREATE FULLTEXT INDEX heading_fulltext_v1 IF NOT EXISTS
FOR (n:Chunk|Section) ON EACH [n.heading];

// Phase 3: Document title fulltext index for efficient REFERENCES resolution
// Supports fuzzy CONTAINS queries on Document.title without O(N) scan
CREATE FULLTEXT INDEX document_title_ft IF NOT EXISTS
FOR (d:Document) ON EACH [d.title];


// ---------------------------------------------------------------------------
// PART 3: VECTOR INDEXES
// ---------------------------------------------------------------------------
// Keep the original 1024-D content vectors (both labels for compatibility)
CREATE VECTOR INDEX section_embeddings_v2_bge_m3 IF NOT EXISTS
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

// NEW (optional): 1024-D title/entity vectors for fusion (populate if used)
CREATE VECTOR INDEX chunk_title_embeddings_v1 IF NOT EXISTS
FOR (c:Chunk)
ON c.title_vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX chunk_entity_embeddings_v1 IF NOT EXISTS
FOR (c:Chunk)
ON c.entity_vector_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
};


// ---------------------------------------------------------------------------
// PART 4: DUAL-LABELING FOR EXISTING SECTIONS (Migration-safe)
// ---------------------------------------------------------------------------

MATCH (s:Section)
WHERE NOT s:Chunk
SET s:Chunk,
    s.is_legacy_section = true;   // BUGFIX: add label (no "= true")


// ---------------------------------------------------------------------------
// PART 5: DOC ID ALIAS BACKFILL (Idempotent)
// ---------------------------------------------------------------------------
// Ensures canonical document_id exists; doc_id is maintained as a legacy alias.

MATCH (c:Chunk)
WHERE c.doc_id IS NULL AND c.document_id IS NOT NULL
SET c.doc_id = c.document_id;

MATCH (c:Chunk)
WHERE c.document_id IS NULL AND c.doc_id IS NOT NULL
SET c.document_id = c.doc_id;

MATCH (s:Section)
WHERE s.doc_id IS NULL AND s.document_id IS NOT NULL
SET s.doc_id = s.document_id;

MATCH (s:Section)
WHERE s.document_id IS NULL AND s.doc_id IS NOT NULL
SET s.document_id = s.doc_id;


// ---------------------------------------------------------------------------
// PART 6: RELATIONSHIP TYPE MARKER (documentation)
// ---------------------------------------------------------------------------

// Phase 2 Cleanup: Removed PREV and SAME_HEADING from marker
// PREV is redundant (use <-[:NEXT]-), SAME_HEADING had O(n²) fanout with no query usage
// Phase 3: Added REFERENCES for cross-document connectivity (Chunk → Document)
// Phase 3: Added PENDING_REF for forward reference tracking (Chunk → PendingReference)
MERGE (m:RelationshipTypesMarker {id: 'chunk_rel_types_v1'})
  ON CREATE SET m.types = ['NEXT','CHILD_OF','MENTIONS','PARENT_OF','REFERENCES','PENDING_REF'],
                m.created_at = datetime()
  ON MATCH SET m.types = ['NEXT','CHILD_OF','MENTIONS','PARENT_OF','REFERENCES','PENDING_REF'],
               m.updated_at = datetime();


// ---------------------------------------------------------------------------
// PART 7: SCHEMA VERSION MARKER (v2.2)
// ---------------------------------------------------------------------------

MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v2.2',
    sv.edition = 'community',
    sv.vector_dimensions = 1024,
    sv.embedding_provider = 'bge-m3',
    sv.embedding_model = 'BAAI/bge-m3',
    sv.updated_at = datetime(),
    sv.description = 'Phase 7E+: Hybrid retrieval enablement (multi-vector, lexical boost, graph expansion) with small-chunk ingestion',
    sv.validation_note = 'Property existence constraints enforced in application layer (Community Edition)',
    sv.backup_source = 'v2.1 clean state upgraded to v2.2 hybrid-ready on 2025-11-05',
    sv.reform_note = 'Chunk sizes converge to ~400 target with ~32 token overlap; dynamic assembly at query time',
    sv.compatibility = 'All v2.1 objects preserved; new indexes are additive';


// ---------------------------------------------------------------------------
// PART 8: OPTIONAL POST-INGEST RELATIONSHIP BUILDERS (run manually)
// ---------------------------------------------------------------------------
// NOTE (dual-label guard): when matching (:Chunk), restrict to true chunks via exists(c.text).

// -- CHILD_OF (Chunk -> Section)  [guarded]
// MATCH (c:Chunk)
// WHERE exists(c.text) AND c.parent_section_id IS NOT NULL
// MATCH (s:Section {id: c.parent_section_id})
// MERGE (c)-[:CHILD_OF]->(s);

// -- PARENT_OF (Section -> Section)
// MATCH (child:Section)
// WHERE child.parent_section_id IS NOT NULL
// MATCH (parent:Section {id: child.parent_section_id})
// MERGE (parent)-[:PARENT_OF]->(child);

// -- NEXT within same document/parent ordered by c.order  [guarded]
// Phase 2 Cleanup: Removed PREV (redundant - use <-[:NEXT]-)
// MATCH (c:Chunk)
// WHERE exists(c.text)
// WITH coalesce(c.document_id, c.doc_id) AS d, c.parent_section_id AS p, c
// ORDER BY d, p, c.order
// WITH d, p, collect(c) AS chunks
// UNWIND range(0, size(chunks)-2) AS i
// WITH chunks[i] AS a, chunks[i+1] AS b
// MERGE (a)-[:NEXT]->(b);

// Phase 2 Cleanup: Removed SAME_HEADING builder entirely
// (O(n²) edge fanout with zero query usage)


// ---------------------------------------------------------------------------
// PART 9: DIAGNOSTICS / VERIFICATION (run manually)
// ---------------------------------------------------------------------------
// SHOW CONSTRAINTS YIELD name, type RETURN count(*) AS constraint_count;
// SHOW INDEXES YIELD name, type RETURN name, type ORDER BY type, name;
// SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' RETURN name;
// SHOW INDEXES YIELD name, type, labelsOrTypes WHERE type = 'VECTOR' RETURN name, labelsOrTypes;
// MATCH (c:Chunk) RETURN exists(c.document_id) AS has_document_id, exists(c.doc_id) AS has_doc_id LIMIT 1;


// ===========================================================================
// PART 10: QDRANT SCHEMA (Reference Only - Apply via Python/REST)
// ===========================================================================
// This section documents the Qdrant vector store schema for completeness.
// It cannot be applied via Cypher - use the Python script or Qdrant REST API.
//
// To apply: python scripts/neo4j/schema_ddl_complete_20251125.py --apply-qdrant
//
// ---------------------------------------------------------------------------
// COLLECTIONS
// ---------------------------------------------------------------------------
// Primary collection: chunks_multi_bge_m3 (Phase 7E+ BGE-M3 embeddings)
// Legacy collection:  chunks_multi (kept for backward compatibility)
//
// DENSE VECTORS (per collection):
//   - content:          1024D, Cosine  (main content embedding)
//   - entity:           1024D, Cosine  (entity-name embedding)
//   - title:            1024D, Cosine  (chunk title/heading embedding)
//   - doc_title:        1024D, Cosine  (parent document title embedding)
//   - late-interaction: 1024D, Cosine, MaxSim (ColBERT-style multi-vector)
//
// SPARSE VECTORS (per collection):
//   - text-sparse:      BM25/BGE-M3 sparse, on-disk index (chunk content)
//   - doc_title-sparse: BM25/BGE-M3 sparse, on-disk index (document title)
//
// HNSW CONFIG:
//   - m: 48                    (graph edges per node)
//   - ef_construct: 256        (construction beam width)
//   - full_scan_threshold: 10000
//   - on_disk: false
//
// OPTIMIZER CONFIG:
//   - deleted_threshold: 0.2
//   - vacuum_min_vector_number: 1000
//   - indexing_threshold: 10000
//   - flush_interval_sec: 5
//
// WAL CONFIG:
//   - wal_capacity_mb: 32
//   - wal_segments_ahead: 0
//
// ---------------------------------------------------------------------------
// PAYLOAD INDEXES (22 fields per collection)
// ---------------------------------------------------------------------------
// Identifiers:
//   - id (keyword)
//   - document_id (keyword)     -- canonical
//   - doc_id (keyword)          -- legacy alias
//   - parent_section_id (keyword)
//   - parent_section_original_id (keyword)
//   - node_id (keyword)         -- BGE-M3 collection only
//   - kg_id (keyword)           -- knowledge graph ID
//
// Structure & ordering:
//   - order (integer)
//   - heading (text)
//   - doc_title (text)          -- parent document title for filtering/display
//
// Metadata filters:
//   - doc_tag (keyword)
//   - tenant (keyword)
//   - lang (keyword)
//   - version (keyword)
//   - source_path (keyword)
//   - snapshot_scope (keyword)
//
// Token/size:
//   - token_count (integer)
//   - is_microdoc (bool)
//
// Embedding audit:
//   - embedding_version (keyword)
//   - embedding_provider (keyword)
//   - embedding_dimensions (integer)
//
// Deduplication:
//   - text_hash (keyword)
//   - shingle_hash (keyword)
//
// Timestamps:
//   - updated_at (integer)      -- epoch seconds
//
// ===========================================================================
// END OF SCHEMA DEFINITION
// ===========================================================================
