// ============================================================================
// WekaDocs GraphRAG Schema v4.0 - :Section DEPRECATED
// ============================================================================
// Created:     2025-11-05
// Updated:     2025-12-12 (Phase 4: Section→Chunk consolidation)
// Base:        v3.0 + Phase 4 (:Section label deprecated, all content now :Chunk)
// Edition:     Neo4j Community Edition (no property-existence constraints)
// Purpose:     Enable Strategy 1 (smaller chunks), Strategy 3 (hybrid retrieval),
//              and full GraphRAG MENTIONS bridge for entity-chunk traversal.
// Canonical:   document_id (canonical) ; doc_id (legacy alias)
// Idempotent:  Yes (IF NOT EXISTS / MERGE everywhere).
//
// PHASE 4 CHANGES (2025-12-12):
//   - DEPRECATED: :Section label - all content nodes are now :Chunk only
//   - DEPRECATED: section_* indexes - kept temporarily for migration
//   - REMOVED: Dual-labeling migration (no longer needed)
//   - All queries now use :Chunk exclusively
//   - Run scripts/migrate_section_to_chunk.py to remove :Section from existing nodes
//
// PHASE 3.5 FEATURES (retained):
//   - GLiNER entity types (Parameter, Component, Protocol, etc.)
//   - MENTIONS relationship with confidence and source properties
//   - MENTIONED_IN inverse edge for bidirectional traversal
//   - Entity-chunk bridge for GraphRAG hybrid retrieval pattern
// ============================================================================


// ---------------------------------------------------------------------------
// PART 0: CORE NODE CONSTRAINTS (same as v2.1)
// ---------------------------------------------------------------------------

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT document_source_uri_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_uri IS UNIQUE;

// DEPRECATED (Phase 4): :Section label replaced by :Chunk
// Constraint kept for migration compatibility - will be removed in future version
CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (s:Section) REQUIRE s.id IS UNIQUE;

// Primary constraint for content nodes (replaces section_id_unique)
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

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

// Phase 3.5: GLiNER entity type constraints (expanded entity model)
CREATE CONSTRAINT protocol_id_unique IF NOT EXISTS
FOR (p:Protocol) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT cloudprovider_id_unique IF NOT EXISTS
FOR (c:CloudProvider) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT storageconcept_id_unique IF NOT EXISTS
FOR (s:StorageConcept) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT version_id_unique IF NOT EXISTS
FOR (v:Version) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT procedurestep_id_unique IF NOT EXISTS
FOR (ps:ProcedureStep) REQUIRE ps.id IS UNIQUE;

CREATE CONSTRAINT capacitymetric_id_unique IF NOT EXISTS
FOR (cm:CapacityMetric) REQUIRE cm.id IS UNIQUE;

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

// ---------------------------------------------------------------------------
// DEPRECATED Section indexes (Phase 4) - kept for migration compatibility
// These will be orphaned after running migrate_section_to_chunk.py
// Remove after migration is complete across all environments
// ---------------------------------------------------------------------------
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

// Phase 3.5: Entity source tracking for GLiNER provenance
CREATE INDEX entity_source IF NOT EXISTS
FOR (e:Entity) ON (e.source);

// Phase 3.5: Composite index for entity lookup by document + name
CREATE INDEX entity_document_name_idx IF NOT EXISTS
FOR (e:Entity) ON (e.document_id, e.name);

// Phase 3.5: Entity source section tracking
CREATE INDEX entity_source_section_idx IF NOT EXISTS
FOR (e:Entity) ON (e.source_section_id);

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

// DEPRECATED (Phase 4): Kept for migration compatibility
CREATE INDEX section_doc_id IF NOT EXISTS
FOR (s:Section) ON (s.doc_id);


// ---------------------------------------------------------------------------
// PART 1B: PHASE 3 STRUCTURAL METADATA INDEXES (markdown-it-py)
// ---------------------------------------------------------------------------
// These support enhanced metadata from markdown-it-py parser

// ---------------------------------------------------------------------------
// DEPRECATED Section structural indexes (Phase 4) - kept for migration only
// ---------------------------------------------------------------------------
CREATE INDEX section_line_start_idx IF NOT EXISTS
FOR (s:Section) ON (s.line_start);

CREATE INDEX section_line_end_idx IF NOT EXISTS
FOR (s:Section) ON (s.line_end);

CREATE INDEX section_has_code_idx IF NOT EXISTS
FOR (s:Section) ON (s.has_code);

CREATE INDEX section_has_table_idx IF NOT EXISTS
FOR (s:Section) ON (s.has_table);

CREATE INDEX section_code_ratio_idx IF NOT EXISTS
FOR (s:Section) ON (s.code_ratio);

CREATE INDEX section_parent_path_idx IF NOT EXISTS
FOR (s:Section) ON (s.parent_path);

// Chunk structural filtering (mirrors Section indexes)
CREATE INDEX chunk_has_code_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.has_code);

CREATE INDEX chunk_has_table_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.has_table);

CREATE INDEX chunk_line_start_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.line_start);

CREATE INDEX chunk_parent_path_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.parent_path);


// ---------------------------------------------------------------------------
// PART 1C: PHASE 3.5 MENTIONS RELATIONSHIP INDEXES (GraphRAG bridge)
// ---------------------------------------------------------------------------
// Critical for hybrid retrieval: Vector search -> Graph expansion via MENTIONS

// Index on MENTIONS confidence for filtering high-quality extractions
CREATE INDEX mentions_confidence_idx IF NOT EXISTS
FOR ()-[r:MENTIONS]-() ON (r.confidence);

// Index on MENTIONS source for distinguishing GLiNER vs structural
CREATE INDEX mentions_source_idx IF NOT EXISTS
FOR ()-[r:MENTIONS]-() ON (r.source);

// Index on MENTIONED_IN (reverse edge) for entity->chunk traversal
CREATE INDEX mentioned_in_confidence_idx IF NOT EXISTS
FOR ()-[r:MENTIONED_IN]-() ON (r.confidence);

CREATE INDEX mentioned_in_source_idx IF NOT EXISTS
FOR ()-[r:MENTIONED_IN]-() ON (r.source);

// Index on PARENT_HEADING level_delta for hierarchy traversal
CREATE INDEX parent_heading_level_delta_idx IF NOT EXISTS
FOR ()-[r:PARENT_HEADING]-() ON (r.level_delta);


// ---------------------------------------------------------------------------
// PART 2: FULL-TEXT INDEXES (lexical tier for hybrid retrieval)
// ---------------------------------------------------------------------------

// Legacy single-field fulltext index (kept for backward compatibility)
CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
FOR (n:Chunk) ON EACH [n.text];

// Primary fulltext index for hybrid retrieval (text + heading)
CREATE FULLTEXT INDEX chunk_text_index_v3_bge_m3 IF NOT EXISTS
FOR (n:Chunk|CitationUnit) ON EACH [n.text, n.heading];

// Note: Includes :Section for backward compatibility during migration
// After migration, consider recreating with :Chunk only
CREATE FULLTEXT INDEX heading_fulltext_v1 IF NOT EXISTS
FOR (n:Chunk|Section) ON EACH [n.heading];

// Phase 3: Document title fulltext index for efficient REFERENCES resolution
// Supports fuzzy CONTAINS queries on Document.title without O(N) scan
CREATE FULLTEXT INDEX document_title_ft IF NOT EXISTS
FOR (d:Document) ON EACH [d.title];


// ---------------------------------------------------------------------------
// PART 3: VECTOR INDEXES
// ---------------------------------------------------------------------------
// DEPRECATED (Phase 4): Section vector index - kept for migration compatibility
// After migration, this index will be orphaned (no :Section nodes)
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
// PART 4: DUAL-LABELING MIGRATION (DEPRECATED - Phase 4)
// ---------------------------------------------------------------------------
// This block was used to add :Chunk label to legacy :Section nodes.
// After Phase 4 migration (migrate_section_to_chunk.py), this is no longer needed.
// The migration script removes :Section label entirely.
//
// REMOVED: The following was the original dual-labeling migration:
// MATCH (s:Section)
// WHERE NOT s:Chunk
// SET s:Chunk,
//     s.is_legacy_section = true;


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

// DEPRECATED (Phase 4): Section doc_id backfill - kept for migration compatibility
// After migration, no :Section nodes will exist
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
// Phase 3: Added REFERENCES for cross-document connectivity (Chunk → Document)
// Phase 3: Added PARENT_HEADING for heading hierarchy traversal
// Phase 3.5: MENTIONS is now (Chunk → Entity) with confidence/source properties
// Phase 3.5: MENTIONED_IN is inverse edge (Entity → Chunk) for bidirectional traversal
MERGE (m:RelationshipTypesMarker {id: 'chunk_rel_types_v1'})
  ON CREATE SET m.types = ['NEXT','CHILD_OF','MENTIONS','MENTIONED_IN','PARENT_OF','PARENT_HEADING','REFERENCES','PENDING_REF'],
                m.created_at = datetime()
  ON MATCH SET m.types = ['NEXT','CHILD_OF','MENTIONS','MENTIONED_IN','PARENT_OF','PARENT_HEADING','REFERENCES','PENDING_REF'],
               m.updated_at = datetime();


// ---------------------------------------------------------------------------
// PART 7: SCHEMA VERSION MARKER (v3.0)
// ---------------------------------------------------------------------------

MERGE (sv:SchemaVersion {id: 'singleton'})
SET sv.version = 'v4.0',
    sv.edition = 'community',
    sv.vector_dimensions = 1024,
    sv.embedding_provider = 'bge-m3',
    sv.embedding_model = 'BAAI/bge-m3',
    sv.updated_at = datetime(),
    sv.description = 'Phase 4: :Section deprecated - all content nodes are now :Chunk only',
    sv.validation_note = 'Property existence constraints enforced in application layer (Community Edition)',
    sv.backup_source = 'v3.0 upgraded to v4.0 Section deprecation on 2025-12-12',
    sv.reform_note = 'MENTIONS: (Chunk)-[:MENTIONS {confidence, source}]->(Entity); :Section label deprecated',
    sv.compatibility = 'Legacy :Section indexes kept for migration; run migrate_section_to_chunk.py to complete',
    sv.phase3_features = 'Source line mapping, parent_path hierarchy, structural flags (has_code, has_table, code_ratio)',
    sv.phase4_features = ':Section deprecated, all queries use :Chunk, migration script provided';


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
// DENSE VECTORS (4 per collection):
//   - content:          1024D, Cosine  (main content embedding)
//   - title:            1024D, Cosine  (chunk title/heading embedding)
//   - doc_title:        1024D, Cosine  (parent document title embedding)
//   - late-interaction: 1024D, Cosine, MaxSim (ColBERT-style multi-vector)
//   NOTE: Dense 'entity' vector REMOVED - was broken (duplicated content embedding)
//         Replaced by entity-sparse for lexical entity name matching
//
// SPARSE VECTORS (4 per collection):
//   - text-sparse:      BM25/BGE-M3 sparse, on-disk index (chunk content)
//   - doc_title-sparse: BM25/BGE-M3 sparse, on-disk index (document title)
//   - title-sparse:     BM25/BGE-M3 sparse, on-disk index (section heading)
//   - entity-sparse:    BM25/BGE-M3 sparse, on-disk index (entity names)
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
// Phase 3: Structural metadata (markdown-it-py):
//   - has_code (bool)              -- for CLI query filtering
//   - has_table (bool)             -- for reference query filtering
//   - code_ratio (float)           -- code density 0.0-1.0
//   - parent_path (text)           -- heading hierarchy path
//   - line_start (integer)         -- source line correlation
//
// Timestamps:
//   - updated_at (integer)      -- epoch seconds
//
// ===========================================================================
// PART 11: MENTIONS RELATIONSHIP SCHEMA (Phase 3.5 GraphRAG Bridge)
// ===========================================================================
// The MENTIONS relationship is the critical bridge for GraphRAG hybrid retrieval:
//   Vector search -> Find relevant chunks -> Graph expansion -> Related entities
//
// DIRECTION CONVENTION:
//   - MENTIONS:     (Chunk)-[:MENTIONS]->(Entity)
//                   "This chunk mentions this entity"
//   - MENTIONED_IN: (Entity)-[:MENTIONED_IN]->(Chunk)
//                   "This entity is mentioned in this chunk" (inverse edge)
//
// RELATIONSHIP PROPERTIES:
//   - confidence (float):  0.0-1.0, extraction confidence score
//                          GLiNER entities: model confidence
//                          Structural entities: default 0.5 or higher
//   - source (string):     "gliner" | "structural"
//                          Enables filtering by extraction method
//   - count (integer):     Co-occurrence count (incremented on re-ingestion)
//
// ENTITY NODE PROPERTIES (for GLiNER entities):
//   - id:          "gliner:{TYPE}:{hash}" (e.g., "gliner:COMMAND:9eafb5aa")
//   - name:        Entity text as extracted
//   - entity_type: GLiNER label (COMMAND, PARAMETER, COMPONENT, etc.)
//   - label:       Neo4j label (Command, Parameter, Component, etc.)
//   - source:      "gliner"
//
// GLINER ENTITY TYPE → NEO4J LABEL MAPPING:
//   COMMAND         → Command
//   PARAMETER       → Parameter
//   COMPONENT       → Component
//   PROTOCOL        → Protocol
//   CLOUD_PROVIDER  → CloudProvider
//   STORAGE_CONCEPT → StorageConcept
//   VERSION         → Version
//   PROCEDURE_STEP  → ProcedureStep
//   ERROR           → Error
//   CAPACITY_METRIC → CapacityMetric
//
// EXAMPLE QUERY PATTERN (GraphRAG hybrid retrieval):
//   MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
//   WHERE c.id IN $chunk_ids_from_vector_search
//     AND r.confidence >= 0.5
//   RETURN c, e, r.confidence
//   ORDER BY r.confidence DESC
//
// ===========================================================================
// END OF SCHEMA DEFINITION
// ===========================================================================
