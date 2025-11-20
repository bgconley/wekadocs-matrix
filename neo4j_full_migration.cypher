// Auto-generated Neo4j schema dump
// Constraints
CREATE CONSTRAINT `answer_id_unique` FOR (n:`Answer`) REQUIRE (n.`answer_id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `command_id_unique` FOR (n:`Command`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `component_id_unique` FOR (n:`Component`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `concept_id_unique` FOR (n:`Concept`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `configuration_id_unique` FOR (n:`Configuration`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `document_id_unique` FOR (n:`Document`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `document_source_uri_unique` FOR (n:`Document`) REQUIRE (n.`source_uri`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `error_id_unique` FOR (n:`Error`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `example_id_unique` FOR (n:`Example`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `parameter_id_unique` FOR (n:`Parameter`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `procedure_id_unique` FOR (n:`Procedure`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `query_id_unique` FOR (n:`Query`) REQUIRE (n.`query_id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `schema_version_singleton` FOR (n:`SchemaVersion`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `section_id_unique` FOR (n:`Section`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `session_id_unique` FOR (n:`Session`) REQUIRE (n.`session_id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `step_id_unique` FOR (n:`Step`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};

// Indexes
CREATE RANGE INDEX `answer_created_at` FOR (n:`Answer`) ON (n.`created_at`);
CREATE CONSTRAINT `answer_id_unique` FOR (n:`Answer`) REQUIRE (n.`answer_id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `answer_user_feedback` FOR (n:`Answer`) ON (n.`user_feedback`);
CREATE RANGE INDEX `chunk_doc_id` FOR (n:`Chunk`) ON (n.`doc_id`);
CREATE RANGE INDEX `chunk_doc_tag` FOR (n:`Chunk`) ON (n.`doc_tag`);
CREATE RANGE INDEX `chunk_document_id` FOR (n:`Chunk`) ON (n.`document_id`);
CREATE RANGE INDEX `chunk_embedding_dimensions` FOR (n:`Chunk`) ON (n.`embedding_dimensions`);
CREATE RANGE INDEX `chunk_embedding_provider` FOR (n:`Chunk`) ON (n.`embedding_provider`);
CREATE RANGE INDEX `chunk_embedding_version` FOR (n:`Chunk`) ON (n.`embedding_version`);
CREATE VECTOR INDEX `chunk_embeddings_v2` FOR (n:`Chunk`) ON (n.`vector_embedding`) OPTIONS {indexConfig: {`vector.dimensions`: 1024,`vector.similarity_function`: 'COSINE'}, indexProvider: 'vector-1.0'};
CREATE VECTOR INDEX `chunk_entity_embeddings_v1` FOR (n:`Chunk`) ON (n.`entity_vector_embedding`) OPTIONS {indexConfig: {`vector.dimensions`: 1024,`vector.similarity_function`: 'COSINE'}, indexProvider: 'vector-1.0'};
CREATE RANGE INDEX `chunk_heading` FOR (n:`Chunk`) ON (n.`heading`);
CREATE RANGE INDEX `chunk_is_microdoc` FOR (n:`Chunk`) ON (n.`is_microdoc`);
CREATE RANGE INDEX `chunk_lang` FOR (n:`Chunk`) ON (n.`lang`);
CREATE RANGE INDEX `chunk_level` FOR (n:`Chunk`) ON (n.`level`);
CREATE RANGE INDEX `chunk_order` FOR (n:`Chunk`) ON (n.`order`);
CREATE RANGE INDEX `chunk_shingle_hash` FOR (n:`Chunk`) ON (n.`shingle_hash`);
CREATE RANGE INDEX `chunk_source_path` FOR (n:`Chunk`) ON (n.`source_path`);
CREATE RANGE INDEX `chunk_tenant` FOR (n:`Chunk`) ON (n.`tenant`);
CREATE FULLTEXT INDEX `chunk_text_fulltext` FOR (n:`Chunk`) ON EACH [n.`text`] OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-no-stop-words',`fulltext.eventually_consistent`: false}, indexProvider: 'fulltext-1.0'};
CREATE RANGE INDEX `chunk_text_hash` FOR (n:`Chunk`) ON (n.`text_hash`);
CREATE FULLTEXT INDEX `chunk_text_index_v3_bge_m3` FOR (n:`Chunk`|`CitationUnit`) ON EACH [n.`text`, n.`heading`] OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-no-stop-words',`fulltext.eventually_consistent`: false}, indexProvider: 'fulltext-1.0'};
CREATE VECTOR INDEX `chunk_title_embeddings_v1` FOR (n:`Chunk`) ON (n.`title_vector_embedding`) OPTIONS {indexConfig: {`vector.dimensions`: 1024,`vector.similarity_function`: 'COSINE'}, indexProvider: 'vector-1.0'};
CREATE RANGE INDEX `chunk_token_count` FOR (n:`Chunk`) ON (n.`token_count`);
CREATE RANGE INDEX `chunk_updated_at` FOR (n:`Chunk`) ON (n.`updated_at`);
CREATE RANGE INDEX `chunk_version` FOR (n:`Chunk`) ON (n.`version`);
CREATE RANGE INDEX `citation_unit_document_id` FOR (n:`CitationUnit`) ON (n.`document_id`);
CREATE RANGE INDEX `citation_unit_order` FOR (n:`CitationUnit`) ON (n.`order`);
CREATE RANGE INDEX `citation_unit_parent_chunk_id` FOR (n:`CitationUnit`) ON (n.`parent_chunk_id`);
CREATE CONSTRAINT `command_id_unique` FOR (n:`Command`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `command_name` FOR (n:`Command`) ON (n.`name`);
CREATE CONSTRAINT `component_id_unique` FOR (n:`Component`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `component_name` FOR (n:`Component`) ON (n.`name`);
CREATE CONSTRAINT `concept_id_unique` FOR (n:`Concept`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `concept_term` FOR (n:`Concept`) ON (n.`term`);
CREATE CONSTRAINT `configuration_id_unique` FOR (n:`Configuration`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `configuration_name` FOR (n:`Configuration`) ON (n.`name`);
CREATE RANGE INDEX `document_doc_tag` FOR (n:`Document`) ON (n.`doc_tag`);
CREATE CONSTRAINT `document_id_unique` FOR (n:`Document`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `document_last_edited` FOR (n:`Document`) ON (n.`last_edited`);
CREATE RANGE INDEX `document_source_type` FOR (n:`Document`) ON (n.`source_type`);
CREATE CONSTRAINT `document_source_uri_unique` FOR (n:`Document`) REQUIRE (n.`source_uri`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `document_title` FOR (n:`Document`) ON (n.`title`);
CREATE FULLTEXT INDEX `document_title_fulltext` FOR (n:`Document`) ON EACH [n.`title`] OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-no-stop-words',`fulltext.eventually_consistent`: false}, indexProvider: 'fulltext-1.0'};
CREATE RANGE INDEX `document_version` FOR (n:`Document`) ON (n.`version`);
CREATE RANGE INDEX `error_code` FOR (n:`Error`) ON (n.`code`);
CREATE CONSTRAINT `error_id_unique` FOR (n:`Error`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `example_id_unique` FOR (n:`Example`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE FULLTEXT INDEX `heading_fulltext_v1` FOR (n:`Chunk`|`Section`) ON EACH [n.`heading`] OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-no-stop-words',`fulltext.eventually_consistent`: false}, indexProvider: 'fulltext-1.0'};
CREATE LOOKUP INDEX `index_343aff4e` FOR (n) ON EACH labels(n);
CREATE LOOKUP INDEX `index_f7700477` FOR ()-[r]-() ON EACH type(r);
CREATE CONSTRAINT `parameter_id_unique` FOR (n:`Parameter`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE CONSTRAINT `procedure_id_unique` FOR (n:`Procedure`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `procedure_title` FOR (n:`Procedure`) ON (n.`title`);
CREATE RANGE INDEX `query_asked_at` FOR (n:`Query`) ON (n.`asked_at`);
CREATE CONSTRAINT `query_id_unique` FOR (n:`Query`) REQUIRE (n.`query_id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `query_turn` FOR (n:`Query`) ON (n.`turn`);
CREATE CONSTRAINT `schema_version_singleton` FOR (n:`SchemaVersion`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `section_doc_id` FOR (n:`Section`) ON (n.`doc_id`);
CREATE RANGE INDEX `section_document_id_idx` FOR (n:`Section`) ON (n.`document_id`);
CREATE VECTOR INDEX `section_embeddings_v2` FOR (n:`Section`) ON (n.`vector_embedding`) OPTIONS {indexConfig: {`vector.dimensions`: 1024,`vector.similarity_function`: 'COSINE'}, indexProvider: 'vector-1.0'};
CREATE CONSTRAINT `section_id_unique` FOR (n:`Section`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `section_level_idx` FOR (n:`Section`) ON (n.`level`);
CREATE RANGE INDEX `section_order_idx` FOR (n:`Section`) ON (n.`order`);
CREATE RANGE INDEX `section_parent_section_id` FOR (n:`Section`) ON (n.`parent_section_id`);
CREATE RANGE INDEX `session_active` FOR (n:`Session`) ON (n.`active`);
CREATE RANGE INDEX `session_expires_at` FOR (n:`Session`) ON (n.`expires_at`);
CREATE CONSTRAINT `session_id_unique` FOR (n:`Session`) REQUIRE (n.`session_id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};
CREATE RANGE INDEX `session_started_at` FOR (n:`Session`) ON (n.`started_at`);
CREATE RANGE INDEX `session_user_id` FOR (n:`Session`) ON (n.`user_id`);
CREATE CONSTRAINT `step_id_unique` FOR (n:`Step`) REQUIRE (n.`id`) IS UNIQUE OPTIONS {indexConfig: {}, indexProvider: 'range-1.0'};

// SchemaVersion metadata
CREATE (:SchemaVersion {compatibility: 'All v2.1 objects preserved; new indexes are additive', edition: 'community', reform_note: 'Chunk sizes converge to ~400 target with ~32 token overlap; dynamic assembly at query time', validation_note: 'Property existence constraints enforced in application layer (Community Edition)', embedding_provider: 'jina-ai', backup_source: 'v2.1 clean state upgraded to v2.2 hybrid-ready on 2025-11-05', embedding_model: 'jina-embeddings-v3', applied_at: '2025-11-12T20:27:19.534000000+00:00', version: 'v2.2', id: 'singleton', updated_at: '2025-11-12T20:26:53.732000000+00:00', description: 'Phase 7E+: Hybrid retrieval enablement (multi-vector, lexical boost, graph expansion) with small-chunk ingestion', vector_dimensions: 1024});

// ---------------------------------------------------------------------------
// OPTIONAL POST-INGEST RELATIONSHIP BUILDERS (from create_graphrag_schema_v2_2_20251105_guard.cypher)
// NOTE: These rely on ingested data (Chunk/Section nodes). Run manually
// after ingestion if you want to backfill relationships across all docs.
// Guarded versions for per-document builders exist in GraphBuilder._build_typed_relationships.
// ---------------------------------------------------------------------------

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

// -- NEXT/PREV within same document/parent ordered by c.order  [guarded]
// MATCH (c:Chunk)
// WHERE exists(c.text)
// WITH coalesce(c.document_id, c.doc_id) AS d, c.parent_section_id AS p, c
// ORDER BY d, p, c.order
// WITH d, p, collect(c) AS chunks
// UNWIND range(0, size(chunks)-2) AS i
// WITH chunks[i] AS a, chunks[i+1] AS b
// MERGE (a)-[:NEXT]->(b)
// MERGE (b)-[:PREV]->(a);

// -- SAME_HEADING among siblings (bounded fanout)  [guarded]
// MATCH (c:Chunk) WHERE exists(c.text) AND c.heading IS NOT NULL
// WITH coalesce(c.document_id, c.doc_id) AS d, c.parent_section_id AS p, c.heading AS h, collect(c) AS chunks
// UNWIND chunks AS a
// UNWIND chunks AS b
// WITH a, b
// WHERE a.id <> b.id AND a.order < b.order AND a.order + 8 >= b.order
// MERGE (a)-[:SAME_HEADING]->(b);
