# 05 — Runtime Flows

## Flow 1: MCP Server Startup

**Entry:** `src/mcp_server/main.py`

```
Module-level imports
├── src/shared/__init__.py
│   ├── src/shared/config.py          → init_config(), get_config(), get_settings()
│   └── src/shared/connections.py     → ConnectionManager, initialize_connections()
├── src/shared/observability/logging.py → setup_logging()
├── src/shared/observability/metrics.py → PrometheusMiddleware, setup_metrics()
├── src/shared/observability/tracing.py → setup_tracing()
├── src/mcp_server/mcp_app.py         → build_mcp_server()
├── src/mcp_server/webhooks.py        → APIRouter (/webhooks/*)
├── src/mcp_server/models.py          → Pydantic response models
├── src/monitoring/health.py          → run_startup_health_checks()
├── src/connectors/manager.py         → ConnectorManager
└── src/query/traversal.py            → TraversalService

FastAPI app creation
├── app = FastAPI(...)
├── app.add_middleware(PrometheusMiddleware)
├── app.include_router(webhooks.router, prefix="/webhooks")
└── app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)

startup_event()
├── init_config()                     → (config, settings)
├── setup_logging("INFO")
├── setup_tracing("weka-mcp-server")
├── initialize_connections()
│   └── ConnectionManager.initialize_all()
│       ├── get_neo4j_driver()        → GraphDatabase.driver(bolt://localhost:7687)
│       ├── get_qdrant_client()       → CompatQdrantClient(host, port, prefer_grpc=True)
│       └── get_redis_client()        → aioredis.Redis(pool)
├── run_startup_health_checks()
│   └── HealthChecker.check_all()
│       ├── _check_neo4j_connection()         → session.run("RETURN 1")
│       ├── _check_neo4j_schema_version()     → MATCH (n:SchemaVersion) RETURN n.version
│       ├── _check_neo4j_constraints()        → SHOW CONSTRAINTS
│       ├── _check_neo4j_indexes()            → SHOW INDEXES
│       ├── _check_neo4j_vector_indexes()     → check 4 vector indexes exist
│       ├── _check_qdrant_connection()        → qdrant_client.get_collections()
│       ├── _check_qdrant_collection()        → qdrant_client.get_collection(collection_name)
│       ├── _check_qdrant_dimensions()        → verify vector size matches config
│       └── _check_embedding_config()         → compare embed_dim/model/provider against plan
├── app.state.mcp_session_manager = StreamableHTTPSessionManager(build_mcp_server(), ...)
│   └── build_mcp_server()
│       ├── Server("wekadocs", ...)
│       ├── @server.list_tools() → _list_tools()
│       ├── @server.call_tool() → _call_tool(name, arguments)
│       ├── @server.list_resources() → _list_resources()
│       ├── @server.read_resource() → _read_resource(uri)
│       ├── @server.list_prompts() → _list_prompts()
│       └── @server.read_prompt() → _get_prompt(name, arguments)
├── app.state.mcp_session_manager.run().__aenter__()
└── _build_connector_manager()
    ├── Redis(sync client)
    ├── ConnectorManager(redis, config)
    └── For each enabled connector:
        ├── ConnectorConfig(...)
        ├── CircuitBreaker(...)
        ├── GitHubConnector(config, queue, circuit_breaker)
        └── await connector_manager.start_polling()
            └── asyncio.create_task(_poll_loop(name, connector))

shutdown_event()
├── connector_manager.stop_polling()
├── connector_manager.close()
├── mcp_session_manager_context.__aexit__()
└── close_connections()
    ├── neo4j_driver.close()
    ├── qdrant_client.close()
    └── redis_client.close()
```

**Key files:** `src/mcp_server/main.py`, `src/shared/config.py`, `src/shared/connections.py`, `src/monitoring/health.py`, `src/mcp_server/mcp_app.py`, `src/connectors/manager.py`

**Key symbols:** `FastAPI`, `StreamableHTTPSessionManager`, `ConnectionManager`, `HealthChecker`, `build_mcp_server()`, `ConnectorManager`

---

## Flow 2: Ingestion Worker Startup

**Entry:** `src/ingestion/worker.py`

```
main()
├── setup_logging("INFO")
├── init_tracing("weka-ingestion-worker", "1.0.0", instrument_redis=True)
├── signal.signal(SIGTERM, handle_shutdown)
├── signal.signal(SIGINT, handle_shutdown)
├── loop.set_exception_handler(global_exception_handler)
├── load_config()                          → (config, settings)
├── redis.Redis.from_url(redis_url, decode_responses=True)
├── JobReaper(redis, timeout, interval, max_retries, action, enabled)
│   └── create_monitored_task(reaper.reap_loop(), name="job_reaper")
│       └── JobReaper.reap_loop()
│           └── while True:
│               ├── reap_once()
│               │   ├── _get_processing_jobs()    → redis.lrange(KEY_PROCESSING, 0, -1)
│               │   └── For each job:
│               │       ├── _get_job_age(job_id)
│               │       ├── _requeue_job()  OR  _fail_job()
│               │       └── _update_status(job_id, status)
│               └── await asyncio.sleep(interval)
├── while not shutdown_requested:
│   ├── raw = brpoplpush(KEY_JOBS, KEY_PROCESSING, timeout=1)
│   ├── If raw:
│   │   ├── IngestJob.from_json(raw)
│   │   ├── _update_status(job_id, "PROCESSING")
│   │   └── await process_job(job)
│   └── await asyncio.sleep(0.1)

process_job(job)
├── Validate: job.kind == "file" and job.path exists
├── parse_file_uri(job.path)              → container path
├── open(container_path, "r").read()      → file content
├── Detect format: .md/.markdown → "markdown", .html/.htm → "html"
├── get_config(), get_connection_manager()
├── neo4j_driver = manager.get_neo4j_driver()
├── qdrant_client = manager.get_qdrant_client()
├── AtomicIngestionCoordinator(neo4j_driver, qdrant_client, config)
├── coordinator.ingest_document_atomic(source_uri, content, format=format)
│   └── See Flow 4 for full ingestion detail
├── On success: ack(raw, job_id)          → redis.lrem(KEY_PROCESSING, 0, raw)
├── On failure: fail(raw, job_id, reason, requeue=True)
│   └── redis.lrem(KEY_PROCESSING, 0, raw)
│       redis.rpush(KEY_DLQ, raw)         → Dead letter queue
└── _update_status(job_id, "DONE" or "FAILED")
```

**Key files:** `src/ingestion/worker.py`, `src/ingestion/atomic.py`, `src/ingestion/auto/queue.py`, `src/ingestion/auto/reaper.py`, `src/shared/config.py`

**Key symbols:** `AtomicIngestionCoordinator`, `JobReaper`, `IngestJob`, `brpoplpush`, `ack()`, `fail()`

---

## Flow 3: Query/Retrieval (kb_search)

**Entry:** `src/mcp_server/mcp_app.py` → `kb_search()`

```
kb_search(query, top_k, cursor, page_size, scope, filters, options, mcp_call_context)
├── _get_deps(ctx)                        → Deps(query=QueryService, scratch=ScratchStore, ...)
├── _resolve_session_id(ctx, session_id)  → effective_session
├── _kb_search_candidates(query, top_k, cursor, page_size, scope, filters, options, deps, effective_session)
│   ├── _normalize_scope(scope)           → validate project_id/environment
│   ├── _merge_scope_filters(filters, normalized_scope)
│   ├── Parse options: mode, include_scores, include_debug, max_snippet_chars, max_per_doc
│   ├── _decode_cursor(cursor)            → base64 decode → integer offset
│   ├── deps.query.search_sections_light(query=query, fetch_k=fetch_k, filters=filters, expand=expand)
│   │   └── QueryService.search_sections_light()
│   │       ├── self._rewrite_keyword_query(query)    → detect keyword-stuffing, reformulate
│   │       ├── self._get_7e_retriever()              → lazy HybridRetriever
│   │       └── retriever.retrieve(query=query, query_original=..., top_k=fetch_k, filters=filters, expand=expand)
│   │           └── HybridRetriever.retrieve()
│   │               ├── Step A: embedder.embed_query(query)
│   │               │   → QueryEmbeddingBundle(dense, sparse, colbert)
│   │               ├── Step B: EntityExtractor.extract_entities(query)
│   │               ├── Step C: classify_query_intent(query)
│   │               ├── Step D: QdrantMultiVectorRetriever.search() for each vector field
│   │               │   → content, title, doc_title, entity (dense)
│   │               │   → text-sparse, doc_title-sparse, title-sparse, entity-sparse (sparse)
│   │               │   → late-interaction (ColBERT multivector)
│   │               ├── Step E: BM25Retriever.search() via Neo4j full-text index
│   │               ├── Step F: _gp.expand_with_graph_signals() if expand=True
│   │               ├── Step G: build_signal_pool() for signal-diverse selection
│   │               ├── Step H: _fp.rrf_fusion() or _fp.weighted_fusion()
│   │               ├── Step I: dedup_chunk_results()
│   │               ├── Step J: _ep.expand_neighbors() bounded adjacency
│   │               ├── Step K: _apply_structural_boost_pure()
│   │               ├── Step L: _rp.rerank_candidates() (cross-encoder)
│   │               └── Step M: _rp.colbert_score_candidates()
│   │               → Returns (List[ChunkResult], metrics_dict)
│   ├── _dedupe_by_doc(chunks, max_per_doc)
│   ├── Slice: deduped[offset : offset + effective_limit]
│   ├── For each chunk:
│   │   ├── Compute score from rerank_score/fused_score/vector_score/bm25_score
│   │   ├── Determine source label: "reranked", "graph_expanded", "rrf_fusion", etc.
│   │   ├── _build_preview(chunk.text, query, max_snippet_chars)
│   │   ├── deps.scratch.put(effective_session, passage_id, scratch_payload)
│   │   └── Build result dict
│   └── Returns (payload, diagnostic_context)
├── _new_budget()                         → ContextBudgetManager(token_budget=14000, byte_budget=524288)
├── _apply_budget(payload, budget, "seeds")
├── _finalize_payload("kb_search", payload, ...)
│   ├── Adds session_id, partial, limit_reason, meta.usage
│   └── Observes Prometheus metrics
└── _emit_diagnostics(...)                → write retrieval diagnostic if enabled
```

**Key files:** `src/mcp_server/mcp_app.py`, `src/mcp_server/query_service.py`, `src/query/hybrid_retrieval.py`, `src/query/fusion_pipeline.py`, `src/query/rerank_pipeline.py`, `src/services/context_budget_manager.py`

**Key symbols:** `kb_search()`, `_kb_search_candidates()`, `QueryService.search_sections_light()`, `HybridRetriever.retrieve()`, `ContextBudgetManager`

---

## Flow 4: Document Ingestion

**Entry:** `src/ingestion/atomic.py` → `AtomicIngestionCoordinator.ingest_document_atomic()`

```
ingest_document_atomic(source_uri, content, format)
├── saga_id = str(uuid.uuid4())
├── _tracer.start_as_current_span("ingest_document", ...)
├── logger.info("ingestion_started", saga_id=saga_id, ...)
├── _prepare_ingestion(source_uri, content, format, ...)
│   ├── Parsing:
│   │   ├── If format == "markdown":
│   │   │   ├── parse_markdown(source_uri, content)
│   │   │   │   ├── Route to markdown_it_parser.parse_markdown() (active)
│   │   │   │   └── OR markdown.parse_markdown() (legacy fallback)
│   │   │   └── Returns {"Document": {...}, "Sections": [...]}
│   │   └── If format == "html":
│   │       └── parse_html(source_uri, content)
│   ├── Extract doc_tag, doc_category, snapshot_scope from headers and path
│   ├── extract_entities(sections)
│   │   ├── extract_commands(section)
│   │   ├── extract_configurations(section)
│   │   ├── extract_procedures(section)
│   │   └── Returns (entities_dict, mentions_list)
│   ├── extract_references(content, doc_chunk_id)
│   │   ├── Extract hyperlinks from raw markdown
│   │   └── If config.references.enabled: extract_chunk_references(sections)
│   ├── get_chunk_assembler(config.ingestion.chunk_assembly)
│   │   ├── Creates StructuredChunker or SemanticChunkerAssembler or GreedyCombiner
│   │   └── assembler.assemble(document_id, sections)
│   ├── If config.ner.enabled: enrich_chunks_with_entities(sections)
│   │   └── GLiNER NER enrichment
│   ├── Creates GraphBuilder(self.neo4j_driver, config, self.qdrant_client)
│   └── Returns {"document": ..., "sections": ..., "entities": ..., "mentions": ..., "references": ..., "builder": ...}
├── Apply structural entity quality gate: is_excluded_structural_entity(entity_name)
├── Merge GLiNER mentions with structural mentions, deduplicate by entity_id
├── Attach _mentions to each section
├── logger.info("document_parsed", ...)
├── logger.info("chunking_complete", ...)
├── Pre-commit validation:
│   └── IngestionValidator.validate_pre_ingest(document_id, chunks, entities, mentions)
│       ├── Validate chunk IDs present and unique
│       ├── Validate entity-chunk references
│       ├── Validate Qdrant collection exists
│       └── Validate document_id consistency
├── _compute_embeddings(document, sections, entities, builder)
│   ├── Build entity_id_to_name lookup
│   ├── Get dense, sparse, ColBERT embedders from ProviderFactory
│   ├── Initialize TokenizerService()
│   ├── Batch sections by token budget (BGE_M3_SAFE_INPUT_TOKENS or EMBEDDING_MAX_TOKENS, default 8000)
│   ├── For each batch:
│   │   ├── dense_embedder.embed_documents(texts) → dense vectors
│   │   ├── If enable_sparse: dense_embedder.embed_sparse(texts) → sparse vectors
│   │   └── If enable_colbert: dense_embedder.embed_colbert(texts) → ColBERT vectors
│   ├── Validate dimensions: validate_embedding_metadata()
│   ├── Canonicalize: canonicalize_embedding_metadata()
│   └── Returns {"sections": {section_id → {content, title, sparse?, colbert?}}, "entities": {}, "stats": {...}}
├── _execute_atomic_saga(saga_id, document, sections, entities, mentions, references, embeddings, builder)
│   ├── SagaContext(saga_id=saga_id, document_id=document_id)
│   ├── Step 1: neo4j_session = self.neo4j_driver.session()
│   ├── Step 2a: _neo4j_upsert_document(neo4j_tx, document)
│   ├── Step 2b: _neo4j_upsert_sections(neo4j_tx, document_id, sections)
│   ├── Step 2c: _neo4j_upsert_entities(neo4j_tx, entities)
│   ├── Step 2d: Prune structural entities with zero mentions
│   ├── Step 2e: _neo4j_create_mentions(neo4j_tx, all_mentions)
│   ├── Step 2f: _neo4j_create_references(neo4j_tx, references)
│   ├── Step 2g: _neo4j_upsert_embedding_metadata(neo4j_tx, sections, embeddings, builder)
│   ├── Step 2h: build_structural_edges_in_tx(neo4j_tx, document_id, skip_has_chunk=True)
│   │   ├── Creates NEXT_CHUNK, PARENT_HEADING, CHILD_OF, PARENT_OF, NEXT relationships
│   ├── Step 3: _qdrant_upsert_vectors(document, sections, embeddings, builder)
│   │   ├── Upserts points with all vector types: content, title, doc_title, text-sparse, title-sparse, entity-sparse, late-interaction
│   ├── Step 4: neo4j_tx.commit()  ← ONLY after Qdrant succeeds
│   ├── Step 5: _create_cross_doc_links(document_id, document, sections, embeddings)
│   │   └── CrossDocLinker.link_document()
│   └── Compensation (exception handler):
│       ├── If Neo4j tx still open: neo4j_tx.rollback()
│       ├── If Qdrant points written: _compensate_qdrant(written_qdrant_points, builder)
│       └── Returns AtomicIngestionResult(success=False, error=str(e), compensated=...)
└── Returns AtomicIngestionResult(success=True, ...)
```

**Key files:** `src/ingestion/atomic.py`, `src/ingestion/parsers/markdown_it_parser.py`, `src/ingestion/extract/__init__.py`, `src/ingestion/chunk_assembler.py`, `src/ingestion/structural_edges.py`, `src/ingestion/saga.py`, `src/providers/factory.py`, `src/services/cross_doc_linking.py`

**Key symbols:** `AtomicIngestionCoordinator.ingest_document_atomic()`, `_prepare_ingestion()`, `_compute_embeddings()`, `_execute_atomic_saga()`, `GraphBuilder`, `IngestionValidator`

---

## Flow 5: Schema Bootstrap

**Entry:** `bootstrap_schema.py`

```
bootstrap_schema.py
├── os.environ.setdefault("EMBEDDINGS_PROFILE", "bge_m3")
├── os.environ["MANAGE_QDRANT_SCHEMA_ON_INIT"] = "true"
├── Reset cached config: config_module._config = None
├── get_config()                          → reload with new env vars
├── get_embedding_settings()              → resolve embedding profile settings
├── qcfg = config.search.vector.qdrant
├── qcfg.enable_sparse = True
├── qcfg.enable_colbert = True
├── Derive namespace suffix: _slugify_identifier(settings.version or settings.profile or "")
├── Append suffix to qcfg.collection_name if not present
├── QdrantClient(url="http://127.0.0.1:6333")
├── GraphBuilder(driver=None, config=config, qdrant_client=client)
│   └── GraphBuilder.__init__()
│       ├── build_qdrant_schema()         → src/shared/qdrant_schema
│       │   ├── Define vector params (dense 1024-D, sparse, ColBERT multivector)
│       │   ├── Define sparse vector params
│       │   ├── HNSW config (m=48, ef_construct=256)
│       │   └── Optimizers config
│       ├── If MANAGE_QDRANT_SCHEMA_ON_INIT=true:
│       │   ├── qdrant_client.create_collection() OR reconcile existing
│       │   ├── Create named vectors: content, title, doc_title, text-sparse, title-sparse, entity-sparse, late-interaction
│       │   └── Create HNSW and skipgram indexes
│       └── ensure_schema_version(neo4j_driver, expected_version)  ← skipped (driver=None)
```

**Key files:** `bootstrap_schema.py`, `src/shared/config.py`, `src/shared/qdrant_schema.py`, `src/shared/schema.py`

**Key symbols:** `GraphBuilder.__init__()`, `build_qdrant_schema()`, `ensure_schema_version()`

---

## Open Questions / Ambiguous Branches

1. **Parser routing:** `parsers/__init__.py` routes between `markdown.py` and `markdown_it_parser.py`. The exact routing logic and fallback behavior needs verification. Is markdown_it_parser always preferred, or is there a config flag?

2. **Chunk assembler selection:** `get_chunk_assembler()` creates different assembler types based on config. The exact config keys and default behavior need verification.

3. **Embedder selection:** `ProviderFactory.create_embedding_provider_for_role()` selects embedders by role (dense, sparse, colbert). The mapping from embedding profile to provider is config-driven and complex. Which providers are actually active for the `bge_m3` profile?

4. **Graph expansion in retrieval:** `HybridRetriever.retrieve()` calls `_gp.expand_with_graph_signals()` if `expand=True`. The expansion logic branches based on query intent and entity extraction results. The exact expansion depth and fanout limits need verification.

5. **Reranking activation:** The reranking pipeline (`rerank_pipeline.py`) is ACTIVE but depends on a reranker being configured. Is the local mxbai-reranker or Jina reranker active in the current deployment?

6. **Cross-document linking:** `_create_cross_doc_links()` is called at the end of ingestion. The `CrossDocLinker` uses dense similarity to find related documents. Is this feature enabled in the current config?

7. **Security gap:** `src/mcp_server/security/auth.py` and `rate_limiter.py` are marked DEAD but exist. Is there a plan to wire them in, or is the MCP server intentionally unauthenticated?
