# E2E v2.2 Production Validation Suite (Specs Only)

This test suite validates the full v2.2 hybrid RAG pipeline against a
hand-selected production snapshot. It is aligned with our architecture: StructuredChunker,
Neo4j v2.2 schema (no APOC), Qdrant `chunks_multi` named vectors, and HybridRetriever
with multi-vector ANN + BM25 + adjacency expansion.

Important:
- These tests assume a snapshot of production Markdown files will be placed into
  `data/ingest/` and auto-ingested by the watcher/worker pipeline exactly as in prod.
- All tests are marked for integration and expect running services (Neo4j, Qdrant, Redis).
- Tests will capture verbose logs (when run) into this directory under `artifacts/<run-ts>/`.
- Do not run these until you have approved the snapshot and doc_tag and ensured services are up.

Environment variables:
- `E2E_PROD_DOCS_DIR` (required): Local path with hand-selected Markdown files.
- `E2E_PROD_SNAPSHOT_ID` (optional): ID for this run; default generated timestamp.
- `E2E_PROD_DOC_TAG` (required for isolation): Doc tag to stamp and filter during retrieval.
- `E2E_INGEST_SPOOL_PATTERN` (optional): Set to `ready` to use `.part`→`.ready` rename pattern; otherwise direct write.
- `NEO4J_URI`, `QDRANT_HOST`, `QDRANT_PORT`, `REDIS_HOST`, `REDIS_PASSWORD` should point to localhost containers when running locally.

Artifacts (when tests run):
- `artifacts/<snapshot_id>/logs/*` – captured container logs and service metrics.
- `artifacts/<snapshot_id>/manifest.json` – doc list and tag used.

See the doc `docs/hybrid-rag-v2_2-e2e-prod-spec.md` for the full plan and acceptance criteria.
