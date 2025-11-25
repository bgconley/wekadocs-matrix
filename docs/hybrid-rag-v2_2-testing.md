# Hybrid RAG v2.2 Testing & Deliverables (Sections 8–9)

This document captures the concrete verification story for the v2.2 integration
work. It links the canonical spec (Sections 8–9) to runnable artifacts so future
sessions can produce the required evidence without reconstructing the test plan
from scratch.
## 1. Environment Prerequisites

1. Bring up the dev data stack (`docker-compose up -d neo4j qdrant redis`).
2. Export the standard dev secrets (`NEO4J_PASSWORD`, `REDIS_PASSWORD`, `JINA_API_KEY`
   if you want real embeddings/reranking). The new test suite can run air-gapped because it
   ships a deterministic embedding provider, but production parity runs should set
   `EMBEDDINGS_PROVIDER=jina-ai` and, when exercising the reranker path, flip
   `search.hybrid.reranker.enabled=true` in your config. You can also set
   `search.hybrid.reranker.provider=jina-ai` / `search.hybrid.reranker.model=jina-reranker-v3`
   (env overrides like `RERANK_PROVIDER` still work for one-off experiments).
3. Ensure the HuggingFace tokenizer cache is present locally (see
   `config/development.yaml` → `embedding`). Set `TRANSFORMERS_OFFLINE=true` to avoid
   network fetches during CI. The tokenizer backend can now be pinned via
   `tokenizer.backend` (either `hf` or `segmenter`) when env `TOKENIZER_BACKEND`
   isn’t set—use this to temporarily flip the segmenter backend during rollout
   validation without touching deployment scripts.
4. Multi-vector Qdrant collections are the default. When upgrading an environment
   that still has a single-vector collection, toggle
   `search.vector.qdrant.allow_recreate=true` during the migration window so the
   ingestion pipeline can rebuild the collection with the `{content,title,entity}`
   schema automatically. Query routing is controlled via
   `search.vector.qdrant.query_vector_name` (default `content`) and the new
   `search.vector.qdrant.query_strategy` knob (`content_only`, `weighted`,
   `max_field`). Use `weighted` to send scaled named vectors that mirror
   `search.hybrid.vector_fields` weights, or `max_field` to force the heaviest field
   when debugging collection drift.
4. Verify schema + collection bootstrap via the existing tooling:
   - `python scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher`
   - `python scripts/qdrant\ and\ helpers/qdrant_setup_chunks_multi-1546.py`
## 2. Multi-Vector Integration Tests (`tests/v2_2/test_hybrid_rag_v22_integration.py`)

| Test | Purpose | Key Assertions |
| --- | --- | --- |
| `test_ingestion_populates_multi_vector_payloads` | Validates Section 4 + Section 5 deliverables. Uses `GraphBuilder` with the deterministic embedding provider to ingest a synthetic document into Neo4j/Qdrant. | `chunks_multi` points contain named vectors (`content`, `title`, `entity`), canonical payload fields (`doc_tag`, tenant/lang/version, text/shingle hashes), `semantic_metadata` placeholders, and v2.2 doc aliases. |
| `test_hybrid_retrieval_returns_multi_vector_results` | Covers Section 6 retrieval guarantees. Runs the actual `HybridRetriever` (BM25 + ANN + fusion + expansion) against the data produced above. | Multi-vector hits appear in the fused result set, metrics report the configured vector fields, and the winning chunk exposes non-zero vector/fused scores. |
| `test_cleanup_script_generates_report` | Exercises the updated cleanup deliverable (Section 8). Runs `scripts/cleanup-databases.py --dry-run` against live services and inspects the generated JSON report to prove metadata preservation logic. |

### Running only these tests

```bash
export ENV=development
poetry run pytest tests/v2_2/test_hybrid_rag_v22_integration.py -m integration -vv
```

Notes:
- The deterministic embedding provider keeps the suite offline-friendly while still
  exercising named-vector payloads. Set `EMBEDDINGS_PROVIDER=jina-ai` plus
  `JINA_API_KEY` to re-run with the production embedder for final validation.
- The fixtures skip automatically if Neo4j/Qdrant/Redis are unavailable; this makes
  it safe to include the suite in CI without breaking faster unit pipelines.

### Semantic enrichment hook checks
- `tests/v2_2/chunking/test_structured_chunker.py::test_semantic_enrichment_disabled_is_noop`
  confirms that when `chunk_assembly.semantic.enabled` remains `false`, the new
  enrichment hook leaves chunks’ `semantic_metadata` empty and the Prometheus
  counter `semantic_enrichment_total{provider="stub",status="success"}` unchanged.
- `tests/v2_2/chunking/test_structured_chunker.py::test_semantic_enrichment_enabled_emits_metadata_and_metrics`
  toggles the config to `true`, verifying that every emitted chunk carries the stub
  metadata (`entities`, `topics`, `summary`) and that the success counter increases
  per chunk.
- Run the guardrails locally with:
  ```bash
  poetry run pytest tests/v2_2/chunking/test_structured_chunker.py -k semantic_enrichment -vv
  ```
  These tests require no services and provide fast signal that the enrichment hook
  remains properly gated and instrumented.
## 3. Cleanup Workflow Validation

1. **Dry Run (safe default)**
   ```bash
   NEO4J_URI=bolt://localhost:7687 \
   QDRANT_HOST=localhost \
   QDRANT_PORT=6333 \
   REDIS_HOST=localhost \
   REDIS_PASSWORD=${REDIS_PASSWORD:-testredis123} \
   python scripts/cleanup-databases.py --dry-run --report-dir reports/cleanup/dryrun
   ```
   - Confirms metadata detection, collection enumeration, and Redis key-scoping.
   - Produces a timestamped JSON report; attach this to release notes as evidence.
   - Latest evidence (2025-11-09): `reports/cleanup/dryrun/cleanup-report-20251109-172409.json`.

2. **Live Reset (only when Qdrant/Neo4j contain expendable data)**
   ```bash
   python scripts/cleanup-databases.py --report-dir reports/cleanup/live
   ```
   - Deletes only data labels (`Document`, `Section`, `Chunk`, etc.) while preserving
     `SchemaVersion`, `_metadata`, and typed relationship markers.
   - Qdrant points are purged using deterministic UUID selectors; collection schemas
     remain intact.
   - Redis cleanup targets DB #1 by default—the epoch keys in DB #0 stay untouched.
## 4. Deliverable Checklist (Spec §9)

- [x] **Code Integration** – Sections 2–6 already landed; these tests ensure the
      multi-vector pipeline remains verifiable end-to-end.
- [x] **Testing Suite (real services)** – The new pytest module drives Neo4j,
      Qdrant, and Redis without mocks. Each test documents its coverage area and
      depends on the deterministic embedder for offline reproducibility.
- [x] **Cleanup Evidence** – Running the dry-run variant emits a structured report
      suitable for change reviews. Attach the latest JSON output when handing off.
- [x] **Documentation** – This page plus the canonical spec/architecture docs cover
      bootstrap steps, commands, and how to interpret the metrics emitted by the
      tests.

## 5. Retrieval Observability & Metrics

Hybrid retrieval now emits a structured metrics dict per query (and logs a summary
line) capturing the signals we monitor in Grafana. Key fields:

| Metric | Meaning | Usage |
| --- | --- | --- |
| `seed_gated` / `seeds_after_gate` | Counts how many ANN seeds were dropped when we enforce single-document focus. | Alert if gating suddenly spikes (possible doc_tag drift). |
| `microdoc_extras` / `microdoc_tokens` | Number of microdoc chunks appended after the primary budget and their token cost. | Track microdoc reliance; spikes can indicate chunking regressions. |
| `microdoc_present` / `microdoc_used` | Whether microdocs were available and how many made it into the final prompt. | Ensure microdocs don’t crowd out primaries; investigate if `present=1` but `used=0`. |
| `reranker_applied` | 1 when the reranker successfully reorders seeds (`reranker_reason` explains skips). | Build a panel that spot-checks activation and alerts if it stays false while enabled. |
| `expansion_reason`, `expansion_count`, `expansion_cap_hit` | Explain why graph expansion fired and whether it hit configured limits. | Feed into adjacency health dashboards. |

### Capturing the metrics
1. **Logs** – Every call ends with `Hybrid retrieval complete … microdocs=X, time=Yms`.
   Running `pytest tests/v2_2/test_retrieval_edge_cases.py::test_forced_expansion_adds_neighbors -vv -s`
   yields:
   ```
   2025-11-09 17:15:22 [info] Filtered seeds by doc_tag=beta-bd7c: kept 1/1
   2025-11-09 17:15:22 [info] Gating seeds to primary document retrieva: kept 1/1
   2025-11-09 17:15:22 [info] Adjacency expansion: query_tokens=7, expanded=1 chunks
   2025-11-09 17:15:22 [info] Hybrid retrieval complete: ..., microdocs=0, time=42.96ms
   ```
   These lines confirm the metrics update flows end-to-end during the test run.
2. **Grafana/Prometheus** – Ship the metrics dict via your existing log-scrape
   pipeline or emit custom counters (e.g., `hybrid_seed_gated_total`) based on the
   dict fields. Suggested panels: seed gating trend, microdoc utilization, reranker
   adoption (once enabled), and expansion-rate gauges.
3. **Tests** – `tests/v2_2/test_retrieval_edge_cases.py` asserts
   `metrics["microdoc_present"] in (0, 1)` and verifies expansion happens, so CI will
   catch regressions in the metrics payload.

By following the log command above (or tailing production logs) you can grab the raw
metrics samples and feed them into Grafana alerts. When wiring dashboards, filter on
`metrics.seed_gated`, `metrics.microdoc_present`, and `metrics.reranker_applied` to
validate that the v2.2 changes are live.

Future improvements (tracked in context files): expand the reranker dashboards
with p95 latency/error ratios and plug semantic chunk metadata into the ingestion
hook so `semantic_metadata` holds real entities/topics.
