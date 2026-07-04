# Active Path Map

## Query/Retrieval Path

1. Docker starts `src.mcp_server.main:app` via Uvicorn.
2. FastAPI startup initializes config/connections and builds the MCP server from `src/mcp_server/mcp_app.py`.
3. MCP tools call handlers in `src/mcp_server/mcp_tools.py` and shared helpers in `src/mcp_server/mcp_search.py` / `src/mcp_server/mcp_utils.py`.
4. `kb.search`/`kb.retrieve_evidence` candidate retrieval routes to `QueryService.search_sections_light()`.
5. `QueryService` creates providers through `ProviderFactory` and constructs `HybridRetriever` through `_get_7e_retriever()`.
6. `HybridRetriever.retrieve()` resolves `graph_assisted` behavior through `src/query/retrieval_plan.py`, runs multi-vector Qdrant search through `QdrantMultiVectorRetriever`, optionally enriches with graph/structure/cross-doc signals, applies signal pool/ColBERT/reranker/specificity adjustments, then returns chunk results.
7. MCP scratch/evidence helpers store candidate payloads and expose excerpt/expansion/read APIs.

## Ingestion Path

1. `src/ingestion/auto/service.py` watches `/app/data/ingest` and enqueues file jobs in Redis.
2. `src/ingestion/worker.py` consumes jobs, maps host file URIs into `/app/...`, reads content, detects markdown/html, and creates `AtomicIngestionCoordinator`.
3. `AtomicIngestionCoordinator.ingest_document_atomic()` parses, extracts doc tags, references, structural entities, assembles chunks, enriches with GLiNER, computes dense/sparse/title/doc-title/entity-sparse/ColBERT embeddings, validates, and executes the saga.
4. `_execute_atomic_saga()` writes Neo4j document/chunks/entities/mentions/references/structural edges, writes Qdrant vectors, commits Neo4j only after Qdrant success, and compensates Qdrant on failure.
5. Cross-document linking runs post-commit through `CrossDocLinker.link_document()` when enabled and corpus size threshold is met.

## Model/Profile Path

- `config/development.yaml` sets active app defaults and feature flags.
- `config/embedding_profiles.yaml` defines provider profiles and a top-level plan. The plan references Qwen dense + SPLADE sparse + ColBERT, while `development.yaml` and env files often say `bge_m3` or mixedbread/Qwen reranker depending on layer.
- `ProviderFactory` normalizes aliases and prefers explicit/env/config/default precedence for rerank providers.

## Active Retrieval Configuration Observed

| Setting surface | Observed value/pattern | Audit interpretation |
|---|---|---|
| `config/development.yaml` app name | `wekadocs-matrix` | Active config is still WEKA-domain. |
| `search.hybrid.profile` | `graph_assisted` | The intended active query path is multi-vector retrieval plus bounded graph/structure/cross-doc assistance, not the old simple BM25 path. |
| `search.hybrid.mode` | `legacy` | Name is misleading because the active orchestrator is modern `HybridRetriever`; the mode still preserves old branching semantics. |
| `search.hybrid.neo4j_disabled` | `false` | Graph expansion is configured available unless runtime/env blocks it. |
| `search.vector.qdrant.collection_name` | `chunks_multi` | Collection is still generic/legacy; namespace behavior depends on embedding profile settings. |
| Qdrant vector flags | dense + sparse + doc-title sparse + title sparse + entity sparse + ColBERT enabled in config | The active retrieval design assumes multi-vector coverage. Missing vectors must be treated as degraded quality, not normal full-quality retrieval. |
| `colbert_rerank_enabled` and cross-doc `colbert_rerank_before_write` | enabled | Both query-time and cross-doc paths can depend on ColBERT availability. |
| `signal_pool.enabled` | enabled | Candidate pool diversity is part of the intended path before final rerank. |
| `bm25.enabled` | false | Lexical matching is expected to come mostly from learned sparse vectors, not Neo4j full text, unless fallback paths engage. |
| Feature flags | `query_api_weighted_fusion`, `graph_garbage_filter`, `graph_rel_types_wired`, `dedup_best_score`, `structure_aware_expansion`, `signal_diverse_rerank_pool`, `signal_pool_before_colbert`, `precision_focused_rerank_text`, `precision_specificity_adjustment` | These flags materially alter scoring and expansion and must be part of any Nutanix acceptance trace. |

## Active Confidence

High confidence on Python call-path shape. Medium confidence on exact live model behavior because no non-mutating runtime smoke was run and env/config layers conflict.
