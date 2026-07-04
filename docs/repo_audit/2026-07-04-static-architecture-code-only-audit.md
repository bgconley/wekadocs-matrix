# Static Architecture And Code-Only Audit

Date: 2026-07-04
Repository: `/Users/brennanconley/vibecode/wekadocs-matrix`
Scope: current worktree, static architecture/code/config/deployment review only.
Explicit exclusion: no live embedder, reranker, Qdrant, Neo4j, Redis, or MCP runtime validation was available for this pass.

GitNexus status: `npx gitnexus status` reported repository `wekadocs-matrix`, indexed commit `7745c92`, current commit `7745c92`, status up-to-date. The GitNexus MCP tools were not exposed in this session, so the authoritative evidence below is direct source/config/deployment citation plus GitNexus CLI status.

Important worktree note: the worktree was already dirty before this report. I did not revert or normalize pre-existing changes.

## 1. Executive Summary

Architecture confidence: High confidence for the active Docker/MCP/ingestion/retrieval path, medium confidence for legacy/dead classification, and unknown for live model quality because embedders and rerankers were unavailable.

Reliable active path identified: Confirmed. The best functional path is:

1. `docker-compose.yml` starts `mcp-server`, `ingestion-service`, `ingestion-worker`, Neo4j, Qdrant, Redis, and Alloy.
2. `mcp-server` runs `python -m uvicorn src.mcp_server.main:app` from `docker/mcp-server.Dockerfile`.
3. `src.mcp_server.main` mounts Streamable MCP at `/_mcp`, builds the low-level MCP server through `src.mcp_server.mcp_app::build_mcp_server`, and defaults legacy REST `/mcp/*` off.
4. Production MCP tool listing exposes `kb.retrieve_evidence`, `kb.read_excerpt`, and `graph.expand`.
5. `kb.retrieve_evidence` calls `_kb_search_candidates`, which calls `QueryService.search_sections_light`, which calls `HybridRetriever.retrieve`.
6. `HybridRetriever` resolves the `graph_assisted` profile, builds `QdrantMultiVectorRetriever`, runs Query API weighted/RRF multi-vector retrieval, graph signals, ColBERT if available, signal-pool/cross-encoder rerank if available, context-budget enforcement, and returns chunks/metrics.
7. The active evidence-pack path returns quotes, coverage, diagnostics, and trace ids. It does not generate a natural-language answer with an answer LLM.

Biggest risks:

- WEKA residue is runtime-connected, not merely documentation residue. It appears in MCP instructions, embedding query prefixes, reranker instructions, query reformulation, query-intent rules, NER labels, cache keys, service names, observability, deployment names, and the current corpus layout.
- Nutanix readiness is not present in the active code/config. Static scan across active source/config/deploy/docs returned zero Nutanix files and 135 WEKA-related files.
- Config authority is split: `config/development.yaml` claims BGE-M3 defaults, while `config/embedding_profiles.yaml` has a manifest `plan` that overrides the effective dense/sparse/ColBERT roles to Qwen3/SPLADE/ColBERT and causes `EMBEDDINGS_PROFILE` to be ignored.
- MCP tool profile filtering only hides tools from `list_tools`; `call_tool` still accepts the full tool map for backward compatibility.
- Several fallbacks silently change quality semantics: Query API to legacy vector search, reranker to fused ordering, ColBERT to pass-through, GLiNER to no entity enrichment, and cross-doc linking to fail-open post-commit.

Biggest cleanup wins:

- Introduce a Nutanix domain pack and remove WEKA strings from all runtime prompts, model prefixes, NER labels, query intent rules, URI names, cache namespaces, telemetry, and deployment names.
- Freeze the active production MCP surface and enforce the tool profile in `call_tool`, not only `list_tools`.
- Consolidate embedding/reranker config into one manifest-owned config surface and delete stale BGE/Qwen/mxbai contradictions.
- Split the god modules after locking the active path with fresh static and live validation.

Safe to refactor incrementally: Yes, with discipline. The active path is identifiable and can be wrapped with contract tests before cutting. Do not begin broad deletion until the active path is frozen and legacy entrypoints are quarantined.

Recommended first implementation milestone: "Nutanix Domain Pack + Active Path Freeze." Create a single `DomainConfig` or equivalent, switch active prompts/instructions/query intent/NER labels/cache URI naming to Nutanix, and add a static guard that fails on WEKA in runtime-connected files except approved compatibility aliases.

## 2. Repository Inventory

| Area | Classification | Evidence | Notes |
| --- | --- | --- | --- |
| `src/mcp_server/` | Active but problematic | `src/mcp_server/main.py:66-82`, `src/mcp_server/main.py:343-352`, `src/mcp_server/mcp_app.py:192-221`, `src/mcp_server/mcp_tools.py:1129-1418` | Active MCP server, evidence tools, scratch/diagnostics. Runtime WEKA prompts remain. |
| `src/query/` | Active but problematic | `src/query/hybrid_retrieval.py:256-323`, `src/query/vector_backends.py:653-680`, `src/query/retrieval_plan.py:150-193` | Main retrieval pipeline. Strong architecture pieces, but fallbacks and domain contamination remain. |
| `src/ingestion/` | Active but god-module-heavy | `src/ingestion/worker.py:177-242`, `src/ingestion/atomic.py:182-242`, `src/ingestion/atomic.py:856-1040`, `src/ingestion/atomic.py:2000-2036` | Redis worker into atomic Neo4j/Qdrant saga. Cross-doc linking is post-commit fail-open. |
| `src/providers/` | Active but config-sprawled | `src/providers/factory.py:53-77`, `src/providers/factory.py:303-465`, `src/providers/embeddings/embedding_service.py:191-234` | Provider factory owns embedding/rerank resolution. Legacy aliases and WEKA query instruction flow remain. |
| `src/services/cross_doc_linking.py` | Active but ancillary/fail-open | `src/ingestion/atomic.py:259-390`, `src/services/cross_doc_linking.py:1106-1264` | Creates `RELATED_TO` edges after main ingestion commit. Useful but should not be treated as core atomic success. |
| `config/development.yaml` | Config source of truth candidate, contradictory | `config/development.yaml:1-3`, `config/development.yaml:12-25`, `config/development.yaml:148-160` | Claims single source of truth and BGE defaults, but effective embedding plan is elsewhere. |
| `config/embedding_profiles.yaml` | Effective embedding plan source | `config/embedding_profiles.yaml:1-12`, `src/shared/config.py:1336-1378` | Manifest `plan` overrides config profile and ignores `EMBEDDINGS_PROFILE`. |
| `docker-compose.yml`, `docker/` | Active deployment source | `docker-compose.yml:126-228`, `docker-compose.yml:230-341`, `docker-compose.yml:343-455`, `docker/mcp-server.Dockerfile:86-87` | Active compose path. Service names remain `weka-*`. |
| `deploy/`, `monitoring/`, `config/alloy/`, `config/grafana/` | Build/deploy/ops artifacts, WEKA-specific | `config/alloy/config.alloy:18-34`, `deploy/monitoring/prometheus-alerts.yaml:2-19` | Useful operational structure but needs Nutanix rename and metric label migration. |
| `docs/architecture/`, `docs/plans/`, `docs/session-notes/`, `docs/cdx-outputs/` | Context/planning artifacts | `docs/architecture/2026-03-04-end-to-end-architecture.md:20-35`, `docs/session-notes/2026-03-04-evidence-pack-mcp-modernization.md:13-24` | Useful hypotheses only. Some claims conflict with current compose/code. |
| `docs/repo_audit/00_*.md` through `13_*.md` | Generated audit artifacts | directory listing shows existing shards | Treat as previous analysis, not source of truth. |
| `scripts/` | Mixed active admin, legacy, eval, migration | `scripts/backfill_cross_doc_edges.py`, `scripts/eval/run_eval.py`, `scripts/neo4j/*` | Keep only explicit admin/eval paths after entrypoint freeze. |
| `patched/` | Legacy/alternate source candidate | `patched/atomic_patched_v3.py` | Quarantine unless proven referenced. |
| `tests/` | Test artifact, ignored as truth | 261 files under `tests/` by inventory | Do not infer intended architecture from tests. Quarantine/rewrite strategy below. |
| `hf-cache/`, `.venv*`, caches, reports | Generated/cached artifacts | directory inventory | Inventory existence only; do not use as architecture truth. |

## 3. Context/Plan Chronology

| Period/file | Key claims | Code confirmation | Classification |
| --- | --- | --- | --- |
| 2025-11 to 2025-12 `docs/cdx-outputs/*` | GraphRAG, atomic ingestion, GLiNER, references, ColBERT, sparse/title/entity vectors, schema migration. | Many concepts remain wired: references extraction in `src/ingestion/atomic.py:926-995`, GLiNER enrichment in `src/ingestion/atomic.py:1013-1029`, sparse/ColBERT plan in `src/shared/config.py:1773-1848`. | Historical but partially active. WEKA-specific legacy. |
| `docs/plans/2026-03-01-codebase-pruning-and-modernization-plan.md` plus June cleanup commits | Cleanup/removal phases and status annotations. | Git log shows June cleanup commits; source still has large god modules and legacy paths. | Useful cleanup intent, incomplete in current worktree. |
| `docs/plans/2026-03-03-evidence-pack-architecture-mcp-modernization-plan.md` | Evidence-pack-first MCP path, production 3 tools, do not touch legacy REST/search_documentation. | Confirmed by `src/mcp_server/mcp_app.py:161-188`, `src/mcp_server/mcp_tools.py:1129-1418`, and legacy REST default off in `src/mcp_server/main.py:74-76`. | Likely current guidance, confirmed with caveats. |
| `docs/session-notes/2026-03-04-evidence-pack-mcp-modernization.md` | Server-orchestrated evidence path, retrieval depth decoupled from quotes, graph enrichment, LLM reformulation/domain prefixes. | Evidence path and depth clamp confirmed in `src/mcp_server/mcp_tools.py:1165-1172`; query reformulator confirmed in `src/mcp_server/query_service.py:646-735`. | Current-ish but WEKA-specific and partly risky. |
| `docs/architecture/2026-03-04-end-to-end-architecture.md` | WekaDocs purpose-built for WEKA, sidecar/GPU gateway topology, model stack, evidence pack. | Evidence pack confirmed. Sidecar topology conflicts with current `docker-compose.yml:7-10` and comment at `docker-compose.yml:17-18` saying sidecars removed. | Useful architecture reference, partially stale, WEKA-specific. |
| `docs/session-notes/2026-03-04-integration-deployment-retrieval-tuning.md` | March live tuning noted metadata retrieval failures and WEKA-specific model behavior. | Static code still contains WEKA query reformulation/instructions. Live result claims not reused as proof. | Historical runtime context only; useful risk hypothesis. |
| Existing `docs/repo_audit/*.md` | Prior repo audit shards. | Not treated as authority in this pass. | Generated artifact; compare later if useful. |

Conflict preserved: March architecture documents describe Tailscale sidecar networking, but current compose explicitly uses bridge `weka-net` and notes sidecars removed (`docker-compose.yml:7-18`). The code/compose is more authoritative for current startup.

## 4. Entrypoint Map

| Entrypoint | Starts how | Loads | Activated path | Status/confidence |
| --- | --- | --- | --- | --- |
| MCP HTTP server | `docker/mcp-server.Dockerfile:86-87` -> `python -m uvicorn src.mcp_server.main:app` | `src/shared/config.py::init_config`, FastAPI app, Streamable MCP session manager | `src.mcp_server.main` -> `build_mcp_server` -> MCP tools | Active, confirmed. |
| MCP Streamable HTTP transport | Mounted at `/_mcp` | `src/mcp_server/main.py:120-151`, startup at `src/mcp_server/main.py:343-352` | Low-level MCP server from `mcp_app.py` | Active, confirmed. |
| Legacy MCP REST | `/mcp/initialize`, `/mcp/tools/list`, `/mcp/tools/call` | `src/mcp_server/main.py:517-660` | `QueryService.search` for `search_documentation` | Legacy, default off, reachable if `MCP_HTTP_LEGACY_REST_ENABLED=true`. |
| STDIO MCP server | `python -m src.mcp_server.stdio_server` | `src/mcp_server/stdio_server.py:118-129` | Same `build_mcp_server` as HTTP | Standalone active candidate, not in compose. High confidence. |
| Ingestion service | `docker-compose.yml:343-455`, command `python -m src.ingestion.auto.service` | `src/ingestion/auto/service.py:27-73`, `FileSystemWatcher` | Watches/enqueues docs into Redis | Active, confirmed. |
| File watcher | Created on startup | `src/ingestion/auto/service.py:86-99`, `src/ingestion/auto/watchers.py:130-210` | `.ready`/direct/auto modes enqueue jobs | Active, confirmed. |
| Ingestion worker | `docker-compose.yml:230-341`, command `python -m src.ingestion.worker` | Config, Redis queue, connection manager | `process_job` -> `AtomicIngestionCoordinator.ingest_document_atomic` | Active, confirmed. |
| Cross-doc backfill/admin scripts | `scripts/backfill_cross_doc_edges.py`, `scripts/batch_crossdoc_link.py` | Neo4j/Qdrant/config | Batch `RELATED_TO` maintenance | Admin/legacy-active candidate. Needs operator entrypoint review before deletion. |
| Evaluation scripts | `scripts/eval/*`, `scripts/run_canonical_retrieval_benchmark.py`, etc. | Ad hoc config/runtime | Retrieval benchmarks | Ambiguous; useful future validation, not active application path. |
| CI | `.github/workflows/ci.yml` | lint/type/test/deploy tasks | CI/CD | Build artifact; WEKA naming remains. |

## 5. Active Application Path

| Stage | Active files/symbols | Upstream -> downstream | Config/env/flags | Evidence/confidence | Issues |
| --- | --- | --- | --- | --- | --- |
| Application startup | `docker-compose.yml` and `docker/mcp-server.Dockerfile` | compose -> uvicorn -> `src.mcp_server.main:app` | `CONFIG_PATH`, `ENV`, DB env, model env | `docker-compose.yml:126-228`, `docker/mcp-server.Dockerfile:86-87`; confirmed | WEKA container/network/service names. |
| Config load | `src.shared.config::load_config`, `apply_embedding_profile`, `get_embedding_plan` | import/startup -> config object/settings | `CONFIG_PATH`, `EMBEDDINGS_PROFILE`, `EMBEDDING_PROFILES_PATH`, strict flags | `src/shared/config.py:1625-1667`, `src/shared/config.py:1336-1378`, `src/shared/config.py:1773-1848`; confirmed | `EMBEDDINGS_PROFILE` ignored when manifest plan exists. |
| MCP server construction | `src.mcp_server.mcp_app::build_mcp_server` | FastAPI startup or STDIO -> MCP server | `MCP_TOOL_PROFILE`, `ENABLE_LEGACY_SEARCH_DOCUMENTATION` | `src/mcp_server/mcp_app.py:192-221`; confirmed | Tool call bypass: full map callable even if not listed. |
| Production tool surface | `TOOL_PROFILES["production"]` | `list_tools` -> 3 tools | `MCP_TOOL_PROFILE=production` default | `src/mcp_server/mcp_app.py:158-188`; confirmed | Runtime instructions say WEKA. |
| Evidence tool | `mcp_tools.kb_retrieve_evidence` | MCP call -> `_kb_search_candidates` | evidence depth/env budget flags | `src/mcp_server/mcp_tools.py:1129-1418`; confirmed | Static only; graph enrichment off by default in evidence tool env. |
| Candidate search helper | `src.mcp_server.mcp_search::_kb_search_candidates` | evidence/search tools -> QueryService | scope, filters, `neo4j_disabled` | `src/mcp_server/mcp_search.py` and `mcp_utils.py:91-93`; high confidence | `_neo4j_disabled` reads the wrong config path in `mcp_utils.py`. |
| Query service | `QueryService.search_sections_light` | MCP helper -> `HybridRetriever.retrieve` | config search/hybrid | `src/mcp_server/query_service.py:287-329`; confirmed | Query rewriting can inject WEKA wording. |
| Query rewriting | `_llm_reformulate`, `_heuristic_reformulate` | QueryService -> gateway or fallback | `EMBEDDING_BASE_URL` | `src/mcp_server/query_service.py:646-735`; confirmed | WEKA prompt and fallback text contaminate Nutanix. |
| Retrieval plan | `resolve_retrieval_plan` | `HybridRetriever.__init__` -> immutable plan | `search.hybrid.profile`, profile overrides, feature flags | `src/query/retrieval_plan.py:150-193`, `config/development.yaml:67-95`; confirmed | Good control-plane idea; legacy flags still present. |
| Query intent | `classify_query_intent` | retriever -> adaptive weights/reranker text | built-in term sets | `src/query/query_intent.py:19-90`, `src/query/query_intent.py:147-154`, `src/query/query_intent.py:224-330`; confirmed | WEKA CLI pattern hardcoded. |
| Vector retrieval | `QdrantMultiVectorRetriever.search` | HybridRetriever -> Qdrant Query API | qdrant Query API flags, field weights | `src/query/vector_backends.py:653-680`, `src/query/vector_backends.py:1124-1402`; confirmed | Query API failure falls back to legacy search silently enough to change semantics. |
| Embedding bundle | `_build_query_bundle` | vector retriever -> dense/sparse/ColBERT providers | embedding plan roles | `src/query/vector_backends.py:985-1042`; confirmed static | Live model behavior unknown in this pass. |
| Graph signals | `HybridRetriever` + `graph_pipeline` | fused candidates -> boosts/RELATED_TO/entity channel | graph profile booleans, Neo4j | `src/query/hybrid_retrieval.py:918-985`, `src/query/retrieval_plan.py:110-125`; high confidence | Domain-specific anchors and graph quality need Nutanix retuning. |
| ColBERT rerank | `_run_colbert`, `rerank_pipeline.colbert_rerank` | candidates -> late interaction ordering | `use_colbert`, Qdrant vectors | `src/query/hybrid_retrieval.py:989-1040`, `src/query/rerank_pipeline.py:443-482`; confirmed static | If vectors unavailable, candidates pass through. Live unavailable. |
| Cross-encoder rerank | `rerank_pipeline.apply_reranker` | signal pool/ColBERT candidates -> ranked seeds | reranker config/env | `src/query/hybrid_retrieval.py:1098-1185`, `src/query/rerank_pipeline.py:112-387`; confirmed static | Uses WEKA instruction; hardcoded 4096 token batch limit conflicts with config 8192/6000. |
| Context assembly | `HybridRetriever._enforce_context_budget`, `ContextAssembler` | final chunks -> budgeted context | `answer_context_max_tokens` | `src/query/hybrid_retrieval.py:1357-1375`, `src/query/context_assembly.py:112-261`; confirmed | Citation labels may degrade to headings only. |
| Evidence output | `_extract_evidence_from_passages`, trace, coverage | evidence tool -> JSON payload | MCP budget env | `src/mcp_server/mcp_tools.py:1338-1418`; confirmed | Good design; domain/URI names still `wekadocs://`. |
| Legacy response path | `QueryService.search` + `build_response` | REST/legacy tool -> formatted response | legacy REST flag; hybrid enabled | `src/mcp_server/query_service.py:737-1128`, `src/query/response_builder.py:213-345`; confirmed legacy-active | Answer is formatted search result, not LLM-generated. |
| Ingestion source watch | `FileSystemWatcher` | ingestion service -> Redis job | `INGEST_WATCH_DIR`, `INGEST_WATCH_MODE` | `src/ingestion/auto/service.py:27-99`, `src/ingestion/auto/watchers.py:130-210`; confirmed | Default tag `wekadocs`; source assumptions need Nutanix. |
| Worker processing | `process_job` | Redis job -> atomic coordinator | Redis/DB env | `src/ingestion/worker.py:177-242`, `src/ingestion/worker.py:386-455`; confirmed | Host path mapping hardcodes `wekadocs-matrix`. |
| Parsing/metadata/chunking | `_prepare_ingestion` | worker -> parser -> assembler | parser/chunk config | `src/ingestion/atomic.py:856-1040`, `config/development.yaml:257-300`; confirmed | doc_tag/category assumptions generic enough but corpus-specific. |
| Embedding/indexing | `_compute_embeddings`, `_execute_atomic_saga` | chunks -> providers -> Neo4j/Qdrant | embedding plan, Qdrant flags, strict modes | `src/ingestion/atomic.py:1128-1253`, `src/ingestion/atomic.py:2000-2036`; confirmed static | Dense required; sparse/ColBERT can degrade depending flags. |
| Cross-doc linking | `_create_cross_doc_links` -> `CrossDocLinker.link_document` | post-commit ingestion -> `RELATED_TO` edges | `ingestion.cross_doc_linking.*` | `src/ingestion/atomic.py:259-390`, `src/services/cross_doc_linking.py:1106-1264`; confirmed | Fail-open and quality-gate bypasses on missing ColBERT vectors. |

## 6. Alternative And Legacy Paths

| Path | Files/symbols | Status | Reuse/delete/quarantine recommendation |
| --- | --- | --- | --- |
| Legacy REST MCP | `src/mcp_server/main.py:517-660` | Default off, env-reachable | Quarantine behind explicit compatibility setting; remove after MCP clients are migrated. |
| `search_documentation` tool | `src/mcp_server/mcp_tools.py:676-750`, appended only at `src/mcp_server/mcp_tools.py:2307-2318` | Legacy, disabled by default | Keep one release only if needed; otherwise delete. |
| Legacy `HybridSearchEngine` | `src/mcp_server/query_service.py:192-255`, `src/mcp_server/query_service.py:996-1050`, `src/query/hybrid_search.py` | Reachable if `search.hybrid.enabled=false` | Quarantine as `legacy_retrieval/` or delete after active path tests. |
| STDIO server | `src/mcp_server/stdio_server.py:118-129` | Standalone, not compose | Keep if Claude Desktop direct transport still needed; it shares active tool server. |
| Backfill scripts | `scripts/backfill_cross_doc_edges.py`, `scripts/batch_crossdoc_link.py` | Operator/admin candidates | Keep only as documented admin commands; move under `src/admin` or `scripts/admin`. |
| Old schema/migration scripts | `scripts/neo4j/*`, `migration/*` | Mixed historical and admin | Archive dated historical scripts; keep only current schema bootstrap. |
| `patched/atomic_patched_v3.py` | alternate patched source | Likely dead | Quarantine/delete after import scan confirms no references. |
| Existing test suite | `tests/` | Test artifact only | Quarantine and rewrite from active path. Do not rely on it for behavior. |
| Existing generated reports/context | `reports/`, `docs/cdx-outputs/`, existing `docs/repo_audit/` | Context/generated | Keep as archive, not source of truth. |

## 7. WEKA-To-Nutanix Migration Residue

Static scan over active source/config/deploy/docs searched in this pass found zero Nutanix files and 135 WEKA-related files.

| Residue | Classification | Evidence | Cleanup action |
| --- | --- | --- | --- |
| MCP production/analyst instructions | Runtime-connected prompt contamination | `src/mcp_server/mcp_app.py:76-103`, `src/mcp_server/mcp_tools.py:74-101` | Replace with Nutanix evidence-first instructions. |
| Server naming | Runtime-connected but mostly harmless until productization | `src/mcp_server/mcp_app.py:192-193`, `src/mcp_server/main.py:107-112` | Rename service/server metadata after client compatibility review. |
| Embedding query prefix | Retrieval contamination | `config/embedding_profiles.yaml:116-124`, `src/providers/embeddings/embedding_service.py:191-234` | Replace with Nutanix query instruction or neutral instruction. |
| Query reformulation | Retrieval/prompt contamination | `src/mcp_server/query_service.py:646-735` | Replace system prompt and heuristic fallback with Nutanix domain pack. |
| Reranker instructions | Runtime-connected prompt contamination | `config/development.yaml:148-160`, `src/providers/factory.py:383-390`, `src/query/rerank_pipeline.py:135-140` | Replace with Nutanix-specific instructions and query-type variants. |
| Query intent terms and CLI pattern | Retrieval contamination | `src/query/query_intent.py:84-90`, `src/query/query_intent.py:147-154` | Create domain-neutral classifier plus Nutanix command vocabulary. |
| NER labels | Metadata/schema contamination | `config/development.yaml:509-535` | Replace WEKA entity examples with Nutanix AHV/Prism/Flow/Files/Objects/etc. taxonomy. |
| Cache/URI namespace | Runtime-connected naming residue | `config/development.yaml:430-446`, `src/mcp_server/mcp_utils.py:95-97`, `src/mcp_server/scratch_store.py:39` | Move to neutral `kb://` or Nutanix-specific namespace with migration aliases. |
| Docker/service/network names | Runtime-connected ops residue | `docker-compose.yml:7-10`, `docker-compose.yml:126-160`, `docker-compose.yml:230-379` | Rename in a controlled ops phase; preserve compatibility aliases temporarily. |
| Monitoring/runbooks | Config/documentation residue | `config/alloy/config.alloy:18-34`, `deploy/monitoring/prometheus-alerts.yaml:2-19` | Rename labels/dashboards/runbooks; avoid breaking dashboards without migration. |
| Current corpus assumptions | Runtime-connected data residue | `src/ingestion/auto/service.py:27-30`, data directory scan showed WEKA ingest samples | Replace corpus with Nutanix documents and retune metadata/doc_tag logic. |

## 8. Nutanix Readiness

Verdict: Not ready as a Nutanix-first RAG pipeline.

| Area | Readiness | Evidence | Required changes |
| --- | --- | --- | --- |
| Source/document assumptions | Low | Watcher is generic markdown/html, but tag defaults to `wekadocs` at `src/ingestion/auto/service.py:27-30`; corpus/data names are WEKA. | Define Nutanix source layout, tags, product families, version metadata. |
| Metadata model | Medium-low | Generic `doc_tag`, `doc_category`, `snapshot_scope` at `src/ingestion/atomic.py:867-921`; NER labels are WEKA-specific. | Add Nutanix metadata schema: product, version, platform, feature, command/API, release train. |
| Prompt readiness | Low | WEKA in MCP instructions, reformulator, reranker instruction, embedding prefix. | Centralize prompts in a Nutanix domain pack. |
| Retrieval readiness | Medium technically, low domain-wise | Graph-assisted pipeline is strong, but query intent and sparse/entity fields are WEKA-tuned. | Rebuild query intent, RRF weights, entity labels, and golden queries for Nutanix. |
| Citation readiness | Medium | Evidence packs return quotes/coverage; context citations may fallback to headings. | Require source URI/title/version in every citation; validate citations against retrieved passage ids. |
| Evaluation readiness | Low | Existing tests untrusted; eval scripts exist but are not current truth. | Build fresh Nutanix golden set and live retrieval/citation smokes. |
| LLM answer readiness | Unknown/low | Legacy response builder says answer is formatted results, not LLM-generated at `src/query/response_builder.py:300-308`; evidence tool returns quote packs. | Decide whether product is evidence-pack retrieval or generated Q&A. If Q&A, add explicit grounded answer layer. |

## 9. Defects And Risks

| ID | Severity | Files | Description | Why it matters | Recommended fix |
| --- | --- | --- | --- | --- | --- |
| RISK-001 | Critical for Nutanix migration | `src/mcp_server/mcp_app.py:76-103`, `config/embedding_profiles.yaml:116-124`, `config/development.yaml:148-160`, `src/mcp_server/query_service.py:646-735` | WEKA prompts/instructions are runtime-connected. | Nutanix queries will be reformulated, embedded, and reranked through WEKA assumptions. | Build a domain pack and remove runtime WEKA strings. |
| RISK-002 | High | `src/mcp_server/mcp_app.py:219-242` | Tool profile filtering only affects `list_tools`; `call_tool` uses `full_tool_map`. | Hidden/analyst/deprecated tools remain callable if a client knows names. | Enforce allowed profile in `call_tool`; add explicit compatibility allowlist. |
| RISK-003 | High | `src/mcp_server/mcp_utils.py:91-93` | `_neo4j_disabled` checks `_config.hybrid.neo4j_disabled`, but config shape is `config.search.hybrid.neo4j_disabled`. | Evidence search may request graph behavior even when retrieval config disabled Neo4j. | Read `getattr(getattr(_config.search, "hybrid", None), "neo4j_disabled", False)`. |
| RISK-004 | High | `config/development.yaml:12-25`, `config/embedding_profiles.yaml:1-12`, `src/shared/config.py:1336-1378` | Config claims BGE default, manifest plan overrides to Qwen3/SPLADE/ColBERT and ignores `EMBEDDINGS_PROFILE`. | Operators can believe they selected one model while another role plan is active. | Make manifest plan the only authority or remove it and rely on one profile key. |
| RISK-005 | Medium-high | `docker/ingestion-service.Dockerfile:59-64`, `docker-compose.yml:343-455`, `src/ingestion/auto/service.py:27-29` | Dockerfile exposes/checks 9108; service code and compose use 8081. | Image-level healthcheck can fail outside compose or mislead operators. | Align Dockerfile to 8081 or make port env-driven. |
| RISK-006 | Medium-high | `docker/ingestion-worker.Dockerfile:35-39`, `docker/ingestion-service.Dockerfile:35-40`, `docker-compose.yml:154-157`, `config/embedding_profiles.yaml:8-12` | Worker/service Dockerfiles prefetch Qwen3 reranker 0.6B tokenizer while runtime env/config points to Qwen3-Reranker-4B. | Offline tokenizer readiness can be false for the runtime model. | Prefetch exactly configured tokenizer/model ids or generate prefetch list from config. |
| RISK-007 | Medium-high | `src/query/vector_backends.py:653-680` | Query API failure falls back to legacy search. | Retrieval semantics and diagnostic meaning change under failure. | Make fallback explicit in user diagnostics; fail closed in strict/prod if Query API is contract-required. |
| RISK-008 | Medium | `src/query/rerank_pipeline.py:249-252`, `config/development.yaml:153-156` | Reranker batching hardcodes 32/4096 despite config max 8192/6000. | Context may be unnecessarily truncated or batching behavior differs from operator expectations. | Read limits from provider/config. |
| RISK-009 | Medium | `src/services/cross_doc_linking.py:1012-1019`, `src/services/cross_doc_linking.py:1034-1046`, `src/services/cross_doc_linking.py:1076-1088` | Cross-doc ColBERT rerank fails open on missing source/target vectors or errors. | RELATED_TO edges may bypass intended quality gate. | Record degraded quality state; optionally fail closed for edge creation in strict mode. |
| RISK-010 | Medium | `src/services/cross_doc_linking.py:923-964` | Structural priors swallow exceptions with `pass`. | Broken priors are invisible and edge quality degrades without diagnostics. | Log debug/warning with prior type and ids; count failures. |
| RISK-011 | Medium | `src/services/cross_doc_linking.py:99-113` | Entity prior counts document frequency via graph traversal for each source entity. | Can be expensive on entity-heavy docs and runs after candidate selection. | Materialize entity doc frequency or store hub status. |
| RISK-012 | Low-medium | `src/services/cross_doc_linking.py:75-88`, `src/services/cross_doc_linking.py:878-902` | Reciprocity result counts mutual reverse rows, not all r1 rows updated. | Metric undercounts unilateral edges touched. | Return separate counts for source edges and mutual reverse edges. |
| RISK-013 | Medium | `src/mcp_server/mcp_utils.py:177-196` | One global scratch session id is used when caller does not provide a session. | Cross-client passage ids and scratch state can collide/leak in shared server contexts. | Derive stable per-client/session ids or require explicit session id. |
| RISK-014 | Medium | `src/query/context_assembly.py:470-503` | Citations can degrade to title/heading fallback only. | Source attribution can be too weak for low-hallucination RAG. | Require URL/path/doc/version/source id in citation payloads. |
| RISK-015 | Medium | `src/ingestion/atomic.py:259-390` | Cross-doc linking is post-commit and intentionally fail-open. | Core ingestion can succeed with missing or stale `RELATED_TO` graph. | Separate "core indexed" from "graph enriched" readiness; add repair queue. |
| RISK-016 | Medium | `src/mcp_server/query_service.py:646-735` | Query reformulator uses `EMBEDDING_BASE_URL` chat endpoint and WEKA prompt. | Embedding gateway doubles as LLM endpoint and domain wording contaminates queries. | Separate `REFORMULATOR_BASE_URL`/provider and Nutanix prompt; add disabled-by-default toggle. |

## 10. God Modules And Refactor Candidates

| Module | Lines | Mixed responsibilities | Target decomposition | Sequence/risk |
| --- | ---: | --- | --- | --- |
| `src/ingestion/atomic.py` | 2417 | orchestration, parsing prep, metadata extraction, references, entity enrichment, embedding generation, batch token logic, Neo4j/Qdrant saga, compensation, cross-doc linking | `ingestion/orchestrator.py`, `ingestion/document_preparer.py`, `ingestion/embedding_stage.py`, `ingestion/saga_writer.py`, `ingestion/cross_doc_stage.py` | High risk. First wrap current public coordinator and add golden ingest smoke. |
| `src/mcp_server/mcp_tools.py` | 2319 | tool schemas, handlers, budgets, scratch, graph tools, evidence extraction, trace recording, backward aliases | `mcp/tools/registry.py`, `mcp/tools/evidence.py`, `mcp/tools/kb.py`, `mcp/tools/graph.py`, `mcp/tools/schemas.py` | Medium-high. Split by tool family after enforcing profile. |
| `src/query/hybrid_retrieval.py` | 2081 | retrieval orchestration, graph expansion, signal pool, rerank, ColBERT, budgeting, diagnostics, logging | `query/pipeline.py`, `query/stages/vector_stage.py`, `graph_stage.py`, `rerank_stage.py`, `budget_stage.py`, `diagnostics.py` | High. Freeze `retrieve()` contract first. |
| `src/shared/config.py` | 2015 | pydantic schemas, env settings, profile manifest, namespace, validation, singleton access | `config/schema.py`, `config/settings.py`, `config/embedding_plan.py`, `config/validation.py` | Medium. Make manifest authority explicit before split. |
| `src/query/vector_backends.py` | 1750 | Qdrant query bundle, Query API, weighted fusion, legacy search, payload conversion | `vector/qdrant_retriever.py`, `vector/query_bundle.py`, `vector/fusion.py`, `vector/payload_mapper.py` | Medium-high. Need live Qdrant smoke after each split. |
| `src/services/cross_doc_linking.py` | 1419 | candidate discovery, ColBERT fetch/MaxSim, structural priors, edge writes, reciprocity, batch stats, deprecated helpers | `cross_doc/discovery.py`, `cross_doc/rerank.py`, `cross_doc/priors.py`, `cross_doc/writer.py`, `cross_doc/batch.py` | Medium. Ancillary path; can split after active ingestion is guarded. |
| `scripts/backfill_cross_doc_edges.py` | 1355 | operator flow, discovery, batch processing, reporting | Move to `src/admin/cross_doc_backfill.py` with thin CLI | Low-medium if kept as admin only. |

## 11. Feature Flag And Configuration Consolidation

| Name | Defined | Read/used | Default/current | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| `MCP_HTTP_STREAMABLE_ENABLED` | env in `src/mcp_server/main.py:66-82` | Streamable mount handler/startup | true | Low | Keep. |
| `MCP_HTTP_LEGACY_REST_ENABLED` | env in `src/mcp_server/main.py:74-76` | legacy REST endpoints | false | Medium | Keep temporarily; add removal date. |
| `MCP_TOOL_PROFILE` | env in `src/mcp_server/mcp_utils.py:88`, `mcp_app.py:158-188` | tool list profile | production | High due call bypass | Enforce in call path. |
| `ENABLE_LEGACY_SEARCH_DOCUMENTATION` | env in `mcp_utils.py:82-84` | legacy tool registration | false | Low-medium | Delete once REST/legacy gone. |
| Evidence budgets (`MCP_EVIDENCE_MAX_QUOTES`, graph expansion, scratch/diag flags) | `mcp_utils.py:55-88` | evidence/search tools | mixed | Medium | Move under `mcp.evidence` config. |
| `search.hybrid.profile` | `config/development.yaml:67-75` | `retrieval_plan.py:150-193` | graph_assisted | Good but central | Keep as main retrieval behavior knob. |
| Legacy graph flags (`graph_channel_enabled`, `graph_enrichment_enabled`, etc.) | `config/development.yaml:84-90` | legacy inference and direct reads | mixed | Medium | Remove after profile is authoritative. |
| Feature flags (`query_api_weighted_fusion`, `signal_diverse_rerank_pool`, etc.) | `config/development.yaml:452-469` | `HybridRetriever`, `retrieval_plan` | mostly true | Medium | Fold into profile presets; leave only experimental overrides. |
| `search.vector.qdrant.enable_sparse/enable_colbert/use_query_api` | `config/development.yaml:44-63` | vector retriever/schema/ingestion | true/true/true | Medium | Derive from embedding plan plus schema validation. |
| Embedding manifest `plan` | `config/embedding_profiles.yaml:1-6` | `src/shared/config.py:1336-1378`, `1773-1848` | qwen3/splade/colbert | High operator confusion | Make this the visible root config, not hidden override. |
| `EMBEDDINGS_PROFILE` | compose + `Settings` | ignored if plan exists | bge_m3 in compose | High confusion | Remove or make invalid when plan exists. |
| `EMBEDDING_BASE_URL`, `BGE_M3_API_URL` | compose/provider | embedding service and query reformulator | remote gateway | Medium | Replace legacy BGE var with unified role-specific URLs. |
| Reranker env (`RERANK_PROVIDER`, `RERANK_MODEL`, `RERANKER_BASE_URL`, tokenizer id) | compose/provider | `ProviderFactory.create_rerank_provider` | Qwen3 4B env, mxbai YAML | High contradiction | One reranker config source. |
| `ANSWER_MODEL` | compose/response builder | provenance only | search-result-formatting | Low but misleading | Rename to `ANSWER_FORMATTER_ID` unless answer LLM is added. |
| Ingestion watcher env | `src/ingestion/auto/service.py:27-69`, compose | file watch/enqueue | auto/direct/ready mix | Low | Keep; rename tags to domain-neutral/Nutanix. |
| Cross-doc linking config | `config/development.yaml:315-340` | `AtomicIngestionCoordinator`, `CrossDocLinker` | enabled RRF+ColBERT | Medium | Split quality/ops config and add strict/degraded modes. |
| Cache config | `config/development.yaml:430-446` | cache/tooling | WEKA prefix | Medium domain residue | Rename namespace with migration. |

Consolidated target surface:

- `domain`: product name, corpus namespace, prompts, query instruction, reranker instructions, NER labels, query-intent vocabulary, URI/cache prefix.
- `runtime`: MCP transports, tool profile, legacy compatibility toggles.
- `models`: dense/sparse/colbert/reranker/reformulator roles with model id, endpoint, tokenizer id, context limits.
- `retrieval_profile`: named profile and minimal allowed overrides.
- `storage`: Neo4j/Qdrant/Redis names, schema versions, namespace.
- `ingestion`: source paths, parser/chunker, strictness, repair queue.
- `observability`: trace/diagnostics/logging/exporter settings.

## 12. RAG Quality And Hallucination Analysis

Strengths:

- Evidence-pack-first tool design is a good low-hallucination interface: retrieval depth is decoupled from quote count (`src/mcp_server/mcp_tools.py:1165-1172`), coverage is returned (`src/mcp_server/mcp_tools.py:1348-1359`), and traces are written (`src/mcp_server/mcp_tools.py:1377-1417`).
- Retrieval uses multiple signals: dense, learned sparse, title/entity sparse, RELATED_TO, graph channel, ColBERT, signal pool, cross-encoder rerank.
- Context assembly is budget-aware and groups by parent (`src/query/context_assembly.py:149-261`).

Risks:

- The intended corpus is Nutanix, but runtime prompts and query instructions still say WEKA. This can produce plausible but unsupported Nutanix answers by steering retrieval toward the wrong conceptual space.
- Query reformulation fallback literally appends "in WEKA" (`src/mcp_server/query_service.py:729-735`).
- Reranker instruction asks about WEKA distributed file system relevance (`config/development.yaml:157-160`).
- Citation output can fallback to headings, reducing auditability (`src/query/context_assembly.py:486-501`).
- Empty/weak retrieval in evidence path returns few/no quotes, but if a downstream agent ignores coverage and uses prior knowledge, hallucination risk remains.
- Conflicting sources are not explicitly detected or summarized; the evidence tool exposes quotes but does not classify contradictions.
- Silent fallbacks can mask degraded retrieval quality: Query API fallback, reranker fallback, ColBERT pass-through, GLiNER fail-open, cross-doc fail-open.

Concrete recommendations:

1. Add a Nutanix corpus contract: every ingested chunk must carry `product`, `version`, `source_uri`, `doc_id`, `doc_title`, `section_path`, `published_at` or release train when available.
2. Make evidence answers refuse unsupported claims by default. Keep the MCP instruction but move it to domain-neutral/Nutanix prompt source.
3. Add a `retrieval_degraded` block to evidence output when Query API, reranker, ColBERT, graph, or entity enrichment falls back.
4. Add citation validation: each quote must map to a scratch passage and original chunk/document source.
5. Add contradiction detection for answers that collect sources with different product/version/release metadata.
6. Separate query reformulator LLM from embedding gateway and make it domain-configured.

## 13. Proposed Clean Architecture

Target module layout:

```text
src/
  app/
    mcp_http.py
    mcp_stdio.py
    ingestion_service.py
    ingestion_worker.py
  domain/
    nutanix.py
    schema.py
    prompts.py
    query_intent.py
    ner_labels.py
  config/
    schema.py
    settings.py
    model_roles.py
    retrieval_profiles.py
    validation.py
  ingestion/
    orchestrator.py
    source_loader.py
    parser.py
    metadata.py
    chunking.py
    embeddings.py
    graph_writer.py
    vector_writer.py
    cross_doc_stage.py
  retrieval/
    orchestrator.py
    query_rewrite.py
    vector_stage.py
    graph_stage.py
    colbert_stage.py
    rerank_stage.py
    context_stage.py
    evidence_pack.py
  storage/
    qdrant.py
    neo4j.py
    redis.py
  mcp/
    registry.py
    tools/evidence.py
    tools/excerpt.py
    tools/graph.py
  evaluation/
    golden_set.py
    retrieval_eval.py
    citation_eval.py
```

What survives:

- `RetrievalPlan` concept from `src/query/retrieval_plan.py`.
- Evidence-pack-first MCP UX from `src/mcp_server/mcp_tools.py::kb_retrieve_evidence`.
- Atomic ingestion saga design from `src/ingestion/atomic.py`.
- Multi-vector Qdrant Query API implementation after extraction.
- Context assembly budget enforcement.
- Cross-doc `RELATED_TO` as a repairable enrichment stage, not core ingest success.

What gets deleted/replaced:

- Runtime WEKA prompts, labels, query prefixes, reformulation strings, and CLI patterns.
- Legacy REST MCP once clients migrate.
- `search_documentation` and underscore aliases after one compatibility window.
- Legacy `HybridSearchEngine` path after active retrieval tests cover the new path.
- Stale generated audits/reports from active docs, moving them to archive.

How Nutanix becomes first-class:

- Add `domain/nutanix.py` as the only source for product vocabulary, prompts, entity labels, command/API patterns, source metadata, and evaluation dimensions.
- Add ingestion metadata fields for Nutanix product family, version, component, deployment context, and source provenance.
- Add Nutanix golden queries and citation ground truth before model/live tuning.

## 14. Cleanup And Refactor Roadmap

| Phase | Goal | Files likely touched | Risk | Validation |
| --- | --- | --- | --- | --- |
| 1. Freeze active path | Document current HTTP/STDIO/evidence/ingestion path and add static guards | `src/mcp_server/*`, `src/query/*`, `src/ingestion/*`, docs | Low-medium | Static import/call smoke, MCP tool-list smoke when services available. |
| 2. Quarantine legacy/test artifacts | Move legacy REST/search docs/tests/generated outputs out of active surface | `src/mcp_server/main.py`, `mcp_tools.py`, `tests/`, `docs/cdx-outputs/`, `patched/` | Medium | No active startup imports removed; fresh smoke. |
| 3. Domain pack | Centralize Nutanix prompts, labels, query intent, URI/cache names | config, `src/domain/*`, `query_service.py`, `query_intent.py`, `mcp_app.py`, `mcp_tools.py` | Medium | Static WEKA guard over runtime files; Nutanix prompt diff review. |
| 4. Config/model consolidation | Make model-role manifest authoritative and remove contradictory env/config | `config/*.yaml`, `src/shared/config.py`, `src/providers/factory.py`, compose/Dockerfiles | High | `/health`, `/v1/models`, embedding/rerank smoke in later live pass. |
| 5. Enforce tool profiles | Apply profile filtering to `call_tool`, not just `list_tools` | `src/mcp_server/mcp_app.py`, tool registry | Medium | MCP call tests: hidden tool denied in production, allowed in analyst/full. |
| 6. Split god modules | Extract stages behind stable facades | atomic/hybrid/mcp_tools/vector_backends/config | High | Contract tests around public facades and live ingestion/retrieval smoke. |
| 7. Harden RAG quality | Add degraded-mode diagnostics, citation validation, contradiction metadata | retrieval/evidence/context/response | Medium | Golden answer/evidence tests. |
| 8. Delete obsolete paths | Remove legacy search engine, old scripts, stale schema migrations | legacy modules/scripts | Medium-high | Import graph, GitNexus detect changes, end-to-end smoke. |

## 15. Fresh Test Strategy

Existing tests are not used as truth. Inventory only: 261 files exist under `tests/` across unit, integration, e2e, query, providers, services, and fixtures. Recommendation: quarantine and rewrite around the frozen active architecture.

New validation strategy:

- Static contract tests:
  - MCP production profile lists only approved tools and denies hidden calls.
  - Runtime-connected files contain no WEKA strings except explicit compatibility allowlist.
  - Config has one model-role authority; env overrides are either honored or rejected loudly.
  - Dockerfile health ports match service ports.
- Unit tests:
  - Domain pack prompt/label/query-intent generation.
  - Retrieval plan resolution.
  - Query bundle construction for dense/sparse/ColBERT roles using fakes.
  - Evidence pack quote extraction and citation mapping.
  - Context assembly budget and citation formatting.
- Integration tests with fakes:
  - MCP `kb.retrieve_evidence` through fake `QueryService`.
  - Ingestion worker job through fake Neo4j/Qdrant writers.
  - Cross-doc linker degraded modes with fake Qdrant/Neo4j.
- Live tests for the next pass:
  - `GET /health` confirms effective model roles.
  - `/v1/models` on gateway confirms served model ids.
  - Dense/sparse/ColBERT embedding smoke.
  - Reranker health and one rerank call.
  - Qdrant schema validation.
  - Neo4j schema/relationship count smoke.
- Golden Nutanix tests:
  - Curated Nutanix docs ingestion reproducibility.
  - Retrieval relevance for 30-50 golden questions.
  - Citation correctness: every quote maps to source doc/path/version.
  - Empty retrieval and unsupported-feature refusal.
  - Conflicting-source/version handling.
  - Metadata filter tests by product/version/component.
- CI:
  - Keep static/fake tests in PR CI.
  - Run live model/database smokes in a separate GPU/service workflow.

## 16. Open Questions And Ambiguities

| Question | Why it matters | Evidence found | What resolves it | Can proceed? |
| --- | --- | --- | --- | --- |
| Which exact Nutanix corpus and metadata taxonomy should replace WEKA? | Domain pack and NER/query intent depend on it. | No Nutanix files found in active scan. | Product/source inventory and curated corpus sample. | Yes for architecture cleanup; no for final tuning. |
| Is product meant to return evidence packs only, or generated answers? | Current active production tool returns evidence quotes; legacy path formats search results. | `kb_retrieve_evidence` returns quotes/coverage; `response_builder.py:300-308` says not LLM-generated. | Product decision. | Yes, but tests differ. |
| Which reranker is canonical: Qwen3-Reranker-4B or mxbai? | Config/env/Dockerfiles disagree. | Compose/env use Qwen3 4B; YAML says mxbai; worker Dockerfiles prefetch Qwen3 0.6B. | Live `/v1/models` and config decision. | Yes for config consolidation design; no for live quality claims. |
| Should Query API fallback be allowed in production? | It changes retrieval semantics under failure. | Fallback exists in `vector_backends.py:668-679`. | Operator/product tolerance decision. | Yes, add strict mode. |
| Are STDIO clients still used? | Determines whether `stdio_server.py` remains active. | Standalone entrypoint imports shared `build_mcp_server`; not in compose. | User/client inventory. | Yes, keep for now. |
| Which existing tests, if any, are worth salvaging? | Rewrite cost. | Only directory/file inventory was done; tests not used as truth. | Post-architecture test audit. | Yes, quarantine first. |
| How stale are existing `docs/repo_audit` shards? | Could contain useful ledgers but may conflict. | Existing shards present; not trusted in this pass. | Diff against this report and source evidence. | Yes. |

## 17. Final Confidence Statement

Confidence in active path identification: High. Multiple independent signals agree: compose commands, Docker CMD, FastAPI startup, MCP builder, tool registry, QueryService, HybridRetriever, provider factory, ingestion service, worker, and atomic coordinator all line up.

Confidence in dead/legacy classification: Medium. Default-off legacy REST and `search_documentation` are clear. Some scripts and generated artifacts need import/usage checks before deletion.

Confidence in Nutanix-readiness assessment: High for static readiness: the repo is not Nutanix-ready because Nutanix is absent from active scan and WEKA is runtime-connected in prompts, config, retrieval logic, ops names, and corpus assumptions. Confidence in live retrieval quality is unknown because live embedders/rerankers/datastores were unavailable.

Top remaining uncertainties:

- Canonical Nutanix corpus/schema and product taxonomy.
- Canonical model-role deployment and served model ids.
- Whether the product should generate answers or provide evidence packs.
- Whether STDIO/legacy clients still require compatibility.

Can implementation/refactor safely begin: Yes, if the first milestone is narrow: freeze the active path, introduce a Nutanix domain pack, enforce MCP tool profiles, and add static/fake validation. Do not delete broad legacy paths or tune ranking until the live model/datastore pass confirms runtime behavior.
