# Legacy And Alternative Paths

## Confirmed Or Suspected Legacy

| Path | Classification | Evidence/Risk |
|---|---|---|
| `src/query/hybrid_search.py` | Legacy retrieval engine | `docs/dependency_graph.md` labels it old/orphaned; active `QueryService.search_sections_light()` uses `HybridRetriever`. |
| `src/query/ranking.py` | Legacy ranker | Dependency graph labels old ranker; modern path uses retriever/reranker pipeline. |
| `patched/atomic_patched_v3.py` | Stale alternate implementation | GitNexus found a duplicate `AtomicIngestionCoordinator` symbol outside active `src/ingestion/atomic.py`. Must not be patched as runtime without proof. |
| `scripts/qdrant and helpers/*` | Historical helpers | Multiple copied helper versions and patch files; useful archaeology, not runtime. |
| `scripts/backfill_cross_doc_edges.py` | Manual backfill path | Reimplements search/linking functions; can diverge from `CrossDocLinker`. Keep only as operator migration tool or replace with thin wrapper around service class. |
| `repo-analysis-artifacts/*` | Prior audit outputs | Useful context, not current proof. |
| `docs/cdx-outputs/*` | Session/context history | Valuable chronology but should not be treated as live architecture. |
| `tests/*` | Untrusted by prompt | Tests may encode stale architecture; do not let them anchor refactor decisions. |

## Stale-But-Reachable Concerns

- STDIO MCP and FastAPI MCP share `build_mcp_server()`; both are reachable if invoked, so domain instruction fixes must cover shared MCP definitions rather than only HTTP wrappers.
- `HybridRetriever` includes legacy and fallback modes, including Qdrant Query API fallback to legacy search. If the preferred path fails, stale behavior may re-enter production silently.
- `SemanticChunkerAssembler` can fall back when Chonkie is unavailable. That is useful operationally but means chunking quality depends on dependency/runtime availability.
- Cross-doc scripts can create/update graph edges outside the ingestion worker's normal post-commit path.

## Quarantine Recommendation

Create a `legacy_manifest.md` before deletion. Anything not proven active should be classified as one of: historical doc, manual migration tool, fallback runtime path, test-only artifact, or dead alternate implementation. Only fallback runtime paths should survive the first cleanup wave.
