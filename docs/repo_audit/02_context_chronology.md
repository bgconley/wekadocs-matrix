# Context Chronology

## Reading Stance

Historical notes were treated as archaeology, not runtime truth. The authoritative runtime map came from active entrypoints, imports, config, Docker, and GitNexus symbol context. Existing tests and old session notes were used only to identify contradictions, legacy paths, and prior intent.

## Chronological Arc

| Period | Evidence | Interpretation |
|---|---|---|
| Early context snapshots (`context-1.md` through `context-31.md`) | Repeated WEKA service names, WEKA docs corpus, Docker health, queue state, Qdrant/Neo4j setup | Repo began and remained a WEKA GraphRAG/MCP system. These files are useful for operational history, but not a Nutanix target contract. |
| 2025-11 retrieval/graph docs under `docs/cdx-outputs/` | Graph channel rehabilitation, BGE-M3 sparse research, hybrid retrieval optimization, atomic ingestion | The system evolved from simpler retrieval into multi-vector and graph-assisted retrieval. Many docs describe intermediate plans that may no longer match active code. |
| 2025-12 phase docs | Cross-doc linking, ColBERT, sparse title/entity vectors, semantic chunking, GLiNER, Qwen reranker, observability | Current architecture accreted advanced features quickly. The active path now includes many of these features, but config defaults and env examples disagree about which model/profile is canonical. |
| 2026-01 MCP/session docs | MCP Claude Desktop fixes, scratch fallback, modern evidence-pack tooling | MCP tool ergonomics improved, especially scratch/evidence handling. Runtime instructions still stayed WEKA-specific. |
| 2026-03 session notes | Evidence-pack modernization, retrieval tuning, architecture notes | These are the closest explanatory docs for the present retrieval path. They confirm active `HybridRetriever`, Query API sparse/lexical routing concerns, and old fallback caveats. |
| Root cleanup/dead-code docs | `DEAD-CODE-MAP.md`, `REFACTOR-PLAN.md`, `docs/dependency_graph.md`, `repo-analysis-artifacts/*` | The repo has already been audited before. Those reports are useful but not enough; the current dirty worktree and Nutanix target require a fresh ledger and target architecture. |

## Contradictions From Chronology

- Several docs call specific old files safe to delete, yet stale alternates still exist and some scripts/tests still import old surfaces.
- Runtime Docker/env files evolved after many docs, so the active model stack cannot be inferred from docs alone.
- WEKA language appears in both historical docs and active production instruction/config files, so it is not merely archival residue.
- There is no comparable Nutanix chronology, corpus, or domain spec in the tree.

## Confidence

High confidence that the system history is WEKA-centered. Medium confidence on exact phase-by-phase order because many context files are snapshots with overlapping dates and self-reported status.
