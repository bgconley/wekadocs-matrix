# Executive Summary

Audit date: 2026-07-04

## Bottom Line

The repository is an advanced WEKA GraphRAG system, not a Nutanix-ready RAG system. The active runtime path is identifiable and technically substantial, but it remains domain-bound to WEKA in container names, MCP instructions, reranker instructions, embedding query instructions, NER/entity heuristics, historical context, telemetry names, sample corpus, and operator scripts.

I found no evidence of a Nutanix-first corpus, Nutanix prompt layer, Nutanix evaluation set, or Nutanix domain configuration in the current working tree. The right remediation is not a global string replacement. It is a controlled domain extraction: preserve the working RAG machinery, isolate WEKA as a legacy domain pack, and introduce Nutanix as an explicit domain profile with its own corpus, prompts, entity schema, evals, and runtime config.

## Highest-Priority Findings

1. Active runtime is still WEKA-branded and WEKA-instructed. `src/mcp_server/mcp_app.py`, `src/mcp_server/mcp_tools.py`, `config/development.yaml`, `config/embedding_profiles.yaml`, `docker-compose.yml`, `.env.*`, and the context corpus all carry WEKA assumptions.
2. Configuration has multiple sources of truth. Defaults disagree across `config/development.yaml`, `config/embedding_profiles.yaml`, `.env.local`, `.env.docker`, `.env.production`, `docker-compose.yml`, and `src/shared/config.py` for embedding/rerank models, provider URLs, and chunking behavior.
3. Active retrieval is reachable through MCP/FastAPI into `QueryService.search_sections_light()` and `HybridRetriever.retrieve()`, but it contains fail-open and legacy fallback behavior that makes quality regressions hard to detect.
4. Active ingestion is reachable through `src/ingestion/auto/service.py` and `src/ingestion/worker.py` into `AtomicIngestionCoordinator`. It is powerful but overcentralized, with parsing, references, GLiNER, chunking, embedding, Neo4j, Qdrant, structural edges, and cross-doc linking coordinated inside one large module.
5. Existing tests were not used as proof, per the prompt. They should be inventoried, quarantined, and rewritten around a fresh Nutanix contract/eval suite after the active architecture is stabilized.
6. GitNexus is indexed and current, but its CLI query path could not build FTS indexes because the local graph database was read-only. Symbol context still worked and surfaced a duplicate `AtomicIngestionCoordinator` in `patched/atomic_patched_v3.py`, reinforcing the need to classify stale alternates.

## Risk Rating

- Current WEKA-system maintainability: medium-high.
- Current Nutanix readiness: low.
- Feasibility of conversion: good, if treated as domain-pack extraction plus fresh Nutanix evals rather than broad rename/refactor.

## Artifacts

This audit created `docs/repo_audit/00` through `13`. The detailed file ledger is in `01_file_coverage_ledger.md`; the remediation sequence is in `10_refactor_roadmap.md`; the fresh test strategy is in `11_fresh_test_strategy.md`.
