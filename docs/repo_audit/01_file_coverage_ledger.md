# File Coverage Ledger

Audit date: 2026-07-04

This ledger is generated from `rg --files --hidden` with `.git`, GitNexus index data, Python/cache folders, Hugging Face cache, node/venv folders, and retrieval diagnostic artifacts excluded. It accounts for first-party source, config, scripts, docs, tests, and local corpus/report artifacts that remain in the working tree. Rows marked metadata were inventoried and classified but not read line-by-line.

## Summary

- Ledger rows: 1105
- Files with WEKA/wekadocs residue by path or content sample: 393
- Files with Nutanix references by path or content sample: 0
- Classification counts: {'active': 88, 'external-corpus': 11, 'historical/context': 412, 'legacy/alternative': 10, 'operator/legacy-mixed': 78, 'unknown': 360, 'unknown-test': 146}
- Category counts: {'compose': 1, 'config': 12, 'corpus/data': 2, 'cypher': 10, 'deployment': 25, 'dockerfile': 4, 'documentation': 415, 'json': 150, 'other': 64, 'python': 20, 'script': 81, 'shell': 4, 'source': 151, 'test': 146, 'text': 14, 'yaml': 6}

## Ledger

| Path | Category | Reviewed | Depth | Reason | Classification | Active path? | WEKA residue? | Nutanix relevant? | Symbols/config/entrypoints | Notes |
|---|---|---:|---|---|---|---|---:|---:|---|---|
| `.claude/agents/db-cleanup.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.coveragerc` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.coveragerc.phase-3` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.env.apply-schema` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `.env.example` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `.env.example.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.github/pull_request_template.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `.github/workflows/ci.yml` | yaml | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `.gitignore` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.gitlint` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.gitmessage` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.pre-commit-config.yaml` | yaml | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.python-version` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.qwen/settings.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.qwen/settings.json.orig` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `.secrets.baseline` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `AGENTS.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `ARCHITECTURE.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `AUDIT-VERIFICATION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `Archive 2.zip` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `Archive.zip` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `CLAUDE.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `CLEANUP-EXECUTION-PLAN.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `CLEANUP-PLAN.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `COMPREHENSIVE-SUMMARY.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `DEAD-CODE-MAP.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | no | no | context/contract |  |
| `DEV-HISTORY.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `Makefile` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `PROGRESS_SUMMARY_2025-10-13.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `REFACTOR-PLAN.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `REPO-MAP.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `SESSION-SUMMARY.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `SESSION_PROGRESS_2025-10-13.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `STATUS.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | no | no | context/contract |  |
| `bootstrap_schema.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `ci/cd/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/files.zip` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `claude-raw/files/phase_7e_app_spec.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/files/phase_7e_expert_guidance.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/files/phase_7e_implementation_plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/files/phase_7e_pseudocode.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/master.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/original-v2-gpt5pro.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7D/create_schema_v2_2_complete.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `claude-raw/phase-7D/full-quad-docs-claude-20251026.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7D/gpt-oss-analysis-20251026.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7D/initial-full-implementation-no-oss-20251026.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7D/migrate_v2_1_to_v2_2_phase7d.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/integration_findings.csv` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/integration_findings.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/integration_guide-2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/integration_guide.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/second eval with reranker/integration_guide.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/second eval with reranker/line_by_line_eval.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/second eval with reranker/phase7e_hybrid_bm25_rrf_rerank.patch` | other | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no |  |  |
| `claude-raw/phase-7E/GPT5-Pro Integration/second eval with reranker/repo_scan_summary.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/GraphRAG_v2.1_Canonical_Spec_and_Implementation__Jina_v3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/cypher schema backup 20251028.zip` | other | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no |  |  |
| `claude-raw/phase-7E/jina-content-preservation-token-fix-plan-2025-01-27__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3.zip` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/GraphRAG_v2.1_Canonical_Spec_and_Implementation__Jina_v3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/claude gpt-5 pro plan review__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/create_schema_v2_1_complete__v3.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/final critiques for chunking and ingestion__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/improved tokenizer chunking fragmentation database schema adjustments__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/improved tokenizer chunking fragmentation solution__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/jina-content-preservation-token-fix-plan-2025-01-27__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/phase-7E/unzip/GraphRAG_v21_canonical_package__Jina_v3/redis invalidation plan and fusion query testing__aligned.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/phase-prompt-file-gpt5.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `claude-raw/planning-and-installation-prerequisites-and-compatibility.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/pseudocode-comp.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/weka_mcp_implementation_continued.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/weka_mcp_implementation_plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/wekadocs-matrix-app-spec-v1-claude.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `claude-raw/wekadocs-matrix-implementation-plan-v1-claude.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `config/alloy/config.alloy` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/development.yaml` | config | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | runtime config |  |
| `config/embedding_profiles.yaml` | config | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | runtime config |  |
| `config/grafana/dashboards/cross-doc-linking.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/dashboards/ingestion-overview.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/dashboards/retrieval-performance.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/dashboards/system-health.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/provisioning/dashboards/default.yaml` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/provisioning/dashboards/wekadocs-infrastructure.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/provisioning/dashboards/wekadocs-ingestion.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/provisioning/dashboards/wekadocs-overview.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `config/grafana/provisioning/dashboards/wekadocs-retrieval.json` | config | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `context-1.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `context-11.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-12.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-13.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-14.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-15.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-16.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-17.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-18.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-19.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-2.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-21.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-22.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-23.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-24.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-25.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-26.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-27.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-3.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-30.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-31.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-4.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-5.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-6.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-7.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-8.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `context-9.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `data/documents/inbox/test-1760726986.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | no | no | context/contract |  |
| `data/documents/spool/62d7e6a2c88621ad2cc183ab65eb51ad4c1a725b857e96dc24dd3c304ef834d9.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | no | no | context/contract |  |
| `data/samples/api_guide.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no | context/contract |  |
| `data/samples/getting_started.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no | context/contract |  |
| `data/samples/performance_tuning.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no | context/contract |  |
| `data/samples/sample_doc.html` | corpus/data | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no |  |  |
| `data/samples/test-oversized-section.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no | context/contract |  |
| `data/samples/test-truly-massive.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | no | no | context/contract |  |
| `data/test/doc_with_references_a.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no | context/contract |  |
| `data/test/doc_with_references_b.md` | documentation | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no | context/contract |  |
| `data/test/golden_queries.json` | corpus/data | no | metadata | corpus payload, not first-party implementation | external-corpus | input-only | yes | no |  |  |
| `db-check.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `dead_code_audit_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `deploy/DR-RUNBOOK.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `deploy/helm/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `deploy/k8s/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `deploy/k8s/base/configmap.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/ingestion-worker-deployment.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/ingress.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/kustomization.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/mcp-server-canary-deployment.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/mcp-server-deployment.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/mcp-server-green-deployment.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/namespace.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/neo4j-statefulset.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/qdrant-statefulset.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/redis-statefulset.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/k8s/base/secrets.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/monitoring/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `deploy/monitoring/RUNBOOK.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `deploy/monitoring/grafana-dashboard-ingestion.json` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/monitoring/grafana-dashboard-overview.json` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/monitoring/grafana-dashboard-query-performance.json` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/monitoring/prometheus-alerts.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/scripts/backup-all.sh` | deployment | yes | partial | entrypoint/config/import scan | legacy/alternative | no | yes | no | runtime config |  |
| `deploy/scripts/blue-green-switch.sh` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/scripts/canary-rollout.sh` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/scripts/dr-drill.sh` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `deploy/scripts/restore-all.sh` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | yes | no | runtime config |  |
| `docker-compose.yml` | compose | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | runtime config |  |
| `docker/ingestion-service.Dockerfile` | dockerfile | yes | full | runtime path or primary contract reviewed | active | yes | no | no | runtime config |  |
| `docker/ingestion-worker.Dockerfile` | dockerfile | yes | full | runtime path or primary contract reviewed | active | yes | no | no | runtime config |  |
| `docker/mcp-server.Dockerfile` | dockerfile | yes | full | runtime path or primary contract reviewed | active | yes | no | no | runtime config |  |
| `docker/mxbai-reranker.Dockerfile` | dockerfile | yes | partial | entrypoint/config/import scan | unknown | unknown | no | no | runtime config |  |
| `docs/Archive.zip` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/FEATURE_SUMMARY_enhanced-responses.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/api-contracts.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/app-spec-phase6.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/architecture/2026-03-04-end-to-end-architecture.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/architecture/retrieval-path-030826.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/blue-green-migration.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/bugfix/bugfix-context-tokenizer-hybrid-logging.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `docs/bugfix/bugfix-neo4j-planner-safety.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `docs/bugfix/bugfix-tokenizer-caching-hybrid-logging-and-cleanup.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `docs/bugfix/explainguard-entity-edge-fix.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `docs/bugfix/reranker-phase7e-readiness.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `docs/canonical-retrieval-benchmark.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/evaluation-comprehensive-retrieval-plan-2025-11-26.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/final-gliner-architecture-20251208.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-2025-12-07-vector-pipeline-prep.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-integration-planning.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-mps-acceleration.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-phase1-implementation.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-phase2-implementation.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-phase3-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-review-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-review.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-phase4-entity-retrieval.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/final-gliner-architecture-20251208.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/gliner-implementation-complete-chronological.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-2025-12-07-vector-pipeline-prep.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-graph-disabled-fixes.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-ingestion-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-integration-planning.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-mps-acceleration.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-phase1-implementation.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-phase2-implementation.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-phase3-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-review-complete.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-review.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-phase4-entity-retrieval.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-retrieval-debugging.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251209-entity-sparse-fix.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251209-gliner-entity-sparse-verification.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/gliner-implementation-sessions/session-context-20251209-rrf-debugging-and-embedding-investigation.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/cdx-outputs/plan-graph-channel-rehabilitation-2025-11-26.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/claude-code-analysis-gpt5pro-phase7-int-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/claude_desktop_config.json` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/coder-guidance-phase6.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/configuration.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/context/2026-01-19-mcp-session-fixes-context.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/context/2026-01-20-session-context-preservation.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/decisions/phase-4-5-atomic-observability.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/decisions/sparse-coverage-policy-B3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/dependency_graph.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | no | no | context/contract |  |
| `docs/evaluation_plan_bge_qdrant.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/expert-coder-guidance.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/feature-spec-enhanced-responses.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/fix-docs/gpt5pro-graph-traversal.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/golden-set-queries.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/gpt5pro-docrag-schema-1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/gpt5pro-enhanced-graph-query-perf-1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/gpt5pro-enhanced-graph-query-perf-jina-2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/gpt5pro-phase7-int-plan-enhanced.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/guides/embedding-profile-matrix.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/guides/embedding-profile-rollout.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/guides/verbosity-usage.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/hybrid-rag-v2_2-architecture.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/hybrid-rag-v2_2-changelog.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/hybrid-rag-v2_2-e2e-prod-spec.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/hybrid-rag-v2_2-handoff.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/hybrid-rag-v2_2-spec.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/hybrid-rag-v2_2-testing.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/implementation-plan-enhanced-responses.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/implementation-plan-phase-6.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/implementation-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/local-env.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/mcp/retrieval_playbook.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/o3-pro/expert-coder-guidance_doc-routing-reranker_v1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/o3-pro/expert-coder-guidance_doc-routing-reranker_v2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/o3-pro/feature-spec_doc-routing-reranker_v1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/o3-pro/feature-spec_doc-routing-reranker_v2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/o3-pro/implementation-plan_doc-routing-reranker_v1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/o3-pro/implementation-plan_doc-routing-reranker_v2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/o3-pro/pseudocode_doc-routing-reranker_v1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/o3-pro/pseudocode_doc-routing-reranker_v2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/phase-7-integration-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/phase3-references-review-consensus.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/phase7-target-phase-tasklist.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/phase7C-tasks-7C3-7C4-summary.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/2026-01-19-cleanup-script-multi-embedder-update.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/2026-01-19-snowflake-arctic-collection-design.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/2026-01-20-diagtool-design.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/2026-02-10-p620-rtx3090-migration.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/2026-03-01-codebase-pruning-and-modernization-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/2026-03-03-evidence-pack-architecture-mcp-modernization-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/atomic_structural_edges_plan_20251213.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/chonkie_semantic_chunking_integration_plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/chunking_architecture_analysis.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/embedder-modulize-plan-20251219.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/gliner_integration_plan_v1.2_codebase_integrated.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/gliner_integration_plan_v1.2_review_findings.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/gliner_rag_implementation_plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/gliner_rag_implementation_plan_gemini_mods.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/gliner_rag_implementation_plan_gemini_mods_apple copy 2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/gliner_rag_implementation_plan_gemini_mods_apple copy 3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/gliner_rag_implementation_plan_gemini_mods_apple copy.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/gliner_rag_implementation_plan_gemini_mods_apple.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/graph_rag_findings-20251210.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/markdown-it-py-integration-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/mcp_http_endpoint_alignment_plan_20251217.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/mcp_retrieval_diagnostics_persistence_and_otel_plan_20251217.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/mcp_stdio_tool_calls_overhaul_plan_20251217.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/multi-embedder-implementation-plan-20251215.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/neo4j_overhaul_gpt52_20251213.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/phase-3.5-cross-doc-linking.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/qwen3-fitness-analysis-20251215.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/qwen3_reranker_multi_backend_plan_20251215 copy.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/qwen3_reranker_multi_backend_plan_20251215.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/plans/redis-streams-implementation-20251215.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/RKG_ENHANCED_FEATURES_SPEC.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/RKG_IMPLEMENTATION_PATTERNS.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/RKG_INTEGRATION_PATTERNS.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/RKG_MCP_SERVER_CORRECTED.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/RKG_MISSING_FEATURES_SPEC.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/RKG_SYSTEM_SPECIFICATION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/corrected_docs_native_mcp/RKG_CONTENT_HANDLING_CORRECTED.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/corrected_docs_native_mcp/RKG_MCP_SERVER_CORRECTED.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/corrected_docs_native_mcp/RKG_MISSING_FEATURES_SPEC.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/corrected_docs_native_mcp/RKG_SYSTEM_SPECIFICATION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/old_docs_pre_mcp_correction/RKG_MISSING_FEATURES_SPEC.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/research-and-session-app/old_docs_pre_mcp_correction/RKG_SYSTEM_SPECIFICATION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/plans/session-context-20251207-vector-architecture-reform.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/plans/vector_pipeline_reform_and_semantic_enrichment_plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/pseudocode-phase6.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/pseudocode-reference.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/quickstart-commands.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/reranking-retrieval-analysis-030326` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/reranking-retrieval-analysis-030326.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/reranking-retrieval-analysis-AND-PLAN-030326.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/session-notes/2026-03-03-signal-pool-provider-cleanup.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | no | no | context/contract |  |
| `docs/session-notes/2026-03-04-evidence-pack-mcp-modernization.md` | documentation | yes | full | runtime path or primary contract reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/session-notes/2026-03-04-integration-deployment-retrieval-tuning.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/session-notes/2026-03-04-retrieval-tuning-graph-prerequisites.md` | documentation | yes | partial | chronology/contradiction sample reviewed | historical/context | no | yes | no | context/contract |  |
| `docs/spec.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/tasks/p1_t1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p1_t2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p1_t3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p1_t4.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p2_t1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p2_t2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p2_t3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p2_t4.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p3_t1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p3_t2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p3_t3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p3_t4.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p4_t1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p4_t2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p4_t3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p4_t4.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p5_t1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p5_t2.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p5_t3.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p5_t4.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `docs/tasks/p6_t1.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/test-smoketest-1760733094.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `docs/wekadocs50_combined.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `gitnexus` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `infra/scripts/deploy-lgtm.sh` | shell | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/scripts/destroy-lgtm.sh` | shell | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/environments/dev/.terraform.lock.hcl` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/environments/dev/backend.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/environments/dev/main.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/environments/dev/terraform.tfvars.example` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/main.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/cloud-init/cloud-config.yaml.tpl` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/cloud-init/docker-compose.yaml.tpl` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/cloud-init/grafana-datasources.yaml` | yaml | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/cloud-init/loki-config.yaml` | yaml | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/cloud-init/mimir-config.yaml` | yaml | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/cloud-init/tempo-config.yaml` | yaml | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/main.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/outputs.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/modules/lgtm-vm/variables.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/outputs.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/variables.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `infra/terraform/versions.tf` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `inventory_neo4j.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `inventory_qdrant.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `migration/EMBEDDING_CANONICALIZATION_REPORT.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `migration/baseline_counts.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `migration/baseline_counts.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `migration/collect_baseline.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `migration/debug_baseline.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `migration/qdrant_inspect.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `migration/verification_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `monitoring/alerts/phase7e_slo_alerts.yaml` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | no | no | runtime config |  |
| `monitoring/dashboards/phase7e_ingestion.json` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | no | no | runtime config |  |
| `monitoring/dashboards/phase7e_retrieval.json` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | no | no | runtime config |  |
| `monitoring/dashboards/phase7e_slos.json` | deployment | yes | partial | entrypoint/config/import scan | unknown | unknown | no | no | runtime config |  |
| `neo4j_full_migration.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `neo4j_schema_dump.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `patched/atomic_patched_v3.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | yes | no |  | GitNexus duplicate of AtomicIngestionCoordinator; classify as stale alternate unless proven otherwise |
| `patched/chonkie_adapter_patched.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no |  |  |
| `patched/semantic_chunker_patched.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no |  |  |
| `phase4-kickoff-fixtures.patch` | other | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no |  |  |
| `progress.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `pytest.ini` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `qdrant_sample.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `qdrant_schema_inventory.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `repo-analysis-artifacts/00-executive-summary.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `repo-analysis-artifacts/01-repo-inventory.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `repo-analysis-artifacts/02-active-vs-deprecated-map.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `repo-analysis-artifacts/03-architecture.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `repo-analysis-artifacts/04-ascii-architecture-diagrams-verified.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `repo-analysis-artifacts/04-ascii-architecture-diagrams.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `repo-analysis-artifacts/05-runtime-flows.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `repo-analysis-artifacts/06-risk-register.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `repo-analysis-artifacts/07-refactor-and-cleanup-plan.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `repo-analysis-artifacts/08-open-questions.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `repo-analysis-artifacts/09-evidence-index.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `repo-analysis-artifacts/comprehensive-analysis-full-reads.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/FIRST_INGESTION_VERIFICATION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/LAUNCH_GATE_REPORT.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/QDRANT_EMPTY_ANALYSIS.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/VOLUME_PERSISTENCE_INVESTIGATION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/baseline/circuit_breaker_analysis.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/baseline/collected_tests.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/baseline/junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/baseline/mypy.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/baseline/provider_architecture_analysis.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/baseline/ruff.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/baseline/summary.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/baseline/summary_real.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/baseline/test_collection.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/baseline/test_collection_fixed.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/baseline/test_suite_analysis.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/baseline_metrics.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/codepath-map.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/community_detection/doc_related_to_louvain_20260304_065448Z.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/community_detection/doc_related_to_louvain_20260304_065448Z.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/community_detection/doc_related_to_louvain_20260304_065612Z.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/community_detection/doc_related_to_louvain_20260304_065612Z.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/full-suite-20251018-191436/consolidated/FULL-SUITE-REPORT.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/full-suite-20251018-191436/consolidated/all-phases.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/full-suite-20251018-191436/consolidated/analysis-report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/full-suite-20251018-191436/consolidated/post-test-state.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/full-suite-20251018-191436/consolidated/pre-cleanup-state.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/global/pytest.out` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/ingest/000d6691-eca7-4d0c-8b82-0cd0b7525207/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/000d6691-eca7-4d0c-8b82-0cd0b7525207/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/02314f75-c34f-4bdc-ad1f-98dbbdd1382f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/02314f75-c34f-4bdc-ad1f-98dbbdd1382f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/03a5c657-d12b-4043-b15a-8407f6cc3027/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/03a5c657-d12b-4043-b15a-8407f6cc3027/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/048327c2-02f4-4616-971a-9ba470e435de/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/048327c2-02f4-4616-971a-9ba470e435de/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/0a2e59e3-b02d-42c0-a0a5-4625afdb9995/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/0a2e59e3-b02d-42c0-a0a5-4625afdb9995/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/0e3c4370-ecc3-4844-a93f-9f608b3ea2fb/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/0e3c4370-ecc3-4844-a93f-9f608b3ea2fb/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/0e44a32e-3b14-46f3-957c-2c5c5499e289/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/0e44a32e-3b14-46f3-957c-2c5c5499e289/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/0ef3d2c2-cf62-4fc3-bd76-0382c216b2a8/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/0ef3d2c2-cf62-4fc3-bd76-0382c216b2a8/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/111e23ce-e2eb-4a1d-b1db-4d828a18b62d/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/111e23ce-e2eb-4a1d-b1db-4d828a18b62d/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/14a6e8a2-db6f-41b3-a2b2-35c5b7094bd8/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/14a6e8a2-db6f-41b3-a2b2-35c5b7094bd8/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/16f282e1-3d63-4bb6-a75e-b52f23a4c88b/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/16f282e1-3d63-4bb6-a75e-b52f23a4c88b/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/1ae3d5b3-bb21-4e79-969e-881648ae74ea/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/1ae3d5b3-bb21-4e79-969e-881648ae74ea/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/1e66cac3-72aa-46a2-8d64-ab2232da29c3/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/1e66cac3-72aa-46a2-8d64-ab2232da29c3/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/20251018_224555_64d442a6/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/20251018_224555_64d442a6/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/20251018_224614_f590c2b9/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/20251018_224614_f590c2b9/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/205062d7-8579-43bc-baf7-5ce67b2e3757/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/205062d7-8579-43bc-baf7-5ce67b2e3757/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/23e89772-c9e7-4fcc-b986-67774b8384f2/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/23e89772-c9e7-4fcc-b986-67774b8384f2/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/260ca493-ba11-4caa-8d63-4f70f083bfff/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/260ca493-ba11-4caa-8d63-4f70f083bfff/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/28c37bbd-053e-42ec-a998-b236ee1c7fec/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/28c37bbd-053e-42ec-a998-b236ee1c7fec/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/2a047714-addd-45c0-ab98-ef0483ef089c/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/2a047714-addd-45c0-ab98-ef0483ef089c/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/2d20e5ac-37bc-432d-a861-745bd0c63589/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/2d20e5ac-37bc-432d-a861-745bd0c63589/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/305354ad-31e0-48a5-9bfa-febb0e1eb685/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/305354ad-31e0-48a5-9bfa-febb0e1eb685/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/339a31be-d139-4674-b6ab-1d46487f019d/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/339a31be-d139-4674-b6ab-1d46487f019d/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/34a298f8-303f-48e4-8573-483c14081397/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/34a298f8-303f-48e4-8573-483c14081397/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/35f3dd41-4bcd-44d8-a3dc-3e1d4bb09cca/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/35f3dd41-4bcd-44d8-a3dc-3e1d4bb09cca/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/35fb45a3-0f90-483a-a0f6-a696243a72ec/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/35fb45a3-0f90-483a-a0f6-a696243a72ec/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/365d521e-bb7f-4c9e-bfd9-0b95062cb182/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/365d521e-bb7f-4c9e-bfd9-0b95062cb182/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/383af2b7-0d1e-47f6-9719-737f68f3aef3/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/383af2b7-0d1e-47f6-9719-737f68f3aef3/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/39a973f5-1913-4aa0-bdb5-50848a7cd014/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/39a973f5-1913-4aa0-bdb5-50848a7cd014/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/3b83dee6-b6a2-4d67-a505-b977893c3ced/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/3b83dee6-b6a2-4d67-a505-b977893c3ced/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/3c42dce0-a0fe-40bc-9ee9-54b5ab5fc13a/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/3c42dce0-a0fe-40bc-9ee9-54b5ab5fc13a/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/3c57a0e2-1d89-488d-86c0-adff87795c1e/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/3c57a0e2-1d89-488d-86c0-adff87795c1e/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/41ddf570-e332-48a2-8581-3c67f2231316/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/41ddf570-e332-48a2-8581-3c67f2231316/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/427cbbbd-6789-4ff7-be3d-2792892923fe/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/427cbbbd-6789-4ff7-be3d-2792892923fe/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/45469c66-a432-4a38-ae92-b3c97e7ff8c3/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/45469c66-a432-4a38-ae92-b3c97e7ff8c3/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/45bbf778-0f89-41e3-a15a-403153889cf1/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/45bbf778-0f89-41e3-a15a-403153889cf1/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/4b010bf2-36a4-430d-8869-b6a1802cd4c2/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/4b010bf2-36a4-430d-8869-b6a1802cd4c2/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/4bd1d770-a7d9-4f77-9c41-272b7c6a27b2/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/4bd1d770-a7d9-4f77-9c41-272b7c6a27b2/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/531a3f24-f430-4f97-bea4-123f3d9332c7/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/531a3f24-f430-4f97-bea4-123f3d9332c7/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/551badd1-19c2-4093-af0b-61623dbdb983/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/551badd1-19c2-4093-af0b-61623dbdb983/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/5794d4c2-2fed-4b4a-ba9e-a4ed706c3c54/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/5794d4c2-2fed-4b4a-ba9e-a4ed706c3c54/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/592e48a9-7104-491b-9d58-2db9c4ed7225/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/592e48a9-7104-491b-9d58-2db9c4ed7225/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/596a45cd-5f3f-4631-b9f0-fee0e4e0a3c6/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/596a45cd-5f3f-4631-b9f0-fee0e4e0a3c6/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/59e09aab-3026-4a9b-8f26-8e1c2186569b/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/59e09aab-3026-4a9b-8f26-8e1c2186569b/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/61097d82-e63a-4709-a5aa-61626edbd9e4/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/61097d82-e63a-4709-a5aa-61626edbd9e4/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/64b950f4-38a7-4d43-87e5-74634986b3d4/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/64b950f4-38a7-4d43-87e5-74634986b3d4/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/6a2d9320-3d9b-445e-91c3-26696df0294f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/6a2d9320-3d9b-445e-91c3-26696df0294f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/6d0916dd-eac5-45b3-9321-eb61d7e99c88/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/6d0916dd-eac5-45b3-9321-eb61d7e99c88/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/6ee8cb2f-da1e-46fc-9f64-faf2300afeaa/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/6ee8cb2f-da1e-46fc-9f64-faf2300afeaa/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/71291453-3651-41d1-b87a-b5d3089069f6/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/71291453-3651-41d1-b87a-b5d3089069f6/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/71f923a6-c8f7-4d6b-8bf9-dec6dea19725/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/71f923a6-c8f7-4d6b-8bf9-dec6dea19725/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/728f3900-1953-457b-9223-9299b5cc529a/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/728f3900-1953-457b-9223-9299b5cc529a/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/73203142-8705-4a67-a0b9-fbb593a6f563/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/73203142-8705-4a67-a0b9-fbb593a6f563/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/7530880b-1616-4af9-a86d-edf60c1e34ac/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/7530880b-1616-4af9-a86d-edf60c1e34ac/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/7d800a6f-c170-42ae-8f03-e5eb77335451/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/7d800a6f-c170-42ae-8f03-e5eb77335451/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/80db651c-70dd-49cd-9eed-2a41c3f9847f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/80db651c-70dd-49cd-9eed-2a41c3f9847f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/81bc0716-bb24-4232-8621-e7fe8606682f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/81bc0716-bb24-4232-8621-e7fe8606682f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/831a12e7-939a-46c6-9879-abff2184e4e1/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/831a12e7-939a-46c6-9879-abff2184e4e1/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/83615e1f-ff00-4769-b242-75b6a24df2ba/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/83615e1f-ff00-4769-b242-75b6a24df2ba/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/842c8a14-cf81-4a88-8c77-c88fe49656bf/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/842c8a14-cf81-4a88-8c77-c88fe49656bf/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/86911e29-e5e1-4a87-9334-30fc4ffa0e1a/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/86911e29-e5e1-4a87-9334-30fc4ffa0e1a/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/86cbc838-3cde-4cac-86cd-e955dc2abdfb/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/86cbc838-3cde-4cac-86cd-e955dc2abdfb/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/88668464-b32c-4c17-9be5-67647816f5d3/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/88668464-b32c-4c17-9be5-67647816f5d3/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/8a7b4211-a5ae-4efb-9743-1e26d5933c6d/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/8a7b4211-a5ae-4efb-9743-1e26d5933c6d/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/8cf7c2d5-814b-4ebe-a51e-57f55e853854/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/8cf7c2d5-814b-4ebe-a51e-57f55e853854/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/900ac8c5-7678-471f-a867-b1a1cfb62662/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/900ac8c5-7678-471f-a867-b1a1cfb62662/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/932b2caa-2b78-4500-a9a1-c05acec901e3/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/932b2caa-2b78-4500-a9a1-c05acec901e3/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/94249539-deb7-4b3a-ac4e-581b2657a9fd/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/94249539-deb7-4b3a-ac4e-581b2657a9fd/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/97998428-8d7e-4a1d-92ed-ff5c90280577/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/97998428-8d7e-4a1d-92ed-ff5c90280577/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/989781c9-6bb8-49f3-a410-0116f8dfdbd7/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/989781c9-6bb8-49f3-a410-0116f8dfdbd7/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/9a3defcd-ec09-4452-881a-623982492f1f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/9a3defcd-ec09-4452-881a-623982492f1f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/9c0ef841-7bc5-46c0-8ecf-6ab99fe30b4a/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/9c0ef841-7bc5-46c0-8ecf-6ab99fe30b4a/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/a92de8d3-0b13-4315-9adc-5b7415b6121f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/a92de8d3-0b13-4315-9adc-5b7415b6121f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/a9cda561-835c-4708-a9d8-60ba7bb3ee74/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/a9cda561-835c-4708-a9d8-60ba7bb3ee74/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/ae4839dd-504b-454b-944b-6349ba1a94dd/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/ae4839dd-504b-454b-944b-6349ba1a94dd/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/b1da7946-a937-48f9-8028-dfda975b1acf/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/b1da7946-a937-48f9-8028-dfda975b1acf/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/b81fe1ec-ce26-44b4-a52f-53fec40bc149/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/b81fe1ec-ce26-44b4-a52f-53fec40bc149/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/b905c8ed-e220-46b3-bc0e-22b27bc3f2f6/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/b905c8ed-e220-46b3-bc0e-22b27bc3f2f6/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/ba590ff3-66d3-4686-84e6-eb4160dd1def/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/ba590ff3-66d3-4686-84e6-eb4160dd1def/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/bdaa47b3-4cd4-4018-bb13-193a2368a18e/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/bdaa47b3-4cd4-4018-bb13-193a2368a18e/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/bddce024-f049-4fb5-b499-4937056a8945/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/bddce024-f049-4fb5-b499-4937056a8945/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/bf220a4f-4326-495d-9746-d7a434507adc/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/bf220a4f-4326-495d-9746-d7a434507adc/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/c27413da-571a-4e49-86a2-5c417ba81806/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/c27413da-571a-4e49-86a2-5c417ba81806/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/c59e9383-bd83-4f09-acea-e585d08ad1e0/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/c59e9383-bd83-4f09-acea-e585d08ad1e0/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/c72d7924-aba4-4137-b4b1-886fba6475e3/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/c72d7924-aba4-4137-b4b1-886fba6475e3/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/c730bfc7-50f3-4551-8636-475007df0942/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/c730bfc7-50f3-4551-8636-475007df0942/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/cf46708f-58ec-43b7-8c03-e0c0eae0f469/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/cf46708f-58ec-43b7-8c03-e0c0eae0f469/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/d827dca8-9c86-444f-b2a0-34e7960c4ad1/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/d827dca8-9c86-444f-b2a0-34e7960c4ad1/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/e2436d35-10f4-4464-a0d6-359904fc1c7f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/e2436d35-10f4-4464-a0d6-359904fc1c7f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/e249fc20-8f7b-4df9-9c30-baead39f53ea/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/e249fc20-8f7b-4df9-9c30-baead39f53ea/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/e647c022-d5c8-4d76-9655-24bac3679ed0/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/e647c022-d5c8-4d76-9655-24bac3679ed0/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/e6802823-3a11-4f4c-86ed-228fb3e69c24/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/e6802823-3a11-4f4c-86ed-228fb3e69c24/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/e70d4413-1473-4ef9-80ed-eb0ad147ef6f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/e70d4413-1473-4ef9-80ed-eb0ad147ef6f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/ec423ff9-79e4-4d84-9cb3-82900767ac11/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/ec423ff9-79e4-4d84-9cb3-82900767ac11/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/ecded8bd-a6bd-457b-a274-d6951ba928ad/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/ecded8bd-a6bd-457b-a274-d6951ba928ad/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/f1e53714-ebb6-45f8-9dcf-bcda05db558f/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/f1e53714-ebb6-45f8-9dcf-bcda05db558f/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/f45ef388-f5b1-4875-831b-a24596a5c7fe/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/f45ef388-f5b1-4875-831b-a24596a5c7fe/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/f573abb7-54c0-492f-a2c5-400e423e3a38/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/f573abb7-54c0-492f-a2c5-400e423e3a38/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/f63ca23e-69a9-4032-a3e9-99c1b1b77e86/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/f63ca23e-69a9-4032-a3e9-99c1b1b77e86/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/f8358610-fef3-4327-8717-8786d5243c9d/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/f8358610-fef3-4327-8717-8786d5243c9d/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/fab0fe77-7c4f-4078-86df-7107de3cc6e0/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/fab0fe77-7c4f-4078-86df-7107de3cc6e0/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/fb89100d-c5d6-4873-992d-9289e409d297/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/fb89100d-c5d6-4873-992d-9289e409d297/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/ingest/fc1eb449-0cff-4350-bd42-695cee317336/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/fc1eb449-0cff-4350-bd42-695cee317336/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/fc8029a2-6543-4827-8998-7c1442ec9473/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/fc8029a2-6543-4827-8998-7c1442ec9473/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/ingest/fdbded5b-93b1-4982-ae73-77987595dfc7/ingest_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/ingest/fdbded5b-93b1-4982-ae73-77987595dfc7/ingest_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-1-integration-test-results.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-1/junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-1/post_fix_metrics.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-1/summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-2/junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-2/junit_t1.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-2/perf_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-2/summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-3/coverage.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-3/junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-3/junit_unit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-3/pytest.out` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-3/summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t1_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t1_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t2_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t2_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t3_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t3_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t4_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/p4_t4_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/phase_4_comprehensive_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-4/summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-5/PHASE_5_COMPLETE.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-5/p5_t1_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-5/p5_t1_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-5/p5_t2_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-5/p5_t2_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-5/p5_t3_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-5/p5_t3_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-5/p5_t4_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-5/p5_t4_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/PHASE_6_COMPLETE.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-6/PHASE_6_GATE_REPORT.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/PHASE_6_INTEGRATION_COMPLETE.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/PHASE_6_STATUS_REPORT.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/junit_all.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-6/p6_t1_completion_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/p6_t1_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t1_refactoring_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t1_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t2_completion_report.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t2_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t2_junit_fixed.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t3_completion_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-6/p6_t3_fix_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/p6_t3_fix_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/p6_t3_fixes.patch` | other | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | yes | no |  |  |
| `reports/phase-6/p6_t3_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-6/p6_t3_quick_reference.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-6/p6_t3_t4_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-6/p6_t4_completion_report.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-6/p6_t4_junit.xml` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-6/p6_t4_summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-6/test_progression.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/DEBUGGING_REPORT.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-7/baseline-summary.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/code/traversal.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/queries/graph-statistics.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/queries/neo4j-test-results.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-7/queries/traversal-procedure-test.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/queries/traversal-query-fixed.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/queries/traversal-query-simple.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/queries/traversal-query-union.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7/queries/traversal-query.cypher` | cypher | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/PHASE0-COMPLETION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-7E/PHASE7E-1-COMPLETION.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-7E/REMEDIATION-PLAN.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `reports/phase-7E/backfill-20251027.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/backfill-corrected.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/backfill-fixed.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/distribution-verified.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/distribution-verified.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-7E/phase-7e-1-verification.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-7E/preflight-second-pass-review.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-7E/preflight.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `reports/phase-7E/preflight_results.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/queries-final.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/phase-7E/validation-20251027.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/validation-corrected.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase-7E/validation-verified.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `reports/phase7c-quality-baseline.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `reports/retrieval_benchmarks/canonical_gpu_20260308.json` | json | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `requirements.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `rrf-fusion-no-neo4j-20251205.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `scripts/QUICKSTART-cleanup.md` | script | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no |  |  |
| `scripts/README-cleanup.md` | script | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no |  |  |
| `scripts/apply_complete_schema_v2_1.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/attribution_test.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/backfill_cross_doc_edges.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/backfill_doc_title_vectors.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/backfill_document_tokens.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/baseline_distribution_analysis.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/batch_crossdoc_link.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/benchmark_parsers.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/ci/check_dead_imports.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/ci/check_phase_gate.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/cleanup-databases.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/compare_fusion_methods.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/dev/seed_minimal_graph.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/diagnose_missing_chunks.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/eval/__init__.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/eval/compare_lexical_modes.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/eval/go_no_go.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/eval/resolve_gold_ids.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/eval/run_eval.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/eval/run_gold_eval.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/evaluate_retrieval.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/generate_massive_test_doc.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/generate_preflight_report.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/hybrid_rag_helpers.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/ingestctl` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/init_schema.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/mcp_transport_parity_check.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/migrate_phase5_payload_indexes.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/migrate_section_to_chunk.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/migration/phase2_cleanup_edges.cypher` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/monitor_ingestion.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/neo4j_structural_migration.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/perf/test_traversal_latency.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/perf/test_verbosity_latency.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/phase0/capture_baseline.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/phase7e_preflight.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/ppi_relationship_build.cypher` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/ppi_run.sh` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant and helpers/1535/hybrid_rag_helpers-1535.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant and helpers/1535/qdrant_setup_chunks_multi-1535.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant and helpers/best-of-both.patch` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant and helpers/hybrid_rag_helpers-1546.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant and helpers/qdrant_setup_chunks_multi-1546.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_setup_snowflake_arctic.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/chunks_multi_bge_m3_schema.json` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/instructions-for-qdrant-schema-scripts-20251206.md` | script | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/qdrant_schema_snapshot.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/qdrant_snapshots/chunks_multi_bge_m3_schema.json` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/qdrant_snapshots/chunks_multi_qwen3_06B_schema.json` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/qdrant_snapshots/chunks_multi_qwen3_0_6b_schema.json` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/qdrant_snapshots_20251206_canonical/qdrant_snapshots/chunks_multi_voyage_context_3_schema.json` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/reset_datastores.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/retrieval_diagnostics/show.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/run_baseline_queries.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/run_canonical_retrieval_benchmark.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/run_local.sh` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/run_phase7e_phase0.sh` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/scaffold.sh` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/smoke_test_golden_queries.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/smoke_test_query.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test/check_phase3_metrics.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test/check_phase4_metrics.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test/debug_explain.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/test/run_phase.sh` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test/summarize.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test/summarize_phase3.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test_doc_title_integration.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/test_jina_integration.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/test_jina_payload_limits.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test_sparse_ingestion.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/test_stdio_container_rrf.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/validate_entity_chunk_sync.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/validate_gds_readiness.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/validate_profile_storage.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/validate_token_accounting.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/verify_dead_code.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/verify_embedding_fields.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `scripts/verify_providers.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | no | no |  |  |
| `scripts/verify_stdio_rrf_wiring.py` | script | yes | partial | entrypoint/config/import scan | operator/legacy-mixed | manual | yes | no |  |  |
| `services/gliner-ner/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no | context/contract |  |
| `services/gliner-ner/requirements.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `services/gliner-ner/run.sh` | shell | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `services/gliner-ner/server.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `services/mxbai-reranker/README.md` | documentation | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no | context/contract |  |
| `services/mxbai-reranker/requirements.txt` | text | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `services/mxbai-reranker/run.sh` | shell | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `services/mxbai-reranker/server.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `src/clients/__init__.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no |  |  |
| `src/clients/embedding_client.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | EmbeddingClientError, EmbeddingClient, __init__, close, _handle_error, embed_dense |  |
| `src/clients/qwen3_embedding_client.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | Qwen3EmbeddingClient, __init__, close, __enter__, __exit__, _headers |  |
| `src/clients/snowflake_embedding_client.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | SnowflakeEmbeddingClient, __init__, close, __enter__, __exit__, _headers |  |
| `src/connectors/README.md` | source | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no |  |  |
| `src/connectors/RUNBOOK.md` | source | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | no | no |  |  |
| `src/connectors/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/connectors/base.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | ConnectorStatus, ConnectorConfig, IngestionEvent, BaseConnector, __init__, fetch_changes |  |
| `src/connectors/github.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | GitHubConnector, __init__, _map_status_to_event, _is_docs_file, _create_event, fetch_changes |  |
| `src/connectors/manager.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | ConnectorManager, __init__, register_connector, get_connector, start_polling, stop_polling |  |
| `src/connectors/queue.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | IngestionQueue, __init__, enqueue, dequeue, get_size, is_backpressure |  |
| `src/ingestion/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/ingestion/atomic.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | retry_with_backoff, decorator, wrapper, AtomicIngestionResult, to_dict, AtomicIngestionCoordinator | active ingestion god module; dirty before audit |
| `src/ingestion/auto/README.md` | source | no | metadata | inventoried for coverage; not opened line-by-line | historical/context | no | yes | no |  |  |
| `src/ingestion/auto/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/ingestion/auto/cli.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | ProgressUI, __init__, render, finish, compute_file_checksum, resolve_targets |  |
| `src/ingestion/auto/progress.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | JobStage, ProgressEvent, ProgressTracker, __init__, emit, advance |  |
| `src/ingestion/auto/queue.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _redis_from_env, JobStatus, IngestJob, to_json, from_json, _ensure_list |  |
| `src/ingestion/auto/reaper.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | JobReaper, __init__, _get_processing_jobs, _get_job_age, _requeue_job, _fail_job |  |
| `src/ingestion/auto/service.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _env_float, _env_bool, _env_str, startup, shutdown, health |  |
| `src/ingestion/auto/watchers.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | FileSystemWatcher, __init__, start, stop, _watch_loop, _scan_directory |  |
| `src/ingestion/build_graph.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | GraphBuilder, __init__, ensure_embedder, _build_section_text_for_embedding, _build_title_text_for_embedding |  |
| `src/ingestion/chunk_assembler.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | ChunkAssembler, assemble, post_process_chunk, _text_hash, _shingle_hash, get_chunk_assembler |  |
| `src/ingestion/extract/__init__.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | extract_entities |  |
| `src/ingestion/extract/commands.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no | extract_commands, _extract_from_code_block, _extract_inline_commands, _extract_documented_commands, _looks_like_command, _parse_command_line |  |
| `src/ingestion/extract/configs.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | extract_configurations, _extract_config_files, _extract_config_parameters, _extract_env_variables, _extract_from_config_code, _is_valid_config_name |  |
| `src/ingestion/extract/ner_gliner.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no | enrich_chunks_with_entities |  |
| `src/ingestion/extract/procedures.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | _hash16, extract_procedures |  |
| `src/ingestion/extract/references.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | slugify_for_id, Reference, ReferenceEdge, extract_references, normalize_filename_to_title, create_reference_edge |  |
| `src/ingestion/neo4j_writers.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | Neo4jWriter, __init__, _sanitize_for_neo4j, _neo4j_upsert_document, _neo4j_upsert_sections, _neo4j_upsert_entities |  |
| `src/ingestion/parsers/__init__.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | get_parser_engine, get_shadow_mode, get_fail_on_mismatch, parse_markdown, _parse_with_engine, _parse_with_shadow_comparison |  |
| `src/ingestion/parsers/html.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | parse_html, _extract_title, _parse_sections, _finalize_section, _compute_document_id, _compute_section_id |  |
| `src/ingestion/parsers/markdown.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _slugify, _normalize_text, _section_checksum, _section_id, _extract_frontmatter, parse_markdown |  |
| `src/ingestion/parsers/markdown_it_parser.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _slugify, _normalize_text, _section_checksum, _section_id, _compute_document_id, _compute_checksum |  |
| `src/ingestion/parsers/shadow_comparison.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | ShadowModeError, __init__, ParserComparisonResult, section_count_differs, section_count_delta, summary |  |
| `src/ingestion/qdrant_writers.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _get_response_handling_exception, _PlaceholderException, QdrantWriter, __init__, _qdrant_upsert_vectors, estimate_point_bytes |  |
| `src/ingestion/run_stats.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | FailedJob, IngestionRunStats, start_new, record_job, _aggregate_success_stats, record_warning |  |
| `src/ingestion/saga.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | SagaContext, ValidationResult, to_dict, IngestionValidator, __init__, validate_pre_ingest |  |
| `src/ingestion/semantic.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | SemanticEnrichmentResult, SemanticEnricher, __init__, enrich, StubSemanticEnricher, enrich |  |
| `src/ingestion/semantic_chunker.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _text_hash, _shingle_hash, SemanticChunkerAssembler, __init__, _initialize_chunker, _get_tokenizer |  |
| `src/ingestion/structural_edges.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | StructuralEdgeStats, to_dict, StructuralEdgeResult, to_dict, build_structural_edges_in_tx, _normalize_parent_path |  |
| `src/ingestion/worker.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | global_exception_handler, log_task_exception_callback, create_monitored_task, parse_file_uri, process_job, _env_bool |  |
| `src/mcp_server/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/mcp_server/main.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _ensure_streamable_accept_headers, _apply_legacy_mcp_deprecation_headers, _mcp_streamable_http_app, _build_connector_manager, correlation_id_middleware, metrics_middleware |  |
| `src/mcp_server/mcp_app.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | lifespan, _summary_for_tool, _invoke_tool, build_mcp_server, _list_tools, _call_tool | active MCP tool registry and WEKA instruction surface |
| `src/mcp_server/mcp_search.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _kb_search_candidates, _extract_evidence_from_passages, _append_quote, _expand_evidence_with_structure, _infer_source, _infer_source_tags |  |
| `src/mcp_server/mcp_tools.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _error_schema, _with_error, search_documentation, kb_search, kb_read_excerpt, kb_expand_excerpt |  |
| `src/mcp_server/mcp_utils.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _encode_cursor, _decode_cursor, _coerce_bool, _new_budget, _apply_budget, _report_progress |  |
| `src/mcp_server/models.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | MCPInitializeRequest, MCPInitializeResponse, MCPTool, MCPToolsListResponse, MCPToolCallRequest, MCPToolCallResponse |  |
| `src/mcp_server/query_service.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | QueryService, __init__, _embedding_diag, _get_embedder, _get_reranker, _get_search_engine |  |
| `src/mcp_server/retrieval_trace.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | TraceCandidate, TraceQuote, TraceFollowup, RetrievalTraceBuilder, __init__, record_query |  |
| `src/mcp_server/scratch_store.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | ScratchEntry, ScratchStore, __init__, build_uri, put, get |  |
| `src/mcp_server/stdio_server.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | _stderr_print, run_stdio_server, main |  |
| `src/mcp_server/webhooks.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | github_webhook, notion_webhook, confluence_webhook, webhook_health |  |
| `src/monitoring/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/monitoring/health.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | HealthStatus, HealthCheckResult, is_ok, SystemHealth, is_ok, get_failures |  |
| `src/monitoring/metrics.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | ChunkMetrics, to_dict, RetrievalMetrics, to_dict, IngestionMetrics, to_dict |  |
| `src/monitoring/slos.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | AlertLevel, SLOType, SLODefinition, SLOViolation, __str__, SLOMonitor |  |
| `src/neo/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/neo/contract_checks.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | GraphContractChecker, __init__, find_documents_needing_repair |  |
| `src/neo/schema.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no |  |  |
| `src/neo/schema_validator.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | SchemaValidationResult, validate_neo4j_schema |  |
| `src/providers/embeddings/__init__.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no |  |  |
| `src/providers/embeddings/arctic_chonkie_adapter.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | ArcticChonkieAdapter, __init__, _normalize_model_name, _create_client, _embed_single_batch, _health_check |  |
| `src/providers/embeddings/base.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | EmbeddingProvider, dims, model_id, provider_name, embed_documents, embed_query |  |
| `src/providers/embeddings/base_chonkie_adapter.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | BaseChonkieAdapter, __init__, _create_client, _embed_single_batch, _health_check, _get_client |  |
| `src/providers/embeddings/chonkie_adapter.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | BgeM3ChonkieAdapter, __init__, _get_client, _get_tokenizer, dimension, embed |  |
| `src/providers/embeddings/contracts.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | SparseEmbedding, MultiVectorEmbedding, QueryEmbeddingBundle |  |
| `src/providers/embeddings/embedding_service.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | _to_sparse_embedding, _to_multivector, _load_embedding_client_symbols, EmbeddingServiceProvider, __init__, dims |  |
| `src/providers/embeddings/jina.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | RateLimiter, __init__, wait_if_needed, JinaEmbeddingProvider, __init__, dims |  |
| `src/providers/embeddings/qwen3_chonkie_adapter.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | Qwen3ChonkieAdapter, __init__, _create_client, _embed_single_batch, _health_check, __repr__ |  |
| `src/providers/embeddings/qwen3_triton.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | Qwen3TritonProvider, __init__, dims, model_id, provider_name, embed_documents |  |
| `src/providers/embeddings/sentence_transformers.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | SentenceTransformersProvider, __init__, _validate_dimensions, dims, model_id, provider_name |  |
| `src/providers/embeddings/snowflake_arctic.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | SnowflakeArcticProvider, __init__, dims, model_id, provider_name, embed_documents |  |
| `src/providers/embeddings/voyage.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | VoyageEmbeddingProvider, __init__, dims, model_id, provider_name, close |  |
| `src/providers/factory.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | ProviderFactory, create_embedding_provider, create_embedding_provider_for_role, _normalize_provider, _apply_legacy_overrides, _build_settings_from_profile |  |
| `src/providers/ner/__init__.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no |  |  |
| `src/providers/ner/gliner_service.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | Entity, to_dict, GLiNERService, __new__, __init__, _get_http_client |  |
| `src/providers/ner/labels.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no | get_default_labels, extract_label_name, get_label_names, is_excluded_entity, normalize_entity_name, is_excluded_structural_entity |  |
| `src/providers/rerank/__init__.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no |  |  |
| `src/providers/rerank/base.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | RerankProvider, model_id, provider_name, rerank, health_check |  |
| `src/providers/rerank/jina.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | JinaRerankProvider, __init__, model_id, provider_name, rerank, health_check |  |
| `src/providers/rerank/local_reranker_service.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | batch_documents, LocalRerankerServiceProvider, __init__, __del__, close, __enter__ |  |
| `src/providers/rerank/noop.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | NoopReranker, __init__, model_id, provider_name, rerank, health_check |  |
| `src/providers/settings.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | EmbeddingCapabilities, EmbeddingSettings, build_embedding_telemetry |  |
| `src/providers/tokenizer_service.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _resolve_local_snapshot_path, TokenizerBackend, count_tokens, encode, decode, HuggingFaceTokenizerBackend |  |
| `src/query/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `src/query/context_assembly.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _normalize_and_order_citations, _sort_key, AssembledContext, ContextAssembler, __init__, assemble |  |
| `src/query/entity_extraction.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | EntityExtractor, __init__, _insert, _should_include, _build_trie, extract_entities |  |
| `src/query/expansion_pipeline.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | neighbor_score, result_id, path_prefix, should_expand, dedup_results, build_expanded_chunk |  |
| `src/query/fusion_pipeline.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | rrf_fusion, weighted_fusion, normalize_scores, apply_doc_continuity_boost |  |
| `src/query/graph_pipeline.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | chunk_from_props, relationships_for_query, relationships_for_query_type, get_query_type_weights, compute_graph_signals, compute_cross_doc_signals |  |
| `src/query/hybrid_retrieval.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | HybridRetriever, __init__, _normalize_filter_value, _normalize_filters, _classify_query_type, _relationships_for_query |  |
| `src/query/hybrid_search.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no | build_metadata_projection, SearchResult, HybridSearchResults, VectorStore, search, QdrantVectorStore | documented legacy path; keep quarantined from active QueryService path |
| `src/query/planner.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | QueryPlan, TemplateLibrary, __init__, _load_templates, _parse_template_versions, _select_default_version |  |
| `src/query/processing/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/processing/disambiguation.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | QueryAnalysis, has_entities, QueryDisambiguator, __init__, labels, threshold |  |
| `src/query/query_intent.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | QueryIntent, _partition_terms, _term_matches, classify_query_intent |  |
| `src/query/ranking.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | legacy/alternative | no | no | no | RankingFeatures, RankedResult, Ranker, __init__, rank, _extract_features | legacy ranker according to dependency graph |
| `src/query/rerank_pipeline.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | get_reranker, apply_reranker, _clean_text, _build_focused_text, _approx_tokens, hydrate_colbert_vectors |  |
| `src/query/response_builder.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | Verbosity, Evidence, Diagnostics, StructuredResponse, to_dict, Response |  |
| `src/query/retrieval_observability.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | log_stage_snapshot |  |
| `src/query/retrieval_plan.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | RetrievalProfile, ResolvedRetrievalPlan, resolve_retrieval_plan, _zero_graph_fields, _infer_from_legacy_flags, _warn_legacy_flag_drift |  |
| `src/query/retrieval_types.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | FusionMethod, ExpandWhen, ChunkResult, __post_init__, _snapshot_top, _deduplicate_entity_metadata |  |
| `src/query/session_tracker.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | SessionTracker, __init__, create_session, ensure_session, create_query, extract_focused_entities |  |
| `src/query/signal_pool.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | SignalPoolResult, build_signal_pool, _add, _has_per_field_scores, _fill_signal_slot, _fill_per_field_slots |  |
| `src/query/structural_retrieval.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | StructuralRetrievalConfig, get_query_type_rrf_weights, build_structural_filter, apply_structural_boost, get_structural_boost_info |  |
| `src/query/templates/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/advanced/.gitkeep` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/advanced/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | TemplateGuardrails, TemplateSchema |  |
| `src/query/templates/advanced/comparison.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/advanced/dependency_chain.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/advanced/impact_assessment.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/advanced/schemas.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | get_template, list_templates |  |
| `src/query/templates/advanced/temporal.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/advanced/troubleshooting_path.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/compare.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/explain.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/search.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/traverse.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/templates/troubleshoot.cypher` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/query/traversal.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | TraversalNode, TraversalRelationship, TraversalResult, to_dict, TraversalService, __init__ |  |
| `src/query/vector_backends.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | BM25Retriever, __init__, _list_indexes, _ensure_fulltext_index, search, QdrantMultiVectorRetriever |  |
| `src/services/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/services/context_assembler.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | SummarizationService, __init__, summarize_neighborhood, ContextAssemblerService, __init__, compute_context_bundle |  |
| `src/services/context_budget_manager.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | BudgetExceeded, __init__, PhaseUsage, ContextBudgetManager, __post_init__, usage |  |
| `src/services/cross_doc_edge_model.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | CandidateSignals, score_final, StructuralPriors, EdgePayload, to_neo4j_params, _compute_quality_tier |  |
| `src/services/cross_doc_linking.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | LinkingResult, to_dict, BatchLinkingStats, from_results, escape_lucene_query, prepare_lucene_phrase_query |  |
| `src/services/delta_cache.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | SessionEntry, touch, SessionDeltaCache, __init__, _purge, _entry |  |
| `src/services/graph_service.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | ProjectionViolation, _encode_cursor, _decode_cursor, GraphResult, GraphService, __init__ |  |
| `src/services/text_service.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | TextFetchResult, TextService, __init__, get_section_text |  |
| `src/shared/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/shared/cache.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | L1Cache, __init__, get, _record_metrics, put, invalidate |  |
| `src/shared/chunk_utils.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | generate_chunk_id, create_chunk_metadata, validate_chunk_schema, create_combined_chunk_metadata, canonicalize_parent_ids |  |
| `src/shared/config.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | yes | no | EmbeddingConfig, model_name, validate_similarity, validate_dims, validate_version, Config |  |
| `src/shared/connections.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | _as_sequence, _convert_point_id, _normalize_points_selector, CompatQdrantClient, __init__, __getattr__ |  |
| `src/shared/embedding_fields.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | canonicalize_embedding_metadata, read_embedding_version_with_fallback, validate_embedding_metadata, ensure_no_embedding_model_in_payload, create_write_payload |  |
| `src/shared/logging.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/shared/models.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | WekaBaseModel |  |
| `src/shared/observability/__init__.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no |  |  |
| `src/shared/observability/exemplars.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | get_trace_context, trace_mcp_tool, trace_cypher_query, trace_vector_search, trace_hybrid_search, trace_graph_expansion |  |
| `src/shared/observability/logging.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no | _setup_otel_logs, get_correlation_id, set_correlation_id, add_correlation_id, add_trace_context, setup_logging |  |
| `src/shared/observability/metrics.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no | PrometheusMiddleware, dispatch, setup_metrics, get_metrics |  |
| `src/shared/observability/retrieval_diagnostics.py` | source | yes | partial | entrypoint/config/import scan | active | yes | no | no | _bool_env, _float_env, _int_env, RetrievalDiagnosticEmitter, __init__, _should_emit |  |
| `src/shared/observability/tracing.py` | source | yes | partial | entrypoint/config/import scan | active | yes | yes | no | setup_tracing, get_tracer, init_tracing |  |
| `src/shared/qdrant_schema.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no | QdrantSchemaPlan, build_qdrant_schema, validate_qdrant_schema |  |
| `src/shared/resilience/__init__.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `src/shared/resilience/circuit_breaker.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | _safe_parse_int, _safe_parse_float, CircuitState, CircuitBreaker, __init__, state |  |
| `src/shared/schema.py` | source | yes | full | runtime path or primary contract reviewed | active | yes | no | no | parse_cypher_statements, create_schema, apply_schema_v2_1, create_vector_indexes, verify_schema, ensure_schema_version |  |
| `src/shared/section_metadata.py` | source | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no | compute_parent_path_depth, compute_dominant_block_type, extract_enhanced_metadata, has_enhanced_metadata, build_context_prefix, merge_enhanced_metadata_to_chunk |  |
| `test_extract_import.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `test_import.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `test_verify_imports.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `tests/baselines/test_phase0_gates.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/clients/__init__.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/clients/test_snowflake_embedding_client.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/conftest.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/contracts/test_cypher_policy.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/contracts/test_graph_v2_contracts.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/contracts/test_mcp_streamable_contracts.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e/test_golden_set.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/README.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/__init__.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/conftest.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/test_prod_chunking_invariants.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/test_prod_graph_alignment.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/test_prod_ingestion_markdown.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/test_prod_mcp_retrieval.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/test_prod_retrieval_hybrid.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/e2e_v22_prod/test_prod_vectors_qdrant.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/baseline_query_set.yaml` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/canonical_retrieval_benchmark.yaml` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/doc_with_references_a.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/doc_with_references_b.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/doc_with_references_c.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/doc_with_references_d.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/golden_query_set.yaml` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/fixtures/procedure_with_steps.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/ingestion/test_build_graph_sparse.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/ingestion/test_namespace_enforcement.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/__init__.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/prod_docs_pack_noprefix/README.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/prod_docs_pack_noprefix/artifacts/phase7e3_prod_docs_report.json` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/prod_docs_pack_noprefix/docs/additional-protocols_s3_s3-information-lifecycle-management_s3-information-lifecycle-management.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/prod_docs_pack_noprefix/runner/prod_docs_runner.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_gds_readiness.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_gliner_config.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_gliner_ingestion_flow.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_gliner_live.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_jina_large_batches.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_phase1_entity_edges.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_phase2_gates.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_phase4_entity_retrieval.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_phase7c_integration.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_precision_retrieval_live.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_profile_ingestion_retrieval.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_profile_matrix_integration.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_references_e2e.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_related_to_integration.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_session_tracking.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/integration/test_sparse_colbert_integration.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/mcp_server_tests/__init__.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/mcp_server_tests/test_evidence_pack.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/mcp_server_tests/test_tool_profiles.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p1_t1_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p1_t2_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p1_t3_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p2_t1_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p2_t3_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p2_t4_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p3_t1_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p3_t2_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p4_t1_complex_patterns_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p4_t1_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p4_t2_optimizer_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p4_t3_cache_perf_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p5_t1_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p5_t2_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p5_t4_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/p6_t1_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/p6_t3_test.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/providers/test_arctic_chonkie_adapter.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/providers/test_bge_m3_service_provider.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/providers/test_profile_matrix.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/providers/test_snowflake_arctic_provider.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/providers/test_voyage_provider.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_context_assembly_partial.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_guardrails_modes.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_hybrid_bridge.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_hybrid_retrieval_text.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_hybrid_search_strategy.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_multivector_sparse_colbert.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_qdrant_multivector_sparse.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_qdrant_vector_store.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_query_api_payload.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_query_intent.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_reranker_integration.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_reranker_mode.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_retrieval_plan.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_signal_pool.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/query/test_vector_store.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/scripts/test_backfill_edge_parity.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/scripts/test_verify_providers.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/services/__init__.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/services/test_mxbai_reranker_service.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_config_reload.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_embedding_plan_fingerprint.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_embedding_profiles.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_namespace_suffix.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_qdrant_schema.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_qdrant_schema_plan_dims.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/shared/test_qdrant_schema_validation.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_chunk_parent_mapping.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_cleanup_validation.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_config_schema_completeness.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_embedding_field_canonicalization.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_graph_contract.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_integration_prephase7.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_jina_adaptive_batching.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase1_foundation.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase2_provider_wiring.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase4_ranking_coverage.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase5_response_schema.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase7c_dual_write.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase7c_provider_factory.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase7c_reranking.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase7c_schema_v2_1.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase7e2_hybrid_retrieval.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_phase7e_phase0.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_query_api_weighted_fusion.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_structure_aware_expansion.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/test_tokenizer_service.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_colbert_observability.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_context_microdoc.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_cross_doc_edge_model.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_cross_doc_linking_v2.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_entity_quality_gating.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_gliner_service.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_markdown_it_parser.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_ner_gliner_enrichment.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_parser_shadow_mode.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_phase1_reranker_batching.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_phase2_schema_cleanup.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_query_disambiguation.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_reciprocity.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_related_to_retrieval.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_related_to_schema.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_related_to_signal_pool.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_section_metadata.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_single_section_chunking.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_source_attribution.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_structural_precision.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_structural_priors.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_structural_retrieval.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/unit/test_tokenizer_overlap.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | yes | no |  | tests intentionally not trusted as architecture proof |
| `tests/v2_2/chunking/test_structured_chunker.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/v2_2/conftest.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tests/v2_2/test_hybrid_ranking_behaviors.py` | test | no | metadata | inventoried for coverage; not opened line-by-line | unknown-test | no | no | no |  | tests intentionally not trusted as architecture proof |
| `tmpfile` | other | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `tools/__init__.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | yes | no |  |  |
| `tools/fusion_ab.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `tools/redis_epoch_bump.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
| `tools/redis_invalidation.py` | python | no | metadata | inventoried for coverage; not opened line-by-line | unknown | unknown | no | no |  |  |
