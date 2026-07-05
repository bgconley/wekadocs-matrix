# WEKA Residue Surface Map - 2026-07-04

Generated from the Nutanix migration WIP branch after syncing current `origin/master`.

## Scan Command

```bash
rg -n --hidden --pcre2 --glob '!.git/**' --glob '!.venv/**' --glob '!__pycache__/**' --glob '!node_modules/**' --glob '!reports/retrieval_diagnostics/**' '(?i)(\bweka\b|wekadocs|weka[-_:]|weka docs|WekaDocs)' .
```

## Summary

- Total residue hits: 39666
- Total files with residue: 309
- Runtime/config/test blocker hits: 219

| Category | Hits | Files | Migration Meaning |
| --- | ---: | ---: | --- |
| runtime_source | 1 | 1 | Active source or scripts; blockers until renamed or intentionally archived. |
| config_deploy_ops | 1 | 1 | Runtime configuration, deployment, CI, or operations surfaces; blockers until renamed. |
| tests | 217 | 41 | Tests and fixtures; blockers unless explicitly historical. |
| docs_active | 8459 | 130 | Active docs/plans; rename or mark historical. |
| generated_reports | 30591 | 84 | Generated analysis/report artifacts; archive, regenerate, or exclude from runtime residue gates. |
| archive_historical | 0 | 0 | Historical archive candidates; allowed only under explicit archive policy. |
| other | 397 | 52 | Needs manual classification. |

## Top Files By Hit Count

| Hits | File |
| ---: | --- |
| 29147 | `./reports/phase-7/queries/neo4j-test-results.txt` |
| 6643 | `./docs/wekadocs50_combined.md` |
| 368 | `./reports/community_detection/doc_related_to_louvain_20260304_065448Z.json` |
| 358 | `./reports/community_detection/doc_related_to_louvain_20260304_065612Z.json` |
| 289 | `./docs/cdx-outputs/gliner-implementation-sessions/gliner-implementation-complete-chronological.md` |
| 145 | `./reports/community_detection/doc_related_to_louvain_20260304_065448Z.md` |
| 134 | `./reports/retrieval_benchmarks/canonical_gpu_20260308.json` |
| 133 | `./reports/community_detection/doc_related_to_louvain_20260304_065612Z.md` |
| 73 | `./docs/superpowers/plans/2026-07-04-stabilize-nutanix-wip-branch.md` |
| 67 | `./claude-raw/planning-and-installation-prerequisites-and-compatibility.md` |
| 44 | `./docs/repo_audit/2026-07-04-static-architecture-code-only-audit.md` |
| 40 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-retrieval-debugging.md` |
| 36 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251209-rrf-debugging-and-embedding-investigation.md` |
| 35 | `./claude-raw/wekadocs-matrix-implementation-plan-v1-claude.md` |
| 34 | `./docs/plans/2026-02-10-p620-rtx3090-migration.md` |
| 34 | `./reports/VOLUME_PERSISTENCE_INVESTIGATION.md` |
| 33 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-ingestion-complete.md` |
| 32 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-complete.md` |
| 32 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-phase1-implementation.md` |
| 32 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-complete.md` |
| 32 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-phase1-implementation.md` |
| 31 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-mps-acceleration.md` |
| 31 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-2025-12-07-vector-pipeline-prep.md` |
| 31 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-mps-acceleration.md` |
| 31 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-2025-12-07-vector-pipeline-prep.md` |
| 30 | `./docs/blue-green-migration.md` |
| 30 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-integration-planning.md` |
| 30 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-integration-planning.md` |
| 30 | `./docs/quickstart-commands.md` |
| 29 | `./docs/architecture/2026-03-04-end-to-end-architecture.md` |
| 29 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-phase3-complete.md` |
| 29 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251209-gliner-entity-sparse-verification.md` |
| 29 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-phase3-complete.md` |
| 28 | `./docs/session-notes/2026-03-04-integration-deployment-retrieval-tuning.md` |
| 28 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-gliner-graph-disabled-fixes.md` |
| 28 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251209-entity-sparse-fix.md` |
| 26 | `./reports/baseline/collected_tests.txt` |
| 25 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-phase4-entity-retrieval.md` |
| 25 | `./docs/cdx-outputs/gliner-implementation-sessions copy/session-context-20251208-gliner-review-complete.md` |
| 25 | `./docs/cdx-outputs/gliner-implementation-sessions/session-context-20251208-phase4-entity-retrieval.md` |

## Runtime/Config/Test Blockers

| Category | File | Line | Text |
| --- | --- | ---: | --- |
| runtime_source | `./tools/__init__.py` | 1 | `# Tools package for wekadocs-matrix` |
| tests | `./tests/test_phase1_foundation.py` | 72 | `"This is a test document about Weka filesystem.",` |
| tests | `./tests/test_phase1_foundation.py` | 92 | `test_query = "How do I configure NFS for Weka?"` |
| tests | `./tests/p6_t3_test.py` | 42 | `weka cluster create` |
| tests | `./tests/p6_t3_test.py` | 111 | `cwd="/Users/brennanconley/vibecode/wekadocs-matrix",` |
| config_deploy_ops | `./.env.example` | 2 | `# WekaDocs Matrix — Environment Configuration` |
| tests | `./tests/e2e_v22_prod/conftest.py` | 196 | `"weka-neo4j",` |
| tests | `./tests/e2e_v22_prod/conftest.py` | 197 | `"weka-qdrant",` |
| tests | `./tests/e2e_v22_prod/conftest.py` | 198 | `"weka-redis",` |
| tests | `./tests/e2e_v22_prod/conftest.py` | 199 | `"weka-mcp-server",` |
| tests | `./tests/e2e_v22_prod/conftest.py` | 200 | `"weka-ingestion-service",` |
| tests | `./tests/e2e_v22_prod/conftest.py` | 201 | `"weka-ingestion-worker",` |
| tests | `./tests/p3_t2_test.py` | 34 | `# Should find weka commands` |
| tests | `./tests/p3_t2_test.py` | 36 | `weka_commands = [c for c in command_names if "weka" in c.lower()]` |
| tests | `./tests/p3_t2_test.py` | 37 | `assert len(weka_commands) > 0` |
| tests | `./tests/p3_t2_test.py` | 98 | `# Should find weka.conf` |
| tests | `./tests/p3_t2_test.py` | 100 | `assert any("weka.conf" in name for name in config_names)` |
| tests | `./tests/p3_t2_test.py` | 127 | `"text": "Set $WEKA_HOME and ${MAX_MEMORY} before starting.",` |
| tests | `./tests/p3_t2_test.py` | 143 | `assert "WEKA_HOME" in config_names or "MAX_MEMORY" in config_names` |
| tests | `./tests/p2_t4_test.py` | 43 | `"name": "weka cluster create",` |
| tests | `./tests/p2_t4_test.py` | 251 | `"how to install weka", "search", sample_ranked_results, {}` |
| tests | `./tests/p2_t4_test.py` | 254 | `assert "how to install weka" in response.answer_markdown` |
| tests | `./tests/p2_t4_test.py` | 356 | `query="how to install weka",` |
| tests | `./tests/e2e/test_golden_set.py` | 32 | `"query": "How do I install Weka on Ubuntu?",` |
| tests | `./tests/e2e/test_golden_set.py` | 38 | `"query": "What are the hardware requirements for Weka?",` |
| tests | `./tests/e2e/test_golden_set.py` | 44 | `"query": "How do I set up a Weka cluster?",` |
| tests | `./tests/e2e/test_golden_set.py` | 50 | `"query": "How do I configure Weka licensing?",` |
| tests | `./tests/e2e/test_golden_set.py` | 57 | `"query": "How do I create a filesystem in Weka?",` |
| tests | `./tests/e2e/test_golden_set.py` | 63 | `"query": "How do I manage users and permissions in Weka?",` |
| tests | `./tests/e2e/test_golden_set.py` | 69 | `"query": "How do I configure networking for Weka?",` |
| tests | `./tests/e2e/test_golden_set.py` | 82 | `"query": "How do I monitor Weka cluster health?",` |
| tests | `./tests/e2e/test_golden_set.py` | 100 | `"query": "How do I collect and analyze Weka logs?",` |
| tests | `./tests/e2e/test_golden_set.py` | 107 | `"query": "How do I optimize SSD performance in Weka?",` |
| tests | `./tests/e2e/test_golden_set.py` | 144 | `"query": "How do I use the Weka REST API?",` |
| tests | `./tests/e2e/test_golden_set.py` | 150 | `"query": "How do I upgrade Weka to a new version?",` |
| tests | `./tests/query/test_vector_store.py` | 32 | `filters = {"doc_tag": "prod.docs", "tenant": "weka"}` |
| tests | `./tests/query/test_vector_store.py` | 44 | `assert "weka" not in query` |
| tests | `./tests/p6_t1_test.py` | 98 | `weka cluster create` |
| tests | `./tests/test_tokenizer_service.py` | 82 | `weka cluster create --name=production --nodes=10 --failure-domain=rack` |
| tests | `./tests/test_tokenizer_service.py` | 100 | `\| weka cluster create \| Create new cluster \| --name, --nodes \|` |
| tests | `./tests/test_tokenizer_service.py` | 101 | `\| weka fs create \| Create filesystem \| --name, --capacity \|` |
| tests | `./tests/test_tokenizer_service.py` | 102 | `\| weka user add \| Add user \| --username, --role \|"""` |
| tests | `./tests/test_tokenizer_service.py` | 526 | `### Command {i}: weka-command-{i}` |
| tests | `./tests/test_tokenizer_service.py` | 532 | `weka command-{i} [--flag1 VALUE] [--flag2 VALUE]` |
| tests | `./tests/test_tokenizer_service.py` | 544 | `weka command-{i} --flag1=production --flag2=100` |
| tests | `./tests/test_tokenizer_service.py` | 547 | `weka command-{i} --flag1=test` |
| tests | `./tests/test_tokenizer_service.py` | 575 | `cmd_text = "weka cluster create --name=prod --nodes=10 --failure-domain=rack"` |
| tests | `./tests/test_tokenizer_service.py` | 586 | `"\| weka fs create \| Create filesystem \| --name, --capacity, --thin \|"` |
| tests | `./tests/p2_t3_test.py` | 60 | `collection_name = "weka_sections"  # Default from config` |
| tests | `./tests/p2_t3_test.py` | 98 | `query_vector = embedder.encode("weka").tolist()` |
| tests | `./tests/shared/test_nutanix_domain.py` | 69 | `assert "WEKA" not in prompt_text` |
| tests | `./tests/shared/test_nutanix_domain.py` | 70 | `assert "weka" not in prompt_text` |
| tests | `./tests/p2_t1_test.py` | 59 | `entities = linker.link("Run weka cluster create to initialize")` |
| tests | `./tests/p2_t1_test.py` | 61 | `assert "weka cluster" in entities["command_name"]` |
| tests | `./tests/p4_t1_complex_patterns_test.py` | 64 | `MERGE (cmd:Command {id:$cmd, name:'weka diag', cli_syntax:'weka diag'})` |
| tests | `./tests/p4_t1_complex_patterns_test.py` | 131 | `assert steps[0]["cmd"] == "weka diag", "First step should execute 'weka diag'."` |
| tests | `./tests/conftest.py` | 139 | `"service.name": "wekadocs-mcp-test",` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 26 | `The WEKA distributed file system provides high-performance storage for` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 30 | `Configuration is managed through the weka CLI tool, which provides` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 35 | `weka cluster status --json` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 36 | `weka fs create myfs --total-capacity 10TiB --ssd-capacity 1TiB` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 37 | `mount -t wekafs backend1/myfs /mnt/weka` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 38 | `chmod 755 /mnt/weka/data` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 39 | `weka local resources --cores 4 --memory 32GiB` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 40 | `weka cluster drive add /dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 41 | `weka alerts list --severity error --since 24h` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 48 | `# Configure WEKA mount` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 49 | `MOUNT_POINT="/mnt/weka"` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 50 | `FS_NAME="${WEKA_FS:-default}"` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 63 | `# Installing WEKA on Ubuntu 22.04` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 75 | `curl -O https://get.weka.io/dist/v4/install.sh` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 82 | `weka status` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 83 | `weka cluster nodes` |
| tests | `./tests/unit/test_tokenizer_overlap.py` | 295 | `text = "weka cluster status --json --verbose --format=table"` |
| tests | `./tests/test_nutanix_hard_rename_guard.py` | 1 | `"""Static guard for runtime-connected WEKA residue."""` |
| tests | `./tests/test_nutanix_hard_rename_guard.py` | 20 | `LEGACY_PATTERN = re.compile(r"\b(?:W[E]KA\|WekaDocs\|wekadocs\|weka-docs\|weka-)\b")` |
| tests | `./tests/p3_t1_test.py` | 86 | `assert "weka" in all_code.lower()` |
| tests | `./tests/unit/test_markdown_it_parser.py` | 12 | `Author: WekaDocs Team` |
| tests | `./tests/fixtures/doc_with_references_b.md` | 16 | `weka fs snapshot policy create --name hourly --schedule "0 * * * *" --retain 24` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 1 | `# Golden Query Test Set for WekaDocs GraphRAG Smoke Testing` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 28 | `expected_keywords: ["weka nfs global-config set", "config-fs", "mountd-port"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 44 | `expected_keywords: ["weka nfs interface-group add", "subnet", "gateway"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 64 | `expected_keywords: ["weka s3 bucket add", "policy", "quota"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 68 | `text: "How to configure AWS CLI for WEKA S3?"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 71 | `- "Configure and use AWS CLI with WEKA S3 storage"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 76 | `text: "What are the S3 API limitations in WEKA?"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 111 | `text: "How to use boto3 with WEKA S3?"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 136 | `text: "weka s3 bucket quota command"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 140 | `expected_keywords: ["weka s3 bucket quota", "set", "reset"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 166 | `text: "How to export WEKA metrics to Prometheus?"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 190 | `text: "What is Local WEKA Home?"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 193 | `- "Local WEKA Home overview"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 194 | `- "WEKA Home - The WEKA support cloud"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 195 | `expected_keywords: ["weka home", "local", "support"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 199 | `text: "How to deploy Local WEKA Home v3?"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 202 | `- "Deploy Local WEKA Home v3.0 or higher"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 203 | `expected_keywords: ["deploy", "local weka home", "v3"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 218 | `text: "List of WEKA alerts and how to fix them"` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 230 | `expected_keywords: ["weka alerts", "cli", "manage"]` |
| tests | `./tests/fixtures/golden_query_set.yaml` | 312 | `text: "Enforce security and compliance in WEKA"` |
| tests | `./tests/fixtures/doc_with_references_c.md` | 8 | `WEKA tiering allows automatic data movement between hot and cold storage tiers.` |
| tests | `./tests/fixtures/doc_with_references_c.md` | 16 | `weka fs tier create --name archive-tier --backend s3://bucket` |
| tests | `./tests/integration/test_gliner_config.py` | 87 | `# Should have WEKA-specific labels` |
| tests | `./tests/integration/test_gliner_config.py` | 88 | `assert any("weka" in label.lower() for label in labels)` |
| tests | `./tests/integration/test_gliner_config.py` | 139 | `assert any("weka" in label.lower() for label in labels)` |
| tests | `./tests/unit/test_structural_retrieval.py` | 69 | `def test_real_weka_path(self):` |
| tests | `./tests/unit/test_structural_retrieval.py` | 70 | `"""Test with realistic WEKA documentation path."""` |
| tests | `./tests/unit/test_structural_retrieval.py` | 71 | `path = "WEKA System Overview > Planning > Networking Requirements"` |
| tests | `./tests/integration/__init__.py` | 1 | `"""Integration tests for wekadocs-matrix."""` |
| tests | `./tests/mcp_server_tests/test_nutanix_query_reformulation.py` | 13 | `assert "WEKA" not in rewritten` |
| tests | `./tests/mcp_server_tests/test_nutanix_query_reformulation.py` | 14 | `assert "weka" not in rewritten` |
| tests | `./tests/fixtures/doc_with_references_d.md` | 9 | `WEKA supports multiple DR strategies using snapshots and replication.` |
| tests | `./tests/fixtures/procedure_with_steps.md` | 8 | `This procedure explains how to configure tiering for your WEKA filesystem.` |
| tests | `./tests/fixtures/procedure_with_steps.md` | 11 | `1. Enable tiering on the filesystem using `weka fs tier enable --filesystem my-fs`. Verify the command completed successfully.` |
| tests | `./tests/fixtures/procedure_with_steps.md` | 13 | `2. Configure the tier policy with an age threshold using `weka fs tier policy set --filesystem my-fs --age 30d`. This moves data older than 30 days to the object store tier.` |
| tests | `./tests/fixtures/procedure_with_steps.md` | 15 | `3. Check that tiering is properly configured using `weka fs tier status --filesystem my-fs`. The output should show tiering enabled with your policy settings.` |
| tests | `./tests/fixtures/procedure_with_steps.md` | 17 | `4. Monitor tiering operations using `weka fs tier progress --filesystem my-fs` to track progress.` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 16 | `text: "how is metadata managed and architected on a weka cluster"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 40 | `text: "what are the metadata limitations in WEKA filesystems"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 54 | `text: "How do I configure a dedicated backend in WEKA?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 68 | `text: "How do I appropriately size the weka drives, compute, and frontends containers?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 76 | `- "Plan the WEKA system hardware requirements"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 91 | `text: "What are the minimum hardware requirements for a WEKA cluster?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 99 | `- "Minimal server configuration for a WEKA cluster"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 107 | `text: "How do I use the weka cluster drive command?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 113 | `any_of: ["weka cluster drive"]` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 118 | `any_of: ["getting-started-with-weka"]` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 126 | `text: "How do I expand storage capacity in a WEKA cluster?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 140 | `text: "what is WEKA deduplication"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 147 | `- "Data reduction **in WEKA filesystems**"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 152 | `- id: local_weka_home_ports_01` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 153 | `label: Local WEKA Home Ports` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 156 | `text: "What ports are required for Local WEKA Home?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 157 | `rationale: "Ports benchmark grounded in the Local WEKA Home deployment docs."` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 163 | `- "Deploy Local WEKA Home v3.0 or higher"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 164 | `- "Change the Local WEKA Home listening ports"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 166 | `note: "Local WEKA Home deployment/ports guidance should appear in the top 5."` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 169 | `label: WEKA GUI Overview` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 172 | `text: "How do I manage the system using the WEKA GUI?"` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 178 | `any_of: ["WEKA GUI overview"]` |
| tests | `./tests/fixtures/canonical_retrieval_benchmark.yaml` | 180 | `note: "WEKA GUI overview should appear in the top 3."` |
| tests | `./tests/fixtures/doc_with_references_a.md` | 8 | `WEKA filesystems support point-in-time snapshots for data protection.` |
| tests | `./tests/fixtures/doc_with_references_a.md` | 13 | `Use the `weka fs snapshot create` command to create a snapshot:` |
| tests | `./tests/fixtures/doc_with_references_a.md` | 16 | `weka fs snapshot create --filesystem my-fs --name daily-backup` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 1 | `# Baseline Query Set for WekaDocs GraphRAG` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 24 | `expected_topics: ["snapshot", "weka fs snapshot", "create"]` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 58 | `text: "How to upgrade WEKA software version?"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 71 | `text: "weka cluster container failure error"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 132 | `text: "Minimum hardware requirements for WEKA cluster"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 144 | `text: "Supported Linux distributions for WEKA"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 151 | `text: "How does WEKA tiering work?"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 157 | `text: "What is the WEKA data protection model?"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 163 | `text: "Explain WEKA cluster architecture"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 169 | `text: "How does WEKA handle data distribution?"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 175 | `text: "What is a WEKA container?"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 182 | `text: "WEKA CLI command reference"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 188 | `text: "API documentation for WEKA"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 201 | `text: "weka fs tier status command"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 203 | `expected_topics: ["weka fs tier", "status"]` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 207 | `text: "How to use weka cluster status"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 209 | `expected_topics: ["weka cluster", "status"]` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 213 | `text: "weka fs snapshot list options"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 215 | `expected_topics: ["weka fs snapshot", "list", "options"]` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 219 | `text: "weka events show command parameters"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 221 | `expected_topics: ["weka events", "show", "parameters"]` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 225 | `text: "weka alerts mute syntax"` |
| tests | `./tests/fixtures/baseline_query_set.yaml` | 227 | `expected_topics: ["weka alerts", "mute"]` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 44 | `Query: 'how is metadata managed and architected on a weka cluster'` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 49 | `#     "how is metadata managed and architected on a weka cluster",` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 65 | `Query: 'How do I appropriately size the weka drives, compute, and frontends containers?'` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 67 | `core WEKA planning/sizing docs.` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 70 | `#     "How do I appropriately size the weka drives, compute, and frontends containers?",` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 80 | `#     "WEKA inode management internals", top_k=10` |
| tests | `./tests/integration/test_precision_retrieval_live.py` | 88 | `#     "WEKA metadata architecture", top_k=10` |
| tests | `./tests/integration/test_gliner_live.py` | 60 | `def test_extract_weka_entities(self):` |
| tests | `./tests/integration/test_gliner_live.py` | 61 | `"""Verify extraction works with WEKA-specific text."""` |
| tests | `./tests/integration/test_gliner_live.py` | 68 | `# Real WEKA documentation text` |
| tests | `./tests/integration/test_gliner_live.py` | 70 | `To mount WEKA filesystem on RHEL 8, install the weka-agent package` |
| tests | `./tests/integration/test_gliner_live.py` | 71 | `and configure NFS exports. Use the weka fs mount command with` |
| tests | `./tests/integration/test_gliner_live.py` | 72 | `--net-apply option. Check /var/log/weka for errors.` |
| tests | `./tests/integration/test_gliner_live.py` | 89 | `expected_finds = ["weka", "rhel", "nfs"]` |
| tests | `./tests/integration/test_gliner_live.py` | 108 | `"Configure AWS S3 backend for WEKA cluster",` |
| tests | `./tests/integration/test_gliner_live.py` | 205 | `text = "Configure WEKA cluster with NFS exports on RHEL 8 servers."` |
| tests | `./tests/integration/test_gliner_live.py` | 231 | `f"Document {i}: Configure WEKA backend {i} with NFS mount on host-{i}"` |
| tests | `./tests/integration/prod_docs_pack_noprefix/README.md` | 14 | `weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 35 | `"text": "Configure NFS exports on RHEL 8 using the weka fs mount command.",` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 63 | `# Chunk 1 should have entities (NFS, RHEL, weka fs mount)` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 99 | `"name": "weka fs",` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 101 | `"entity_id": "cmd:weka_fs",` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 108 | `"text": "Use weka fs mount to attach the filesystem on RHEL.",` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 120 | `assert regex_mentions[0]["name"] == "weka fs"` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 203 | `"entity_id": "cmd:weka_fs",` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 204 | `"name": "weka fs",` |
| tests | `./tests/integration/test_gliner_ingestion_flow.py` | 257 | `assert "weka fs" in names` |
| tests | `./tests/integration/test_phase1_entity_edges.py` | 228 | `"text": f"Document {i} about WEKA filesystem operations including snapshots, tiering, and data protection."` |
| tests | `./tests/test_embedding_field_canonicalization.py` | 73 | `collection_name = os.getenv("QDRANT_COLLECTION", "weka_sections_v2")` |
| tests | `./tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` | 14 | `**Command:** `weka fs tier s3 attach`` |
| tests | `./tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` | 18 | ``weka fs tier s3 attach <fs-name> <obs-name> [--mode mode]`` |
| tests | `./tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` | 30 | `**Command:** `weka fs tier s3 detach`` |
| tests | `./tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` | 34 | ``weka fs tier s3 detach <fs-name> <obs-name>`` |
| tests | `./tests/integration/prod_docs_pack_noprefix/docs/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md` | 43 | `Note: To [recover from a snapshot](../../snap-to-obj#creating-a-filesystem-from-a-snapshot-using-the-cli) uploaded when two `local` object stores have been attached, use the `--additional-obs` parameter in the `weka fs d` |
| tests | `./tests/integration/test_phase7c_integration.py` | 60 | `Before configuring NFS for Weka, ensure the following:` |
| tests | `./tests/integration/test_phase7c_integration.py` | 61 | `- Weka cluster version 4.2 or higher` |
| tests | `./tests/integration/test_phase7c_integration.py` | 69 | `To enable the NFS protocol on your Weka cluster:` |
| tests | `./tests/integration/test_phase7c_integration.py` | 72 | `weka nfs enable` |
| tests | `./tests/integration/test_phase7c_integration.py` | 80 | `weka nfs export create --name my-export --filesystem my-fs --path /data` |
| tests | `./tests/integration/test_phase7c_integration.py` | 88 | `mount -t nfs weka-cluster:/my-export /mnt/weka` |
| tests | `./tests/integration/test_phase7c_integration.py` | 105 | `3. Check Weka cluster status with `weka status`` |
| tests | `./tests/integration/test_phase7c_integration.py` | 152 | `query = "How do I configure NFS for Weka?"` |
| tests | `./tests/integration/test_phase7c_integration.py` | 353 | `session_id, "How do I configure NFS for Weka?", turn=1` |
| tests | `./tests/integration/test_phase7c_integration.py` | 358 | `query_id, "How do I configure NFS for Weka?"` |
| tests | `./tests/integration/test_session_tracking.py` | 151 | `test_session_id, "How do I configure NFS for Weka?", 1` |
| tests | `./tests/integration/test_session_tracking.py` | 156 | `query_id=query_id, query_text="How do I configure NFS for Weka?"` |
| tests | `./tests/integration/test_session_tracking.py` | 283 | `answer_text = "To configure NFS for Weka, follow these steps..."` |
| tests | `./tests/integration/test_session_tracking.py` | 353 | `query="How do I configure NFS for Weka?",` |
| tests | `./tests/integration/test_session_tracking.py` | 382 | `assert record["query_text"] == "How do I configure NFS for Weka?"` |
| tests | `./tests/test_integration_prephase7.py` | 279 | `test_text = "This is a test document about Weka filesystem configuration."` |
| tests | `./tests/integration/prod_docs_pack_noprefix/artifacts/phase7e3_prod_docs_report.json` | 102 | `"src_uri": "tests://prodpack/PRODPACK-02/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md",` |
| tests | `./tests/integration/prod_docs_pack_noprefix/artifacts/phase7e3_prod_docs_report.json` | 103 | `"filename": "weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md",` |

## Notes

- This map does not claim the migration is complete.
- Because this map is generated after syncing `origin/master`, it includes any WEKA strings reintroduced by that merge.
- Live embedding, reranking, Qdrant, Neo4j, MCP behavior, and quality tuning remain explicitly out of scope for this architecture/code-only pass.
- Historical WEKA material is allowed only when moved to explicit archive paths excluded from runtime residue gates.
