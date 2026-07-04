# Session: Retrieval Pipeline Tuning + RELATED_TO v2 Graph Prerequisites

**Date:** 2026-03-04
**Branch:** `multi-embedder-reranker`
**Predecessor sessions:** `2026-03-04-integration-deployment-retrieval-tuning.md`
**Commits produced:** `eb211a9`, `6b5bbd5`, `ee76615`
**Author:** Full retrieval tuning and graph structure upgrade session

---

## Session Overview

This session executed two major workstreams:

1. **SPLADE-Dominant Retrieval Tuning** — Research-driven rebalancing of the 6-field weighted RRF fusion pipeline to address conceptual query degradation, plus entity quality gating to reduce NER noise in the entity-sparse retrieval signal.

2. **RELATED_TO v2 Edge Model for GDS Readiness** — Complete upgrade of the cross-document linking pipeline from minimal (doc_id, score) tuples to a full signal-provenance edge model with reciprocity tracking, structural priors, relationship indexes, and a GDS readiness validation suite.

---

## Part 1: Research Phase — Retrieval Quality Investigation

### Problem Statement

Conceptual queries like "How does a WEKA cluster manage inodes and metadata?" returned generic cluster-overview content instead of the specific metadata-management documentation. Root cause analysis from the prior session identified that SPLADE (text-sparse) correctly ranked the metadata content at position 3, but the 6-field weighted RRF fusion buried it because three correlated dense signals from the same 0.6B embedding model (content=2.0, title=1.5, doc_title-sparse=0.8) triple-voted for generic content while SPLADE had a single vote at weight=0.5.

### Research Methodology

Three parallel research agents were launched to investigate:

1. **RRF k-constant and fusion research** — Consulted 14 sources including the Cormack et al. 2009 original paper, Qdrant/Milvus/Azure documentation, and recent (2024-2026) preprints on hybrid retrieval fusion.

2. **SPLADE + small dense model fusion** — Consulted 20 sources including Zhang et al. 2024 (ByteDance/Tsinghua), the Qwen3-Embedding paper, SPLADE++ (Formal et al. 2024, ACM TOIS), and production RAG system documentation.

3. **NER entity quality gating** — Consulted 15 sources including the GLiNER paper (NAACL 2024), Knowledgator production documentation, "Less is More: Denoising Knowledge Graphs for RAG" (Zheng et al. 2025), and NER noise reduction research.

### Key Research Findings

**RRF k-Constant (Cormack 2009):**
- The MAP curve is remarkably flat between k=20 and k=100 (total variation ~0.6%)
- k=60 was declared "not critical" by the authors
- Qdrant's own default is k=2 — our k=60 was 30x more conservative than Qdrant's design center
- Recommendation: k=30 provides 1.9x amplification of early-rank contributions with minimal risk

**SPLADE + Small Dense Fusion (Multiple Sources):**
- Zhang et al. 2024: After distribution alignment, optimal weights shift toward equal weighting (alpha approximately 0.5)
- Weinberg 2026: alpha=0.5 yields up to 580% recall improvement over dense-only
- dbi-services 2025: For domain-specific corpora, (0.3 dense, 0.7 sparse) often performs best
- Qwen3-Embedding-0.6B scores 61.41 on MTEB Retrieval vs 69.60 for 4B — a significant capacity gap
- Dense content, title, and doc_title fields from the same 0.6B model are correlated signals violating RRF's independence assumption

**NER Entity Quality (GLiNER + RAG Research):**
- GLiNER's own documentation uses threshold=0.5 as default; Knowledgator uses 0.7 in production
- The current 0.45 threshold admitted entities where GLiNER was less than half-confident
- Zheng et al. 2025 showed removing 30-40% of noisy entities improved downstream RAG quality
- Per-label confidence thresholds are strongly recommended for heterogeneous label sets (ZERONER, ACL Findings 2025)
- Entity-sparse text generation in atomic.py had zero confidence gating — every entity passing the 0.45 global threshold flowed into the SPLADE encoding

### Quantitative Impact Analysis

For the metadata query with current vs proposed settings:

**Current (k=60, content=2.0, text-sparse=0.5, title=1.5):**
- Metadata chunk total RRF score: ~0.043
- Cluster overview chunk total RRF score: ~0.063 (cluster wins by 47%)

**Proposed (k=30, content=1.2, text-sparse=2.0, title=0.3):**
- Metadata chunk total RRF score: ~0.081
- Cluster overview chunk total RRF score: ~0.078 (metadata wins by ~4%)

The ranking flips. Additionally, the signal pool guarantees the metadata chunk a dedicated reranker slot where the 4B cross-encoder can evaluate it directly.

### Research: Per-Field RRF Weight Theory

The standard RRF formula is `score(d) = sum(weight_i / (k + rank_i))` where rank_i is the 0-based rank in each sub-list and weight_i is the per-field weight. With the original weights, a content rank-1 hit contributed `2.0 * 1/61 = 0.033` while a text-sparse rank-1 hit contributed `0.5 * 1/61 = 0.008`. The dense field had 4x the influence per rank position.

The key theoretical issue: RRF assumes input rankings are **independent signals**. Dense `content`, `title`, and `doc_title` all come from the same 0.6B embedding model — they are highly correlated, not independent. This violates RRF's core assumption and effectively gives the dense model 3+ votes vs SPLADE's single vote.

The query-type specific weights (conceptual: text-sparse=0.3) made this **worse**. The original design assumed a larger dense model (BGE-M3, 567M params but with 250K vocabulary giving it effectively larger capacity for domain terms). The 0.6B Qwen3 model has less capacity for cross-domain conceptual reasoning (mapping "manage inodes" -> "metadata limitations").

### Research: Entity-Sparse Noise Propagation Path

The entity-sparse noise propagation was traced through four code locations:

1. **GLiNER extraction** (`gliner_service.py:240`): Entities extracted with global threshold 0.45
2. **Enrichment** (`ner_gliner.py:162-181`): Every entity above 0.45 flows into `_mentions` with confidence stored but **never checked**
3. **Entity-sparse text** (`atomic.py:1691-1701`): Every mention's name collected with `" ".join(entity_names)` — no confidence check, no filtering, no cap
4. **SPLADE encoding**: The entity text is encoded by SPLADEv3 into a sparse vector. Noisy entities like "system" and "cluster" activate the same SPLADE term weights as legitimate entities like "NFS" and "inode"

With entity-sparse at RRF weight 0.8, a noisy entity signal contributed ~13% of the total RRF signal (0.8 out of 6.4 total weighted signals). If 30% of entity-sparse matches were from noisy/generic entities, approximately 4% of the final ranking signal was noise — enough to shift 2-5 positions in a top-20 results list.

### Research: Prefetch Limit Truncation Bias

The `_build_prefetch_entries` method in hybrid_retrieval.py hardcoded `limit=50` for auxiliary sparse fields (doc_title-sparse, title-sparse, entity-sparse) while dense and text-sparse got 200. This created asymmetric truncation bias in RRF — a document ranked 51st in an auxiliary field got zero RRF contribution from that field.

At k=30 (proposed), three lost auxiliary appearances at rank 50 equals `3 * weight * 1/80 = 3 * 0.5 * 0.0125 = 0.019` in lost score. One rank-1 appearance in content is `1.2 * 1/31 = 0.039`. So losing three auxiliary rank-50 contributions equals nearly half a rank-1 appearance — material at the margin where metadata and cluster chunks are competing.

### Research: Signal Pool Architecture Validation

The signal pool (`src/query/signal_pool.py`, 298 lines, 17 tests from the 2026-03-03 session) was designed to address exactly the SPLADE-surfaced-but-RRF-buried problem. Verification confirmed the activation requires THREE conditions:
1. `feature_flags.signal_diverse_rerank_pool: true`
2. `search.hybrid.signal_pool.enabled: true`
3. `feature_flags.query_api_weighted_fusion: true` (already active)

All three are AND-ed at `hybrid_retrieval.py:2361-2368`. With 30 dedicated text_sparse_slots, the signal pool guarantees the SPLADE rank-3 metadata hit reaches the reranker even when RRF fusion buries it.

---

## Part 2: SPLADE-Dominant RRF + Entity Quality Gating (Commit 6b5bbd5)

### Phase A: Config-Only Changes (No Re-Ingestion Required)

**A.1 — RRF k from 60 to 30** (`config/development.yaml:87`, `config/production.yaml:87`)

Amplifies early-rank contributions by ~1.9x. SPLADE rank-3 metadata hit contribution doubles from 0.016 to 0.030 per unit weight.

**A.2 — Base RRF Field Weights Rebalanced** (`config/*.yaml:93-99`)

| Field | Old | New | Rationale |
|---|---|---|---|
| content | 2.0 | 1.2 | Dense semantic — moderate for 0.6B |
| title | 1.5 | 0.3 | Correlated with content — tiebreaker only |
| text-sparse | 0.5 | 2.0 | SPLADE — primary discriminative signal |
| doc_title-sparse | 0.8 | 0.3 | Auxiliary tiebreaker |
| title-sparse | 0.8 | 0.5 | Section headings — secondary |
| entity-sparse | 0.8 | 0.5 | Reduced until entity quality improved |

**A.3 — Query-Type Adaptive RRF Weights** (`src/query/structural_retrieval.py:48-103`, `src/query/hybrid_retrieval.py:3148-3168`)

All 6 query types (conceptual, cli, config, procedural, troubleshooting, reference) rebalanced to SPLADE-dominant. The conceptual type was the worst offender at text-sparse=0.3 — raised to 2.5.

Critical code fix: `DEFAULT_QUERY_TYPE_RRF_WEIGHTS` was defined in structural_retrieval.py but never wired into the retrieval path. Added `get_query_type_rrf_weights()` import and temporary weight override with try/finally in `HybridRetriever.retrieve()` before the vector search call. The query classifier determines the type, and the weights are applied per-search then restored.

**A.4 — Auxiliary Sparse Prefetch Limits 50 to 100** (`hybrid_retrieval.py:1716,1726,1736`)

At k=30, truncation at rank 50 in auxiliary sparse fields (doc_title-sparse, title-sparse, entity-sparse) loses nearly one full rank-1 contribution worth of score. Raised to 100 to eliminate this bias.

**A.5 — Reranker Pool 20 to 60** (`config/*.yaml:143`)

With 6-field RRF producing 180+ fused candidates, sending only 20 to the 4B reranker wasted its capacity. Raised to 60 to ensure the metadata chunk reaches the cross-encoder.

**A.6 — Signal Pool Activated** (`config/*.yaml: feature_flags + signal_pool section`)

Added `signal_diverse_rerank_pool: true` to feature_flags and a `signal_pool:` section under `search.hybrid:` with SPLADE-boosted slot allocations (30 text_sparse_slots out of 200 total). Both the feature flag AND `signal_pool.enabled: true` are required (verified in `hybrid_retrieval.py:2361-2368`).

**A.7 — Broadened Reranker Instruction** (`config/*.yaml:147`)

Added "Match the specificity of the query", "system architecture, filesystem internals, data management concepts, metadata handling" to the Qwen3-Reranker-4B instruction. This lets the 4B model distinguish query intent rather than always favoring config/CLI content.

**BM25 Disabled** (`config/*.yaml:168`)

SPLADE provides better learned lexical matching than raw BM25 for this corpus. BM25 keyword matches dominated the RRF fusion with generic "cluster" hits. The retrieval path with BM25 off was verified: `_rrf_fusion(bm25_results=[], vec_results)` handles empty BM25 cleanly, and the reranker fires as long as `fused_results` is non-empty.

### Phase B: Entity Quality Gating (Requires Re-Ingestion)

**B.1 — GLiNER Threshold 0.45 to 0.55** (`config/*.yaml:502`)

**B.2 — Expanded Entity Exclusion List** (`src/providers/ner/labels.py:54-80`)

From 5 WEKA brand entries to ~30 entries covering generic domain vocabulary (system, server, cluster, node, service, data, file, process, etc.), over-generic measurement terms (performance, capacity, size), and over-generic procedure words (step, click, select, run, enter). Domain-discriminative terms (NFS, SMB, S3, inode, metadata, tiering, snapshot, POSIX) explicitly NOT excluded.

**B.3 — Per-Label Confidence Floors** (`labels.py:74-97`, `ner_gliner.py:143-152`)

10-entry `RETRIEVAL_CONFIDENCE_FLOORS` dict with per-label thresholds. STORAGE_CONCEPT and CAPACITY_METRIC at 0.70, PROTOCOL and CLOUD_PROVIDER at 0.60-0.65, COMMAND/PARAMETER/VERSION/ERROR at 0.55. Applied in `enrich_chunks_with_entities()` — entities below the floor are still recorded in `entity_metadata` (informational) but excluded from `_mentions` and `_embedding_text` (retrieval-critical).

**B.4 — Entity Cap 8 Per Chunk** (`src/ingestion/atomic.py:1689-1698`)

Entity-sparse text generation now sorts mentions by confidence (descending) and caps at 8. Prevents long noisy entity tails from diluting the SPLADE encoding.

### Signal Pool title_sparse Fix

During verification, a signal pool slot mapping bug was discovered and fixed: `title_sparse_slots` was reading `doc_title_sparse_score` instead of the actual title-sparse score. Added `title_sparse_score: Optional[float]` to `ChunkResult`, populated from `vec_score_by_id[(pid, "title-sparse")]` at both call sites in hybrid_retrieval.py, and updated `signal_pool.py:192` to use the correct field.

### Test Results

54 new tests in `tests/unit/test_entity_quality_gating.py` — all passing. 14 existing signal pool tests — no regressions.

---

## Part 3: RELATED_TO v2 Edge Model + GDS Prerequisites (Commit ee76615)

### Motivation

The RELATED_TO edges between Document nodes are the foundation for future Neo4j GDS algorithms (community detection via Louvain, centrality via PageRank, similarity projections). Current edges carried minimal properties (score, method, phase, timestamps, optional colbert_score), candidates were collapsed to `(doc_id, score)` tuples losing signal provenance, the backfill script had drifted from the service (bare SET vs ON CREATE/ON MATCH causing timestamp clobber), there were no relationship indexes, and RELATED_TO wasn't even in the guard DDL marker.

### Codebase Research Findings

Three parallel explore agents analyzed the complete cross-doc linking pipeline:

**Cross-Doc Linking Service** (`src/services/cross_doc_linking.py`, 1283 lines):
- Class `CrossDocLinker` with `link_document()` (line 1017), `link_all_documents()` (line 1198)
- Edge writer `_create_edge()` at line 738 wrote only: score, method, phase, created_at, updated_at
- ColBERT reranker `_rerank_edges_colbert()` at line 893 added colbert_score, reranked_at via a write-then-update pattern
- Candidate representation: `List[Tuple[str, float]]` — simple (doc_id, score) tuples losing all signal provenance
- RRF fusion at line 251 with k=60, fusing dense + sparse doc_title vectors

**Backfill Script** (`scripts/backfill_cross_doc_edges.py`, 1270 lines):
- Independent `create_related_to_edge()` at line 710 using bare `SET r.created_at = datetime()` — **clobbered the original creation timestamp on every re-run**
- Independent copies of `aggregate_chunks_to_documents()` (line 643) and `reciprocal_rank_fusion()` (line 668) — drifted from service
- Three processing pipelines: dense (line 764), RRF (line 836), title FT (line 923)

**Neo4j Schema Infrastructure:**
- Guard DDL (`scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher`): RELATED_TO NOT in RelationshipTypesMarker, zero relationship indexes on RELATED_TO
- 6 existing relationship indexes: MENTIONS (3), MENTIONED_IN (2), PARENT_HEADING (1)
- Health checker: did NOT validate RELATED_TO indexes
- Schema version: v4.0

**Existing Graph Patterns:**
- REFERENCES edges: Chunk->Document with type, reference_text, confidence (0.70-0.95)
- MENTIONS edges: Chunk->Entity with confidence, start, end, source_section_id
- Document properties: id, title, source_uri, doc_tag, doc_category, snapshot_scope
- Graph expansion in mcp_app.py used NEXT_CHUNK + siblings only — NOT RELATED_TO
- Graph direction: `(Document)-[:HAS_CHUNK]->(Chunk)` per atomic.py:2542
- No entity hub detection existed

### Research: Cross-Doc Linking Signal Loss

The current cross-doc linking pipeline collapsed all signal information into a single `(doc_id, score)` tuple at the aggregation stage. This meant that by the time an edge was written to Neo4j, all provenance about which signals contributed (dense similarity, sparse overlap, title full-text match) was lost. The only surviving signal was the final score and a method string like "rrf_fusion".

For GDS projections, this is insufficient. Community detection algorithms like Louvain need meaningful edge weights to produce useful communities. A score of 0.03 from RRF fusion (which represents a combination of rank positions across signals) has different quality characteristics than a score of 0.85 from dense cosine similarity. Without signal provenance, GDS cannot differentiate high-confidence single-signal edges from lower-confidence multi-signal consensus edges.

The v2 edge model preserves all signal components (score_dense, score_sparse, score_rrf, score_colbert, score_title_ft) alongside a computed score_final for backward compatibility. This enables future GDS projections to use either score_final (simple) or custom-weighted combinations of individual signals (advanced).

### Research: Backfill Script Drift Analysis

The backfill script at `scripts/backfill_cross_doc_edges.py` was originally copy-pasted from the service code and had drifted in several ways:

1. **Timestamp clobber**: The edge writer used bare `SET r.created_at = datetime()` (line 744) instead of `ON CREATE SET / ON MATCH SET`. Every re-run of the backfill overwrote the original creation timestamp, making edge age tracking impossible.

2. **RRF constant duplication**: `RRF_K = 60` at line 118 was independent from the service's `DEFAULT_RRF_K = 60` at line 59. If either changed, they'd diverge silently.

3. **Aggregation function duplication**: `aggregate_chunks_to_documents()` at line 643 and `reciprocal_rank_fusion()` at line 668 were independent copies that could diverge from the service versions.

4. **No ColBERT pre-write**: The backfill's ColBERT reranking (`process_colbert_rerank()` at line 998) operated on existing edges using the write-then-update pattern, creating intermediate edge states.

### Research: Structural Priors Design Space

Three candidate prior signals were identified from the existing graph structure:

**REFERENCES edges** (`src/ingestion/extract/references.py`): Four regex patterns detect hyperlinks (confidence 0.95), "see also" patterns (0.85), "related" patterns (0.80), and "refer to" patterns (0.70). These represent explicit author-intended cross-document links and are the highest-quality prior signal.

**Entity MENTIONS edges** (GLiNER + structural extractors): Shared entities between documents indicate topical overlap. However, high-degree entity hubs (e.g., "NFS" appearing in 50 documents) would make every NFS-related document appear related to every other. Document-frequency-based hub suppression was chosen over raw mention count to avoid chunking artifacts — a long document split into 20 chunks could produce 20 MENTIONS edges for a single entity occurrence.

**Document taxonomy** (doc_tag, doc_category properties): Documents with the same `doc_tag` (from explicit DocTag: header or filename pattern) share a topic scope. Documents with the same `doc_category` (from directory path) share a broader category. These are the weakest but most stable prior signals.

### Implementation: 8 Stages

**Stage 1: Shared Data Model** (`src/services/cross_doc_edge_model.py`, ~250 lines, NEW)

Created the canonical shared module with:
- `CandidateSignals` dataclass replacing `Tuple[str, float]` — carries `score_dense`, `score_sparse`, `score_rrf`, `score_colbert`, `score_title_ft`, `dense_rank`, `sparse_rank` with a `score_final` property implementing coalesce(colbert > rrf > dense > title_ft > sparse > 0.0)
- `StructuralPriors` dataclass — `prior_reference`, `prior_entity`, `prior_taxonomy`
- `EdgePayload` dataclass — complete v2 property set with signal scores, structural priors, provenance (method, method_version, phase), reciprocity (is_mutual, mutual_score), quality classification (quality_tier), and `to_neo4j_params()` that omits None values and forces `score = score_final` for backward compat
- `aggregate_chunks_to_candidates()` — max-score-wins per document returning CandidateSignals
- `reciprocal_rank_fusion_v2()` — RRF preserving component scores and setting ranks
- `build_edge_merge_cypher()` — canonical MERGE Cypher with ON CREATE/ON MATCH SET
- 15 unit tests, all passing

**Stage 2: CrossDocLinker Refactor** (`src/services/cross_doc_linking.py`)

Major architectural change: **rerank-before-write**. Previous flow created edges then ran ColBERT to update or prune them. New flow runs ColBERT on candidates first, so edges are born with complete signal provenance in a single write.

- Replaced `_create_edge(source, target, score, method, phase)` with `_create_edge(source, target, payload: EdgePayload)` using `r += $props` Cypher pattern with `datetime()` directly in Cypher (matching existing pattern)
- Replaced `_rerank_edges_colbert()` with `_rerank_candidates_colbert()` operating on CandidateSignals before edge creation
- Removed `_update_edge_colbert_score()` and `_prune_edge()` — superseded by rerank-before-write
- `link_document()` flow: candidates -> ColBERT rerank -> EdgePayload.from_candidate() -> _create_edge()
- Added `LinkingResult.reciprocity_updated: int = 0`
- Old functions kept as deprecated wrappers for backward compatibility
- Gated by `colbert_rerank_before_write: true` config flag
- Cypher uses `r.updated_at IS NULL AS was_created` for clean creation detection (avoids the subtle edge case in the old `r.created_at = r.updated_at` comparison)
- 11 unit tests, all passing

**Stage 3: Backfill Alignment** (`scripts/backfill_cross_doc_edges.py`)

- Replaced independent `aggregate_chunks_to_documents()`, `reciprocal_rank_fusion()`, and `create_related_to_edge()` with imports from shared module
- Fixed the `created_at` clobber bug: bare `SET r.created_at = datetime()` replaced with `ON CREATE SET / ON MATCH SET`
- Updated all 3 processing pipelines (dense, RRF, title_ft) to use `CandidateSignals` and `EdgePayload`
- Updated ColBERT reranking to build EdgePayload incorporating existing edge props plus ColBERT score
- Added `--reciprocity` CLI mode for bulk is_mutual/mutual_score reconciliation
- Removed duplicate `RRF_K = 60` constant (imports `DEFAULT_RRF_K` from shared module)
- CLI interface unchanged (--dry-run, --execute, --method, etc.)
- 11 parity tests, all passing

**Stage 4: Schema/Index Prerequisites** (`scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher`, `src/monitoring/health.py`)

- Added PART 1D with 4 RELATED_TO relationship indexes: `related_to_score_final_idx`, `related_to_method_idx`, `related_to_quality_tier_idx`, `related_to_is_mutual_idx`
- Updated RelationshipTypesMarker to include RELATED_TO
- Bumped SchemaVersion to v4.1 with description "RELATED_TO v2 edge model with signal provenance + GDS indexes"
- Updated health.py: `REQUIRED_SCHEMA_VERSION = "v4.1"`, GDS indexes checked at DEGRADED level (not UNHEALTHY) for graceful rollout
- Updated config YAML schema version references to v4.1
- 6 schema tests, all passing

**Stage 5: Reciprocity Reconciliation** (`src/services/cross_doc_linking.py`)

- Added `_reconcile_reciprocity(doc_id)` method with Cypher that sets `is_mutual` and `mutual_score = avg(fwd.score_final, rev.score_final)` for all edges involving a document
- Uses `coalesce(r1.score_final, r1.score, 0.0)` for backward compatibility with existing edges
- Called at end of `link_document()` after all edges created
- Gated by `compute_reciprocity: true` config
- Failures logged but never fail ingestion
- Batch mode: `--reciprocity` flag on backfill script runs bulk reconciliation with `WHERE r1.is_mutual IS NULL` guard for efficiency
- 4 unit tests, all passing

**Stage 6: Structural Priors Materialization** (`src/services/cross_doc_linking.py`)

Three graph-based prior signals computed per candidate, stored on edges (Option A: store only, don't blend into score_final):

- `prior_reference`: `max(ref.confidence)` from `(Document)-[:HAS_CHUNK]->(Chunk)-[:REFERENCES]->(Document)` path
- `prior_entity`: Shared entity ratio with document-frequency hub suppression — entities appearing in > `entity_hub_threshold` (default 20) distinct documents are excluded to prevent chunking artifacts from skewing hub detection
- `prior_taxonomy`: doc_tag alignment (1.0 same tag, 0.5 same category, 0.0 no match)

Each prior query is independent — failures in one don't affect others. Gated by `compute_priors: true` config. Performance: 3 lightweight Neo4j queries per candidate x max_edges_per_doc=5 = 15 queries per document linking.

7 unit tests, all passing.

**Stage 7: Config Knobs** (`src/shared/config.py`, `config/*.yaml`)

Extended `CrossDocLinkingConfig` with: `edge_model_version` ("2.0"), `colbert_rerank_before_write` (true), `compute_reciprocity` (true), `compute_priors` (true), `entity_hub_threshold` (20), `quality_tier_high` (0.040), `quality_tier_medium` (0.028).

**Stage 8: GDS Readiness Validation Suite** (`scripts/validate_gds_readiness.py`, NEW)

Standalone script with 8 gate queries (pass/fail) and 5 metrics queries (informational):

Gates: related_to_edges_exist, score_final_populated, method_version_v2, reciprocity_computed, indexes_exist, quality_tiers_populated, schema_version_v41, marker_includes_related_to.

Metrics: edge_count, score_distribution (p50/p90/p99/mean), reciprocity_ratio, quality_tier_distribution, prior_coverage (ref/ent/tax percentages), degree_distribution (avg/max/p90).

Exit code 0 on all pass, 1 on any failure. 4 integration tests (skipped without NEO4J_URI).

---

## Part 4: Code Review Fixes

A code review identified 5 issues after the initial implementation:

1. **Schema version v4.0/v4.1 mismatch** — Guard DDL set v4.1 but health.py still required v4.0 and YAML configs declared v4.0. Fixed all four locations.

2. **Missing --reciprocity mode in backfill** — Added to CLI method choices, implemented bulk Cypher reconciliation with `WHERE r1.is_mutual IS NULL` guard, skips Qdrant connection (only needs Neo4j).

3. **Missing test files** — Created `tests/unit/test_related_to_schema.py` (6 tests), `tests/scripts/test_backfill_edge_parity.py` (11 tests), `tests/integration/test_gds_readiness.py` (4 integration tests).

4. **Snapshot refresh** — Pending deployment. Requires guard DDL applied to live Neo4j, then `python scripts/neo4j/neo4j_schema_snapshot.py snapshot`.

5. **Stale doc text in health.py** — Updated class docstring from "GraphRAG v4.0" to "GraphRAG v4.1" with RELATED_TO v2 mentions.

---

## Part 5: Files Changed (Full Inventory)

### Commit eb211a9 — Production Ingestion Fixes (Carry-Forward)

| File | Change |
|---|---|
| `config/embedding_profiles.yaml` | Broadened embedding query instruction |
| `docker-compose.yml` | Configurable HF_HUB_OFFLINE, INGEST_WATCH_MODE |
| `src/ingestion/semantic_chunker.py` | qwen3_0_6b adapter alias |
| `src/providers/embeddings/chonkie_adapter.py` | Dual health check path |
| `src/providers/ner/gliner_service.py` | Dual health check path |
| `scripts/neo4j/neo4j_snapshots/*` | Updated schema snapshots |

### Commit 6b5bbd5 — SPLADE-Dominant RRF + Entity Quality Gating

| File | Change |
|---|---|
| `config/development.yaml` | rrf_k=30, SPLADE-dominant weights, signal_pool, reranker instruction, BM25 disabled, threshold 0.55 |
| `config/production.yaml` | Same as development.yaml |
| `src/query/structural_retrieval.py` | 6 query-type weight maps rebalanced SPLADE-dominant |
| `src/query/hybrid_retrieval.py` | Adaptive RRF weights (try/finally), prefetch 50->100, title_sparse_score field |
| `src/query/signal_pool.py` | title_sparse slot corrected to use title_sparse_score |
| `src/providers/ner/labels.py` | Expanded exclusions (~30 terms), per-label confidence floors |
| `src/ingestion/extract/ner_gliner.py` | Per-label confidence gating on _mentions and _embedding_text |
| `src/ingestion/atomic.py` | Top-8 entity cap by confidence |
| `tests/unit/test_entity_quality_gating.py` | 54 new tests (NEW) |

### Commit ee76615 — RELATED_TO v2 Edge Model + GDS Prerequisites

| File | Change |
|---|---|
| `src/services/cross_doc_edge_model.py` | Shared data model: CandidateSignals, EdgePayload, RRF v2 (NEW) |
| `src/services/cross_doc_linking.py` | Rerank-before-write, CandidateSignals pipeline, reciprocity, priors |
| `scripts/backfill_cross_doc_edges.py` | Shared module alignment, timestamp fix, --reciprocity mode |
| `scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher` | 4 RELATED_TO indexes, marker update, v4.1 |
| `src/monitoring/health.py` | GDS index checks (DEGRADED), v4.1 |
| `src/shared/config.py` | CrossDocLinkingConfig v2 fields |
| `config/development.yaml` | Schema v4.1, cross_doc_linking v2 knobs |
| `config/production.yaml` | Same as development.yaml |
| `scripts/validate_gds_readiness.py` | GDS readiness validation suite (NEW) |
| `tests/unit/test_cross_doc_edge_model.py` | 15 tests (NEW) |
| `tests/unit/test_cross_doc_linking_v2.py` | 11 tests (NEW) |
| `tests/unit/test_reciprocity.py` | 4 tests (NEW) |
| `tests/unit/test_structural_priors.py` | 7 tests (NEW) |
| `tests/unit/test_related_to_schema.py` | 6 tests (NEW) |
| `tests/scripts/test_backfill_edge_parity.py` | 11 tests (NEW) |
| `tests/integration/test_gds_readiness.py` | 4 tests (NEW) |

---

## Part 6: Test Summary

| Test File | Count | Status |
|---|---|---|
| test_entity_quality_gating.py | 54 | All pass |
| test_cross_doc_edge_model.py | 15 | All pass |
| test_cross_doc_linking_v2.py | 11 | All pass |
| test_backfill_edge_parity.py | 11 | All pass |
| test_structural_priors.py | 7 | All pass |
| test_related_to_schema.py | 6 | All pass |
| test_reciprocity.py | 4 | All pass |
| test_gds_readiness.py | 4 | Skipped (needs NEO4J_URI) |
| test_signal_pool.py (existing) | 14 | All pass (no regressions) |
| **Total** | **126** | **112 pass, 4 skipped, 0 fail** |

---

## Part 7: Remaining Work (Next Session)

### Part 2: RELATED_TO Retrieval Integration (Stages 9-17)

Wire RELATED_TO edges into the live retrieval pipeline for immediate quality uplift:

- **Stage 9**: Config knobs in `ReferencesQueryConfig` for RELATED_TO retrieval (enable, weight, seed docs, max docs, chunks per doc, min edge score)
- **Stage 10**: 4 `related_to_*` score fields on `ChunkResult`
- **Stage 11**: `_compute_related_to_doc_signals()` — doc-level RELATED_TO signal function using `coalesce(r.score_final, r.colbert_score, r.score, 0.0)` for backward compat with current and v2 edges
- **Stage 12**: `_expand_from_related_docs()` — expand candidates with chunks from RELATED_TO documents, constrained vector search
- **Stage 13**: Blend RELATED_TO into `_apply_graph_reranker()` with per-query-type lambda (0.15 conceptual, 0.00 cli)
- **Stage 14**: Signal pool `related_to_slots` (10 slots, rebalance from per_doc_depth)
- **Stage 15**: Retrieval metrics and trace support for RELATED_TO expansion
- **Stage 16**: Unit + integration tests for retrieval integration
- **Stage 17**: `_relationships_for_query_type()` type safety fix — currently `_compute_cross_doc_signals` passes a query type string to `_relationships_for_query` which re-classifies it as query text

### Deployment Tasks

- Apply guard DDL to live Neo4j (both dev Mac and GPU server)
- Refresh schema snapshots after DDL application
- Re-ingest all 260 documents to populate v2 entity-sparse vectors (Phase B changes)
- Run backfill `--method rrf --execute` to create v2 RELATED_TO edges
- Run backfill `--method reciprocity --execute` to populate is_mutual/mutual_score
- Run `scripts/validate_gds_readiness.py` to verify all 8 gates pass
- Test metadata query on GPU server to verify retrieval quality improvement

### Legacy Edge Migration

One-time Cypher to upgrade existing edges to v2 schema:
```cypher
MATCH ()-[r:RELATED_TO]->()
WHERE r.method_version IS NULL
SET r.score_final = coalesce(r.score, 0.0),
    r.score_dense = CASE WHEN r.method CONTAINS 'dense' THEN r.score END,
    r.score_rrf = CASE WHEN r.method CONTAINS 'rrf' THEN r.score END,
    r.score_colbert = r.colbert_score,
    r.method_version = '2.0',
    r.quality_tier = CASE WHEN r.score >= 0.040 THEN 'high'
                          WHEN r.score >= 0.028 THEN 'medium' ELSE 'low' END,
    r.last_seen_at = datetime()
RETURN count(r) AS migrated
```

---

## Part 8: Operational Procedures for Next Session

### Deploying Phase A (Retrieval Tuning) to GPU Server

Phase A changes are config + retrieval code only — no re-ingestion required:

```bash
SSH_KEY=~/vibecode/infx/ubuntu24_ed25519
GPU=bgconley@10.25.0.50

# Pull latest code
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && git pull"

# Restart MCP server (picks up bind-mounted src changes + config)
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && docker compose --env-file .env.production restart mcp-server"

# Test the metadata query
ssh -i $SSH_KEY $GPU "curl -s http://localhost:8000/health | python3 -m json.tool"

# Check retrieval trace for SPLADE-dominant behavior
# Look for: rrf_k: 30, text-sparse weight 2.0, signal_pool_enabled: true
```

### Deploying Phase B (Entity Quality) to GPU Server

Phase B changes require full re-ingestion to rebuild entity-sparse vectors:

```bash
# Flush Redis checksums and clear Neo4j data (keep SchemaVersion)
ssh -i $SSH_KEY $GPU "docker exec weka-redis redis-cli -a testredis123 FLUSHDB"
ssh -i $SSH_KEY $GPU "echo 'MATCH (n) WHERE NOT n:SchemaVersion DETACH DELETE n;' | docker exec -i weka-neo4j cypher-shell -u neo4j -p testpassword123"

# Delete Qdrant points
ssh -i $SSH_KEY $GPU "curl -X POST 'http://localhost:6333/collections/chunks_multi_qwen3_0_6b/points/delete' -H 'Content-Type: application/json' -d '{\"filter\":{\"must\":[]}}'"

# Re-create SchemaVersion singleton
ssh -i $SSH_KEY $GPU "echo 'MERGE (sv:SchemaVersion {id: \"singleton\"}) SET sv.version = \"v4.1\";' | docker exec -i weka-neo4j cypher-shell -u neo4j -p testpassword123"

# Touch files to trigger watcher
ssh -i $SSH_KEY $GPU "find ~/wekadocs-matrix/data/ingest -name '*.md' -exec touch {} +"

# Monitor ingestion
ssh -i $SSH_KEY $GPU "docker logs -f weka-ingestion-worker"
```

### Applying Guard DDL and Running GDS Validation

```bash
# Apply guard DDL to Neo4j
ssh -i $SSH_KEY $GPU "cat ~/wekadocs-matrix/scripts/neo4j/create_graphrag_schema_v2_2_20251105_guard.cypher | docker exec -i weka-neo4j cypher-shell -u neo4j -p testpassword123"

# Verify indexes created
ssh -i $SSH_KEY $GPU "echo 'SHOW INDEXES YIELD name WHERE name STARTS WITH \"related_to_\" RETURN collect(name);' | docker exec -i weka-neo4j cypher-shell -u neo4j -p testpassword123"

# Refresh schema snapshot (from inside the container or locally)
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && python scripts/neo4j/neo4j_schema_snapshot.py snapshot"

# Run GDS readiness validation
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 python scripts/validate_gds_readiness.py"
```

### Running Cross-Doc Backfill with v2 Edges

```bash
# RRF edge discovery with v2 edge model
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && python scripts/backfill_cross_doc_edges.py --execute --method rrf --verbose"

# Reciprocity reconciliation
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && python scripts/backfill_cross_doc_edges.py --execute --method reciprocity"

# Legacy edge migration (one-time Cypher)
ssh -i $SSH_KEY $GPU "echo '
MATCH ()-[r:RELATED_TO]->()
WHERE r.method_version IS NULL
SET r.score_final = coalesce(r.score, 0.0),
    r.method_version = \"2.0\",
    r.quality_tier = CASE WHEN r.score >= 0.040 THEN \"high\" WHEN r.score >= 0.028 THEN \"medium\" ELSE \"low\" END,
    r.last_seen_at = datetime()
RETURN count(r) AS migrated;
' | docker exec -i weka-neo4j cypher-shell -u neo4j -p testpassword123"

# Final GDS readiness check
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword123 python scripts/validate_gds_readiness.py"
```

---

## Part 9: Architecture Decisions and Design Rationale

> Note: Formerly Part 8 — renumbered after operational procedures inserted.

### Why SPLADE-Dominant for a 0.6B Dense Model

The 0.6B dense embedding model (Qwen3-Embedding-0.6B, 1024-dim) has limited capacity to encode domain-specific term relationships. It defaults to broad semantic patterns ("cluster" = "computing infrastructure") rather than specific term-level matching ("inode management" = "metadata allocation"). SPLADE's learned term expansion mechanism generates precisely the kind of term associations that domain queries need. The empirical evidence from multiple independent sources converges on equal-to-sparse-dominant weighting for small dense models on domain-specific corpora.

### Why Rerank-Before-Write for Cross-Doc Linking

The previous create-then-update pattern wrote edges with incomplete signal provenance, then ran ColBERT to update or prune them. This meant edges were born with only score/method/phase, and the ColBERT score was added in a separate transaction. With rerank-before-write, edges are born with the complete v2 property set in a single atomic write. This eliminates orphaned half-updated edges and ensures the GDS validation gate (`score_final_populated`) passes for all new edges.

### Why Option A for Structural Priors

Storing priors without blending into score_final (Option A) provides:
- Observability: priors are queryable and debuggable independently
- Safety: no risk of priors degrading existing edge quality during initial rollout
- Flexibility: blending weights can be tuned later based on empirical data
- GDS compatibility: GDS projections can use score_final as weight without prior noise

Option B (blending) can be activated later by changing `score_final = coalesce(score_colbert, score_rrf, ...) * (1 + prior_weight * max(priors))`.

### Why Document-Frequency Hub Suppression

Raw chunk mention counts for entity hub detection are skewed by chunking artifacts — a long document with many small chunks may produce 15 MENTIONS edges for a single entity that only appears once in the text. Document frequency (count of distinct documents mentioning the entity) is a stable measure that doesn't depend on chunking granularity. Entities appearing in >20 documents (default `entity_hub_threshold`) are too common in the corpus to be discriminative for inter-document similarity.

---

## Part 10: Configuration Reference (End of Session)

### Retrieval Pipeline (production.yaml)

| Setting | Value | Location |
|---|---|---|
| rrf_k | 30 | search.hybrid.rrf_k |
| content weight | 1.2 | search.hybrid.rrf_field_weights.content |
| text-sparse weight | 2.0 | search.hybrid.rrf_field_weights.text-sparse |
| title weight | 0.3 | search.hybrid.rrf_field_weights.title |
| entity-sparse weight | 0.5 | search.hybrid.rrf_field_weights.entity-sparse |
| BM25 enabled | false | search.hybrid.bm25.enabled |
| reranker top_n | 60 | search.hybrid.reranker.top_n |
| signal_pool enabled | true | search.hybrid.signal_pool.enabled |
| text_sparse_slots | 30 | search.hybrid.signal_pool.text_sparse_slots |
| NER threshold | 0.55 | ner.threshold |
| schema version | v4.1 | schema.version |

### Cross-Doc Linking (development.yaml)

| Setting | Value | Location |
|---|---|---|
| edge_model_version | "2.0" | cross_doc_linking.edge_model_version |
| colbert_rerank_before_write | true | cross_doc_linking.colbert_rerank_before_write |
| compute_reciprocity | true | cross_doc_linking.compute_reciprocity |
| compute_priors | true | cross_doc_linking.compute_priors |
| entity_hub_threshold | 20 | cross_doc_linking.entity_hub_threshold |
| quality_tier_high | 0.040 | cross_doc_linking.quality_tier_high |
| quality_tier_medium | 0.028 | cross_doc_linking.quality_tier_medium |
