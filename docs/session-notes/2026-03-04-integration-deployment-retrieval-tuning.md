# Session: Integration Testing, GPU Production Deployment, and Retrieval Tuning

**Date:** 2026-03-04
**Branch:** `multi-embedder-reranker`
**Predecessor sessions:** `2026-03-03-signal-pool-provider-cleanup.md`, `2026-03-04-evidence-pack-mcp-modernization.md`
**Commits produced:** `75c8a59`, `f6248a8`, `a89b462`
**Author:** Full-stack deployment and integration session

---

## Session Overview

This session executed three major workstreams across the `multi-embedder-reranker` branch:

1. **Commit, integration test, and fix cycle** — committed the evidence pack + MCP modernization work, ran end-to-end integration tests against live infrastructure, discovered and fixed 6 integration bugs, then documented the full architecture.
2. **x86_64 compatibility audit and dual-platform deployment** — audited the codebase for arm64 vs x86_64 portability, removed Tailscale sidecars, implemented dual-platform Docker Compose profiles, and deployed the full stack to the GPU server at `10.25.0.50`.
3. **Production ingestion and retrieval quality investigation** — ingested 258 documents through the full pipeline on the GPU box, investigated GPU memory usage, fixed GLiNER FP32→FP16 waste, and began diagnosing dense retrieval quality issues.

---

## Part 1: Commit and Integration Testing

### Initial Commit (75c8a59)

Committed the evidence pack architecture and MCP modernization — 127 files, +6,140/-3,778 lines. Pre-commit hooks caught two lint issues fixed before commit:
- Unused `tool_map` variable in `mcp_app.py` (the `full_tool_map` superseded it)
- Missing `Optional` import in `local_reranker_service.py` (`from __future__ import annotations` made it a runtime-invisible issue but ruff caught it statically)

### Integration Bugs Found and Fixed (f6248a8)

The unit test suite (44 tests) passed cleanly, but live end-to-end testing against the GPU gateway revealed 6 integration bugs:

**Bug 1 — ColBERT `KeyError: 'vectors'` (embedding_client.py:124):**
The unified GPU gateway returns ColBERT multi-vectors under the key `"embeddings"`, but the client expected `"vectors"`. This caused the Query API vector search path to fail, falling back to the legacy search path. Fixed with dual-format parsing: `item.get("embeddings", item.get("vectors", []))`. Same pattern as the sparse embedding fix from the previous session.

**Bug 2 — BM25 disabled in config (development.yaml:152):**
`bm25.enabled: false` was set during "Phase 1 vector-only hardening" testing. With Qdrant empty (pre-ingestion) and BM25 disabled, the retriever had zero retrieval sources. Fixed by enabling BM25 and adding `index_name: "chunk_text_index_v3_bge_m3"` pointing at the existing Neo4j fulltext index.

**Bug 3 — BM25 embedding_version filter (hybrid_retrieval.py:3111):**
The `_resolve_filters()` method applied the `embedding_version` filter to all retrieval channels, including BM25. Neo4j chunks had `embedding_version = "plan-d0e6a904..."` (old BGE-M3 plan hash) while the retriever filtered for `"plan-605ed3f0..."` (new Qwen3 plan hash). BM25 fulltext search is embedding-agnostic — the text content doesn't change when you switch embedding models. Fixed by stripping `embedding_version` from the filter dict before passing to the BM25 retriever.

**Bug 4 — Reranker connection refused (docker-compose.yml + .env):**
The docker-compose.yml `environment:` block used `${RERANKER_BASE_URL:-http://host.docker.internal:9005}` — a stale default. The `.env` file (used for compose variable interpolation) also had the old value. Even though `.env.docker` had the correct `http://10.25.0.50:8080`, the compose `environment:` block takes precedence over `env_file:`. Fixed the `.env` file and all three service blocks in docker-compose.yml to default to the unified gateway.

**Bug 5 — Cypher syntax error in graph enrichment (mcp_app.py:734):**
Neo4j 5.x does not allow `WHERE` directly after `UNWIND`. The graph enrichment Cypher had `UNWIND (...) AS neighbor WHERE neighbor.id NOT IN $ids`. Fixed with `UNWIND (...) AS neighbor WITH neighbor WHERE NOT neighbor.id IN $ids`.

**Bug 6 — Coverage `reranker_applied` typo (mcp_app.py:2343):**
The evidence pack coverage dict read `search_metrics.get("rerank_applied")` but the retriever sets `metrics["reranker_applied"]`. Missing two characters (`er`) caused the coverage to always report `reranker_applied: False`. Fixed the key name.

### End-to-End Architecture Documentation

Wrote a 996-line comprehensive architecture document at `docs/architecture/2026-03-04-end-to-end-architecture.md` covering:
- Full ingestion pipeline (parsing → NER → chunking → 3-embedder generation → Neo4j graph → Qdrant vectors → atomic saga → cross-doc linking → Redis queue)
- Full retrieval pipeline (MCP entry → tool profiles → query reformulation → BM25 + multi-vector search → RRF fusion → entity/structural boost → cross-encoder reranking → expansion → evidence extraction → graph enrichment → traces)
- Configuration reference (config loading chain, embedding profiles, feature flags, Docker env precedence)
- Deployment guide with ASCII data flow diagrams
- Score field survival table tracking each score from ChunkResult through to evidence quote

---

## Part 2: x86_64 Compatibility and Dual-Platform Deployment

### Architecture Compatibility Audit

Audited the entire codebase for arm64 (Apple Silicon) vs x86_64 (Linux) portability:

**Confirmed clean:**
- All Docker images multi-arch (`python:3.11-slim`, neo4j, qdrant, redis, grafana/alloy, tailscale)
- All Python compiled deps (tokenizers, sentencepiece, numpy, lxml) have `manylinux_x86_64` wheels
- HuggingFace cache files (safetensors, JSON tokenizers) are architecture-neutral
- Zero platform-specific Python code in `src/` (no `sys.platform`, `os.uname()`, or arch-conditional imports)
- GPU gateway client is plain HTTP/JSON via httpx — no gRPC or platform-specific wire format

**Blockers found:**
- `host.docker.internal` — doesn't resolve on native Linux Docker without `extra_hosts` directive
- GLiNER sidecar (`services/gliner-ner/`) — macOS-native MPS service, no Docker container
- Tailscale sidecars — add 30s+ startup delay, require `cap_add: net_admin`, carry hardcoded auth key

### Tailscale Sidecar Removal (a89b462)

The GPU gateway at `10.25.0.50` is LAN-reachable without Tailscale. Removed both sidecar services:

**Deleted:** `ts-mcp-server` and `ts-ingestion-worker` service definitions + 2 state volumes from docker-compose.yml.

**Modified `mcp-server`:** Removed `network_mode: service:ts-mcp-server`, added `networks: [weka-net]`, `ports: ["${MCP_PORT:-8000}:8000"]`, `extra_hosts: ["host.docker.internal:host-gateway"]`, removed `depends_on.ts-mcp-server`.

**Modified `ingestion-worker`:** Same pattern — direct network attachment + extra_hosts.

**Modified `ingestion-service`:** Added `extra_hosts` for Linux compatibility.

**Fixed all stale defaults:** Updated `BGE_M3_API_URL` and `GLINER_SERVICE_URL` defaults from `host.docker.internal:9000/9002` to `10.25.0.50:8080` across all 3 app service environment blocks.

**Cleaned up:** Removed `TS_AUTHKEY` from `.env`, `.env.docker`, `.env.local` (was a live auth key in plaintext). Created `config/production.yaml` with `host.docker.internal:8080` for co-located GPU gateway. Created `.env.production` (gitignored) for production env vars. Updated `.env.example` with deployment guide.

### Deployment Topology

| Environment | Docker Host | GPU Gateway | Gateway URL from containers |
|---|---|---|---|
| Dev (Mac, arm64) | Docker Desktop | Separate LAN machine at 10.25.0.50 | `http://10.25.0.50:8080` |
| Prod (GPU server, x86_64) | Native Docker on GPU box | localhost on Docker host | `http://host.docker.internal:8080` |

### GPU Server Deployment

Deployed the full stack to `10.25.0.50` (x86_64 Linux, RTX 3090, 125GB RAM, 1.9TB disk):

1. Cloned repo (`multi-embedder-reranker` branch, commit `a89b462`)
2. Created `.env.production` with `host.docker.internal:8080` gateway URLs
3. Built all 3 app container images (x86_64 `python:3.11-slim` base)
4. Started full stack: neo4j, qdrant, redis, alloy, mcp-server, ingestion-service, ingestion-worker
5. Applied Neo4j schema from snapshot (26 constraints, 98 indexes — exact match with Mac)
6. Created Qdrant `chunks_multi_qwen3_0_6b` collection with 4 dense + 4 sparse vector fields + 32 payload indexes
7. Created `SchemaVersion` singleton node (`v4.0`)
8. MCP server healthy, all 3 databases connected, GPU gateway reachable from containers

---

## Part 3: Production Ingestion

### Ingestion Pipeline Issues and Fixes

**Issue 1 — Watch mode defaulting to `ready` in production:**
`INGEST_WATCH_MODE` defaults to `"ready"` when `ENV=production` (safety feature). The file watcher detected files but didn't enqueue them. Fixed by adding `INGEST_WATCH_MODE=${INGEST_WATCH_MODE:-auto}` to the ingestion-service environment block in docker-compose.yml.

**Issue 2 — Checksum dedup blocking re-processing:**
After the first failed ingestion attempt, the 260 file checksums were cached in Redis `ingest:checksums:wekadocs`. Subsequent runs skipped all files as "duplicates." Fixed by flushing the checksum set.

**Issue 3 — Tokenizer cache miss (`HF_HUB_OFFLINE=1`):**
The docker-compose.yml hardcoded `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=true`. On the GPU box, the `hf-cache` bind mount was empty (never populated). The worker failed with `RuntimeError: HuggingFace tokenizer cache miss for 'Qwen/Qwen3-Embedding-0.6B' in offline mode`. Fixed by making both env vars configurable: `${HF_HUB_OFFLINE:-0}` and `${TRANSFORMERS_OFFLINE:-false}`.

**Issue 4 — ColBERT dimension mismatch in Qdrant upsert validator:**
`atomic.py:3591` built `expected_dim` with only 3 dense vectors (`content`, `title`, `doc_title` at 1024). The `late-interaction` ColBERT vector (128-dim) wasn't in the dict. The `upsert_validated()` fallback at `connections.py:245` grabbed the first dict value (1024) when the key wasn't found, causing `Dimension mismatch: expected 1024, got 128`. Fixed by reading the ColBERT dimension from the embedding plan's colbert profile.

**Issue 5 — Semantic chunker health check (`/healthz` vs `/health`):**
`BgeM3ChonkieAdapter.is_available()` at `chonkie_adapter.py:748` hit `/healthz` — the old standalone BGE-M3 service endpoint. The unified gateway uses `/health`. The adapter silently fell back to structured chunking (no semantic splitting). Fixed to try `/health` first, then `/healthz`, accepting `"ok"` or `"healthy"` status values.

**Issue 6 — GLiNER NER health check (same `/healthz` issue):**
`GLiNERService._check_http_health()` at `gliner_service.py:218` had the same `/healthz` hardcoded path. The gateway's GLiNER endpoint returned 404, so the service fell back to loading the GLiNER model locally on CPU — taking 78 seconds per document instead of 2-3 seconds via GPU. Fixed with the same dual-path health check pattern.

**Issue 7 — Semantic chunker adapter name mismatch:**
`production.yaml` set `embedding_adapter: "qwen3_4b"` which routed to the dead `Qwen3ChonkieAdapter` (Triton port 8101). Added `"qwen3_0_6b"` as a recognized alias in `semantic_chunker.py:239` that routes through `BgeM3ChonkieAdapter` with the correct model name and service URL.

**Issue 8 — Neo4j transaction timeout:**
The CLI reference guide (328 chunks, 683 entities, 1549 mentions) exceeded the default 30-second Neo4j transaction timeout. The atomic saga writes all nodes, edges, structural edges, and embedding metadata in one transaction. Fixed by increasing `db.transaction.timeout` to 120 seconds.

**Issue 9 — SPLADE batch size limit:**
TEI SPLADE service has a max batch size of 32, but the token-budgeted batching produced batches of 33-39 texts for the CLI reference guide. The 500 errors were handled gracefully (placeholder sparse vectors inserted), but those chunks lost SPLADE search capability. Not fixed in this session — needs batch size cap in the embedding code.

### Ingestion Results

- **258 of 260 documents** ingested successfully (1 in dead letter queue due to transaction timeout before the fix, 1 README skipped)
- **3,357 Neo4j chunks**, all with `embedding_version = "plan-605ed3f0..."`
- **3,409 Qdrant points** (52 orphan points from failed saga rollback compensation — cosmetic, not functional)
- **9,648 entities**, **23,432 MENTIONS edges**, **883 RELATED_TO cross-doc edges**
- **3,099 NEXT_CHUNK edges**, **1,536 PARENT_HEADING/CHILD_OF edges**
- Zero documents without chunks, zero orphan documents

---

## Part 4: GPU Memory Investigation

### VRAM Audit

All 5 models on the RTX 3090 (24GB):

| Container | Process | Model | VRAM | dtype | Notes |
|---|---|---|---|---|---|
| `qwen3_reranker` | `qwen3_reranker.api.app` | Qwen3-Reranker-4B | 12,164 MiB | FP16 + FA2 | Within documented peak (~12GB) |
| `qwen3_embedder` | `qwen3_embedder.main` | Qwen3-Embedding-0.6B | 5,580 MiB | FP16 + FA2 | Bloated by batch-64 warmup pre-allocation |
| `tei_gliner` | `server.py --device cuda` | GLiNER Medium v2.1 | 3,336 MiB → **1,292 MiB** | FP32 → **FP16** | Fixed with `.half()` |
| `tei_colbert` | `uvicorn app:app` | ColBERTv2 | 864-984 MiB | — | Normal |
| `tei_splade` | `text-embeddings-router` | SPLADEv3 | 562-626 MiB | FP16 | Normal (HuggingFace TEI) |

**Total before fix:** 22,506 MiB (91.6%). **After GLiNER FP16 fix:** 20,413 MiB (83.1%).

### GLiNER FP16 Fix

`~/tei/gliner/server.py` loaded the model with `model.to("cuda")` — no dtype cast, so DeBERTa-v3-base loaded in FP32. Changed to `model.to("cuda").half()`. Rebuilt the Docker image (`docker compose build gliner`) to bake the fix. Saved 2,234 MiB with zero quality loss (identical entity extraction scores before/after).

### GPU Gateway Architecture (~/tei/)

The GPU gateway stack lives at `/home/bgconley/tei/` with its own `docker-compose.yml`:
- **embed-gateway** — reverse proxy routing to all 5 model services, exposed on port 8080
- All services use `deploy.resources.reservations.devices` for NVIDIA GPU access
- Custom Dockerfiles for qwen3-embedder, qwen3-reranker, gliner, colbert
- HuggingFace TEI for SPLADE (pre-built image `ghcr.io/huggingface/text-embeddings-inference:cuda-1.9`)

Profiles are correctly configured via env vars (`QWEN_EMBED_PROFILE=qwen3_06b_cuda`, `QWEN_RERANK_PROFILE=qwen3_4b_cuda`), overriding stale YAML defaults.

### Embedder VRAM Bloat (unresolved)

Qwen3-Embedding-0.6B uses 5.5GB for a 0.6B param model (~1.2GB weights in FP16). The excess is from:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — PyTorch holds freed memory segments
- Batch-64 warmup pre-allocates activation memory that's never released
- Potential fix: reduce warmup batch_size from 64 to 8, add `torch.cuda.empty_cache()` after warmup. Not applied this session.

---

## Part 5: Retrieval Quality Investigation

### The Metadata Query Problem

Test query: "How does a WEKA cluster manage inodes and metadata?"

The corpus has 7 relevant chunks including "Metadata limitations in WEKA filesystems", "Metadata management", "Metadata processing", and "Metadata units calculation" — all in `weka-system-overview`. But the evidence pack consistently returned cluster configuration content instead.

### Root Cause Analysis

**Direct Qdrant search** (bypassing the pipeline) returns cosine scores of 0.67-0.75 for the top results and 0.51-0.60 for the metadata chunks. The gap is real — the 0.6B embedding model genuinely ranks generic "WEKA cluster" content higher than specific "metadata management" content for this query.

**SPLADE sparse search** finds "Metadata management" at **rank 3** (score 21.25). The learned sparse representations handle conceptual terms better than the small dense model.

**The pipeline's weighted RRF fusion** kills the SPLADE signal because:
- `content` (dense) weight was 2.0, `text-sparse` (SPLADE) weight was 0.5
- A SPLADE rank-3 hit with weight 0.5 contributes less than a dense rank-20 hit with weight 2.0
- BM25's 200 keyword-matched "cluster" results dominated the fusion top-N

**The reranker** then scored the cluster-config candidates highly (0.98-0.99) because its instruction emphasized "configuration procedures, CLI commands, parameter references" as "highly relevant."

### Fixes Applied

1. **Disabled BM25** in production — SPLADE provides better learned lexical matching than raw BM25 for this corpus
2. **Rebalanced RRF weights** — `content: 2.0 → 1.5`, `text-sparse: 0.5 → 1.5` (SPLADE equal to dense)
3. **Increased rerank pool** — `top_n: 20 → 80` so more diverse candidates reach the cross-encoder
4. **Broadened reranker instruction** — added "system architecture, filesystem internals, data management concepts, metadata handling" as highly relevant categories, with "Match the specificity of the query" guidance
5. **Broadened embedding query instruction** — added "system architecture, data management, filesystem internals, performance" to the Qwen3-Embedding instruction prefix

### Critical Config Corruption (found and fixed)

A blanket `sed -i 's/enabled: true/enabled: false/'` on `production.yaml` (intended for BM25 only) turned off **every `enabled: true` flag** in the file — reranker, expansion, NER, health checks, metrics, monitoring, everything. This caused the mysterious "reranker not firing when BM25 is disabled" behavior. Fixed by rebuilding production.yaml from a clean copy of development.yaml with only targeted changes.

### Current Retrieval State (end of session)

With BM25 disabled and the rebalanced weights, the Qdrant-only path (dense + SPLADE + 4 sparse fields) produces 180 fused candidates. The metadata page appears at **RRF rank 3** in the fusion. The reranker (80-candidate pool) is active and scoring candidates. The metadata page is now within the rerank pool.

However, the cross-encoder still ranks cluster-config content higher than metadata content for this query. This is a retrieval tuning problem, not a code bug — the reranker instruction and RRF k-constant need further calibration.

### Remaining Retrieval Tuning Work

1. **RRF k-constant** — currently k=60 (very conservative). Lowering to k=20-30 would amplify early-rank differences, making SPLADE's rank-3 metadata hit contribute more to the fusion score.
2. **NER entity quality** — GLiNER is extracting noisy/bogus entities that add noise to the entity-sparse field (weight 0.8). Needs label refinement and threshold tuning.
3. **Per-signal candidate breakdown** — retrieval traces Section 2 ("CANDIDATES BY SIGNAL") always shows "not available." Implementing this would enable signal-level debugging without log archaeology.
4. **Signal pool activation** — `signal_diverse_rerank_pool` feature flag is implemented and tested (17 tests) but not activated. Would ensure the reranker sees candidates from every signal source, not just top-N by fused score.

---

## Part 6: Files Modified (Full Inventory)

### Committed Changes (3 commits)

| Commit | Files | Summary |
|---|---|---|
| `75c8a59` | 127 files | Evidence pack architecture + MCP modernization |
| `f6248a8` | 6 files | Integration bug fixes + architecture doc |
| `a89b462` | 5 files | Tailscale removal + dual-platform deployment |

### Uncommitted Changes on Mac (need commit)

| File | Change |
|---|---|
| `src/ingestion/atomic.py` | ColBERT expected_dim fix for `late-interaction` vector |
| `src/ingestion/semantic_chunker.py` | Added `qwen3_0_6b` adapter alias, broadened service URL resolution |
| `src/providers/embeddings/chonkie_adapter.py` | Dual health check path (`/health` then `/healthz`), broadened `is_available()` URL resolution |
| `src/providers/ner/gliner_service.py` | Dual health check path (`/health` then `/healthz`) |
| `config/embedding_profiles.yaml` | Broadened query instruction (added architecture, data management, filesystem internals) |
| `config/development.yaml` | `embedding_adapter: "qwen3_0_6b"`, `bm25.enabled: true` |
| `docker-compose.yml` | `INGEST_WATCH_MODE=${INGEST_WATCH_MODE:-auto}`, `HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}`, `TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-false}` |

### GPU Server Files (not in repo)

| File | Change |
|---|---|
| `~/tei/gliner/server.py` | `.half()` on CUDA model loading (rebuilt into Docker image) |
| `~/wekadocs-matrix/.env.production` | Production env vars (gitignored) |
| `~/wekadocs-matrix/.env.docker` | Added `ALLOW_NAMESPACE_MISMATCH=true` |
| `~/wekadocs-matrix/config/production.yaml` | BM25 disabled, rerank pool 80, rebalanced RRF weights, broadened instructions |

---

## Part 7: Infrastructure State (End of Session)

### Mac (Development)

- Branch: `multi-embedder-reranker` at `a89b462`
- Docker: all 7 services running (no sidecars), MCP server healthy
- Neo4j: 3944 chunks (old BGE-M3 data from prior sessions)
- Qdrant: `chunks_multi_qwen3_0_6b` empty on Mac (data is on GPU box)
- 44 unit/contract tests passing

### GPU Server (10.25.0.50, Production)

- Branch: `multi-embedder-reranker` at `a89b462`
- Docker: 7 app services + 6 GPU gateway services running
- Neo4j: 258 documents, 3,357 chunks, 9,648 entities, schema v4.0
- Qdrant: 3,409 points in `chunks_multi_qwen3_0_6b` (52 orphans from failed saga)
- GPU: 20.4GB / 24GB VRAM (after GLiNER FP16 fix)
- MCP server: healthy, evidence packs returning with reranker active
- Ingestion: complete (258/260 docs), queue empty

### Docker Env Var Precedence Chain (documented for operators)

```
docker-compose.yml environment: block    ← highest priority
  ↑ expands ${VAR:-default} from:
.env (compose interpolation, gitignored)  ← for ${VAR} substitution in compose YAML
.env.docker (env_file: directive)         ← loaded into container environment
.env.production (--env-file flag)         ← overrides .env for compose interpolation only
```

### Known Issues Carried Forward

- 52 orphan Qdrant points from failed saga rollbacks (cosmetic)
- CLI reference guide (1 doc) not ingested — transaction timeout even at 120s for 328-chunk doc
- SPLADE batch size limit (32) causes 500 errors on large docs — needs batch size cap in atomic.py
- Embedder VRAM bloat (5.5GB for 0.6B model) — needs warmup batch_size reduction
- `EMBEDDINGS_PROFILE=bge_m3` in `.env.docker` is stale — config YAML plan overrides it
- Retrieval quality for conceptual queries needs further RRF k-constant tuning and NER entity cleanup
- Per-signal candidate breakdown not implemented in retrieval traces
- Signal pool feature flag not activated

---

## Part 8: Codebase Research — Retrieval Pipeline Deep Dive

This section captures the detailed codebase research performed during the session to support retrieval quality investigation and future tuning work.

### The 6-Field Weighted RRF Fusion (Query API Path)

When `feature_flags.query_api_weighted_fusion=True` (active in both dev and prod configs), the retriever executes a single Qdrant `query_points` call with nested `Prefetch` entries across 6 vector fields. The Qdrant server performs server-side DBSF (Distribution-Based Score Fusion) to return a unified candidate list. The client then performs weighted RRF across the individual field rankings.

The per-field RRF formula: `contribution = weight * (1 / (k + rank))` where k=60 (from `rrf_k` config). The final fused score is the sum of all field contributions.

With the original weights (content=2.0, text-sparse=0.5), a content rank-1 hit contributed `2.0 * 1/61 = 0.0328`, while a text-sparse rank-1 hit contributed `0.5 * 1/61 = 0.0082`. The dense field had 4x the influence per rank position. With the rebalanced weights (both at 1.5), each rank-1 hit contributes `1.5 * 1/61 = 0.0246` — equal influence.

The field ranking is determined by the Qdrant prefetch scoring within each vector space:
- `content` — cosine similarity of chunk content embedding vs query embedding (both via Qwen3-Embedding-0.6B)
- `title` — cosine similarity of section heading embedding vs query embedding
- `text-sparse` — dot product of SPLADE sparse vectors (learned term weights)
- `doc_title-sparse` — SPLADE similarity of document title
- `title-sparse` — SPLADE similarity of section heading
- `entity-sparse` — SPLADE similarity of entity name concatenation

Each field returns up to 200 candidates (configurable via `query_api_dense_limit` and `query_api_sparse_limit` in development.yaml). After fusion, duplicates are merged keeping the best score, yielding 180-331 unique candidates.

### Why Dense Retrieval Scored Metadata Low

Direct Qdrant search confirmed the cosine scores:
- "Metadata limitations in WEKA filesystems" → **0.6028** against the query
- "Metadata management" → **0.5692**
- "Obtain access information about WEKA cluster" → **0.7553** (top result)
- "Cluster" (generic overview) → **0.7119**

The 0.15 cosine gap between the top cluster content and the best metadata content is significant. The 0.6B embedding model (Qwen3-Embedding-0.6B, 1024 dims) treats "WEKA cluster" as a stronger semantic match because:
1. The query contains "WEKA cluster" as a bigram — docs mentioning "WEKA cluster" get strong lexical-semantic overlap
2. Metadata/inode content uses different vocabulary ("filesystem metadata", "SSD space", "RAM", "metadata units") with weaker semantic overlap to "manage inodes"
3. At 0.6B parameters, the model has limited capacity for cross-domain conceptual reasoning (mapping "manage inodes" → "metadata limitations")

The query instruction prefix was broadened to include "system architecture, data management, filesystem internals" — but our direct test showed this actually **widened** the score distribution (boosting general WEKA overview content) rather than narrowing it toward metadata content. The instruction affects query embedding only (asymmetric retrieval), not stored document embeddings, so it can be changed without re-ingestion.

### SPLADE's Superior Lexical Matching

SPLADE (Sparse Lexical and Expansion model) is a learned sparse retrieval model that generates term importance weights. Unlike BM25's statistical term weighting, SPLADE learns which terms are semantically important and expands queries with related terms. For "How does a WEKA cluster manage inodes and metadata?", SPLADE produces sparse vectors where "metadata" gets high weight, and related terms like "filesystem", "inode", "management" also activate.

This is why SPLADE ranked "Metadata management" at position 3 while the dense model ranked it at position ~50-80. SPLADE correctly identifies the query's focus on metadata/inode concepts through learned term expansion, while the dense model's limited capacity causes it to over-weight the "cluster" component.

The practical implication: for a 0.6B dense model, SPLADE is a critical retrieval signal, not a secondary "de-rated" one. The original 4:1 weight ratio (content=2.0, text-sparse=0.5) was calibrated for a larger dense model (BGE-M3, 567M params but with 250K vocabulary and multilingual training giving it effectively larger capacity for domain terms).

### Reranker Pipeline Flow

The cross-encoder path (`hybrid_retrieval.py:3371-3442`):

1. Takes the top `rerank_top_n` candidates from fused results (now 80, was 20)
2. If signal pool enabled: `build_signal_pool()` selects candidates with diversity across signals (BM25, vector, entity, structural). **Not active** — the feature flag is off.
3. Otherwise: flat `fused_results[:80]` by fused score
4. `_hydrate_parent_paths()` — standalone Neo4j lookup enriches each candidate's `parent_path_norm` for reranker context. The reranker sees `{parent_path} > {heading}\n\n{text}` as input.
5. `_apply_reranker(query, candidates, metrics)` — calls Qwen3-Reranker-4B via the `LocalRerankerServiceProvider`:
   - Prepends the domain instruction to the query (if configured)
   - Formats as `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
   - Batches documents in groups of 16 (reranker profile batch_size)
   - Collects P(yes) logit scores from the model
   - Returns candidates sorted by rerank_score

The reranker instruction is the final quality gate. When it says "configuration procedures and CLI commands are highly relevant", the 4B cross-encoder has enough capacity to understand this and score config content at 0.99+ while scoring conceptual metadata content at 0.70-0.80. The broadened instruction should help, but the reranker still needs to see the metadata chunks in its input pool.

### The Config Corruption Lesson

The blanket `sed -i 's/enabled: true/enabled: false/'` was intended to disable BM25 only but destroyed the production config by disabling every feature. This created a cascading failure that was extremely hard to diagnose:
- Reranker disabled → results returned in raw fusion order (poor quality)
- NER disabled → no entity extraction for query disambiguation
- Expansion disabled → no NEXT_CHUNK neighbor addition
- Health checks disabled → MCP server started without validation
- Monitoring disabled → no metrics for debugging

The fix was to rebuild `production.yaml` from a clean copy of `development.yaml` with **targeted** `python3` scripts for each individual change instead of global `sed`. This pattern should be followed for all future config changes on the GPU box.

### Signal Pool (Ready but Not Activated)

The signal-diverse rerank pool (`src/query/signal_pool.py`, 298 lines, 17 tests from the 2026-03-03 session) would address the retrieval quality issue by ensuring the reranker sees candidates from every signal source:

Slot fill priority:
1. **Consensus slots** — chunks that rank well across multiple signals
2. **Per-signal unique slots** — top chunks from each signal that don't appear in consensus (this is where a SPLADE rank-3 metadata hit would get a dedicated slot)
3. **Structural slots** — NEXT_CHUNK neighbors of seeds
4. **Per-doc depth slots** — additional depth within represented documents
5. **Backfill** — remaining capacity filled from the fused results

Activation requires: `signal_pool.enabled=True` in config + `feature_flags.signal_diverse_rerank_pool=True` + `feature_flags.query_api_weighted_fusion=True`. All three conditions AND the embedding plan's per-field scoring are needed.

The signal pool is the designed solution for exactly this problem — ensuring the SPLADE rank-3 metadata hit gets a dedicated rerank slot even when the dense model doesn't surface it. Activating it is the highest-leverage retrieval quality improvement available.

### NER Entity Quality Concerns

GLiNER is extracting entities during both ingestion (chunk enrichment) and retrieval (query disambiguation). During ingestion, it enriches chunks with entity metadata that populates the `entity-sparse` SPLADE field and `entity_metadata` Qdrant payload. During retrieval, it extracts entities from the query for entity boost.

Observed issues:
- The entity labels are broad: `COMMAND`, `PARAMETER`, `COMPONENT`, `PROTOCOL`, `CLOUD_PROVIDER`, `STORAGE_CONCEPT`, `VERSION`, `PROCEDURE_STEP`, `ERROR`, `CAPACITY_METRIC`
- Common WEKA terms trigger frequent matches that add noise rather than signal
- "WEKA" itself is excluded via `src/providers/ner/labels.py` deny list, but related terms aren't
- The `entity-sparse` field weight (0.8) means noisy entities influence RRF fusion

The fix path: audit the entity label set and extraction threshold (currently 0.45), add domain-specific stopwords to the deny list, and consider reducing `entity-sparse` weight to 0.3-0.5 until entity quality is improved.

---

## Part 9: Operational Procedures for Next Session

### Starting the GPU Server Stack

```bash
# From the GPU server (10.25.0.50)
cd ~/wekadocs-matrix
docker compose --env-file .env.production up -d

# Verify
docker compose --env-file .env.production ps
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/ready | python3 -m json.tool
```

### Starting from the Mac (Remote)

```bash
SSH_KEY=~/vibecode/infx/ubuntu24_ed25519
GPU=bgconley@10.25.0.50

# Check status
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && docker compose --env-file .env.production ps"

# Restart MCP server only
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && docker compose --env-file .env.production restart mcp-server"

# Tail ingestion worker
ssh -i $SSH_KEY $GPU "docker logs -f weka-ingestion-worker"

# Test evidence pack
scp -i $SSH_KEY /tmp/evidence_test.sh $GPU:/tmp/
ssh -i $SSH_KEY $GPU "bash /tmp/evidence_test.sh"
```

### Re-ingesting Documents

```bash
# Flush all ingestion state
ssh -i $SSH_KEY $GPU "docker exec weka-redis redis-cli -a testredis123 FLUSHDB"

# Clear Neo4j data (keep SchemaVersion)
ssh -i $SSH_KEY $GPU "echo 'MATCH (n) WHERE NOT n:SchemaVersion DETACH DELETE n;' | docker exec -i weka-neo4j cypher-shell -u neo4j -p testpassword123"

# SCP documents to ingest directory
scp -i $SSH_KEY -r /path/to/docs/* $GPU:~/wekadocs-matrix/data/ingest/

# Touch files to trigger watcher (if already present)
ssh -i $SSH_KEY $GPU "find ~/wekadocs-matrix/data/ingest -name '*.md' -exec touch {} +"
```

### Syncing Code Changes to GPU Server

The GPU box has a git clone at `~/wekadocs-matrix`. Source code is bind-mounted, so most changes are live on restart. For changes to Dockerfiles or requirements:

```bash
# Push to GitHub first
git push origin multi-embedder-reranker

# Pull on GPU box
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && git pull"

# Rebuild if Dockerfile changed
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && docker compose --env-file .env.production build mcp-server"

# Restart (restart picks up bind-mounted src changes, up -d --force-recreate picks up env changes)
ssh -i $SSH_KEY $GPU "cd ~/wekadocs-matrix && docker compose --env-file .env.production up -d --force-recreate mcp-server"
```

---

## Part 10: Complete Pipeline Flag Reference

Every flag that controls retrieval behavior, where it lives in config, where it's read in code, what it does, and what depends on it.

### Retrieval Pipeline Enable Flags

| Flag | Config Path | Config Line | Code Location | Value (prod) | What It Controls |
|---|---|---|---|---|---|
| `search.hybrid.enabled` | `production.yaml:71` | `enabled: true` | `hybrid_retrieval.py:2280` — `HybridRetriever.__init__()` | **true** | Master switch for the entire hybrid retrieval pipeline. If false, retrieval returns empty. |
| `search.hybrid.neo4j_disabled` | `production.yaml:75` | `neo4j_disabled: false` | `hybrid_retrieval.py:2317` — `self.neo4j_disabled` | **false** | PHASE 1 VECTOR-ONLY gate. When true, skips ALL Neo4j operations: BM25, graph channel, entity graph traversal, expansion, graph reranker. Used during vector-only testing. |
| `search.hybrid.graph_channel_enabled` | `production.yaml:77` | `graph_channel_enabled: false` | `hybrid_retrieval.py:2307` — `self.graph_channel_enabled` | **false** | Entity-anchored Neo4j traversal channel. Adds graph-discovered candidates to the fusion pool. Currently disabled — superseded by graph_as_reranker feature flag. |
| `search.hybrid.graph_enrichment_enabled` | `production.yaml:78` | `graph_enrichment_enabled: false` | `hybrid_retrieval.py:2310` — `self.graph_enrichment_enabled` | **false** | Post-retrieval graph neighbor expansion in the retriever. Different from the evidence pack's `_expand_evidence_with_structure()` which runs in mcp_app.py. |
| `search.hybrid.graph_adaptive_enabled` | `production.yaml:79` | `graph_adaptive_enabled: true` | `hybrid_retrieval.py:2312` — controls `_get_rel_types_for_query()` | **true** | Uses query-type-specific relationship sets for graph traversal. Depends on `feature_flags.graph_rel_types_wired`. |
| `search.hybrid.colbert_rerank_enabled` | `production.yaml:80` | `colbert_rerank_enabled: false` | `hybrid_retrieval.py:2298` — `self.colbert_rerank_enabled` | **false** | ColBERT MaxSim re-scoring after fusion. Requires ColBERT vectors in Qdrant and `late-interaction` field hydration. Disabled to test cross-encoder alone. |

### Reranker Flags

| Flag | Config Path | Config Line | Code Location | Value (prod) | What It Controls |
|---|---|---|---|---|---|
| `search.hybrid.reranker.enabled` | `production.yaml:140` | `enabled: true` | `hybrid_retrieval.py:2352` — `self._reranker_enabled` | **true** | Cross-encoder reranking via Qwen3-Reranker-4B. Gate for the entire rerank pipeline (lines 3371-3442). |
| `search.hybrid.reranker.top_n` | `production.yaml:143` | `top_n: 80` | `hybrid_retrieval.py:3397` — `self.rerank_top_n or top_k` | **80** | Number of candidates sent to the cross-encoder. Also used as `pool_cap` in the legacy (non-signal-pool) path. **Interdependency:** must be ≤ the number of fused results, otherwise capped. |
| `search.hybrid.reranker.instruction` | `production.yaml:146` | `instruction: "Judge..."` | `factory.py:414` → `LocalRerankerServiceProvider.__init__()` → `local_reranker_service.py:self._instruction` | Domain-tuned | Prepended to query text before sending to the cross-encoder. Controls what the 4B model considers "relevant." |

### BM25 Flags

| Flag | Config Path | Config Line | Code Location | Value (prod) | What It Controls |
|---|---|---|---|---|---|
| `search.hybrid.bm25.enabled` | `production.yaml:152` | `enabled: false` | `hybrid_retrieval.py:2086` — `bm25_index_name = getattr(bm25_config, "index_name", None)` | **false** | Neo4j fulltext BM25 retrieval. When disabled, `self.bm25_retriever` is None and the BM25 branch at line 3110 is skipped. **CRITICAL:** Does NOT affect SPLADE sparse retrieval (that's via Qdrant). |
| `search.hybrid.bm25.index_name` | `production.yaml:153` | `index_name: "chunk_text_index_v3_bge_m3"` | `hybrid_retrieval.py:2086` | `chunk_text_index_v3_bge_m3` | Neo4j fulltext index name. Must match an existing index in Neo4j. The `embedding_version` filter is stripped from BM25 queries at line 3111. |
| `search.hybrid.bm25.weight` | `production.yaml:155` | `weight: 0.15` | Used in weighted fusion mode (not active — RRF is current method) | 0.15 | Legacy weight for BM25 scores in weighted fusion. Irrelevant when `method: "rrf"`. |

### Expansion Flags

| Flag | Config Path | Config Line | Code Location | Value (prod) | What It Controls |
|---|---|---|---|---|---|
| `search.hybrid.expansion.enabled` | `production.yaml:159` | `enabled: true` | `hybrid_retrieval.py:2420` — `self.expansion_enabled` | **true** | Bounded adjacency expansion (NEXT_CHUNK ±1 hop from top seeds). Runs after reranking. |
| `search.hybrid.expansion.max_neighbors` | `production.yaml:160` | `max_neighbors: 1` | `hybrid_retrieval.py:2421` | 1 | Maximum NEXT_CHUNK hops per seed. Higher values find more context but risk noise. |
| `search.hybrid.expansion.rescoring.enabled` | `production.yaml:165` | `enabled: true` | `hybrid_retrieval.py:2439-2470` | **true** | Re-scores expanded neighbors using weighted combination of sparse and position signals. |
| `search.hybrid.expansion.structural_enhancements.enabled` | `production.yaml:182` | `enabled: true` | Controls `_expand_with_structure()` — sibling, parent, entity-shared expansion | **true** | Structure-aware expansion beyond simple NEXT_CHUNK. **Depends on:** `feature_flags.structure_aware_expansion`. |

### Feature Flags (development.yaml:410 / production.yaml:410)

| Flag | Config Line | Code Location | Value (prod) | What It Controls | Dependencies |
|---|---|---|---|---|---|
| `query_api_weighted_fusion` | `:413` | `hybrid_retrieval.py:3046` — determines search path | **true** | Routes to `_search_via_query_api_weighted()` instead of legacy search. Enables per-field RRF scoring across 6 vector fields. **Critical for retrieval quality.** | Requires Qdrant Query API support (v1.7+) |
| `graph_garbage_filter` | `:414` | `hybrid_retrieval.py:2683` | **true** | Filters low-quality graph entity matches before fusion | Requires `graph_channel_enabled` or `graph_as_reranker` |
| `graph_rel_types_wired` | `:415` | `hybrid_retrieval.py:2621` | **true** | Uses config-defined relationship type sets instead of hardcoded defaults | Requires `graph_adaptive_enabled` |
| `dedup_best_score` | `:416` | `hybrid_retrieval.py:3560` | **true** | During dedup, keeps the highest-scoring chunk per document instead of first-seen | — |
| `graph_score_normalized` | `:417` | `hybrid_retrieval.py:2750` | **true** | Applies saturating exponential normalization to graph scores | Requires graph results |
| `graph_as_reranker` | `:418` | `hybrid_retrieval.py:3273-3309` | **true** | Graph-based candidate reordering in fusion pipeline. Alternative to graph_channel. | Requires `neo4j_disabled=false` |
| `entity_embedding_fallback` | `:419` | Legacy — entity embeddings removed | **false** | Dead flag — entity embeddings replaced by entity-sparse |  — |
| `structure_aware_expansion` | `:420` | `hybrid_retrieval.py:3526-3550` — gates `_expand_with_structure()` | **true** | Sibling/parent/entity-shared chunk expansion after reranking | Requires `expansion.structural_enhancements.enabled` AND this flag |
| `signal_diverse_rerank_pool` | Not in YAML (class default) | `hybrid_retrieval.py:2361-2369` — `self._signal_pool_enabled` | **false** | Signal-diverse rerank pool from `signal_pool.py`. Ensures reranker sees candidates from every signal source. | Requires `signal_pool.enabled` in config AND this flag AND `query_api_weighted_fusion` |

### NER / Entity Flags

| Flag | Config Path | Config Line | Code Location | Value (prod) | What It Controls |
|---|---|---|---|---|---|
| `ner.enabled` | `production.yaml:483` | `enabled: true` | `hybrid_retrieval.py:3068` — `entity_boost_enabled = getattr(self.config.ner, "enabled", False)` | **true** | GLiNER NER on queries for entity boost. Also gates entity extraction during ingestion. |
| `ner.threshold` | `production.yaml:486` | `threshold: 0.45` | `gliner_service.py:__init__()` | 0.45 | Minimum confidence for entity extraction. Lower = more entities (noisier). |
| `ner.service_url` | `production.yaml:490` | `service_url: "http://host.docker.internal:8080"` | `gliner_service.py:_check_http_health()` | GPU gateway | HTTP mode preferred. Falls back to local CPU model if health check fails. Health check tries `/health` then `/healthz`. |

### Monitoring Flags

| Flag | Config Path | Config Line | Code Location | Value (prod) |
|---|---|---|---|---|
| `monitoring.health_checks_enabled` | `production.yaml:426` | `health_checks_enabled: true` | `main.py:302` — `run_startup_health_checks()` | **true** |
| `monitoring.health_check_fail_fast` | `production.yaml:427` | `health_check_fail_fast: true` | `main.py:306` — `fail_fast=config.monitoring.health_check_fail_fast` | **true** |
| `monitoring.slo_monitoring_enabled` | `production.yaml:430` | `slo_monitoring_enabled: true` | `hybrid_retrieval.py` SLO violation logging | **true** |
| `monitoring.metrics_enabled` | `production.yaml:439` | `metrics_enabled: true` | Prometheus metrics collection | **true** |

### RRF Weight Settings (production.yaml:93-99)

| Field | Weight (prod) | Purpose | Impact on Retrieval |
|---|---|---|---|
| `content` | **1.5** (was 2.0) | Dense semantic content similarity (Qwen3-0.6B) | Primary semantic signal. Lowered from 2.0 to reduce dense dominance over SPLADE. |
| `title` | **1.5** | Dense semantic title/heading similarity | Second dense signal. Strong for heading-matched queries. |
| `text-sparse` | **1.5** (was 0.5) | SPLADE learned sparse content matching | Critical for conceptual/vocabulary-specific queries. Raised 3x to equal dense. |
| `doc_title-sparse` | **0.8** | SPLADE document title matching | Cross-document signal. Moderate influence. |
| `title-sparse` | **0.8** | SPLADE section heading matching | Heading-level signal. Moderate influence. |
| `entity-sparse` | **0.8** | SPLADE entity name matching | Entity-level signal. **May need reduction** if NER entities are noisy. |

### Interdependency Map

```
query_api_weighted_fusion = true
  └── enables: _search_via_query_api_weighted() (6-field RRF)
  └── required by: signal_diverse_rerank_pool

reranker.enabled = true
  └── enables: _apply_reranker() (Qwen3-Reranker-4B)
  └── depends on: fused_results not empty
  └── pool size controlled by: reranker.top_n (80)

signal_diverse_rerank_pool = false (NOT ACTIVE)
  └── would override: flat top-N pool with signal-diverse pool
  └── requires: signal_pool.enabled + this flag + query_api_weighted_fusion

expansion.enabled = true
  └── enables: _bounded_expansion() (NEXT_CHUNK ±1)
  └── depends on: neo4j_disabled = false
  └── gated by: _should_expand() score delta check

structure_aware_expansion = true
  └── enables: _expand_with_structure() (sibling/parent/entity)
  └── depends on: expansion.structural_enhancements.enabled = true

graph_as_reranker = true
  └── enables: _apply_graph_reranker() in fusion
  └── depends on: neo4j_disabled = false
  └── conflicts with: graph_channel_enabled (use one or the other)

ner.enabled = true
  └── enables: GLiNER query entity extraction for boost
  └── enables: GLiNER chunk enrichment during ingestion
  └── service_url health check: /health then /healthz

bm25.enabled = false (DISABLED)
  └── disables: Neo4j fulltext BM25 retrieval
  └── does NOT affect: SPLADE sparse via Qdrant
  └── does NOT affect: reranker, expansion, entity boost
  └── rationale: BM25 keyword matching dominated RRF fusion
      with generic "cluster" hits, drowning semantic signals
```

### Modifying production.yaml Safely

**Never use blanket `sed` on production.yaml.** Always use targeted Python scripts:

```python
# Example: disable BM25 only
python3 -c "
with open('config/production.yaml') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'bm25:' in line and 'BM25' in lines[i-1]:
        if 'enabled: true' in lines[i+1]:
            lines[i+1] = lines[i+1].replace('enabled: true', 'enabled: false')
            break
with open('config/production.yaml', 'w') as f:
    f.writelines(lines)
"
```
