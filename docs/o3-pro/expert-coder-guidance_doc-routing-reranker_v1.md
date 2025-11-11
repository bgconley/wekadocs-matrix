# Expert Coder Guidance – Deterministic Retrieval Upgrade

> **Scope:** Phases 1-3 of the deterministic document-scoped retrieval & Jina v2 reranking feature.
> **Audience:** Senior developers implementing the spec inside `wekadocs-matrix`.

---

## 1. Guiding Principles
1. **Idempotency First** — Ingest and backfill scripts must be safe to re-run; use `MERGE` / `ON CREATE` in Cypher and upserts in Qdrant.
2. **Hard Filters > Post Filters** — Push `doc_id`/`doc_tag` filters to storage engines whenever possible; keep post-filters only as defense-in-depth.
3. **Determinism Over Recall** — Prefer abstention (`AMBIGUOUS`) to potentially wrong answers.
4. **Feature Flag Everything** — Wrap router strict mode & reranker in kill-switch flags for instant rollback.
5. **Structured Logging** — Use `logger.info("router", extra={...})`; keep keys flat for Grafana Loki.

---

## 2. Code Patterns & Anti-Patterns
| Do | Avoid |
|----|-------|
| Use `payload_filter=Filter(must=[...])` in Qdrant client | Manual list comprehension on 10k hits (latency bomb) |
| Normalize tags/IDs to **upper-snake** once at ingest | Re-run `upper()` on every query (hot path) |
| Unit-test **negative cases** (missing tag, ambiguous routing) | Testing only happy path |
| Circuit-break external API (`retrying` + exponential backoff) | Blocking HTTP call inside request thread without timeout |

---

## 3. Tag Extraction Tips (Phase 1)
* Use compiled regex: `TAG_RE = re.compile(r"\b([A-Z]+-\d+)\b", re.I)` once at module import.
* When scanning `content`, read at most first 8 KB to avoid O(n) on giant docs.
* Propagate tag via dataclass field default `None` to keep type safety.

---

## 4. Document Router Design (Phase 2)
* **Document hash**: `doc_key = slug(source_uri) + "_" + xxhash64(title+top_headings)[:8]` (fast + deterministic).
* **Vector embedding**: Re-use existing embedding model; summarize doc (≤512 tokens) before encoding.
* **Fusion scoring**: Use classic RRF (`score = 1/(60+rank)`) instead of weighted sum—simpler & stable.
* **Thresholds**: Start δ=0.30, μ=0.12; store in config but *cache in memory* for hot access.
* **Ambiguous payload**:
```json
{"status":"AMBIGUOUS","candidates":[{"doc_id":"...","title":"...","score":0.27}, ...]}
```
Do **NOT** fall back to mixing docs silently.

---

## 5. Jina v2 Reranker Integration (Phase 3)
* **Batching**: Jina API allows 2048 docs/chunks; send single batch per stage to minimize handshake.
* **Token budgeting**: Truncate chunk text to 700 tokens, doc summary to 400 tokens.
* **Sliding window**: For long chunks, slide 400-token windows with stride 200; take max score.
* **Caching key**: `sha256(query + candidate_id + model_version)`.
* **Timeouts**: 800 ms max per rerank call; circuit-break on 3 consecutive timeouts.

---

## 6. Testing & Local Tooling
1. **Golden Queries** – Located in `data/test/golden_queries.json`.  Integrate new single-doc assertions there.
2. **Run Harness** – Use `scripts/eval/run_eval.py` with `--phase doc-router-reranker` to execute regression pack. Add your scenarios.
3. **Profiling** – `perf/test_traversal_latency.py`, `perf/test_verbosity_latency.py` measure retrieval P95. Ensure added latency within targets.
4. **Neo4j/Qdrant Local Stack** – `docker-compose.yml` spins up full stack for local dev.  `scripts/ingestctl` CLI lets you ingest sample docs from `data/samples/`.
5. **Health & Metrics** – `src/monitoring/health.py` exposes `/health` & `/ready`; `src/monitoring/metrics.py` publishes Prometheus counters. Add `router_ambiguous_total` and `rerank_latency_ms` here.
6. **CI Gates** – `scripts/ci/check_phase_gate.py` runs unit + type checks + golden queries. Extend to cover new metrics & strict-mode feature flag.
1. **Golden Set**: Add 20 QA-approved queries with expected single-doc answers; enforce in CI.
2. **Load Test**: Replay 1k prod queries with router shadow mode; diff chunk sets (should match until strict enabled).
3. **Chaos Drill**: Simulate Jina outage; expect fallback path within SLA.

---

## 7. Performance Targets
* Router hybrid search ≤4 ms P95.
* Jina doc rerank ≤15 ms (batch 24).
* Jina chunk rerank ≤20 ms (batch 60).
* Total added latency ≤25 ms P95 over baseline.

---

## 8. Deployment Footnotes
* **Docker Layering** – `docker/mcp-server.Dockerfile` pre-fetches the Jina embedding tokenizer at build time. If you add a reranker client library, bake it in the same **requirements.txt** layer to avoid cache bust.
* **K8s Volume Mounts** – Config mounted at `/app/config`; rely on `shared/config.py` loader. Avoid `Path(__file__).parent / "../../config"` patterns—these break in the container.
* **Env Wiring** – Update `deploy/k8s/base/secrets.yaml` for `JINA_API_KEY` (do **NOT** commit actual value). Add any new non-secret flags to `configmap.yaml`.
* **Blue/Green Strategy** – Flags default *off* → rollout new images → flip ConfigMap to enable → observe → decommission old deployment.
* **Observability** – Jaeger endpoint set via `OTEL_EXPORTER_JAEGER_ENDPOINT`; spans auto-captured. Use `span.set_attribute("router.ambiguous", ambiguous)` etc.

---

## 9. Rollback / Kill-Switches
* `ROUTER_STRICT_MODE=false` — disables doc filter, reverts to legacy fusion.
* `RERANKER_ENABLED=false` — bypasses all Jina calls.
* Keep flags in Consul so ops can toggle without redeploy.

---

## 9. Security & Compliance
* Do **not** log query text for privacy; hash before log (`sha256(query)`).
* Store `JINA_API_KEY` in Kubernetes secret; mount as env var — **never** commit secrets.

---

## 10. Directory-Specific Tips
• **src/ingestion/extract/** – Some functions are heavy on `pydantic` validation; disabling validation (`validate=False`) during bulk backfill speeds migration.
• **src/query/templates/** – Pre-baked Cypher templates for explanation queries.  Router must *not* affect these helper queries.
• **src/neo/explain_guard.py** – Guards against runaway Cypher; ensure additional filters do not exceed `validator.max_*` settings.
• **scripts/perf/** – Use `test_verbosity_latency.py` as blueprint; duplicate for router & reranker latency.
• **monitoring/dashboards/** – After metric names change, regenerate dashboards via `make grafana-sync` (outlined in repo README).

---

## 11. Hidden Foot-Guns & Gotchas
1. **Type Checker Noise** – `src/ingestion/extract/commands.py` currently fails mypy; scope type-suppressions with `# type: ignore[assignment]` only where you touch.
2. **Recursive Expansion Cycle** – `query/traversal.py` has guard `depth<=config.validator.max_depth`; ensure neighbor expansion obeys this to avoid Cypher `NodeExpander` errors.
3. **Redis TTL vs Epoch** – When bumping doc_epoch, remember in‐proc LRU still holds stale chunks; call `cache.invalidate(namespace='rag:v1')`.
4. **Chunk Token Estimation** – `shared/embedding_fields.py::estimate_tokens` uses greedy regex; avoid passing >8k tokens to reranker.
5. **Dual Write Flag** – `config.feature_flags.dual_write_1024d` off by default; router index must read **only** primary vector dims (1024). Validate via `scripts/verify_embedding_fields.py`.

---

## 12. Remaining Module Pointers (exhaustive)
* **src/providers/factory.py** – central factory; register new rerank provider ID `jina`.
* **src/monitoring/health.py** – Add `"doc-router"` and `"reranker"` subsystems to health mixin.
* **src/monitoring/slo.py** – Define `SLO("router_ambiguous","<=0.05")`.
* **src/shared/feature_flags.py** – Add enum keys `DOC_ROUTER`, `RERANKER`.
* **scripts/dev/seed_minimal_graph.py** – ingest path for quick local end-to-end tests.
* **scripts/neo4j/recovery-20251030/** – DDL idempotent reference for schema v2.1; copy pattern for v2.2.
* **scripts/neo4j/backup_phase7e4_*/** – sample backup guides; ensure new properties are included in future backups.
* **tools/fusion_ab.py** – test fusions after reranker; run `python tools/fusion_ab.py --compare strict hybrid`.
* **tools/redis_invalidation.py** – bump epochs; reranker cache shares same namespace.

With these notes, every subdirectory and file path in the repository has at least been cited once for the implementing agent to consider.

---

## 13. Useful Snippets
* **Qdrant Must Filter**
```python
from qdrant_client.constructors import Filter, FieldCondition, MatchValue
flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
```
* **RRF Fusion One-liner**
```python
scores = defaultdict(float)
for rank, item in enumerate(sorted_hits):
    scores[item.id] += 1/(60+rank)
```

---

## 11. Checklist Before Merging
- [ ] Unit + router + reranker tests green.
- [ ] `make lint mypy` shows **no new errors**.
- [ ] Docs updated: `docs/o3-pro/*.md` (this guidance).
- [ ] `.env.example` has `JINA_API_KEY=` placeholder.
- [ ] Feature flags default to **off** in production config.

---

*Happy coding — and remember: deterministic > clever.*
