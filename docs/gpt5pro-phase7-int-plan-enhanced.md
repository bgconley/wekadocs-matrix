1.  **Make safety non‑optional in 7a.** Add minimal EXPLAIN‑plan/timeout guards in 7a so we never regress on stability during validation.
2.  **Strengthen the 7a gate.** Use a small _golden set_ of real queries instead of a single “WEKA inode” query; add an explicit hallucination/provenance check.
3.  **Unify graph responsibilities.** Keep 7a “graph mode = 1‑hop evidence context” but define its output schema now so 7b’s `traverse_relationships` doesn’t change payload shapes.
4.  **Move perf baselining earlier (7a→7c overlap).** Start collecting latency/size metrics as soon as verbosity is enabled; you need that data to make 7b/8a decisions.
5.  **Plan for multi‑index embeddings in 8b.** If you dual‑write Jina + OpenAI embeddings, you’ll need parallel vector indexes and routing; call that out explicitly.
6.  **Tighten acceptance thresholds.** Define “Go/No‑Go” numerically for each gate (quality, perf, cost), and pre‑wire the kill‑switches.
7.  **Clarify relation allow‑list.** In Phase 7 keep `MENTIONS` + `HAS_SECTION` as you propose; in Phase 9, evolve to `HAS_CHUNK`/`HAS_TOPIC`/`RELATED_TO` with a bridge plan (no silent breaks).

Everything else in your plan is strong and aligns with the desired end state.

* * *

Line‑by‑line critique & upgrades
--------------------------------

### Phase 7a — Fix & validate E1–E7 basics

**What’s great**

*   Focused scope (verbosity modes + small graph preview).
*   Hard caps (32 KB full text; 50 nodes in graph).

**Gaps / Risks**

*   **Safety**: No EXPLAIN guard yet; a malformed Cypher or unlabeled pattern can still blow up.
*   **Validation bias**: A single test query is too narrow.
*   **Ambiguity**: 32 KB — bytes or tokens? Full‑text sizes often need token‑aware limits.

**Enhancements**

*   **Add minimal safety now** (no perf hit):
    *   Driver timeout (≤1500 ms) and _basic_ EXPLAIN‑plan reject for `Expand(All)` and label‑less traversals.
    *   Allow‑listed labels: `Section|Entity` (your current graph) and relations: `MENTIONS|HAS_SECTION`.
*   **Golden set (10–20 queries)** spanning:
    *   API error messages, feature lookups, file systems (e.g., WEKA), networking, “how‑to” patterns, and ambiguous queries.
*   **Provenance check**: For each full answer, verify that every claim is supported by at least one returned section ID (even if you don’t render citations yet).
*   **Make the 32 KB limit explicit**:
    *   Server‑cap by **bytes** _and_ by **tokens** (e.g., 16 KB bytes hard cap _or_ 2,000 tokens; whichever triggers first).
    *   Truncate at paragraph boundaries; return `truncation:true` in metadata.

**Revised Gate 7a**

*   ✅ P95 latency: `snippet≤200 ms`, `full≤350 ms`, `graph(1‑hop)≤450 ms` under default caps.
*   ✅ Golden‑set _completeness_ improves ≥30% over “snippet”.
*   ✅ **0** unbounded plans (all queries pass EXPLAIN guard).
*   ✅ Provenance: 100% of assertions traceable to at least one section ID.
*   ⚠️ If _completeness≥80%_ **and** P95 targets met → optional stop.

* * *

### Phase 7b — `traverse_relationships` tool

**What’s great**

*   BFS with server caps, separate MCP tool, circuit breaker.

**Gaps / Risks**

*   Payload shape not specified; 7a “graph mode” might not match 7b output.
*   No explainability (why a node made it into the frontier) until Phase 8c.

**Enhancements**

*   **Freeze the schema now** (so 7a graph and 7b tool match):
    ```json
    {
      "nodes": [{"id":"…","label":"Section|Entity","title":"…","score":null}],
      "edges": [{"src":"…","dst":"…","type":"MENTIONS|HAS_SECTION","depth":1}],
      "budget": {"expanded": N, "depth_reached": D, "skipped": M},
      "explain": {"reason": "bfs", "caps":{"max_depth":3,"max_nodes":100},"stop":"depth_limit|timeout|node_cap"}
    }
    ```
*   **Return IDs deterministically** (stable sort by `depth, type, id`) to simplify caching and tests.
*   **Add _light_ dedupe** at 7b (ID‑based; score not needed yet) to avoid obvious repeats.

**Revised Gate 7b**

*   ✅ Depth=2 traversal on a known seed returns **≥1** relevant entity + **≥2** relevant sections.
*   ✅ P95 depth=2 ≤350 ms; depth=3 ≤600 ms (caps: max\_nodes=100).
*   ✅ Circuit breaker verified (two forced timeouts trigger disable for 60 s).

* * *

### Phase 7c — Perf baseline & monitoring

**What’s great**

*   Scripts and counters planned.

**Enhancements**

*   **Start counters in 7a** (don’t wait): `verbosity_total`, `response_bytes`, `graph_nodes_returned`, `explain_reject_total`, `timeout_total`.
*   **Cost probe** (even if $0 now): record _estimated_ token/compute cost per query so Phase 8 comparisons are apples‑to‑apples.
*   **Artifacts**: Store CSV + Markdown summary per run under `/reports/phase-7/`.

**Revised Gate 7c**

*   ✅ Baseline tables for P50/P95 per mode by query family.
*   ✅ At least 30 golden queries × 3 modes recorded.
*   ✅ Known bottlenecks documented (driver, Cypher, vector search, I/O).

* * *

### Phase 8a — Provider abstraction

**What’s great**

*   Interfaces, config, caching, A/B harness.

**Enhancements**

*   **Deterministic hashing** for cache keys: `sha256(normalize(text)+model_version)`.
*   **Cache warmers**: optional nightly job to pre‑embed top N frequent sections.
*   **Fallback routing table** (ordered): `jina → openai → hf` with per‑provider circuit breaker windows.
*   **Telemetry parity**: identical metrics dimensions for each provider (`provider`, `model`, `lat_ms`, `err_kind`).

**Gate 8a**

*   ✅ Cache hit rate ≥60% on second pass (you set 50%—raise it).
*   ✅ A/B harness records NDCG/precision _and_ cost and latency with the same IDs.

* * *

### Phase 8b — Jina integration

**What’s great**

*   Dual‑write + canary rollout + rollback.

**Critical enhancement (often missed)**

*   **Parallel vector indexes.** If Jina dimensions differ from current:
    *   Create `docs_embed_openai` and `docs_embed_jina` fields with separate HNSW/IVF indexes; route queries by provider flag.
    *   **Backfill plan**: rolling batch job; route to provider only after N% backfilled (e.g., >80%), else fall back to the other index for that doc.
    *   **Version pinning**: store `embedding_provider`, `embedding_model`, `embedding_version` alongside vectors.

**Gate 8b (tighten)**

*   ✅ NDCG@10 improvement **≥15%** on golden‑set and a live query sample.
*   ✅ P95 embedding latency ≤150 ms (batch size 16) measured end‑to‑end.
*   ✅ <0.5% provider errors after circuit breaker warmup.

* * *

### Phase 8c — Frontier gating & intelligent traversal

**What’s great**

*   MMR‑style novelty, delta thresholds, explainability.

**Enhancements**

*   **Budget‑aware scoring**: incorporate remaining node/depth budget in the score (stop earlier when marginal utility drops).
*   **Cold‑start fallback**: if embeddings unavailable, revert to BFS with strict caps and set `explain.reason="fallback_bfs"`.
*   **Audit trail**: log `pruned_count`, `expanded_count`, and top 3 “would‑be” expansions with scores for tuning.

**Gate 8c**

*   ✅ Node expansions reduced **≥40%** with **≤2%** NDCG drop vs. BFS baseline (or improved).
*   ✅ P95 traversal depth=3 ≤300 ms with gating **on**.

* * *

### Phase 8d — EXPLAIN‑plan validation & safety

**What’s great**

*   Good scope (plan checks, timeouts, rate limits).

**Enhancements**

*   **Label‑whitelist enforcement** at query level: `Section|Entity` only in Phase 7/8; DocRAG labels appear only in Phase 9.
*   **Depth syntax checks**: reject `*..` variable‑length unless bounded and allow‑listed.
*   **Graceful degradation**: downgrade to `snippet` and return `diagnostics.degraded=true` when capped.

**Gate 8d**

*   ✅ 100% of dynamic Cypher calls pass EXPLAIN guard or are rejected with structured error.
*   ✅ Rate‑limit denials <1% of requests; all denials logged with user/session budget.

* * *

### Phase 9a — DocRAG schema bridge (optional)

**What’s great**

*   Dual‑labeling migration and ephemeral `Session|Query|Answer`.

**Enhancements**

*   **Idempotent migrations** with `dry_run=true` mode that prints counts only.
*   **Bridge edges only** (read‑only from doc graph to ops graph) in this phase to reduce blast radius.
*   **Canary queries**: keep old and new templates and diff outputs.

**Gate 9a**

*   ✅ No P95 regression; ✅ provenance check passes; ✅ rollback script verified in staging.

* * *

### Phase 9b / 9c — Observability, rollout

**Enhancements**

*   **Unified trace**: span names like `embed`, `vector_search`, `rerank`, `graph_expand`, `explain_guard`, `neo4j_run`.
*   **Error budget SLO**: if error budget burns >25% in a day, auto‑reduce canary traffic.
*   **Runtime config**: a single `/config` endpoint (or file) advertising current flags to clients (helps support).

* * *

API & contract refinements (keep payloads stable)
-------------------------------------------------

**`search_documentation`** (Phase 7a; additions bolded):

```json
{
  "query": "string",
  "top_k": 8,
  "verbosity": "snippet" | "full" | "graph",
  "providers": { "embedding": "auto", "reranker": "auto" },
  "limits": {
    "full_text_bytes": 16384,
    "full_text_tokens": 2000,
    "graph": { "max_depth": 1, "max_nodes": 50, "rels": ["MENTIONS","HAS_SECTION"] }
  },
  "diagnostics": { "return": true }  // NEW: opt-in diagnostics in 7a
}
```

**Response (stable across 7a→8c):**

```json
{
  "evidence": [
    {
      "section_id":"…",
      "title":"…",
      "snippet":"…",
      "full_text":"…",             // only when verbosity=full
      "metadata":{"doc_uri":"…","anchor":"…","score":0.71,"truncation":false}
    }
  ],
  "graph": {                        // present for verbosity=graph
    "nodes":[{"id":"…","label":"Section|Entity","title":"…","score":null}],
    "edges":[{"src":"…","dst":"…","type":"MENTIONS|HAS_SECTION","depth":1}],
    "budget":{"expanded":37,"depth_reached":1,"skipped":12},
    "explain":{"reason":"bfs","caps":{"max_depth":1,"max_nodes":50},"stop":"depth_limit"}
  },
  "diagnostics":{
    "retrieval":{"provider":"openai|jina|hf","reranker":"…","lat_ms":94},
    "safety":{"explain_ok":true,"timeout_ms":1200,"degraded":false}
  }
}
```

**`traverse_relationships`** (Phase 7b; stable through 8c):

```json
{
  "seed_ids":["…"],
  "direction":"both",
  "max_depth":3,
  "max_nodes":100,
  "rel_whitelist":["MENTIONS","HAS_SECTION"],
  "label_whitelist":["Section","Entity"],
  "frontier": null,                 // becomes object in 8c
  "query_text": null                // becomes string in 8c for scoring
}
```

* * *

Metrics & SLOs (tightened, realistic)
-------------------------------------

*   **Latency** (staging targets; prod a bit looser until 8c):
    *   `snippet` P50≤120 ms / P95≤250 ms
    *   `full` P50≤180 ms / P95≤350 ms
    *   `graph` P50≤220 ms / P95≤450 ms (1‑hop)
    *   `traverse depth=3` P95≤600 ms (7b BFS) → ≤300 ms (8c frontier on)
*   **Quality**
    *   `ndcg@10 ≥0.75`, `precision@5 ≥0.80` (same as yours)
    *   **Provenance**: 100% claims supported by evidence IDs
    *   **Hallucination**: <2% on golden‑set adjudication
*   **Efficiency**
    *   Cache hit rate: ≥60% run‑2; ≥75% by end of 8a with warmers
    *   Node expansion reduction: ≥40% (8c) with ≤2% NDCG delta
*   **Cost**
    *   Per‑query cost tracked by provider; alerts at 50/75/90% of weekly budget

* * *

Validation gates (recast as checklists)
---------------------------------------

Each gate ships **artifacts**:

*   JSON schema snapshots
*   JUnit/pytest results
*   Perf CSVs + Markdown report
*   Config snapshot (flags, provider routing)

**Gate 7a Checklist**

*    Verbosity modes pass schema validation
*    EXPLAIN guard & timeouts active
*    Golden‑set report (completeness, provenance, latency)
*    Flags documented: `FEATURE_VERBOSITY_ENABLED`, `FEATURE_GRAPH_MODE_ENABLED`

**Gate 7b Checklist**

*    `traverse_relationships` schema frozen & documented
*    BFS caps enforced; circuit breaker test run
*    Deterministic ordering verified

**Gate 8a Checklist**

*    Provider interfaces + Redis cache + hit‑rate report
*    A/B harness results with cost/latency/quality

**Gate 8b Checklist**

*    Dual indexes online + routing verified
*    Backfill progress tracker; ≥80% before default flip
*    Canary 10%→50%→100% with rollback rehearsed

**Gate 8c Checklist**

*    Frontier gating on; pruned/expanded metrics
*    Explainability fields populated

**Gate 8d Checklist**

*    Rate‑limits & budgets enforced; denial logs sampled
*    Degradation path tested (fallback to snippet)

**Gate 9a–c Checklist**

*    Migration dry‑run counts; rollback tested
*    No perf regression; provenance strict
*    SLOs met 7 consecutive days; docs + runbook complete

* * *

Risks you called out (and how your plan now mitigates them better)
------------------------------------------------------------------

*   **Neo4j query explosion** → EXPLAIN guard already in 7a; hard caps and label allow‑lists hold across phases.
*   **Embedding migration pitfalls** → parallel indexes + routing; version pinning; backfill tracking.
*   **Costs** → per‑provider cost telemetry; budget alerts; provider fallback & cache warmers.

* * *

Updated schedule (minor tweaks)
-------------------------------

*   **Week 1**
    *   Mon–Tue: 7a with **minimal safety** + initial metrics
    *   Wed–Thu: 7b BFS tool (in parallel: keep metrics running)
    *   Fri: Consolidate Gate 7a/7b artifacts
*   **Week 2**
    *   Mon: 7c perf baselines (already collecting; publish)
    *   Tue–Wed: 8a provider abstraction
    *   Thu–Fri: 8b Jina canary + backfill start
*   **Week 3**
    *   Mon–Wed: 8c frontier gating
    *   Thu–Fri: 8d safety hardening (rate limits, budgets)
*   **Week 4 (optional)**
    *   9a DocRAG prototype in test; A/B vs. current
*   **Weeks 5–6**
    *   9b observability & 9c rollout with error budget SLOs

* * *

Final recommendations (actionable)
----------------------------------

1.  **Adopt the safety “starter pack” in 7a** (EXPLAIN, timeouts, allow‑lists). It’s cheap and prevents backtracking later.
2.  **Define and run a golden‑set** for Gate 7a; include provenance checks.
3.  **Freeze the graph payload schema now** so 7a graph preview and 7b traversal align.
4.  **Plan the dual‑index route** before 8b; it’s the most common migration trap.
5.  **Hold every gate to numeric thresholds** (quality, perf, and cost) with pre‑wired kill‑switches.

Your phased approach is solid. With these adjustments, you’ll preserve the low‑risk rollout while improving the odds that you can **stop earlier** if 7a/8a already solve the user‑visible problem—without compromising safety or forward‑compatibility.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
