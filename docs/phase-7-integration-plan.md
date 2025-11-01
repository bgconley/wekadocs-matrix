


Below is a **Phase 7 integration pack** that folds in:

*   **Enhanced responses** (verbosity controls + graph exploration),
*   **Performance hardening for graph queries** (frontier gating, EXPLAIN-plan checks, budgets/caps),
*   **Jina-based retrieval quality boosts** (embeddings + reranker, with provider abstractions),
*   **DocRAG lean schema alignment** to keep provenance and multi‑turn context clean.

It’s written in the exact shape an _agentic coder_ (or a team) can execute to completion without further prompts.

* * *

📄 feature-spec-phase-7 — _Enhanced Responses + Perf + Jina + DocRAG alignment_
-------------------------------------------------------------------------------

**Status:** Design-approved for Phase 7
**Scope key:** `verbosity + traverse_relationships + graph-perf + provider.jina + docrag-bridge`
**Compatibility:** Backward compatible with Phases 1‑6; defaults preserve current API behavior.

### 0) Problem & goals

*   **Problem.** Current answers return short snippets only; no mode to fetch full context or _explicitly_ explore connected graph evidence. Graph expansion can be slow/expensive on bigger neighborhoods. Retrieval quality can be improved with stronger embeddings and reranking. We also want a **lean DocRAG subgraph** with strict provenance and multi‑turn memory that doesn’t pollute the long‑term ops graph.
*   **Goals.**
    1.  Add `verbosity` modes to `search_documentation`: `snippet` | `full` | `graph`.
    2.  Add a new MCP tool `traverse_relationships` for controlled graph walks with guardrails.
    3.  Enforce **server-side** performance protections (EXPLAIN-plan checks, timeouts, hard caps, relationship/label allow-lists, frontier gating).
    4.  Introduce **provider interfaces** for embeddings/rerankers + **Jina** implementations.
    5.  Align with **DocRAG v3** lean schema (Document/Chunk/Entity/Topic + Session/Query/Answer edges) without disrupting the existing v2 ops/runtime graph.
*   **Non-goals.**
    *   UI revamp, editor features, or multi-tenant isolation.
    *   Replacing your existing ingestion; Phase 7 is read/query-centric.
    *   Vendor lock-in. Jina is pluggable via provider interface.

### 1) User stories

*   **S1 — “Quick look-up” (verbosity=`snippet`).**
    As a user I get 3–10 best sections with 200‑char snippets and IDs I can click into.
*   **S2 — “Give me everything” (verbosity=`full`).**
    As a user I get the same sections but **full text & metadata** (titles, anchors, doc URIs) to craft a thorough answer.
*   **S3 — “Follow the graph” (verbosity=`graph`).**
    As a user/agent I get evidence **plus** a compact subgraph (nodes, edges, labeled paths, titles) around those sections, bounded by policy.
*   **S4 — “Go deeper on these IDs” (tool=`traverse_relationships`).**
    As an agent I seed the walk with `section_ids`/`entity_ids` and receive a **policy-gated** multi-hop expansion (scored, capped, explainable).

### 2) Interfaces

#### 2.1 `search_documentation` (existing) — **extended**

**Request (additions):**

```json
{
  "query": "string",
  "top_k": 8,
  "verbosity": "snippet" | "full" | "graph",  // NEW (default: "snippet")
  "providers": {                               // NEW (optional)
    "embedding": "jina|openai|hf|auto",
    "reranker": "jina|cohere|none"
  },
  "limits": {                                  // NEW (optional overrides)
    "graph": {
      "max_depth_soft": 3,
      "max_depth_hard": 5,
      "max_nodes_hard": 200,
      "rel_whitelist": ["MENTIONS","HAS_TOPIC","RELATED_TO"]
    }
  }
}
```

**Response (union by mode):**

```json
{
  "answer": "optional freeform markdown from your LLM layer",
  "evidence": [
    {
      "section_id": "…",
      "title": "…",
      "snippet": "…200 chars…",
      "full_text": "…",             // when verbosity=full
      "metadata": { "doc_uri": "…", "anchor": "…", "score": 0.71 }
    }
  ],
  "graph": {                        // when verbosity=graph
    "nodes": [{ "id": "…", "label": "Chunk|Entity|Topic|Document", "title": "…", "score": 0.62 }],
    "edges": [{ "src": "…", "dst": "…", "type": "MENTIONS|HAS_TOPIC|RELATED_TO", "path": ["…","…"], "depth": 1 }],
    "budget": { "expanded": 37, "skipped": 18, "depth_reached": 3 }
  },
  "diagnostics": {
    "retrieval": { "provider": "jina", "reranker": "jina", "lat_ms": 94 },
    "graph": { "expanded": 37, "pruned": 81, "reason": "delta<τ" },
    "plan": { "explain_ok": true, "timeout_ms": 150 }
  }
}
```

#### 2.2 `traverse_relationships` (NEW MCP tool)

**Request:**

```json
{
  "seed_ids": ["…node ids…"],
  "direction": "out|in|both",
  "max_depth": 3,
  "max_nodes": 150,
  "rel_whitelist": ["MENTIONS","HAS_TOPIC","RELATED_TO"],
  "label_whitelist": ["Chunk","Entity","Topic","Document"],
  "frontier": { "top_k": 20, "delta_threshold": 0.05, "novelty_penalty": 0.2 }, // optional
  "filters": { "topic": ["…"], "entity": ["…"] }
}
```

**Response:**

```json
{
  "nodes": [{ "id":"…", "label":"…", "title":"…", "score":0.71 }],
  "edges": [{ "src":"…","dst":"…","type":"…","depth":2 }],
  "explain": { "scoring":"cos(query, node_embed)+…-novelty", "caps": { "max_depth":3,"max_nodes":150 } }
}
```

### 3) Data model alignment (DocRAG v3)

*   Keep your **v2 ops/runtime graph** intact (Servers, NICs, Pipelines, etc.) but run **DocRAG as a compact subgraph**:
    **Nodes:** `Document`, `Chunk` (your existing Section nodes can carry `:Chunk` as a **second label**), `Entity`, `Topic`, `Session`, `Query`, `Answer`.
    **Edges:**
    `(:Document)-[:HAS_CHUNK]->(:Chunk)`
    `(:Chunk)-[:MENTIONS]->(:Entity)`
    `(:Chunk|:Document)-[:HAS_TOPIC]->(:Topic)`
    `(:Query)-[:RETRIEVED {rank,score_*}]->(:Chunk)`
    `(:Query)-[:FOCUSED_ON {score}]->(:Entity)`
    `(:Answer)-[:SUPPORTED_BY {rank}]->(:Chunk)`
*   **Bridge-only edges** to the ops graph where useful (read-only in Phase 7): e.g., `(:Chunk)-[:REFERS_TO]->(:Feature|:APIEndpoint)`.
*   **Provenance rule:** _no chunk → no claim_. Your LLM layer must cite `SUPPORTED_BY` chunks for every assertion.

### 4) Performance & safety requirements

*   **EXPLAIN-plan guard:** Reject Cypher with high `Expand(All)` or plan-estimated rows/hops beyond thresholds; reject label scans outside allow-listed labels; enforce server-side **timeouts**.
*   **Hard caps:** `MAX_DEPTH_HARD=5`, `MAX_NODES_HARD=200`, `TIMEOUT_MS≤2000` for traversal queries; configurable and enforced regardless of client request.
*   **Frontier gating:** Score candidate neighbors with embeddings and (optionally) an MMR-style novelty term; **expand top‑K** only if they improve current best by `Δ≥τ`.
*   **Rate limits & budgets:** IP/user budgets for `graph` mode and `traverse_relationships`; audit logs for escalations.
*   **SLOs:** P50 < 200 ms, P95 < 500 ms for `snippet`; `full` and `graph` allowed higher but must keep P95 ≤ 1.2 s under default caps.

### 5) Providers & Jina

*   Define **provider interfaces**:
    *   `EmbeddingProvider.embed(texts: List[str], model: str) -> List[Vector]`
    *   `Reranker.rerank(query: str, docs: List[str]) -> List[int] (rank order)`
*   Implement **Jina providers** (default): `jina-embeddings-v3` and `jina-reranker-v3`. Keep **OpenAI/HF** adapters for portability.
*   **Caching:** SHA256(text + model\_version) → vector cache in Redis/SQLite; TTL+versioning.

### 6) Observability

*   **Metrics (Prometheus):**
    *   `query_verbosity_total{mode}` counter
    *   `graph_expand_nodes_total`, `graph_pruned_nodes_total`, `graph_depth_histogram`
    *   `cypher_explain_reject_total`, `cypher_timeout_total`
    *   Latency histograms per mode/provider
*   **Tracing (OTel):** spans around: embedding, rerank, Neo4j query, explain-validate, frontier loop.

### 7) Rollout & flags

*   Flags:
    `FEATURE_VERBOSITY_GRAPH` (on by default in staging)
    `FEATURE_TRAVERSAL_TOOL`
    `PROVIDER_JINA_ENABLED`
*   Canary 10% → 50% → 100% with dashboards and kill-switches.

* * *

🛠️ implementation-plan-phase-7 — _What to change, where, and how_
------------------------------------------------------------------

**Duration:** ~2–3 engineer-days (including perf harness & docs)
**All tests:** _no mocks, end‑to‑end against local stack_, as in earlier phases.

### Repo touchpoints (paths illustrative; keep your v2 layout)

```
/src/
  mcp_server/
    tools.py                 # extend search_documentation; add traverse_relationships
    routers.py               # wire new tool
    validation.py            # add EXPLAIN-plan checks + thresholds
  services/
    query_service.py         # wire verbosity modes; graph packer
    traversal/
      frontier.py            # scoring/gating loop
      bfs.py                 # bounded traversal with allow-lists
  providers/
    embeddings/base.py
    embeddings/jina.py
    embeddings/openai.py
    rerankers/base.py
    rerankers/jina.py
  neo/
    cypher_templates.py      # safe templates for DocRAG subgraph
  config/
    defaults.yaml            # new caps/flags/providers
    providers.yaml           # provider credentials & model names
/tests/e2e/
  test_verbosity_modes.py
  test_traverse_relationships.py
  test_explain_guards.py
  test_perf_phase7_latency.py
/docs/phase-7/
  feature-spec-phase-7.md
  implementation-plan-phase-7.md
  coder-guidance-phase-7.md
  pseudocode-phase-7.md
```

### Tasks

#### **T7.1 – Provider interfaces & Jina adapters**

*   **Add** `providers/embeddings/base.py`, `…/rerankers/base.py`.
*   **Implement** `embeddings/jina.py` and `rerankers/jina.py`.
*   **Config:** `config/providers.yaml`
    ```yaml
    providers:
      embedding:
        default: jina
        jina: { base_url: "...", api_key: "${JINA_API_KEY}", model: "jina-embeddings-v3", timeout_ms: 800 }
        openai: { model: "text-embedding-3-large" }
      reranker:
        default: jina
        jina: { model: "jina-reranker-v3", timeout_ms: 600 }
    ```
*   **Cache:** Redis key `embed:{model}:{sha256(text)}`.

**DoD:** Embedding & rerank smoke tests pass; cache hits observable.

#### **T7.2 – Extend `search_documentation` for `verbosity`**

*   **Update** request DTO and handlers; default `snippet`.
*   **`full` mode:** fetch full section text & metadata (title, doc\_uri, anchor).
*   **`graph` mode:** call `TraversalService.expand(evidence_ids, policy)`; add `graph` block to response.

**DoD:** E2E confirms three modes; JSON schema validated.

#### **T7.3 – New `traverse_relationships` tool**

*   **Add** MCP tool handler and schema.
*   **Impose** server caps and allow-lists even if not provided by client.
*   **Return** scored nodes, edges, and `explain` with budget usage.

**DoD:** Walk respects caps; returns deterministic output on fixed seed.

#### **T7.4 – Traversal frontier gating**

*   **Implement** `frontier.score(candidate)` using:
    *   node meta embedding (title+trail),
    *   optional chunk synopsis embedding,
    *   novelty penalty vs. collected set (MMR-ish).
*   **Expand** top‑K per layer if `improvement ≥ delta_threshold`.

**DoD:** Unit/e2e shows fewer expansions with equal/better relevance, CPU down ≥25% on hot paths.

#### **T7.5 – EXPLAIN-plan validation & timeouts**

*   **Add** `validation.explain_guard(query, params)`:
    *   Reject plans with `Expand(All)`/deep variable-length patterns beyond thresholds.
    *   Reject label scans outside `Chunk|Entity|Topic|Document` (for DocRAG queries).
    *   Enforce **driver timeout**; surface `PlanTooExpensive`.
*   **Wrap** all dynamic Cypher calls with explain-check → run-with-timeout.

**DoD:** Bad patterns are rejected; good templates pass.

#### **T7.6 – DocRAG v3 schema bridge**

*   **Dual-label** existing `Section` nodes with `:Chunk` (idempotent migration).
*   Ensure edges exist: `HAS_CHUNK`, `MENTIONS`, `HAS_TOPIC` in the _doc subgraph_.
*   Add **Session/Query/Answer** ephemeral nodes/edges for retrieval traces; TTL clean-up job.

**DoD:** Cypher templates for DocRAG-only queries produce the same top‑K as Phase 6 baseline (±5%) with better explainability.

#### **T7.7 – Observability & budgets**

*   **Counters/histograms** listed in the spec.
*   **Feature flags** + **auditable** denials (rate limit, plan reject, timeout).
*   Dashboards: “verbosity mix”, “graph depth distribution”, “explain rejects”.

**DoD:** Dashboards live; alarms configured.

#### **T7.8 – Perf harness**

*   Python harness to run 25 queries × 3 modes; report p50/p95, expansion counts, rejects, timeouts.
*   Store CSV + MD in `/reports/phase-7/`.

**DoD:** P50/P95 within targets; regressions documented if any.

### Risks & mitigations

*   **Graph blow-ups** → hard caps + frontier gating + EXPLAIN guard.
*   **Vendor outages** (Jina) → provider fallback to OpenAI/HF; circuit breaker.
*   **Token bloat (`full`)** → config max sections; truncate full\_text server-side.

### Acceptance & phase gate

*   All e2e tests pass; perf targets met or exceptions approved.
*   Reports produced in `/reports/phase-7/` with raw artifacts (CSV, JUnit).

* * *

👩‍💻 coder-guidance-phase-7 — _Do this / Pitfalls / DoD / Checklist / Notes_
-----------------------------------------------------------------------------

### A) Do this (in order)

1.  **Providers**
    *   Create `embeddings/base.py`, `embeddings/jina.py`, `rerankers/base.py`, `rerankers/jina.py`.
    *   Wire config & Redis caching; add smoke tests.
2.  **Verbosity**
    *   Extend `search_documentation` DTO & handler; implement `snippet|full|graph`.
    *   Add `providers` override in request, but always apply **server defaults** and **hard caps**.
3.  **Traversal**
    *   Add `traverse_relationships` MCP tool with server caps + allow-lists.
    *   Implement `TraversalService.expand()` using **frontier gating**. Keep **breadth** ≤ `top_k` per layer.
4.  **Perf/Safety**
    *   Wrap every dynamic Cypher with `explain_guard()` then `run_with_timeout()`.
    *   Add metrics and OTel spans; rate-limit graph features.
5.  **DocRAG bridge**
    *   Add `:Chunk` label to Sections; ensure `HAS_CHUNK`, `MENTIONS`, `HAS_TOPIC` edges exist for the doc subgraph.
    *   Introduce ephemeral `Session/Query/Answer` with TTL (cron).
6.  **Tests & Reports**
    *   Implement e2e tests for 3 modes, traversal caps, explain rejects.
    *   Run the perf harness; publish `/reports/phase-7/` artifacts.

### B) Pitfalls (avoid these)

*   Letting clients dictate `max_depth`/`max_nodes` beyond policy. **Server wins**.
*   Using **Expand(All)** wildcards or unlabeled traversals in Cypher.
*   Forgetting **novelty** → you’ll add duplicates and waste budget.
*   Returning **full text** without a server-side size cap → token explosion.
*   Skipping **rerank** on `snippet` mode; it still improves first-page quality.

### C) Definition of Done

*   `search_documentation` returns correct shapes for each mode; JSON schema validated in tests.
*   `traverse_relationships` respects caps and outputs explainability block.
*   EXPLAIN-guard demonstrably rejects heavy queries.
*   Jina provider works; fallback path tested by toggling flag.
*   DocRAG subgraph queries return evidence with strict provenance.

### D) Reviewer checklist

*   ✅ Feature flags present and default-safe.
*   ✅ All new queries pass `explain_guard()`; timeouts set.
*   ✅ Metrics/OTel spans visible; budgets/rate limits in place.
*   ✅ Tests: e2e (3 modes), traversal caps, explain reject, perf harness CSV.

### E) Notes (performance & safety)

*   Keep `MAX_DEPTH_HARD ≤ 5`, `MAX_NODES_HARD ≤ 200`.
*   Keep provider **timeouts** conservative; add circuit breakers.
*   All external responses must be **schema-validated** before use.

* * *

🔣 pseudocode-phase-7 — _Core algorithms & shapes_
--------------------------------------------------

> Syntax follows your v2 pseudocode style.

### 1) Providers

```pseudocode
interface EmbeddingProvider:
  procedure embed(texts: List[str], model: str) -> List[Vector]

class JinaEmbeddingProvider implements EmbeddingProvider:
  procedure embed(texts, model="jina-embeddings-v3"):
    out := []
    for t in texts:
      key := "embed:" + model + ":" + sha256(t)
      if cache.has(key): out.append(cache.get(key)); continue
      v := http.post(CONFIG.providers.embedding.jina.base_url, {text:t, model:model}, timeout=CONFIG.providers.embedding.jina.timeout_ms)
      cache.set(key, v, ttl=CONFIG.cache.embed_ttl_s)
      out.append(v)
    return out

interface Reranker:
  procedure rerank(query: str, docs: List[str]) -> List[int]  // returns indices in best→worst order

class JinaReranker implements Reranker:
  procedure rerank(q, docs): return http.post(...).ranking
```

### 2) EXPLAIN-plan guard

```pseudocode
procedure explain_guard(cypher: str, params: Map) -> ValidatedQuery
  plan := neo4j.EXPLAIN(cypher, params)
  if plan.has_expand_all_over(CONFIG.limits.expand_all_threshold): raise PlanTooExpensive
  if plan.estimated_rows > CONFIG.limits.estimated_rows_max: raise PlanTooExpensive
  if plan.uses_label_scans_outside(ALLOW_LABELS): raise PlanRejected
  return ValidatedQuery(query=cypher, params=params)

procedure run_safe(cypher: str, params: Map, timeout_ms: int) -> Result
  v := explain_guard(cypher, params)
  return neo4j.RUN(v.query, v.params, timeout=timeout_ms)
```

### 3) Frontier-gated traversal

```pseudocode
record FrontierPolicy:
  top_k: int = 20
  delta_threshold: float = 0.05
  novelty_penalty: float = 0.2
  max_depth_soft: int = 3
  max_depth_hard: int = 5
  max_nodes_hard: int = 200
  rel_whitelist: Set[str]
  label_whitelist: Set[str]

procedure traverse(seed_ids: List[ID], policy: FrontierPolicy, query_text: Optional[str]) -> GraphPack
  init:
    visited := Set(seed_ids)
    frontier := Queue(seed_ids.map(id -> (id, depth=0)))
    collected := []
    nodes, edges := Map(), List()
    embeddings := EmbeddingProvider.auto()  // resolve from config

  while not frontier.empty():
    (u, d) := frontier.pop()
    if d >= policy.max_depth_hard or size(nodes) >= policy.max_nodes_hard: break

    // 1) enumerate neighbors with allow-lists
    neigh := run_safe(CYPHER.NEIGHBORS_WHITELISTED, {u, rels:policy.rel_whitelist, labels:policy.label_whitelist}, timeout_ms=CONFIG.limits.cypher_timeout_ms)

    // 2) score candidates
    scored := []
    for v in neigh:
      if v.id in visited: continue
      s_node := cosine(embed(query_text), embed(v.meta))   // provider-backed
      s_novel := novelty_penalty(collected, v)             // MMR-like
      score := s_node - policy.novelty_penalty*s_novel
      if improves_over_best(score, delta=policy.delta_threshold):
        scored.append((v, score))

    // 3) choose top-K and expand
    chosen := top_k(scored, k=policy.top_k)
    for (v, s) in chosen:
      visited.add(v.id)
      nodes[v.id] := { id:v.id, label:v.label, title:v.title, score:s }
      edges.append({ src:u, dst:v.id, type:v.rel_type, depth:d+1 })
      collected.append(v)
      if d+1 < policy.max_depth_soft: frontier.push((v.id, d+1))

    if size(nodes) >= policy.max_nodes_hard: break

  return GraphPack(nodes=list(nodes.values()), edges=edges, budget={ expanded:size(nodes), depth_reached=max_depth(nodes) })
```

### 4) `search_documentation` handler (modes)

```pseudocode
procedure search_documentation(query, top_k=8, verbosity="snippet", providers=None, limits=None) -> Response
  // 0) retrieve candidates
  docs := vector_search(query, top_k=top_k*3)         // wide
  ranked := Reranker.auto().rerank(query, docs.texts) // best→worst
  evidence := take(docs, ranked, k=top_k)

  // 1) pack evidence
  out := []
  for e in evidence:
    item := { section_id:e.id, title:e.title, snippet=truncate(e.text, 200),
              metadata:{ doc_uri:e.uri, anchor:e.anchor, score:e.score } }
    if verbosity == "full":
      item.full_text := cap_length(e.full_text, CONFIG.limits.full_text_bytes)
    out.append(item)

  // 2) optional graph
  graph := null
  if verbosity == "graph":
    policy := limits.graph ⊕ CONFIG.defaults.graph_policy       // server caps override
    graph := traverse(seed_ids=evidence.map(_.id), policy=policy, query_text=query)

  // 3) diagnostics
  diag := { retrieval:{ provider:Providers.embedding.current(), reranker:Providers.rerank.current(), lat_ms:last_latency() } }

  return { answer:null, evidence:out, graph:graph, diagnostics:diag }
```

### 5) `traverse_relationships` tool handler

```pseudocode
procedure mcp.traverse_relationships(args) -> GraphPack
  req := validate(args, schema=TRAVERSE_SCHEMA)
  policy := clamp_to_server_caps(req.policy)
  return traverse(seed_ids=req.seed_ids, policy=policy, query_text=req.query or "")
```

### 6) DocRAG templates (safe)

```pseudocode
CYPHER.NEIGHBORS_WHITELISTED = """
MATCH (u) WHERE id(u) = $u
CALL {
  WITH u
  MATCH (u)-[r]->(v)
  WHERE type(r) IN $rels AND any(l in labels(v) WHERE l IN $labels)
  RETURN v, type(r) AS rel_type
  UNION
  MATCH (v)-[r]->(u)
  WHERE type(r) IN $rels AND any(l in labels(v) WHERE l IN $labels)
  RETURN v, type(r) AS rel_type
}
RETURN v, rel_type
LIMIT $k
"""
```

* * *

✅ What you’ll ship (artifacts)
------------------------------

*   `/docs/phase-7/feature-spec-phase-7.md` (above)
*   `/docs/phase-7/implementation-plan-phase-7.md` (above)
*   `/docs/phase-7/coder-guidance-phase-7.md` (above)
*   `/docs/phase-7/pseudocode-phase-7.md` (above)
*   `/tests/e2e/*.py` (three modes, traversal caps, explain rejects)
*   `/reports/phase-7/` (perf harness CSV + markdown summary)

* * *

🧪 Test matrix (excerpt)
------------------------

| Test | Mode | Expectation |
| --- | --- | --- |
| `test_snippet_returns_evidence` | snippet | 3–10 items, 200‑char snippets, scores present |
| `test_full_returns_full_text` | full | full\_text present & size‑capped, same IDs as snippet |
| `test_graph_respects_caps` | graph | nodes≤200, depth≤5, pruned>0 with frontier gating |
| `test_traverse_whitelist_enforced` | tool | only whitelisted labels/relations; deterministic on seed |
| `test_explain_rejects_bad_plan` | any | malicious/unbounded patterns rejected |
| `test_perf_p50_p95` | all | P50/P95 within targets; report written |

* * *

📦 Config (defaults)
--------------------

```yaml
limits:
  full_text_bytes: 16384
  cypher_timeout_ms: 1500
  graph:
    max_depth_soft: 3
    max_depth_hard: 5
    max_nodes_hard: 200
    top_k: 20
    delta_threshold: 0.05
    novelty_penalty: 0.2
allow_lists:
  labels: ["Chunk","Entity","Topic","Document"]
  rels:   ["MENTIONS","HAS_TOPIC","RELATED_TO"]
flags:
  FEATURE_VERBOSITY_GRAPH: true
  FEATURE_TRAVERSAL_TOOL: true
  PROVIDER_JINA_ENABLED: true
```

* * *

### Why this integrates your attached docs cleanly

*   **Enhanced responses (verbosity + traversal):** adds structured modes and a dedicated traversal tool with bounded, explainable graph output consistent with your evidence-first philosophy.
*   **Perf hardening:** EXPLAIN-plan checks, _frontier gating_ with embeddings/novelty, server caps, and rate limits match the recommended safeguards.
*   **Jina integration:** done via provider interfaces, keeping you portable while gaining stronger embeddings/rerank.
*   **DocRAG lean schema:** subgraph alignment (dual-label `Section`→`:Chunk`) preserves provenance and multi-turn session traces without polluting the ops graph.

* * *

If you’d like, I can also drop these into **four Markdown files** under `/docs/phase-7/` in your repo structure.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
