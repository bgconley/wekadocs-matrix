Thanks—this is a sharp direction. I read the three docs you attached:

*   **Feature Spec – Enhanced Response Verbosity & Graph Traversal** (v1.0, 2025‑10‑18): introduces `verbosity: snippet|full|graph`, the `traverse_relationships` tool, hard caps (depth≤3, ≤100 nodes), a relationship whitelist, and new Prometheus metrics.
*   **Implementation Plan** (2025‑10‑18): details code touchpoints (e.g., `TraversalService`, `QueryService`, `response_builder`), the relationship whitelist (`MENTIONS`, `CONTAINS_STEP`, `HAS_PARAMETER`, `REQUIRES`, `AFFECTS`, `RESOLVES`, `RELATED_TO`, `HAS_SECTION`, `EXECUTES`), latency targets (e.g., P95 `graph` <150 ms, `traverse(depth=2)` <200 ms), and risk controls (rate limits, audit logs, hard caps).
*   **Feature Summary**: confirms scope (verbosity + graph traversal), out‑of‑scope note that **entity vector search** is Phase‑4, and reiterates controls and metrics.

> If you uploaded **other** files earlier, I only see these three. If you want me to use additional docs, please re‑upload them.

Below are answers to your two questions and a set of concrete enhancements you can drop into this design.

* * *

1) Should traversal depth be auto‑adjustable by the LLM (beyond depth 3)?
-------------------------------------------------------------------------

**Short answer:** Yes—with safeguards. Let the LLM _propose_ deeper traversals, but keep the server in charge via hard budgets and a scoring gate. Think “LLM suggests, server decides.”

**Why:** A fixed depth of 2–3 is a good default for latency and safety, but technical questions vary. “What does flag X do in subcommand Y?” may need deeper hops (e.g., `CLI → subcommand → flag → parameter → example`). Conversely, “What is component Z?” doesn’t.

**How to do it safely (drop‑in policy):**

*   **Default depth:** keep your current `MAX_DEPTH = 3` _as the soft default_.
*   **Server‑enforced ceilings:** add `MAX_DEPTH_HARD = 5` and `MAX_NODES_HARD = 200`. Never exceed these regardless of what the LLM asks for.
*   **Auto‑depth policy switch:** extend the `traverse_relationships` tool schema:
    ```json
    {
      "name": "traverse_relationships",
      "params": {
        "start_ids": ["..."],
        "edge_types": ["..."],
        "mode": "fixed|auto",
        "max_depth": 3,
        "depth_budget_tokens": 2000,
        "time_budget_ms": 120,
        "frontier_k": 8
      }
    }
    ```
    *   `mode=auto` means: the LLM can request additional depth, but the server will gate expansions (see scoring below).
    *   Budgets keep latency/token usage bounded, independent of hop count.
*   **Frontier scoring + stop conditions (server‑side):**
    *   Maintain a frontier of candidate next hops. For each candidate node/edge, compute an **Expected Gain** score (details in §3).
    *   Expand only if **(a)** `gain >= gain_threshold`, **(b)** **∆gain vs previous hop ≥ delta\_threshold** (diminishing returns), and **(c)** you are within **token/time budgets**.
    *   Stop when any stop condition triggers: budget exhaustion, no frontier above threshold, or hitting `MAX_DEPTH_HARD/MAX_NODES_HARD`.
*   **Observability you already planned** can be extended to this:
    *   `mcp_traverse_auto_depth_requested_total`
    *   `mcp_traverse_auto_depth_realized_total` (actual max depth reached)
    *   `mcp_traverse_frontier_pruned_total`
    *   Distributions: gain scores by depth, expansions per edge type, abort reasons.

**Result:** The model gets flexibility when it needs it, and you keep latency, cost, and safety under strict control.

* * *

2) Should we “vectorize” deeper nodes/sections to decide whether to keep traversing?
------------------------------------------------------------------------------------

**Yes—this makes sense, with a hierarchical embedding strategy.** Use vector signals to _route_ traversal and predict yield before you pay for deeper hops.

**Key points:**

*   **Embed at multiple granularities** to avoid very long‑context embedding pitfalls:
    1.  **Section synopsis** (300–600 tokens): generated or extracted summary per section.
    2.  **Chunk‑level windows** (e.g., 300–800 tokens) for precise matching.
    3.  **Node meta embedding**: title, headers, path, and a few salient keyphrases.
    4.  (Optional Phase‑4) **Edge textualization**: a compact natural language gloss of the relation (e.g., “Section A explains parameter `--foo` used by Section B’s example”), with an embedding per edge.
*   **Score the frontier with vectors before expanding**:
    *   For each candidate neighbor, compute `s_node = cos(query, section_synopsis_embed)` and `s_meta = cos(query, node_meta_embed)`.
    *   If you textualize edges, also compute `s_edge = cos(query, edge_gloss_embed)`.
    *   Combine (see §3 scoring) and expand the top‑K only if the score clears the threshold and improves on the current best by `∆ ≥ delta_threshold`.
*   **Add a _novelty_ term** (anti‑redundancy): penalize candidates that are semantically near content you already gathered for this query (e.g., MMR or a simple cosine penalty vs. collected snippets).
*   **Rerank with a cross‑encoder** before final inclusion: vectors for routing, cross‑encoder for accuracy (especially in technical docs).

This lets the LLM ask “should I go deeper?” and the server answer with data: “yes, the next hop is semantically promising (score 0.78), within budget; proceeding one level deeper.”

* * *

3) A concrete, low‑risk controller you can implement now
--------------------------------------------------------

**Frontier Expected‑Gain scoring (server‑side):**

```
gain(candidate) =
  α * s_node
+ β * s_meta
+ γ * s_edge
+ η * novelty_bonus
+ ζ * edge_type_prior
- κ * cost_penalty
```

*   `s_node`: cosine(query, section synopsis)
*   `s_meta`: cosine(query, node meta)
*   `s_edge`: cosine(query, edge gloss) (if available)
*   `novelty_bonus`: +MMR bonus vs. already-collected context
*   `edge_type_prior`: small prior by relation type (e.g., `HAS_PARAMETER`, `CONTAINS_STEP` > `MENTIONS`)
*   `cost_penalty`: expected token/time cost of expanding this branch (based on fan‑out, depth remaining, and chunk sizes)

**Defaults (tune later):** α=.45, β=.25, γ=.15, η=.10, ζ=.05, κ=.10.
**Thresholds:** `gain_threshold=.55`, `delta_threshold=.05`.
**Search discipline:** best‑first over the frontier with `frontier_k` (e.g., 8–16).
**Budgets:** enforce `time_budget_ms`, `depth_budget_tokens`, and `MAX_*_HARD`.

* * *

4) Suggested schema and code touchpoints (fits your plan)
---------------------------------------------------------

*   **Tools**
    *   `search_documentation(query, verbosity=snippet|full|graph, ...)` (as in your spec)
    *   `traverse_relationships(start_ids, edge_types, mode=fixed|auto, max_depth, depth_budget_tokens, time_budget_ms, frontier_k)`
    *   _(Optional, Phase‑4)_ `score_frontier(nodes_or_edges)` to let an LLM ask “which way is promising?” without doing the traversal itself.
*   **Where to put things (matching your tree)**
    *   `src/query/traversal.py`: add the **Controller** (`FrontierManager`, `ExpansionPolicy`) and scoring utilities.
    *   `src/query/embeddings.py`: hierarchical embeddings + caching (synopsis, meta, edge gloss).
    *   `src/shared/observability/metrics.py`: new counters/histograms above.
    *   `src/mcp_server/tools.py`: accept `mode=auto`, budgets, and expose reasons for stopping (helps the LLM plan next steps).
*   **Hard safety remains** (from your spec/plan):
    *   Relationship whitelist (keep yours).
    *   Rate limits, query timeouts, audit logging.
    *   `MAX_DEPTH_HARD`, `MAX_NODES_HARD` (server‑enforced).

* * *

5) How this lines up with your current docs
-------------------------------------------

*   You already have **hard caps at depth=3** and **≤100 nodes** with a whitelist and latency targets. Keep those as **defaults**; add `mode=auto` so depth is _negotiated but bounded_ by budgets & hard caps.
*   The **Feature Summary** calls entity‑level vector search _out of scope_. The proposal here keeps vectors in scope by using **section/edge embeddings only to _route traversal_** (no entity index required), so you can ship it with the current release and optionally widen to entity vectors in Phase‑4.
*   Your **observability** section is a perfect base—add the auto‑depth counters and a histogram for **gain scores by depth** to debug behavior and regressions.

* * *

6) Practical details & pitfalls
-------------------------------

*   **Embedding length:** Don’t embed “entire sections” verbatim if they’re long—create a **synopsis** (extractive or short LLM summary) and embed that; keep chunk embeddings for precision later.
*   **Edge glossing:** If your KG comes from structure (TOC, includes, parameter references), you can auto‑generate a 1–2 sentence gloss per edge at index time; cache its embedding.
*   **Hybrid ranker:** Use BM25/lexical as a parallel signal for frontier scoring (cheap, high precision on tecky tokens like flags, error codes).
*   **Return explainability:** Include a `trace` in the tool response showing `(depth, edge_type, gain)` for each hop you took. The LLM can cite _why_ it went deeper without exposing internal chain‑of‑thought.
*   **Latency:** Cap `frontier_k` and tune thresholds first; add an adaptive `gain_threshold` when time budget is nearly exhausted (be stricter late in the search).
*   **Hallucination control:** Heavily down‑weight paths when **no text** exists at the candidate node (graph‑only nodes). Don’t expand into textual deserts unless mandated by the LLM and still within budgets.

* * *

7) Minimal pseudo‑code (server side)
------------------------------------

```python
def auto_traverse(query, start_ids, edge_types, budgets):
    frontier = seed_frontier(start_ids, edge_types)
    collected, visited = [], set()
    start = now()
    best_gain = 0.0

    while frontier and within_budgets(budgets, start, collected):
        cand = frontier.pop_best()  # based on gain()
        if cand.id in visited:
            continue
        visited.add(cand.id)

        g = gain(query, cand)  # α s_node + β s_meta + γ s_edge + η novelty + ζ prior - κ cost
        if g < budgets.gain_threshold or (g - best_gain) < budgets.delta_threshold:
            continue

        materialize(cand)           # read + cache text, create snippets/chunks
        collected.append(cand)
        best_gain = max(best_gain, g)

        if cand.depth < budgets.max_depth_soft:
            children = expand(cand, edge_types, limit=budgets.frontier_k)
            for ch in children:
                ch.gain = gain(query, ch)
                frontier.push(ch)

        if exceeded_hard_caps(budgets, visited, collected):
            break

    return to_traversal_result(collected, trace=True)
```

* * *

8) Answering your questions explicitly
--------------------------------------

1.  **Auto‑adjustable traversal depth?**
    **Yes**, but keep the server in control:
    *   Add `mode=auto`, budgets, and scoring‑based expansion.
    *   Keep hard ceilings (`MAX_DEPTH_HARD`/`MAX_NODES_HARD`).
    *   Log decisions and stop reasons for debuggability.
2.  **Vectorize deeper nodes to decide whether to continue?**
    **Yes**, with a hierarchical approach:
    *   Embed **section synopses**, **chunk windows**, and optional **edge glosses**.
    *   Use these embeddings to **score the frontier** and expand only when useful.
    *   Rerank final candidates with a cross‑encoder before sending to the LLM.

* * *

9) A few “nice‑to‑have next” ideas
----------------------------------

*   **Query intent router** (definition vs. how‑to vs. error triage) to set priors on edge types and depth budgets up front.
*   **Per‑relation fan‑out caps** (e.g., `MENTIONS` expands at most 3 per hop; `HAS_PARAMETER` up to 10).
*   **Learning‑to‑stop:** offline fit a shallow model to predict when an extra hop improves answer F1; use its probability as `gain`.
*   **Cold‑start cache:** persist the top N high‑centrality nodes’ synopses+embeddings in memory; they’re common entry points (e.g., “installation”, “authentication”, “CLI reference”).
*   **De‑dup windows** via locality‑sensitive hashing to keep the context fresh and avoid repetitive snippets.

* * *



## Prompt:
Fantastic—here are **three complete, drop‑in docs** that merge your original “Enhanced Responses” idea with the improvements we discussed (dynamic auto‑depth; vector‑guided traversal via frontier scoring; hierarchical embeddings; traceability, budgets, and hard caps). I aligned structure, naming, and guardrails to your canonical v2 stack so an **agentic coder** can implement without cross‑referencing. Where I lean on existing app behaviors (templates‑first, validator+EXPLAIN, data model, SLOs, interfaces), I cite your v2 docs inline.

> **Heads‑up about sources:** I can’t open your earlier “enhanced‑responses” trio (feature spec, implementation plan, summary)—they appear expired on my side. I rebuilt them from your **v2 canonical** spec/plan/guidance/pseudocode and your **Phase 6 addenda**; if you’d like me to reconcile against those exact v1/v1.0 files, please re‑upload them.
>
> spec

* * *

1) `/docs/feature-spec-enhanced-responses.md` — **Enhanced Responses (GraphRAG + Dynamic Depth + Vector‑Guided Traversal)**
---------------------------------------------------------------------------------------------------------------------------

**Status:** Proposed v2 extension (supersedes prior “Enhanced Responses”).
**Owners:** Retrieval (primary), Graph Eng, Platform/Observability.
**Compatibility:** Backwards‑compatible with v2 tools and response schema.
**Inherits:** Templates‑first NL→Cypher and validator with EXPLAIN plan gates; hybrid retrieval baseline; Section‑level evidence and JSON output; SLOs.

spec

spec

spec

### 0) Purpose & scope

Provide **richer, safer answers** to technical questions by allowing the LLM client (via MCP) to **propose deeper graph traversals** when needed, while the server **keeps control** using budgets, **frontier scoring**, and hard caps. Traversal is **vector‑guided** (use embeddings to predict yield before expanding), with **explainable traces** and unchanged security posture (templates‑first + validator). Baseline hybrid retrieval remains: vector seed → bounded typed expansion → optional bridge paths → rank/return.

spec

### 1) What ships

1.  **Adjustable traversal depth**
    *   `mode=fixed|auto` on `traverse_relationships`, with **server‑enforced** ceilings (`MAX_DEPTH_HARD`, `MAX_NODES_HARD`) and **time/token budgets**.
*   Stop conditions: gain threshold, diminishing‑returns threshold, budget exhaustion, hard caps.
    2.  **Vector‑guided frontier scoring**
    *   Rank next‑hop candidates with a **Expected‑Gain** function that blends query‑to‑section synopsis similarity, node meta similarity, optional edge‑gloss similarity, novelty (MMR), relation priors, and expansion cost penalties.
*   Expand only if predicted gain justifies the cost.
    3.  **Hierarchical embeddings**
    *   **Section synopsis** embeddings (300–600 tokens) for routing;
*   **Chunk** embeddings (300–800 tokens) for precision;
*   Optional **edge gloss** embeddings (1–2 lines) to describe relations (e.g., “Section A explains `--foo` used by Section B’s example”).
    4.  **Explainable traces**
    *   Each traversal returns a `trace[]` with `(depth, node_id, edge_type, gain, stop_reason)` plus the **relationship path** you already expose; answers keep **Section‑level evidence** in JSON (`evidence[]`) as in v2.
    spec
    5.  **Stricter guardrails preserved**
    *   Still **templates‑first** NL→Cypher with **validator + EXPLAIN** plan checks; typed edges only; early LIMITs and timeouts; per‑client rate limits; audit.
    spec

### 2) User stories

*   _Deep flag behavior_: “What does `--consistency-check` do under `wkctl backup`? Show the sequence and caveats.” → Might need `HAS_PARAMETER/CONTAINS_STEP/REQUIRES` two–four hops deep.
*   _Impact/side effects_: “If I change `max_shards`, what else does it affect?” → Prioritize `AFFECTS/DEPENDS_ON` edges with deeper hops if vectors predict relevant content.

### 3) Interfaces

#### 3.1 MCP tools (additions)

*   **`traverse_relationships` (updated)** – server decides expansions
    ```json
    {
      "name": "traverse_relationships",
      "params": {
        "start_ids": ["sec:..."],
        "edge_types": ["MENTIONS","CONTAINS_STEP","HAS_PARAMETER","REQUIRES","AFFECTS","RESOLVES","RELATED_TO","EXECUTES"],
        "mode": "fixed|auto",
        "max_depth": 3,
        "depth_budget_tokens": 2000,
        "time_budget_ms": 120,
        "frontier_k": 8,
        "include_trace": true
      }
    }
    ```
    _Notes:_ Edge types come from your **derived‑edge whitelist**; keep typed expansions only.
    spec
*   **`search_documentation` (unchanged wire, smarter server)** – continues to do vector‑seed → bounded hop expansion; server may internally use the same **frontier scoring** to pick expansions, but **returns the same response schema** (Markdown + JSON with evidence + diagnostics).
    spec
    spec
*   **`explain_ranking` (existing utility)** – extend `diagnostics.ranking_features` to include top `trace.gain` reasons when `mode=auto`.
    spec

#### 3.2 Response schema (incremental)

`answer_json` adds non‑intrusive fields:

```json
{
  "answer": "...",
  "evidence": [{"section_id":"sec:...","path":"Doc > Section > ..."}],
  "confidence": 0.78,
  "diagnostics": {
    "ranking_features": {"semantic":0.64,"graph":0.22,"recency":0.14},
    "trace": [
      {"depth":1,"node_id":"sec:...","via":"HAS_PARAMETER","gain":0.71},
      {"depth":2,"node_id":"step:...","via":"CONTAINS_STEP","gain":0.63,"stop_reason":"delta_below_threshold"}
    ],
    "budgets":{"time_ms":118,"tokens_used":930}
  }
}
```

(Compatible with your v2 **dual output** contract.)

spec

### 4) Retrieval controller & scoring

**Frontier Expected‑Gain** (server‑side):

```
gain(c) = α*s_node + β*s_meta + γ*s_edge + η*novelty - κ*cost + ζ*edge_type_prior
```

*   `s_node`: cosine(query, section synopsis)
*   `s_meta`: cosine(query, node meta: title+headers+keyphrases)
*   `s_edge`: cosine(query, edge gloss) (if present)
*   `novelty`: MMR bonus vs already‑collected evidence
*   `cost`: predicted expansion cost (fan‑out × size × depth remaining)
*   `edge_type_prior`: small prior favoring semantically “tight” edges like `HAS_PARAMETER`, `CONTAINS_STEP`, `REQUIRES` over `MENTIONS`.

**Default weights:** α=.45, β=.25, γ=.15, η=.10, κ=.10, ζ=.05 (configurable).
**Stop conditions:** `gain < threshold` (e.g., .55), or `(gain - best_gain) < delta_threshold` (e.g., .05), or budgets/hard caps exceeded.

**Safety & planning:** This runs **after** the v2 template planner and **inside** the validator/limits envelope—no raw model‑driven Cypher escapes guardrails.

spec

### 5) Data & embedding changes

*   **New per‑Section synopsis** (short summary) + embedding.
*   **Existing Section embedding** stays for coarse search; keep as primary vector SoT (Qdrant or Neo4j vectors per config).
    pseudocode-reference
*   **Optional per‑edge gloss** (textual 1–2 sentences) + embedding cached at ingest or lazy‑built on first reference.
*   **Chunk windows** for final rerank/snippet selection—unchanged storage pattern.
*   **Reconciliation** still monitors graph ↔ vector parity by `embedding_version` (<0.5% drift target).
    pseudocode-reference

### 6) Limits, SLOs, and guardrails

*   **Hard caps:** `MAX_DEPTH_HARD=5`, `MAX_NODES_HARD=200`.
*   **Budgets:** `time_budget_ms` (per request), `depth_budget_tokens`.
*   **Typed edges only** from your whitelist and derived edges with provenance.
    spec
*   **SLOs (inherited):** P50 < 200 ms, P95 < 500 ms end‑to‑end for common queries (warmed). New traversal work must stay within the same SLO envelope.
    spec

### 7) Observability

New metrics (Prometheus), complementing existing latency, vector, and reconciliation metrics:

*   `mcp_traverse_auto_depth_requested_total`, `mcp_traverse_auto_depth_realized_total`
*   `mcp_traverse_frontier_size{depth}` histogram
*   `mcp_traverse_gain_histogram{depth}`
*   `mcp_traverse_stop_reason_total{reason}`
*   `synopsis_embed_build_seconds`, `edge_gloss_embed_build_seconds`
    (Integrates with your OTel/Prom stack).
    pseudocode-reference

### 8) Security

No changes: JWT, rate limiting (token bucket), **parameterized Cypher only**, **EXPLAIN** plan gating, audit logging with correlation IDs.

spec

### 9) Acceptance criteria

*   Auto‑depth expands only when gain≥threshold **and** within budgets; stop reasons are traceable.
*   JSON includes `diagnostics.trace` and budgets when `mode=auto`.
*   P95 latency consistently ≤ 500 ms on a warmed 3–5 hop query set with realistic docs.
    spec
*   Reconciliation drift stays ≤ 0.5% after enabling synopsis/edge embeddings.
    pseudocode-reference

* * *

2) `/docs/implementation-plan-enhanced-responses.md` — **Implementation Plan**
------------------------------------------------------------------------------

**Status:** Ready to execute. Uses your v2 repo layout, phase/task conventions, and **no‑mocks** gatekeeping.

implementation-plan

spec

### Workstreams & tasks

#### A) Retrieval Controller & Frontier Scoring (Owner: Retrieval)

*   **Deliverables**
    *   `src/query/traversal_controller.py` — `FrontierManager`, `ExpansionPolicy`, `gain()`
    *   `src/query/frontier_scoring.py` — cosine/MMR/priors/cost penalties
    *   `src/query/config.yaml` additions: weights, thresholds, budgets, caps
*   **Steps**
    1.  Implement `FrontierManager` (best‑first queue; `frontier_k` cap).
    2.  `gain()` = α_s\_node + β_s\_meta + γ_s\_edge + η_novelty − κ_cost + ζ_prior.
    3.  Stop conditions + traces; integrate with existing **hybrid retrieval** loop.
        pseudocode-reference
*   **DoD**: Gains computed for all expanded nodes; traces emitted; stop reasons accurate; expansions stay within budgets.

#### B) Embeddings & Ingestion Extensions (Owner: Ingestion)

*   **Deliverables**
    *   `src/ingestion/synopsis.py` (generate short per‑Section synopsis; cache/emit),
    *   `src/ingestion/edge_gloss.py` (optional derived‑edge gloss),
    *   Vector upsert: `synopsis_sections`, `edge_glosses` collections if Qdrant SoT; if Neo4j vector SoT, new properties with versioned prefixes.
        pseudocode-reference
*   **Steps**
    1.  Add synopsis generator (extractive first; allow LLM‑summary behind a flag).
    2.  Add optional edge gloss builder for derived edges from provenance.
        spec
    3.  Reuse **embedding\_version**; update reconciliation to include new stores; alert at drift >0.5%.
        pseudocode-reference
*   **DoD**: Fresh ingest creates synopses & (optional) glosses; nightly reconciliation sees parity for all enabled stores.

#### C) MCP Tools & Response Builder (Owner: Backend)

*   **Deliverables**
    *   `src/mcp_server/tools/traverse_relationships.py` (new `mode`, budgets, trace) — start from your existing traversal tool scaffolding.
        pseudocode-reference
    *   `src/query/response_builder.py` — enrich `diagnostics.trace` and budgets (non‑breaking).
        pseudocode-reference
*   **Steps**
    1.  Extend tool schema: `mode=fixed|auto`, `depth_budget_tokens`, `time_budget_ms`, `frontier_k`, `include_trace`.
    2.  Thread budgets through controller; add `trace[]` to JSON when requested.
    3.  Keep **templates‑first** planner and **validator** flow unchanged.
        spec
*   **DoD**: Backwards‑compatible tool contracts; clients that don’t set `mode=auto` see no change.

#### D) Observability (Owner: Observability/Platform)

*   **Deliverables**
    *   Prometheus counters/histograms (prefix `mcp_traverse_*`), OTel spans on expansions; Grafana panels for **gain** and **stop reasons**.
*   **Steps**
    1.  Add metrics + exemplars; propagate `correlation_id` in traces (you already do this).
    2.  Dashboards/alerts (e.g., low gain but high expansions → regression).
        pseudocode-reference
*   **DoD**: Panels show frontier size/gain distributions; alerts wired.

#### E) Config & Flags (Owner: Platform)

*   **Deliverables**: `config/search.yaml` new keys
    ```yaml
    traversal:
      mode_default: fixed
      max_depth_hard: 5
      max_nodes_hard: 200
      frontier_k: 8
      time_budget_ms: 120
      depth_budget_tokens: 2000
      thresholds:
        gain: 0.55
        delta: 0.05
      weights:
        s_node: 0.45
        s_meta: 0.25
        s_edge: 0.15
        novelty: 0.10
        cost: 0.10
        prior: 0.05
      edge_type_priors:
        HAS_PARAMETER: 0.05
        CONTAINS_STEP: 0.05
        REQUIRES: 0.03
        AFFECTS: 0.02
        MENTIONS: -0.03
    ```

#### F) Tests (NO MOCKS) & Gates

*   **Perf**: Warm caches; 100 real prompts spanning shallow/deep intents; assert end‑to‑end P95 ≤ 500 ms.
    spec
*   **Safety**: Negative queries must be blocked by validator (`EXPLAIN` plan checks, param enforcement, depth caps).
    spec
*   **Correctness**: For a labeled set: enabling `mode=auto` yields higher recall at constant precision vs fixed depth=2 (e.g., NDCG lift).
*   **Drift**: After enabling synopsis/edge vectors, parity by `embedding_version` holds (≤0.5% miss).
    pseudocode-reference

**Artifacts:** `/reports/phase-2/…` (retrieval), `/reports/phase-3/…` (ingestion), extended with an **Enhanced Responses** report bundle per run as you do in v2.

implementation-plan

### Rollout & migration

*   **Feature flags**: `traversal.auto_depth` (default off), `embed.synopsis` (default on), `embed.edge_gloss` (default off).
*   **Zero‑downtime**: backfill synopses lazily; compute on first access + cache; nightly job can precompute.
*   **Canary**: enable for 5% of traffic; monitor gain/stop patterns and P95; roll forward if steady. (Follow your CI/CD + canary pattern.)
    pseudocode-reference

### Risks & mitigations

*   **Over‑expansion** → enforce budgets + priors; raise thresholds under time pressure.
*   **Embedding overhead** → summarize once; cache; stagger backfills.
*   **Explainability debt** → always return `trace` when `mode=auto`; keep JSON small (cap trace length).
*   **Security** → no raw model‑driven Cypher; validator remains the gate.
    spec

### Reference pseudocode (server‑side)

```pseudocode
procedure auto_traverse(query, starts, edge_types, budgets):
  frontier := seed_frontier(starts, edge_types, k=budgets.frontier_k)
  collected := []; visited := set(); best_gain := 0.0; tic := now()

  while frontier.not_empty() and within_budgets(budgets, tic, collected):
    cand := frontier.pop_best()          // by gain
    if cand.id in visited: continue
    g := gain(query, cand)               // α*s_node + β*s_meta + γ*s_edge + η*novelty - κ*cost + ζ*prior
    if g < budgets.gain_threshold or (g - best_gain) < budgets.delta_threshold: continue

    materialize(cand)                    // read + cache text/snippet
    collected.append(cand); visited.add(cand.id); best_gain := max(best_gain, g)

    if cand.depth < budgets.max_depth_soft:
      for child in expand_typed(cand, edge_types, limit=budgets.frontier_k):
        child.gain := gain(query, child); frontier.push(child)

    if exceeded_hard_caps(visited, collected, budgets): break

  return build_traversal_result(collected, trace=true, budgets)
```

(Works within your planner/validator/response pipeline; see hybrid retrieval baseline.)

pseudocode-reference

* * *

3) `/docs/feature-summary-enhanced-responses.md` — **One‑Pager**
----------------------------------------------------------------

**What & why**
Enhanced Responses allow the MCP server to **go deeper when it matters**—controlled, explainable, and still fast. We add **auto‑depth traversal** with **vector‑guided frontier scoring** so the system can find the exact flag/step/impact you asked for, without blowing up latency. It preserves our security/validator posture and the user‑facing response format (Markdown + JSON with evidence & confidence).

spec

**How it works**

*   LLM sets `mode=auto` on `traverse_relationships`; server evaluates next hops using **Expected‑Gain**, considering **semantic similarity to section synopses**, **node meta**, optional **edge gloss**, **novelty**, **edge‑type priors**, and **cost**.
*   Expand while **gain is high** and **budgets permit**; stop otherwise; return a `trace` explaining choices.

**APIs & data**

*   MCP tool `traverse_relationships`: `mode`, `frontier_k`, budgets, `include_trace`.
*   New embeddings: **Section synopsis**; optional **edge gloss**; existing Section embeddings unchanged.
*   Typed edges only, using the v2 derived‑edge set with provenance.
    spec

**SLOs & safety**

*   Maintain P95 ≤ 500 ms (warmed); hard caps: depth ≤ 5, nodes ≤ 200; validator + **EXPLAIN** plan checks stay mandatory.
    spec
    spec

**Observability**

*   New metrics: auto‑depth requested/realized, frontier size/gain, stop‑reasons; OTel spans across expansions.
    pseudocode-reference

**Rollout**

*   Flags: `traversal.auto_depth`, `embed.synopsis`, `embed.edge_gloss`; lazy backfill; 5% canary then ramp.
    pseudocode-reference

**Success**

*   Higher recall on deep questions at equal or better precision; evidence density unchanged; zero uptick in unsafe plans; drift ≤ 0.5%.
    pseudocode-reference

* * *

### Source notes & alignment

*   **Data model & derived edges** are unchanged and remain provenance‑first: `REQUIRES`, `AFFECTS`, `RESOLVES`, `CONTAINS_STEP`, `EXECUTES`, `RELATED_TO`, `HAS_PARAMETER`, etc.
    spec
*   **Hybrid retrieval baseline** (vector seed → 1–2 typed hops → bridges → rank) is preserved; enhanced traversal plugs into that loop.
    pseudocode-reference
*   **Planner/validator** discipline (templates‑first, EXPLAIN plan checks, parameterization) remains your guardrail.
    spec
*   **SLOs** continue to target P50/P95/P99 as in v2.
    spec
*   **Phase terminology:** I didn’t tie this to specific phases per your instruction; the plan references your repo layout and gate discipline so you can refactor across phases cleanly.
    spec

* * *



I reviewed the three originals you just uploaded and **merged their concrete targets, limits, and observability details** into the enhanced versions I wrote earlier. Below are the **fully updated** (copy‑ready) documents—**feature spec**, **implementation plan**, and **feature summary**—now aligned with your earlier “Enhanced Responses” trio (verbosity modes, depth‑3 defaults, 64 KB total size cap, specific P95 targets, metric names) and still integrating the new **auto‑depth + vector‑guided frontier** capabilities and guardrails we discussed. Where I cite existing norms (templates‑first + validator/EXPLAIN, no‑mocks gates, SLO envelope, ingestion parity/drift), I reference your canonical docs inline.

> **What I pulled in from your originals:**
> • Exact P95 targets by mode (snippet 70 ms; full < 100 ms; graph < 150 ms; traverse depth=2 < 200 ms) and the global P95 < 500 ms guardrail.
>
> FEATURE\_SUMMARY\_enhanced-respon…
>
>
> • Response size policies (≈ 32 KB per full section, **64 KB** total).
>
> FEATURE\_SUMMARY\_enhanced-respon…
>
>
> • Depth defaults and whitelist enforcement for `traverse_relationships` (hard cap **3** in fixed mode; whitelist & max nodes = 100).
>
> FEATURE\_SUMMARY\_enhanced-respon…
>
>
> • Concrete Prometheus metric names and example instrumentation blocks.
>
> implementation-plan-enhanced-re…
>
> implementation-plan-enhanced-re…
>
>
> • Test matrix & rollout steps (staging canary, E2E via `/mcp/tools/call`).
>
> implementation-plan-enhanced-re…

I also preserved the **templates‑first, validator + EXPLAIN** discipline and your **no‑mocks** testing and gate conventions.

expert-coder-guidance

spec

implementation-plan

* * *

1) `/docs/feature-spec-enhanced-responses.md` — **Enhanced Responses (Auto‑Depth + Graph‑Vector Frontier)**
===========================================================================================================

**Version:** v3.1 (merged) • **Status:** Ready for implementation
**Owners:** Retrieval (primary), Graph Eng, Platform/SRE
**Compat:** Backwards‑compatible with existing tools & response schema

0) Purpose
----------

Enable **complete, evidence‑rich answers** for technical docs by adding:

1.  **Verbosity controls** to `search_documentation` (`snippet | full | graph`) so the LLM can ask for more or less context without changing tools.
    FEATURE\_SUMMARY\_enhanced-respon…
2.  A **bounded** `traverse_relationships` tool, PLUS a server‑controlled **auto‑depth** mode that uses **vector‑guided frontier scoring** to go deeper **only** when it’s likely to pay off.
3.  **Explainable traces**, strict **budgets & hard caps**, and expanded **observability**—while preserving **templates‑first NL→Cypher, validator, EXPLAIN gating**.
    expert-coder-guidance

1) Scope / Non‑goals
--------------------

*   **In scope:** verbosity modes; fixed‑depth traversal; **auto‑depth traversal with frontier scoring**; synopsis & optional edge‑gloss embeddings; traces; metrics.
    feature-spec-enhanced-responses
*   **Non‑goals:** raw LLM‑generated Cypher; entity‑vector index beyond Sections in this release (can be added later).
    FEATURE\_SUMMARY\_enhanced-respon…

2) User value
-------------

*   Get **full section text** (not just 200‑char snippets) when needed.
    FEATURE\_SUMMARY\_enhanced-respon…
*   Follow **dependencies/parameters** across 2–5 hops when it’s **useful** and **within budget** (e.g., Step → Flag → Side‑effect → Remediation).

3) Interfaces (MCP tools)
-------------------------

### 3.1 `search_documentation` (add `verbosity`, optional `include_trace`)

```json
{
  "name": "search_documentation",
  "params": {
    "query": "string",
    "filters": { "doc_tags": ["..."], "labels": ["Section","Command","Configuration"] },
    "verbosity": "snippet|full|graph",
    "max_results": 20,
    "max_context_tokens": 3000,
    "include_trace": false
  }
}
```

*   **Default** `verbosity="snippet"` preserves current behavior.
    FEATURE\_SUMMARY\_enhanced-respon…
*   **Graph mode** folds in related entities/sections at distance ≤ 1, limited to **20 related items / Section**.
    FEATURE\_SUMMARY\_enhanced-respon…
*   **Size limits:** up to ~**32 KB** per `full` section payload, **64 KB** total response cap (enforced).
    FEATURE\_SUMMARY\_enhanced-respon…

### 3.2 `traverse_relationships` (fixed **and** auto‑depth)

```json
{
  "name": "traverse_relationships",
  "params": {
    "start_ids": ["sec:..."],
    "edge_types": ["MENTIONS","CONTAINS_STEP","HAS_PARAMETER","REQUIRES","AFFECTS","RESOLVES","RELATED_TO","DEPENDS_ON","EXECUTES","HAS_SECTION"],
    "mode": "fixed|auto",
    "max_depth": 3,                 // fixed-mode default & soft cap
    "max_nodes_hard": 100,          // fixed-mode hard cap
    "max_depth_hard": 5,            // auto-mode hard ceiling (server-enforced)
    "time_budget_ms": 150,
    "depth_budget_tokens": 2000,
    "frontier_k": 12,
    "gain_threshold": 0.55,
    "delta_threshold": 0.05,
    "include_text": true,
    "return": "nodes|nodes+paths",
    "include_trace": true
  }
}
```

*   **Fixed mode:** respects **depth≤3** and **≤100 nodes** from your original spec.
    FEATURE\_SUMMARY\_enhanced-respon…
*   **Auto mode:** allows going beyond depth 3 **only** if frontier scores justify it and budgets permit; **server** always enforces ceiling (depth≤**5**).
*   **Response:** `nodes`, `relationships`, `paths`; optional `full_text` for traversed Sections; `trace` of hops with stop reasons.
    feature-spec-enhanced-responses

4) Retrieval policy: vector‑guided auto‑traversal
-------------------------------------------------

**Expected‑Gain (EG) score** for candidate `c`:

```
EG(c) = α*s_node + β*s_meta + γ*s_edge + η*novelty - κ*cost + ζ*rel_prior
```

*   `s_node`: cosine(query, **Section.synopsis\_embedding**)
*   `s_meta`: cosine(query, node‑meta embedding (title+headers+keyphrases))
*   `s_edge`: cosine(query, **edge\_gloss\_embedding**) (optional)
*   `novelty`: MMR vs. collected context
*   `cost`: predicted time/tokens (fan‑out × depth remaining)
*   `rel_prior`: priors favoring `HAS_PARAMETER`, `CONTAINS_STEP`, `REQUIRES` over `MENTIONS`
    **Default weights:** α=.45, β=.25, γ=.15, η=.10, κ=.10, ζ=.05 (config‑driven)

**Discipline:** best‑first expansion of **frontier\_k** candidates; expand only if `EG≥gain_threshold` **and** `(EG - best) ≥ delta_threshold`; stop on **budgets** or **hard caps**.

5) Data model updates
---------------------

*   **Section.synopsis** (300–600 tokens) + **synopsis\_embedding** for routing—not full text.
*   **Edge.gloss** (1–2 lines) + **gloss\_embedding** for traversable relations (optional).
*   Vector **indexes** configured via existing vector SoT settings—no hard‑coded dims.
    implementation-plan-phase-6
*   Preserve `embedding_version` and extend reconciliation drift checks (≤ 0.5%).
    implementation-plan

6) Observability
----------------

**Metrics (existing + new):**

*   Existing counters/histos for verbosity & size: `mcp_search_verbosity_total`, `mcp_search_response_size_bytes`.
    implementation-plan-enhanced-re…
*   Existing traversal basics: `mcp_traverse_depth_total`, `mcp_traverse_nodes_found`.
    implementation-plan-enhanced-re…
*   **New (auto‑depth):**
    *   `mcp_traverse_auto_requests_total`
    *   `mcp_traverse_auto_depth_reached{le=…}` (histogram)
    *   `mcp_traverse_frontier_candidates_total{rel_type}`
    *   `mcp_traverse_pruned_total{reason="budget|threshold|cap"}`
    *   `mcp_traverse_stop_reason_total{reason}`

OTel spans around frontier scoring and expansions. Production dashboards show **depth reached**, **stop reasons**, **frontier size/gain**.

7) SLOs & performance targets
-----------------------------

*   Per original targets: snippet **P95=70 ms** (baseline), full **<100 ms**, graph **<150 ms**, traverse (depth=2) **<200 ms**—all well under global **P95 < 500 ms** envelope.
    FEATURE\_SUMMARY\_enhanced-respon…
    spec
*   Auto‑depth must stay within the same global envelope; budgets/thresholds are tuned to do so.

8) Safety
---------

*   **Templates‑first** NL→Cypher; **validator** enforces parameterization, depth/expansion limits, and **EXPLAIN** plan checks.
    expert-coder-guidance
*   Rate limits, audit logging; hard caps cannot be overridden by clients.
    spec

9) Response contract
--------------------

Dual output remains: **Markdown** + **JSON** `{answer, evidence[{section_id,path}], confidence, diagnostics{ranking_features}, trace[]}`. Evidence & confidence remain mandatory.

spec

10) Acceptance criteria
-----------------------

*   Fixed mode matches existing behavior & caps; auto‑depth expands only when `EG` clears thresholds **and** budgets.
*   Traces include `(depth, via, EG, stop_reason)` for each hop; **no chain‑of‑thought**.
*   P95 targets met per mode; global P95 guardrail holds.
    FEATURE\_SUMMARY\_enhanced-respon…

* * *

2) `/docs/implementation-plan-enhanced-responses.md` — **Implementation Plan (v3.1 merged)**
============================================================================================

**Conventions:** Owner/Deps/Deliverables/DoD/Tests(NO‑MOCKS)/Artifacts/Gate; phase gates apply.

implementation-plan

A) Data & embeddings (Ingestion)
--------------------------------

**A.1 Section synopses & embeddings**

*   **Do this:** compute `Section.synopsis` (extractive or brief LLM summary behind a flag); embed+upsert; add vector index; keep `embedding_version`.
*   **DoD:** idempotent re‑ingest; reconciliation drift ≤ 0.5%.
    implementation-plan
*   **Tests (NO‑MOCKS):** edit one section → minimal graph/vector delta; parity holds after reconciliation.
    implementation-plan

**A.2 Edge gloss (optional)**

*   Build `edge_gloss` for traversable relations and embed; cache lazily; version embedding.

B) Traversal controller & frontier scoring (Retrieval)
------------------------------------------------------

*   **Files:** `src/query/traversal.py` (FrontierManager, ExpansionPolicy), `src/query/frontier_scoring.py`, `src/query/embeddings.py`.
*   **Do this:** implement EG scoring; enforce `frontier_k`, thresholds, **time/token budgets**; **hard caps** (fixed: depth≤3, nodes≤100; auto: depth≤5); produce `trace`.
*   **Pitfalls:** revisits; dense relations; no cost penalty.
*   **DoD:** expansions stop for the right reasons; traces complete; budgets honored.

C) MCP tools & planner (Backend)
--------------------------------

*   **Tool updates:**
    *   `search_documentation`: `verbosity`, `include_trace`.
    *   `traverse_relationships`: add `mode`, budgets, thresholds, `include_text`, `include_trace`. (Server retains **final say**.)
    *   Keep security: parameterized Cypher behind templates + validator + EXPLAIN.
        expert-coder-guidance
*   **Planner flow:** use templates‑first; when deeper evidence is likely, call `traverse_relationships(mode=auto, …)` with seeds from hybrid search.

D) Response builder
-------------------

*   Add `trace[]` and `budgets` to `diagnostics`; keep evidence & confidence.
    spec

E) Observability (SRE)
----------------------

*   **Metrics:** keep your existing **four** (verbosity counter, size histogram, depth counter, nodes histogram).
    implementation-plan-enhanced-re…
*   **Add auto‑depth metrics** listed in the spec.
*   **Instrumentation examples:** (as in your originals) in `QueryService` and `TraversalService`.
    implementation-plan-enhanced-re…

F) Performance & size limits
----------------------------

*   Enforce per‑mode P95 targets and **64 KB** total response cap; cap per‑section **~32 KB** for `full`.
    FEATURE\_SUMMARY\_enhanced-respon…

G) Tests (NO‑MOCKS) & matrix
----------------------------

*   **Unit:** verbosity modes; depth limits; whitelist; size limits.
    implementation-plan-enhanced-re…
*   **Integration:** E2E via `/mcp/tools/call` incl. chained `search → traverse`.
    implementation-plan-enhanced-re…
*   **Performance:** supplied scripts for latency by mode and traversal depth; assert per‑mode P95s and global P95 guardrail.
    implementation-plan-enhanced-re…
*   **Security:** injection prevention in traversal; validator/EXPLAIN plan caps.
    implementation-plan-enhanced-re…
*   **Artifacts:** write to `/reports/phase-enhanced/` using your canonical schema.
    implementation-plan

H) Rollout
----------

*   **Flags:** `search.auto_traversal.enabled`, `embed.synopsis.enabled`, `embed.edge_gloss.enabled`.
*   **Canary:** 10% → 50% → 100%; alert if P95>400 ms (safety margin) as in your plan.
    implementation-plan-enhanced-re…

I) Runbook (Ops)
----------------

*   **Depth runaway:** raise `gain_threshold`/`delta_threshold`, lower `frontier_k`, enforce fixed mode per‑tenant.
*   **Vector drift:** freeze embed jobs; run reconciliation; backfill synopses.
*   **Hot relations:** add per‑relation fan‑out cap, increase negative prior on `MENTIONS`.

* * *

3) `/docs/FEATURE_SUMMARY_enhanced-responses.md` — **One‑Pager (v3.1 merged)**
==============================================================================

**What:** More complete answers via **verbosity controls** and a **bounded traversal tool**, plus an **auto‑depth** mode that goes deeper only when it scores as promising—**within strict budgets**. Evidence & confidence remain mandatory.

spec

**Why:** Fixed depth (2–3) misses some real‑world answers; this keeps flexibility without breaking the **P95 < 500 ms** envelope.

FEATURE\_SUMMARY\_enhanced-respon…

**How:**

*   `search_documentation(verbosity=snippet|full|graph)`; `graph` returns related entities/sections (≤ 20/section) with size limits (≈ 32 KB/section, **64 KB total**).
    FEATURE\_SUMMARY\_enhanced-respon…
*   `traverse_relationships`: **fixed** depth≤3 (≤100 nodes) or **auto** depth with frontier EG scoring and server‑side budgets/caps; returns `nodes`, `relationships`, `paths`, and `trace`.
    FEATURE\_SUMMARY\_enhanced-respon…
    feature-spec-enhanced-responses

**Performance targets:** snippet 70 ms; full < 100 ms; graph < 150 ms; traverse (depth=2) < 200 ms; global P95 < 500 ms.

FEATURE\_SUMMARY\_enhanced-respon…

**Observability:** existing metrics for verbosity/size/depth/nodes plus **auto‑depth** metrics for depth reached, frontier prunes, stop reasons; OTel spans across expansions.

implementation-plan-enhanced-re…

**Safety:** templates‑first; validator + **EXPLAIN** caps; parameterized Cypher only; rate limits; audit logs.

expert-coder-guidance

**Rollout:** behind flags; dev→staging→prod canary; alert if P95>400 ms during ramp.

implementation-plan-enhanced-re…

**Success (30 days):** adoption of full/graph/traverse as in your original targets; higher completeness with stable error rate and P95; drift ≤ 0.5%.

FEATURE\_SUMMARY\_enhanced-respon…

implementation-plan

* * *

Notes & deltas vs. your originals
---------------------------------

*   **Kept:** three‑mode verbosity, fixed‑depth `traverse_relationships` with depth≤3 and ≤100 nodes, 64 KB total cap, metric names & code snippets, P95 targets per mode.
    FEATURE\_SUMMARY\_enhanced-respon…
    implementation-plan-enhanced-re…
*   **Added:** **auto‑depth** with **frontier EG scoring** (server‑controlled), synopsis & edge‑gloss embeddings, traces, budgets/thresholds, new metrics, and runbooks; hard ceiling depth≤5 only in **auto** mode so we don’t regress your fixed‑mode guarantees.
*   **Preserved:** security posture (templates‑first, validator + EXPLAIN), no‑mocks gates, artifact schema and reporting.
    expert-coder-guidance
    spec
    implementation-plan

* * *

If you want, I can also **write these three Markdown files into `/docs/`** in your workspace with these exact contents.



---
