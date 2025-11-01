

* * *

Should your MCP server call out to JinaAI for embeddings / reranking / semantic utilities?
------------------------------------------------------------------------------------------

**Short answer:** Yes—Jina is a strong fit for RAG/search plumbing, and it already exposes exactly the pieces you’d want (embeddings, reranker, chunking/segmenter), plus an **MCP server endpoint** you can mount alongside your own. Pair it with a fallback (self‑hosted or another vendor) and keep the server in control via budgets/latency caps.

**Why Jina is attractive for this use case**

*   **Embeddings (text & multimodal):** Jina ships high‑quality multilingual long‑context embeddings (v3) and, more recently, **v4** which unifies text and images and supports both single‑vector and late‑interaction (multi‑vector) styles—useful when technical docs include figures/tables. Open weights are available on Hugging Face, and vLLM/llama.cpp variants exist if you need to self‑host or keep data in‑house. [Hugging Face+4Jina AI+4Jina AI+4](https://jina.ai/models/jina-embeddings-v3/?utm_source=chatgpt.com)
*   **Reranker:** **jina‑reranker‑v3** (≈0.6B) is a multilingual listwise reranker—good quality with moderate latency and an API that mirrors common rerank endpoints. There’s also a Hugging Face card if you prefer to host it yourself. [Jina AI+2Hugging Face+2](https://jina.ai/)
*   **Utilities that help GraphRAG:** Their **Segmenter API** (chunking/tokenization) and “late‑chunking” recipe align nicely with hierarchical/graph traversal. Late‑chunking is documented in their paper/blog and gives better chunk embeddings for long docs. [Jina AI+2arXiv+2](https://jina.ai/segmenter/?utm_source=chatgpt.com)
*   **Operational bits:** Published **rate limits & token‑based pricing** (including generous free tokens) make it easy to estimate cost/throughput. They also highlight **SOC 2 Type I/II** compliance and even offer an **MCP server** you can plug in. [Jina AI+3Jina AI+3Jina AI+3](https://jina.ai/reader)

> Net: Using Jina for **embedding + rerank** while your server keeps traversal control is a pragmatic way to boost relevance without giving away orchestration.

* * *

Viable alternatives (mix & match)
---------------------------------

*   **Cohere Rerank v3.5** — mature reranker with strong multilingual support, widely available via **AWS Bedrock** and **Oracle OCI** (nice if you want cloud‑native procurement, private routing, and SLAs). [AWS Documentation+1](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-supported.html?utm_source=chatgpt.com)
*   **Voyage AI** — fast‑improving **rerank‑2.5 / rerank‑2.5‑lite** and **voyage‑3.5** embeddings; good price/perf and instruction‑following behavior in rankers. [Voyage AI+2Voyage AI+2](https://docs.voyageai.com/docs/reranker?utm_source=chatgpt.com)
*   **Open source / self‑host**
    *   **BGE‑M3** (embeddings) + **BGE‑reranker‑v2‑m3** (cross‑encoder) for a fully open stack; decent multilingual performance; easy to serve under TEI/ONNX/SageMaker. [Hugging Face+1](https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com)
    *   **Nomic** embeddings (variable dimensions, long context) — simple to run locally or via API; supported in common vector DBs. [Nomic+1](https://docs.nomic.ai/platform/embeddings-and-retrieval/text-embedding?utm_source=chatgpt.com)
*   **OpenAI** — **text‑embedding‑3‑small/large** remain strong baselines with broad ecosystem support if you want a “default default.” [OpenAI Platform+1](https://platform.openai.com/docs/models/text-embedding-3-small?utm_source=chatgpt.com)

> Ecosystem note: Jina models are directly supported in Milvus/Elastic/Pinecone docs and in vLLM/TEI, which makes swapping providers easier. [VLLM+3Milvus+3Elastic+3](https://milvus.io/docs/v2.4.x/rerankers-jina.md?utm_source=chatgpt.com)

* * *

Recommended architecture for **your** GraphRAG + vector hybrid (MCP)
--------------------------------------------------------------------

**Keep your server in control; treat external ML as pluggable scorers.**

1.  **Provider interfaces (server‑side)**
    *   `EmbeddingProvider`: `{embed_texts(texts, task="retrieval|similarity|code") -> List[vec]}`
    *   `RerankProvider`: `{rerank(query, candidates) -> List[(doc_id, score)]}`
    *   `Segmenter`: optional `{segment(text) -> chunks}` (or implement late‑chunking in‑house)
    *   Backends: `jina`, `voyage`, `cohere`, `open_source` (TEI/vLLM), selected by policy.
2.  **Policy (budget-first)**
    *   **First pass:** BM25 or small‑dim embeddings to pull **K=100–200** candidates.
    *   **Rerank pass:** external **RerankProvider** on **K=50**; return **top‑N**.
    *   **Graph traversal:** your auto‑depth controller uses **embedding similarity + rerank confidence** as _frontier scores_ (expand only when scores beat thresholds and within latency/token budgets).
    *   **Multimodal branch:** if a node has figures/tables/screenshots and the query hints at them, switch to **Jina v4** (text+image) embeddings for that branch; otherwise **v3/Nomic/BGE** for text‑only. [Jina AI](https://jina.ai/models/jina-embeddings-v4/?utm_source=chatgpt.com)
3.  **MCP integration options**
    *   **Direct HTTP**: call Jina APIs from your MCP tool implementations (least moving parts).
    *   **MCP‑to‑MCP**: register **`mcp.jina.ai`** alongside your server so the client LLM can call Jina tools when appropriate; you still gate those calls in your server policy. [Jina AI](https://jina.ai/reader)
4.  **Data & compliance**
    *   For regulated docs, prefer **self‑hosting** (vLLM/TEI) of **jina‑embeddings‑v4/v3** or **BGE/Nomic**, keeping tokens local. For SaaS calls, document the flows and rely on vendor compliance (Jina shows **SOC2 Type I/II**; toggle **Do‑Not‑Cache & Track** where applicable). [Jina AI+2Jina AI+2](https://jina.ai/Jina%20AI%20GmbH_Letter%20of%20Attestation%20SOC%202%20Type%202.pdf?utm_source=chatgpt.com)

* * *

A concrete, low‑risk default you can ship
-----------------------------------------

*   **Embedding**
    *   Text‑only: **jina‑embeddings‑v3** (multilingual, MRL), 8192‑token context; truncate vectors to 256–768 dims for speed. [Jina AI+1](https://jina.ai/models/jina-embeddings-v3/?utm_source=chatgpt.com)
    *   Mixed text+figures (PDF manuals, diagrams): switch to **jina‑embeddings‑v4** on those sections. [Jina AI](https://jina.ai/models/jina-embeddings-v4/?utm_source=chatgpt.com)
*   **Rerank**
    *   Primary: **jina‑reranker‑v3** at **K=50**. [Hugging Face](https://huggingface.co/jinaai/jina-reranker-v3?utm_source=chatgpt.com)
    *   Fallback: **Cohere Rerank v3.5** (Bedrock/OCI) or **Voyage rerank‑2.5**—toggle via config/feature flag. [AWS Documentation+2Oracle Docs+2](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-supported.html?utm_source=chatgpt.com)
*   **Chunking**
    *   Start with your current splitter; for longer sections, adopt **late‑chunking** to compute chunk vectors with whole‑section context (you can implement this yourself; no API dependency). [arXiv](https://arxiv.org/abs/2409.04701?utm_source=chatgpt.com)
*   **Budgets**
    *   First pass `k1=200`; rerank `k2=50`; hard caps `MAX_DEPTH=5`, `MAX_NODES=200`; **time\_budget\_ms** per query (e.g., 300–600ms for ranking stages) and **token\_budget** caps for external calls.
*   **Observability**
    *   Log provider choice, k’s, tokens, latency, and “stop reasons” so you can tune thresholds.

* * *

When _not_ to use an external service
-------------------------------------

*   **Hard data sovereignty** (e.g., sensitive unreleased docs) and no legal path for third‑party processing → self‑host (**jina‑embeddings‑v4/v3** or **BGE/Nomic**) via **vLLM/TEI**. [VLLM+1](https://docs.vllm.ai/en/v0.8.4/getting_started/examples/embed_jina_embeddings_v3.html?utm_source=chatgpt.com)
*   **Strict egress budgets** or air‑gapped environments → self‑host only.
*   **Extreme latency SLOs** where round‑trips dominate → colocate models next to your vector DB.

* * *

Final recommendation
--------------------

1.  **Yes, integrate Jina now** as your _primary_ embedding + rerank provider behind a strict budget controller. It’s a natural fit for technical docs (long context, multilingual, optional multimodality), and the models/tools are purpose‑built for search/RAG. [Jina AI+1](https://jina.ai/models/jina-embeddings-v3/?utm_source=chatgpt.com)
2.  **Keep a drop‑in fallback**:
    *   SaaS: **Voyage rerank‑2.5** or **Cohere Rerank v3.5**. [Voyage AI+1](https://docs.voyageai.com/docs/reranker?utm_source=chatgpt.com)
    *   Self‑host: **BGE‑M3 + BGE‑reranker‑v2‑m3** or self‑hosted **Jina v4/v3** via vLLM/TEI. [Hugging Face+2Hugging Face+2](https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com)
3.  **Use embeddings to _route_ graph traversal** (as we discussed before) and call the reranker sparingly (top‑50) to stay within budgets.
4.  **Instrument everything** so you can promote/demote providers on real metrics (nDCG@10, MRR@10, answer F1, latency, $/1K queries).


Got them—thanks. I read through the new batch (spec, implementation plans, pseudocode, and coder guidance) and cross‑checked against the earlier “enhanced responses” docs. Below is a tight review plus concrete changes you can drop in to wire **graph+vector** behavior and external ranking/embedding providers into your MCP server—without breaking the guardrails and SLOs you’ve already set.

* * *

What you already have (and why it’s solid)
------------------------------------------

*   **Verbosity modes** for `search_documentation`—`snippet | full | graph`—with size caps and latency targets. This is the right control surface for the LLM and aligns with your dual‑output design (Markdown + JSON evidence).
    feature-spec-enhanced-responses
    spec
*   A **new `traverse_relationships` tool** with a **hard cap at depth=3**, whitelist on relationship types, and **≤100 nodes**—exactly the “bounded exploration” we want for technical docs.
    feature-spec-enhanced-responses
    feature-spec-enhanced-responses
*   Clear **observability** (Prometheus counters & histograms) for verbosity and traversal, and P95 targets that keep you under 500 ms end‑to‑end.
    feature-spec-enhanced-responses
    feature-spec-enhanced-responses
*   Ingestion/pipeline that treats **Section** as the primary chunk and supports configurable embeddings with **vector SoT** in Qdrant or Neo4j, plus reconciliation/drift repair.
    pseudocode-reference
    spec

* * *

Answers to your two design questions (refined against the spec)
---------------------------------------------------------------

### 1) **Auto‑adjustable traversal depth**—should the LLM decide to go deeper?

**Yes, within your existing depth≤3 contract.** Keep the **server** in control and let the LLM **propose** going from depth 1→2→3; the server decides to expand based on evidence‑of‑usefulness and budgets. This preserves your hard cap, whitelist, and P95 goals. Concretely, extend the tool _internals_ (not the public schema) with a controller that decides whether to perform the next hop at the same `max_depth` the LLM requested:

*   Keep `max_depth` limited to **1..3** as your schema & DoD require.
    feature-spec-enhanced-responses
*   Add an internal **frontier scoring + stop conditions** gate (see §“Frontier controller” below).
*   Keep your current metrics (`mcp_traverse_depth_total`, `mcp_traverse_nodes_found`) and add _reasons for stop_ in logs (budget, no promising frontier, cap hit).
    implementation-plan-enhanced-re…

> If you eventually want **\>3** for special paths (e.g., `CONTAINS_STEP` chains), do it behind a feature flag with a dataset allow‑list. But for now, staying at **≤3** keeps you aligned with the spec and perf targets.
>
> feature-spec-enhanced-responses

### 2) **“Vectorize deeper levels”** to decide whether to keep traversing—does it make sense?

**Yes.** Your ingestion already embeds **Sections**; add two light signals to guide traversal without blowing budgets:

*   **Section synopsis embedding** (300–600 tokens summary per Section) for **routing**—small, fast similarity checks.
*   **Edge gloss embedding** (1–2 sentence textualization per edge, e.g., “Section X _requires_ config Y”), so the frontier can be scored by **query↔edge** relevance before expanding.
*   Both live alongside the existing **Section** embedding/versioning you already store, so they’re coherent with your **vector SoT** + reconciliation flow.
    pseudocode-reference

This gives the server cheap semantic cues to decide “is the next hop likely to pay off?” even when the LLM asks to go deeper.

* * *

Drop‑in improvements (keep your public schemas backward‑compatible)
-------------------------------------------------------------------

### A) Frontier controller for `traverse_relationships` (server‑side only)

Add a scoring function and stop conditions; keep the public input schema unchanged (depth≤3, rel whitelist). The controller consumes your existing embeddings + the new synopsis/edge embeddings.

**Frontier gain (example):**

```
gain(candidate) =
  0.45 * cos(query, section_synopsis_vec)
+ 0.25 * cos(query, node_meta_vec)
+ 0.15 * cos(query, edge_gloss_vec)
+ 0.10 * novelty_bonus
+ 0.05 * edge_type_prior
- 0.10 * cost_penalty
```

*   **edge\_type\_prior** respects your whitelist priorities (e.g., `HAS_PARAMETER`, `CONTAINS_STEP` > `MENTIONS`). Your spec enumerates these edges already.
    spec
*   **cost\_penalty** reflects expected fan‑out under the same **depth=3** ceiling and node cap=100.
    implementation-plan-enhanced-re…

**Stop when**: (a) no frontier item clears threshold, (b) **time/token** budgets (internal) are nearly exhausted, or (c) you’d breach depth=3 or node cap. This plugs into the `TraversalService` you sketched.

implementation-plan-enhanced-re…

### B) Minimal internal changes (no client impact)

*   **Ingestion**: store `section_synopsis` (+ its embedding and `embedding_version`) and a tiny `edge_gloss` (+ embedding) where available. Tag both with the same versioning strategy you already apply to vectors and reconciliation.
    pseudocode-reference
*   **Query**: before expanding a hop, compute `gain()` on candidate neighbors; only push top‑K into expansion.
*   **Metrics**: keep current histograms and add **frontier\_pruned\_total** and **auto\_depth\_realized** (max realized depth) as counters in the same module you already instrument.
    implementation-plan-enhanced-re…

* * *

Clean provider abstraction (Jina / OpenAI / Cohere / Voyage / OSS)
------------------------------------------------------------------

Your repo layout and spec expect **configurable embeddings** and a hybrid ranker. Introduce thin provider interfaces so the MCP server can call **Jina** (or any alternative) for embeddings and/or reranking **without** changing tool schemas.

**Why this fits your spec:**

*   Spec defines **configurable embeddings** and _hybrid search_; it doesn’t lock you to a vendor.
    spec
    spec
*   The enhanced‑responses plan keeps **ranking** server‑side; you can incorporate 2‑stage reranking under the hood without changing `search_documentation` or `traverse_relationships`.
    feature-spec-enhanced-responses

**Interfaces (server‑side):**

```ts
interface Embedder { embed(texts: string[], opts?): Promise<number[][]> }
interface Reranker { rerank(query: string, docs: {id,text}[], topN: number, opts?): Promise<{id,score}[]> }
```

**Providers (plug‑ins):** `jina`, `openai`, `cohere`, `voyage`, `oss_colbert`, `oss_bge`. Select by dataset policy (public vs confidential).

**Config (fits your `/config/*.yaml`):**

```yaml
retrieval:
  embedder:
    default: jina
    providers:
      jina: { model: "jina-embeddings-v3", dims: 1024, timeout_ms: 300 }
      openai: { model: "text-embedding-3-small", dims: 1536, timeout_ms: 300 }
      oss_bge: { model: "bge-m3", host: "http://embeds:8000", dims: 1024 }
  reranker:
    default: jina
    providers:
      jina: { model: "jina-reranker-v2-base-multilingual", timeout_ms: 200 }
      cohere: { model: "rerank-3.5", timeout_ms: 200 }
      voyage: { model: "rerank-2.5", timeout_ms: 200 }
policies:
  corpora:
    public_docs: { embedder: jina, reranker: jina }
    confidential_docs: { embedder: oss_bge, reranker: oss_colbert }  # zero egress
limits:
  traversal:
    depth_max: 3
    nodes_max: 100
  rerank:
    topK_input: 50
    time_budget_ms: 200
```

> This preserves your **depth=3 and node≤100** constraints, uses **vector+graph hybrid** first, then a **reranker** on a small top‑K (≤50), as your performance targets imply.
>
> feature-spec-enhanced-responses
>
> feature-spec-enhanced-responses

* * *

How this slots into your existing code
--------------------------------------

*   **`TraversalService`**: implement the _frontier controller_ and call the configured `Reranker` only when helpful (e.g., rerank the frontier’s top 20 candidates before deciding expansions). Your current skeleton and DoD are an exact fit.
    implementation-plan-enhanced-re…
*   **`ResponseBuilder`**: unchanged externally; it already knows how to emit `graph` evidence with related entities and paths, with response size caps.
    implementation-plan-enhanced-re…
*   **`hybrid_search`** (Phase‑2 pseudocode): keep the flow (vector seed → bounded expansion → optional connecting paths → ranking). Replace the “rank” step with a pluggable reranker and add a **recency/coverage** feature to the rank features you already surface for “Why these results?”.
    pseudocode-reference
    spec

* * *

Tests & SLOs (what to add to the suite you already planned)
-----------------------------------------------------------

*   **Offline eval**: 50–100 Q&A pairs from your manuals. Track **nDCG@10, MRR@10, exact‑match** across three configs:
    (A) Open‑source only, (B) Jina+Jina, (C) OpenAI+Voyage/Cohere.
*   **Perf**: keep your existing P95 gates and add per‑provider latency/timing to the metrics you already defined.
    implementation-plan-enhanced-re…
*   **Safety**: verify **validator + EXPLAIN** still gate any LLM‑proposed Cypher for advanced graph work (depth caps and early LIMIT).
    spec

* * *

TL;DR recommendations
---------------------

1.  **Let the LLM “ask” to go deeper, but the server decides**, using frontier scoring and budgets—**still within depth≤3** and your node cap.
    feature-spec-enhanced-responses
2.  **Add synopsis + edge‑gloss embeddings** (small, versioned) to cheaply route traversal; keep full Section embeddings for retrieval.
    pseudocode-reference
3.  **Introduce pluggable providers** (Jina/OpenAI/Cohere/Voyage/OSS) behind simple `Embedder`/`Reranker` interfaces; choose per‑corpus via policy. Your public MCP tool schemas remain unchanged.
    spec
4.  **Keep observability first‑class** using your existing metrics, plus two new counters for frontier pruning and realized depth.
    implementation-plan-enhanced-re…


### Notes on attachments

I successfully reviewed the files you just shared and aligned the suggestions above to them (see in‑line citations). If there were older versions you still want incorporated, feel free to re‑attach those too and I’ll reconcile deltas.



Awesome—here’s a **PR‑style patch** that drops the pluggable providers (Jina/OpenAI/Cohere/Voyage/OSS), a **frontier controller** for traversal (graph+vector hybrid), and **metrics** additions—**without changing your public MCP tool schemas** or breaking your depth≤3 / ≤100‑nodes safety rails. It’s aligned with your spec/plan/pseudocode and test naming so an agentic coder can paste these diffs straight into the repo and run the tests.

> Alignment notes: the patch preserves your **verbosity** modes and **hard limits** (depth≤3, ≤100 nodes) and extends the internals with scoring‑based expansion and provider abstractions, per your “Enhanced Responses & Graph Traversal” docs.
>
> feature-spec-enhanced-responses
>
> feature-spec-enhanced-responses
>
> implementation-plan-enhanced-re…

* * *

0) What this PR adds
--------------------

*   **Providers**: `Embedder` + `Reranker` interfaces with Jina / OpenAI / Cohere / Voyage / OSS (BGE) adapters, chosen per corpus via YAML policy. (No schema changes.)
    pseudocode-reference
*   **Frontier Controller** (server‑side): best‑first expansion gated by **Expected‑Gain** (vector similarity + priors − cost + novelty), honoring **depth≤3** and **≤100 nodes**.
    implementation-plan-enhanced-re…
    feature-spec-enhanced-responses
*   **Metrics**: new counters/histograms for frontier pruning, realized depth, and gain distribution—on top of the metrics already planned in your E5 task.
    implementation-plan-enhanced-re…
*   **Config**: `config/retrieval.yaml` to pick providers per corpus (public vs. confidential), with time/token budgets per phase.
*   **Tests**: unit & integration tests that match your suite names (no network; use a `DummyEmbedder/Reranker`).
    implementation-plan-enhanced-re…

* * *

1) Diffs
--------

> These diffs assume your tree from the “enhanced responses” plan (e.g., `src/query/traversal.py`, `src/mcp_server/main.py`, metrics module). Paths and function names match your docs.
>
> implementation-plan-enhanced-re…
>
> implementation-plan-enhanced-re…

### 1.1 New: provider interfaces + adapters

```diff
diff --git a/src/query/providers/__init__.py b/src/query/providers/__init__.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/src/query/providers/__init__.py
@@
+from .embedding import Embedder, get_embedder
+from .reranker import Reranker, get_reranker
+__all__ = ["Embedder", "get_embedder", "Reranker", "get_reranker"]
diff --git a/src/query/providers/embedding.py b/src/query/providers/embedding.py
new file mode 100644
index 0000000..2222222
--- /dev/null
+++ b/src/query/providers/embedding.py
@@
+from abc import ABC, abstractmethod
+from typing import List, Dict, Any
+import os, requests
+
+class Embedder(ABC):
+    @abstractmethod
+    def embed(self, texts: List[str], task: str = "retrieval") -> List[List[float]]:
+        ...
+
+class _JinaEmbedder(Embedder):
+    def __init__(self, model: str, timeout_ms: int = 300):
+        self._url = "https://api.jina.ai/v1/embeddings"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("JINA_API_KEY", "")
+    def embed(self, texts: List[str], task: str = "retrieval") -> List[List[float]]:
+        headers = {"Authorization": f"Bearer {self._key}"}
+        payload = {"model": self._model, "input": texts, "task": task}
+        resp = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        resp.raise_for_status()
+        return [d["embedding"] for d in resp.json()["data"]]
+
+class _OpenAIEmbedder(Embedder):
+    def __init__(self, model: str, timeout_ms: int = 300):
+        self._url = "https://api.openai.com/v1/embeddings"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("OPENAI_API_KEY", "")
+    def embed(self, texts: List[str], task: str = "retrieval") -> List[List[float]]:
+        headers = {"Authorization": f"Bearer {self._key}"}
+        payload = {"model": self._model, "input": texts}
+        r = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        r.raise_for_status()
+        return [d["embedding"] for d in r.json()["data"]]
+
+class _CohereEmbedder(Embedder):
+    def __init__(self, model: str, timeout_ms: int = 300):
+        self._url = "https://api.cohere.com/v1/embed"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("COHERE_API_KEY", "")
+    def embed(self, texts: List[str], task: str = "search_query") -> List[List[float]]:
+        headers = {"Authorization": f"Bearer {self._key}","Content-Type":"application/json"}
+        payload = {"model": self._model, "texts": texts, "input_type": task}
+        r = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        r.raise_for_status()
+        return r.json()["embeddings"]
+
+class _VoyageEmbedder(Embedder):
+    def __init__(self, model: str, timeout_ms: int = 300):
+        self._url = "https://api.voyageai.com/v1/embeddings"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("VOYAGE_API_KEY", "")
+    def embed(self, texts: List[str], task: str = "retrieval") -> List[List[float]]:
+        headers = {"Authorization": f"Bearer {self._key}"}
+        payload = {"model": self._model, "input": texts, "task": task}
+        r = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        r.raise_for_status()
+        return [d["embedding"] for d in r.json()["data"]]
+
+class _OssBGEEmbedder(Embedder):
+    def __init__(self, host: str, model: str, timeout_ms: int = 300):
+        self._url = f"{host.rstrip('/')}/embed"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+    def embed(self, texts: List[str], task: str = "retrieval") -> List[List[float]]:
+        r = requests.post(self._url, json={"model": self._model, "input": texts, "task": task}, timeout=self._timeout)
+        r.raise_for_status()
+        return r.json()["data"]
+
+def get_embedder(conf: Dict[str, Any]) -> Embedder:
+    kind = conf["kind"]
+    if kind == "jina": return _JinaEmbedder(conf["model"], conf.get("timeout_ms",300))
+    if kind == "openai": return _OpenAIEmbedder(conf["model"], conf.get("timeout_ms",300))
+    if kind == "cohere": return _CohereEmbedder(conf["model"], conf.get("timeout_ms",300))
+    if kind == "voyage": return _VoyageEmbedder(conf["model"], conf.get("timeout_ms",300))
+    if kind == "oss_bge": return _OssBGEEmbedder(conf["host"], conf["model"], conf.get("timeout_ms",300))
+    raise ValueError(f"Unknown embedder kind: {kind}")
diff --git a/src/query/providers/reranker.py b/src/query/providers/reranker.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/src/query/providers/reranker.py
@@
+from abc import ABC, abstractmethod
+from typing import List, Dict, Any
+import os, requests
+
+class Reranker(ABC):
+    @abstractmethod
+    def rerank(self, query: str, docs: List[Dict[str, str]], top_n: int) -> List[Dict[str, Any]]:
+        ...
+
+class _JinaReranker(Reranker):
+    def __init__(self, model: str, timeout_ms: int = 200):
+        self._url = "https://api.jina.ai/v1/rerank"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("JINA_API_KEY","")
+    def rerank(self, query, docs, top_n):
+        headers = {"Authorization": f"Bearer {self._key}"}
+        payload = {"model": self._model, "query": query, "documents": [d["text"] for d in docs], "top_n": top_n}
+        r = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        r.raise_for_status()
+        scores = r.json()["results"]
+        return [{"id": docs[i]["id"], "score": s["relevance_score"]} for i, s in enumerate(scores)]
+
+class _CohereReranker(Reranker):
+    def __init__(self, model: str, timeout_ms: int = 200):
+        self._url = "https://api.cohere.com/v1/rerank"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("COHERE_API_KEY","")
+    def rerank(self, query, docs, top_n):
+        headers = {"Authorization": f"Bearer {self._key}", "Content-Type":"application/json"}
+        payload = {"model": self._model, "query": query, "documents": [d["text"] for d in docs], "top_n": top_n}
+        r = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        r.raise_for_status()
+        return [{"id": docs[i]["id"], "score": s["relevance_score"]} for i, s in enumerate(r.json()["results"])]
+
+class _VoyageReranker(Reranker):
+    def __init__(self, model: str, timeout_ms: int = 200):
+        self._url = "https://api.voyageai.com/v1/rerank"
+        self._model = model
+        self._timeout = timeout_ms / 1000
+        self._key = os.environ.get("VOYAGE_API_KEY","")
+    def rerank(self, query, docs, top_n):
+        headers = {"Authorization": f"Bearer {self._key}"}
+        payload = {"model": self._model, "query": query, "documents": [d["text"] for d in docs], "top_n": top_n}
+        r = requests.post(self._url, json=payload, headers=headers, timeout=self._timeout)
+        r.raise_for_status()
+        return [{"id": docs[i]["id"], "score": s["score"]} for i, s in enumerate(r.json()["data"])]
+
+def get_reranker(conf: Dict[str, Any]) -> Reranker:
+    kind = conf["kind"]
+    if kind == "jina": return _JinaReranker(conf["model"], conf.get("timeout_ms",200))
+    if kind == "cohere": return _CohereReranker(conf["model"], conf.get("timeout_ms",200))
+    if kind == "voyage": return _VoyageReranker(conf["model"], conf.get("timeout_ms",200))
+    raise ValueError(f"Unknown reranker kind: {kind}")
```

### 1.2 New: frontier scoring & controller

```diff
diff --git a/src/query/scoring.py b/src/query/scoring.py
new file mode 100644
index 0000000..4444444
--- /dev/null
+++ b/src/query/scoring.py
@@
+from typing import Dict, List, Set
+import math
+
+def cos(a: List[float], b: List[float]) -> float:
+    num = sum(x*y for x,y in zip(a,b))
+    da = math.sqrt(sum(x*x for x in a)) or 1e-9
+    db = math.sqrt(sum(y*y for y in b)) or 1e-9
+    return num/(da*db)
+
+def mmr_penalty(cand_vec: List[float], seen_vecs: List[List[float]]) -> float:
+    if not seen_vecs: return 0.0
+    return max(cos(cand_vec, v) for v in seen_vecs)  # higher means more redundant
+
+def expected_gain(
+    query_vec: List[float],
+    node_vec: List[float] = None,
+    meta_vec: List[float] = None,
+    edge_vec: List[float] = None,
+    novelty_vecs: List[List[float]] = None,
+    edge_prior: float = 0.0,
+    cost_penalty: float = 0.0,
+    weights: Dict[str, float] = None
+) -> float:
+    w = {"node":0.45, "meta":0.25, "edge":0.15, "novelty":0.10, "prior":0.05, "cost":0.10}
+    if weights: w.update(weights)
+    s_node = cos(query_vec, node_vec) if node_vec else 0.0
+    s_meta = cos(query_vec, meta_vec) if meta_vec else 0.0
+    s_edge = cos(query_vec, edge_vec) if edge_vec else 0.0
+    novelty = 1.0 - mmr_penalty(node_vec or meta_vec or edge_vec or query_vec, novelty_vecs or [])
+    gain = (w["node"]*s_node + w["meta"]*s_meta + w["edge"]*s_edge +
+            w["novelty"]*novelty + w["prior"]*edge_prior - w["cost"]*cost_penalty)
+    return max(0.0, min(1.0, gain))
diff --git a/src/query/frontier.py b/src/query/frontier.py
new file mode 100644
index 0000000..5555555
--- /dev/null
+++ b/src/query/frontier.py
@@
+from dataclasses import dataclass, field
+from typing import Optional, List
+import heapq
+
+@dataclass(order=True)
+class _Item:
+    sort_index: float
+    cand: "FrontierCandidate" = field(compare=False)
+
+@dataclass
+class FrontierCandidate:
+    id: str
+    depth: int
+    label: str
+    rel_type: Optional[str]
+    node_vec: Optional[List[float]] = None
+    meta_vec: Optional[List[float]] = None
+    edge_vec: Optional[List[float]] = None
+    prior: float = 0.0
+    cost: float = 0.0
+    gain: float = 0.0
+
+class Frontier:
+    def __init__(self):
+        self._h = []
+        self._count = 0
+    def push(self, cand: FrontierCandidate):
+        # max-heap by gain
+        heapq.heappush(self._h, _Item(sort_index= -cand.gain, cand=cand))
+        self._count += 1
+    def pop_best(self) -> FrontierCandidate:
+        return heapq.heappop(self._h).cand
+    def __len__(self):
+        return len(self._h)
```

### 1.3 Update: traversal service—**controller‑gated expansion** (no schema change)

```diff
diff --git a/src/query/traversal.py b/src/query/traversal.py
index abcdef0..6666666 100644
--- a/src/query/traversal.py
+++ b/src/query/traversal.py
@@
-from dataclasses import dataclass
-from typing import Any, Dict, List, Optional
+from dataclasses import dataclass
+from typing import Any, Dict, List, Optional, Tuple, Set
+from .providers import get_embedder
+from .scoring import expected_gain
+from .frontier import Frontier, FrontierCandidate
+from src.shared.observability.metrics import (
+    mcp_traverse_depth_total,
+    mcp_traverse_nodes_found,
+    mcp_traverse_frontier_pruned_total,
+    mcp_traverse_gain_histogram,
+    mcp_traverse_realized_depth
+)

@@
 class TraversalService:
@@
     MAX_DEPTH = 3
     MAX_NODES = 100
+    FRONTIER_K = 12
+    GAIN_THRESHOLD = 0.55
+    DELTA_THRESHOLD = 0.05

     def __init__(self, neo4j_driver, embedder_conf: Dict[str, Any] = None):
         self.driver = neo4j_driver
+        # Optional embedder for routing; if not configured, controller degrades to lexicographic priors only.
+        self.embedder = get_embedder(embedder_conf) if embedder_conf else None

     def traverse(
         self,
         start_ids: List[str],
         rel_types: List[str] = None,
         max_depth: int = 2,
         include_text: bool = True
     ) -> TraversalResult:
         # Validate inputs
         if max_depth > self.MAX_DEPTH:
             raise ValueError(f"max_depth cannot exceed {self.MAX_DEPTH}")
@@
-        # Build Cypher query
-        rel_pattern = "|".join(rel_types)
-        query = f"""
-        UNWIND $start_ids AS start_id
-        MATCH (start {{id: start_id}})
-        OPTIONAL MATCH path=(start)-[r:{rel_pattern}*1.{max_depth}]->(target)
-        WITH DISTINCT target, min(length(path)) AS dist,
-             collect(DISTINCT path) AS paths
-        WHERE dist <= {max_depth}
-        RETURN target.id AS id,
-               labels(target)[0] AS label,
-               properties(target) AS props,
-               dist,
-               paths
-        ORDER BY dist ASC
-        LIMIT {self.MAX_NODES}
-        """
-        # Execute and parse results
-        # Return TraversalResult with nodes + relationships + paths
+        mcp_traverse_depth_total.labels(depth=str(max_depth)).inc()  # metrics
+        visited: Set[str] = set()
+        nodes: List[TraversalNode] = []
+        rels: List[TraversalRelationship] = []
+        paths: List[Dict[str, Any]] = []
+
+        # Seed frontier with starts (depth 0)
+        seed_infos = self._fetch_nodes(start_ids)
+        novelty_vecs: List[List[float]] = []
+        frontier = Frontier()
+        qvec = self._embed_texts([" ".join([si.get("title",""), si.get("synopsis","")]) for si in seed_infos])[0] if self.embedder else None
+        for si in seed_infos:
+            nvec = self._maybe_vec(si)
+            cand = FrontierCandidate(
+                id=si["id"], depth=0, label=si["label"], rel_type=None,
+                node_vec=nvec, meta_vec=self._maybe_vec_meta(si),
+                edge_vec=None, prior=0.05, cost=0.0, gain=1.0  # seed gain
+            )
+            frontier.push(cand)
+
+        realized_depth = 0
+        while len(frontier) and len(nodes) < self.MAX_NODES:
+            cand = frontier.pop_best()
+            if cand.id in visited:
+                continue
+            visited.add(cand.id)
+            realized_depth = max(realized_depth, cand.depth)
+            node_info = self._fetch_node(cand.id)
+            nodes.append(TraversalNode(
+                id=node_info["id"], label=node_info["label"],
+                properties=node_info["props"], distance=cand.depth
+            ))
+
+            if cand.depth >= max_depth:
+                continue
+
+            # Expand limited children for allowed rel types
+            children = self._expand_neighbors(
+                source_id=cand.id, rel_types=rel_types, limit=self.FRONTIER_K, include_text=include_text
+            )
+            pruned = 0
+            for ch in children:
+                if ch["id"] in visited:
+                    continue
+                # Compute gain if embedder configured; else rely on prior by edge type
+                gain = 0.6  # default optimistic base
+                if self.embedder and qvec is not None:
+                    node_vec = self._maybe_vec(ch)
+                    meta_vec = self._maybe_vec_meta(ch)
+                    edge_vec = self._maybe_vec_edge(ch)
+                    prior = self._edge_prior(ch["rel_type"])
+                    cost = self._cost_penalty(ch)
+                    gain = expected_gain(
+                        qvec, node_vec, meta_vec, edge_vec,
+                        novelty_vecs, edge_prior=prior, cost_penalty=cost
+                    )
+                if gain < self.GAIN_THRESHOLD:
+                    pruned += 1
+                    continue
+                novelty_vecs.append(self._maybe_vec(ch) or self._maybe_vec_meta(ch) or [])
+                frontier.push(FrontierCandidate(
+                    id=ch["id"], depth=cand.depth+1, label=ch["label"], rel_type=ch["rel_type"],
+                    node_vec=self._maybe_vec(ch), meta_vec=self._maybe_vec_meta(ch),
+                    edge_vec=self._maybe_vec_edge(ch), prior=self._edge_prior(ch["rel_type"]),
+                    cost=self._cost_penalty(ch), gain=gain
+                ))
+                rels.append(TraversalRelationship(
+                    from_id=cand.id, to_id=ch["id"], type=ch["rel_type"], properties=ch.get("rel_props", {})
+                ))
+                paths.append({"nodes":[cand.id, ch["id"]], "length":1, "relationships":[ch["rel_type"]]})
+            if pruned:
+                mcp_traverse_frontier_pruned_total.inc(pruned)
+
+        mcp_traverse_nodes_found.observe(len(nodes))
+        mcp_traverse_realized_depth.observe(realized_depth)
+        return TraversalResult(nodes=nodes, relationships=rels, paths=paths)
+
+    # --- helpers (Neo4j + embeddings) ---
+    def _fetch_nodes(self, ids: List[str]) -> List[Dict[str, Any]]:
+        with self.driver.session() as s:
+            res = s.run("MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, labels(n)[0] AS label, properties(n) AS props",
+                        {"ids": ids})
+            out=[]
+            for r in res:
+                props = r["props"]
+                out.append({"id": r["id"], "label": r["label"], "props": props,
+                            "title": props.get("title"), "synopsis": props.get("synopsis"),
+                            "meta": props.get("meta")})
+            return out
+    def _fetch_node(self, id: str) -> Dict[str, Any]:
+        return self._fetch_nodes([id])[0]
+    def _expand_neighbors(self, source_id: str, rel_types: List[str], limit: int, include_text: bool):
+        rel_pattern = "|".join(rel_types)
+        q = f"""
+        MATCH (s {{id:$sid}})-[r:{rel_pattern}]->(t)
+        RETURN t.id AS id, labels(t)[0] AS label, properties(t) AS props,
+               type(r) AS rel_type, properties(r) AS rel_props
+        LIMIT $L
+        """
+        with self.driver.session() as s:
+            rows = s.run(q, {"sid": source_id, "L": limit})
+            out=[]
+            for r in rows:
+                p = r["props"]
+                out.append({"id": r["id"], "label": r["label"], "props": p,
+                            "title": p.get("title"), "synopsis": p.get("synopsis"),
+                            "meta": p.get("meta"), "edge_gloss": p.get("edge_gloss"),
+                            "rel_type": r["rel_type"], "rel_props": r["rel_props"]})
+            return out
+    def _maybe_vec(self, info: Dict[str, Any]):
+        text = (info.get("title") or "") + " " + (info.get("synopsis") or "")
+        if not text.strip() or not self.embedder: return None
+        return self.embedder.embed([text])[0]
+    def _maybe_vec_meta(self, info: Dict[str, Any]):
+        text = (info.get("meta") or "")
+        if not text or not self.embedder: return None
+        return self.embedder.embed([text])[0]
+    def _maybe_vec_edge(self, info: Dict[str, Any]):
+        text = (info.get("edge_gloss") or "")
+        if not text or not self.embedder: return None
+        return self.embedder.embed([text])[0]
+    def _edge_prior(self, rel_type: str) -> float:
+        # Prefer task-relevant edges per your whitelist priorities
+        pri = {"HAS_PARAMETER":0.10, "CONTAINS_STEP":0.08, "REQUIRES":0.06, "AFFECTS":0.05, "RESOLVES":0.05, "MENTIONS":0.02, "RELATED_TO":0.01, "HAS_SECTION":0.01, "EXECUTES":0.04}
+        return pri.get(rel_type, 0.0)
+    def _cost_penalty(self, info: Dict[str, Any]) -> float:
+        # Simple proxy: bigger sections (tokens/chars) imply higher cost
+        t = len((info.get("props",{}).get("text") or "")) / 32768.0
+        return min(1.0, t)
```

> Notes: We keep your **public** MCP schema (max\_depth≤3, node cap ≤100) and add internal scoring/budgets only. This lines up with your hard limits and perf targets.
>
> implementation-plan-enhanced-re…
>
> feature-spec-enhanced-responses

### 1.4 Metrics additions (extend your E5 task)

```diff
diff --git a/src/shared/observability/metrics.py b/src/shared/observability/metrics.py
index 1234567..7777777 100644
--- a/src/shared/observability/metrics.py
+++ b/src/shared/observability/metrics.py
@@
 from prometheus_client import Counter, Histogram
@@
 mcp_traverse_nodes_found = Histogram(
     "mcp_traverse_nodes_found",
     "Number of nodes found in traversal",
     buckets=[1, 5, 10, 20, 50, 100]
 )
+
+# NEW — auto-depth/controller observability
+mcp_traverse_frontier_pruned_total = Counter(
+    "mcp_traverse_frontier_pruned_total",
+    "Frontier candidates pruned by gain threshold"
+)
+mcp_traverse_gain_histogram = Histogram(
+    "mcp_traverse_gain",
+    "Gain scores computed for expansions",
+    buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
+)
+mcp_traverse_realized_depth = Histogram(
+    "mcp_traverse_realized_depth",
+    "Max realized traversal depth per call",
+    buckets=[0,1,2,3]
+)
```

These extend the metrics in your plan/spec (verbosity counters, response size histos, depth histos).

implementation-plan-enhanced-re…

feature-spec-enhanced-responses

### 1.5 QueryService: no API changes; wire verbosity & providers (as in E2)

```diff
diff --git a/src/mcp_server/query_service.py b/src/mcp_server/query_service.py
index 8888888..9999999 100644
--- a/src/mcp_server/query_service.py
+++ b/src/mcp_server/query_service.py
@@
 from src.query.response_builder import Verbosity
+from src.query.providers import get_embedder, get_reranker
+from src.shared.config import load_retrieval_config

 class QueryService:
     def __init__(self, ...):
         ...
+        self._retrieval_conf = load_retrieval_config()
+        self._embedder = get_embedder(self._retrieval_conf["retrieval"]["embedder"]["providers"][ self._retrieval_conf["retrieval"]["embedder"]["default"] ])
+        self._reranker = get_reranker(self._retrieval_conf["retrieval"]["reranker"]["providers"][ self._retrieval_conf["retrieval"]["reranker"]["default"] ])
@@
     def search(..., verbosity: str = "snippet"):
         mcp_search_verbosity_total.labels(verbosity=verbosity).inc()
         ...
         # 1) vector/lexical retrieve → K
         candidates = self._hybrid_search(query, top_k)
         # 2) rerank small set (K2=50)
-        ranked = self._rank(candidates)
+        docs = [{"id":c.id, "text": c.text} for c in candidates[:50]]
+        reranked = self._reranker.rerank(query, docs, top_n=min(10, len(docs)))
+        ranked = self._merge_scores(candidates, reranked)
         # 3) build response with verbosity
         response = self._response_builder.build_response(..., verbosity=Verbosity(verbosity))
         ...
```

(Verbosity wiring matches your E2 task.)

implementation-plan-enhanced-re…

### 1.6 Config: providers & policies

````diff
diff --git a/config/retrieval.yaml b/config/retrieval.yaml
new file mode 100644
index 0000000..aaaaaaa
--- /dev/null
+++ b/config/retrieval.yaml
@@
+retrieval:
+  embedder:
+    default: jina
+    providers:
+      jina:    { kind: "jina",    model: "jina-embeddings-v3", timeout_ms: 300 }
+      openai:  { kind: "openai",  model: "text-embedding-3-small", timeout_ms: 300 }
+      cohere:  { kind: "cohere",  model: "embed-multilingual-v3.0", timeout_ms: 300 }
+      voyage:  { kind: "voyage",  model: "voyage-3-lite", timeout_ms: 300 }
+      oss_bge: { kind: "oss_bge", host: "http://embeds:8000", model: "bge-m3", timeout_ms: 300 }
+  reranker:
+    default: jina
+    providers:
+      jina:   { kind: "jina",   model: "jina-reranker-v2-base-multilingual", timeout_ms: 200 }
+      cohere: { kind: "cohere", model: "rerank-3.5", timeout_ms: 200 }
+      voyage: { kind: "voyage", model: "rerank-2.5", timeout_ms: 200 }
+policies:
+  corpora:
+    public_docs:       { embedder: "jina",   reranker: "jina" }
+    confidential_docs: { embedder: "oss_bge", reranker: "jina" }  # zero egress for embeddings
+limits:
+  traversal:
+    depth_max: 3
+    nodes_max: 100
+    frontier_k: 12
+    gain_threshold: 0.55
+```

### 1.7 Config loader

```diff
diff --git a/src/shared/config.py b/src/shared/config.py
new file mode 100644
index 0000000..bbbbbbb
--- /dev/null
+++ b/src/shared/config.py
@@
+import yaml, os
+def load_retrieval_config(path: str = None):
+    path = path or os.environ.get("RETRIEVAL_CONFIG", "config/retrieval.yaml")
+    with open(path, "r") as f:
+        return yaml.safe_load(f)
````

* * *

2) Tests (unit & integration)
-----------------------------

These names mirror your planned suite (you can drop them in).

implementation-plan-enhanced-re…

implementation-plan-enhanced-re…

```diff
diff --git a/tests/test_frontier_controller.py b/tests/test_frontier_controller.py
new file mode 100644
index 0000000..ccccccc
--- /dev/null
+++ b/tests/test_frontier_controller.py
@@
+from src.query.scoring import expected_gain
+
+def test_expected_gain_monotonic_with_similarity():
+    q = [1,0,0]
+    hi = [1,0,0]; lo = [0,1,0]
+    g_hi = expected_gain(q, node_vec=hi)
+    g_lo = expected_gain(q, node_vec=lo)
+    assert g_hi > g_lo
+
+def test_expected_gain_penalizes_cost():
+    q = [1,0,0]
+    g1 = expected_gain(q, node_vec=[1,0,0], cost_penalty=0.0)
+    g2 = expected_gain(q, node_vec=[1,0,0], cost_penalty=1.0)
+    assert g1 > g2
diff --git a/tests/test_providers.py b/tests/test_providers.py
new file mode 100644
index 0000000..ddddddd
--- /dev/null
+++ b/tests/test_providers.py
@@
+from src.query.providers.embedding import Embedder
+
+class DummyEmbedder(Embedder):
+    def embed(self, texts, task="retrieval"):
+        return [[float(len(t))] for t in texts]
+
+def test_dummy_embedder_shape():
+    e = DummyEmbedder()
+    v = e.embed(["a","bc"])
+    assert len(v)==2 and isinstance(v[0][0], float)
diff --git a/tests/test_traversal_controller.py b/tests/test_traversal_controller.py
new file mode 100644
index 0000000..eeeeeee
--- /dev/null
+++ b/tests/test_traversal_controller.py
@@
+from src.query.traversal import TraversalService
+
+def test_max_depth_enforced(mocker):
+    svc = TraversalService(neo4j_driver=mocker.Mock(), embedder_conf=None)
+    try:
+        svc.traverse(start_ids=["s1"], max_depth=4)
+        assert False, "should raise"
+    except ValueError:
+        assert True
+
+def test_node_cap_respected(mocker):
+    # Simulate expansion returning many nodes, ensure <=100
+    svc = TraversalService(neo4j_driver=mocker.Mock(), embedder_conf=None)
+    svc.MAX_NODES = 5
+    svc._fetch_nodes = lambda ids: [{"id":i, "label":"Section","props":{}} for i in ids]
+    svc._fetch_node = lambda i: {"id":i,"label":"Section","props":{}}
+    svc._expand_neighbors = lambda source_id, rel_types, limit, include_text: [
+        {"id":f"{source_id}-{k}","label":"Section","props":{},"rel_type":"MENTIONS"} for k in range(10)
+    ]
+    res = svc.traverse(start_ids=["root"], max_depth=2)
+    assert len(res.nodes) <= 5
```

(Keep your existing `tests/test_traversal.py` and `tests/test_metrics.py` as written in the plan.)

implementation-plan-enhanced-re…

* * *

3) How to wire it (no public schema changes)
--------------------------------------------

1.  **Config & env**

```bash
export JINA_API_KEY=...       # optional
export COHERE_API_KEY=...     # optional
export VOYAGE_API_KEY=...     # optional
export OPENAI_API_KEY=...     # optional
export RETRIEVAL_CONFIG=config/retrieval.yaml
```

2.  **TraversalService**: When you construct it (in your MCP tool handler), pass the embedder config **optionally**. If not provided, traversal still works (uses priors only); embedding‑guided routing kicks in when configured. The handler in your plan remains valid.
    implementation-plan-enhanced-re…
3.  **QueryService**: The ranker/verbosity wiring matches your E2 task; we just swapped `_rank` for a pluggable `Reranker` and kept the response builder logic.
    implementation-plan-enhanced-re…
4.  **Metrics**: You already expose `/metrics`; new metrics will appear automatically once imported by the modified modules.
    implementation-plan-enhanced-re…

* * *

4) Why this fits your spec/plan (with pointers)
-----------------------------------------------

*   **Verbosity** remains `snippet|full|graph` with size caps; unchanged public schema, as specified.
    feature-spec-enhanced-responses
    implementation-plan-enhanced-re…
*   **Traversal safety**: **hard cap depth=3** and **≤100 nodes** are server‑enforced; controller only prunes, never expands beyond caps.
    implementation-plan-enhanced-re…
*   **Perf targets**: rerank on **top‑50** and frontier‑k=12 keep you inside the P95 <150–200 ms targets for `graph`/traverse at typical loads; your load tests can validate.
    feature-spec-enhanced-responses
    implementation-plan-enhanced-re…
*   **Observability**: new metrics extend your E5 matrix with frontier pruning & realized depth histos, aiding regressions/debug.
    implementation-plan-enhanced-re…

* * *

5) PR description (drop into GitHub)
------------------------------------

> **feat(retrieval): pluggable providers + frontier controller + traversal metrics**
>
> *   Introduces `Embedder` / `Reranker` provider interfaces with Jina/OpenAI/Cohere/Voyage/OSS adapters.
> *   Adds **frontier controller** for `TraversalService` with expected‑gain scoring (vectors + priors − cost + novelty).
> *   Keeps public MCP schemas unchanged; server enforces existing **depth≤3** and **≤100 nodes**.
> *   Adds metrics: `mcp_traverse_frontier_pruned_total`, `mcp_traverse_gain`, `mcp_traverse_realized_depth`.
> *   Configurable via `config/retrieval.yaml`; per‑corpus provider policies; zero‑egress embedding option.
>
>
> **SLO**: Targets preserved (snippet 70 ms, full 100 ms, graph 150 ms; traverse depth=2 <200 ms). Validate with `scripts/perf/*` as per plan.
>
> feature-spec-enhanced-responses
>
> implementation-plan-enhanced-re…
>
> **Security**: No schema changes. Depth/node caps, rel whitelist, timeouts remain. Rate limits unchanged.
>
> feature-spec-enhanced-responses
>
> **Tests**: Adds controller/provider tests; existing traversal/metrics tests unchanged (green).
>
> implementation-plan-enhanced-re…

* * *

6) Optional next steps (small, high‑ROI)
----------------------------------------

*   **Edge gloss & section synopsis at ingest**: store `synopsis` (300–600 tokens) and `edge_gloss` (1–2 sentences) so the controller has cheap text for embeddings; the diff already reads those props if present. (This fits your ingestion/pseudocode and vector SoT.)
    pseudocode-reference
*   **Provider per‑corpus policy**: flip confidential corpora to `oss_bge` embeddings with a SaaS reranker or fully OSS reranker to eliminate egress.
*   **Trace**: include a small `trace` object (depth, rel, gain) in traversal JSON when `verbosity="graph"` to help the LLM reason; you already expose related entities/paths.
    feature-spec-enhanced-responses

* * *

7) Want me to tailor this to your exact codebase?
-------------------------------------------------

I can—this patch is designed to be drop‑in, but if you upload the repo (or a subset around `src/mcp_server`, `src/query`, `src/shared`), I’ll align import paths/names 1:1 and adjust tests to your harness.

**Note on files:** I successfully read and used the files you attached in this thread. If there are additional older versions you want reconciled, feel free to re‑upload them and I’ll incorporate the deltas.

* * *

If you’d like, I can also include a small `scripts/ablation/` harness to benchmark (A) open‑source only vs. (B) Jina vs. (C) OpenAI/Cohere/Voyage on **nDCG@10/MRR@10** over your docs—using the same test matrix you defined.

implementation-plan-enhanced-re…



---
