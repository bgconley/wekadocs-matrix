Below is a “drop-in” prompt you can give to an agentic coder. It’s written to be explicit about *what to build*, *how to structure it*, and *how we’ll know it’s done*, while leaving room for the agent to adapt to your repo’s actual language/framework.

---

## Prompt for agentic coder: Modularize RAG pipeline + make embedders/rerankers hot-swappable via profiles

### Context

We have a Retrieval-Augmented Generation (RAG) pipeline that needs better modularization and the ability to rapidly swap **embedding models** and **rerankers** in and out for experimentation.

We recently introduced the concept of “profiles” to simplify configuration, but in practice the profile concept isn’t fully centralized. Right now:

* A profile change requires edits in multiple places across the application.
* Component names (especially embedder names) are **hardcoded** in several modules.
* We have “wrapper” / “client” classes (HTTP clients and/or provider clients) where the **wrapper name/identity doesn’t reliably match the actual underlying embedder** being used.

  * Example failure mode: a wrapper originally built for Embedder A gets reused and Embedder B is “slotted in”, so logs/config say A but the runtime is actually B.
* This creates confusion, breaks experimentation, and makes it easy to accidentally run mixed configurations.

### Goal (high-level)

Refactor the code so that the RAG pipeline is built from **composable, testable modules** with clean interfaces, and so that embedders and rerankers are **swappable by configuration** (profile selection) without needing to touch application code. Eliminate hardcoded component names and prevent wrapper/model mismatch.

### Non-goals (explicit)

* Do not redesign our entire product or UI. Keep changes focused on the RAG subsystem and component wiring.
* Do not change the external behavior of retrieval/reranking unless required for correctness or to support the new architecture.
* Do not add “yet another config system” if we already have one; instead consolidate and enforce a single source of truth.

---

## Requirements

### 1) Standardize component interfaces (contracts)

Create explicit interfaces (or abstract base classes / protocols) for the key RAG modules at minimum:

**Embedder**

* Purpose: convert text → vector embeddings
* Must support:

  * `embed_documents(texts: List[str]) -> List[Vector]`
  * `embed_query(text: str) -> Vector`
  * Optional but strongly preferred:

    * `dimension: int` (or a method to get it)
    * `model_id` / `component_id` / `provider` metadata
    * deterministic “fingerprint” of configuration (used to version indices)

**Reranker**

* Purpose: reorder candidate docs given query (cross-encoder or LLM reranker, etc.)
* Must support:

  * `rerank(query: str, candidates: List[Document]) -> List[ScoredDocument]`
  * ability to set `top_k`, maybe `max_candidates` behavior

Also define (if not already clearly defined) lightweight types:

* `Document` (id, text, metadata)
* `ScoredDocument` (document + score + optional explanations)
* `Vector` representation (list/array)

**Acceptance requirement:** There is a single, obvious interface for embedders and rerankers and all implementations adhere to it.

---

### 2) Introduce a component registry / factory that builds from a profile (single source of truth)

Implement a central “component builder” layer:

* A **registry** of known embedder implementations and reranker implementations (or “provider adapters”).
* A **factory** that takes a `Profile` (config object) and returns instantiated components.

Key rules:

* No application module should instantiate a specific embedder/reranker implementation directly.
* No module should hardcode a model name like `"text-embedding-..."` / `"bge-..."` / `"cohere-rerank-..."`.
* All component selection must flow through profiles/config and the factory/registry.

**Acceptance requirement:** To switch embedder or reranker, I change only the profile (or a single CLI/env var) and everything else stays untouched.

---

### 3) Fix wrapper/model identity mismatch by design

We need to stop having wrappers whose names don’t match the actual underlying model.

Implement *one* of these robust patterns (choose what best fits the repo):

* **Pattern A (preferred):** wrappers do not have a “model name” baked in at all; they accept a `ComponentSpec` (provider+model+params) and expose that exact spec in `metadata`.
* **Pattern B:** wrappers are per-model and not reused across models. If a wrapper is reused, it must be a generic provider client whose identity is the provider/base URL only, not the model.

Additionally:

* Add validation at startup/build-time:

  * If a wrapper claims model X but config says model Y, fail fast with a clear error.
* Ensure logs/traces always include the true runtime `component_id` and `model_id`.

**Acceptance requirement:** It is impossible (or at least very difficult) for config/logging to claim one model while runtime uses another.

---

### 4) Profiles: centralize and normalize

We already have “profiles”, but they are not consistently applied.

Refactor so that:

* There is a single definition of a profile (dataclass / struct / schema).
* A profile fully specifies:

  * embedder spec (provider, model, dims if known, batch size, endpoints, auth env var ref, timeouts, etc.)
  * reranker spec (provider, model, top_k, etc.)
  * retrieval settings (top_k, chunking options if relevant, hybrid retrieval toggles if applicable)
  * vector DB namespace/collection/index naming rules (see next requirement)
* Profile selection happens once (startup / request entrypoint), and the constructed components are injected downstream.

Also add:

* A mapping mechanism for backwards compatibility (aliases):

  * old profile names or model names map to new canonical identifiers
  * emit warnings when using deprecated names

**Acceptance requirement:** There is a canonical profile + component spec format; everything reads from it.

---

### 5) Embedder swapping must be safe with vector indices (index versioning)

Swapping an embedder usually implies the vector index must match that embedder. We need guardrails.

Implement:

* A stable embedder “fingerprint” (hash of provider+model+dimension+pooling+normalization+any relevant params).
* Store this fingerprint in vector DB collection metadata or alongside the index (depending on our DB).
* At query time (or pipeline init), verify:

  * the selected embedder fingerprint matches the index fingerprint
  * if mismatch:

    * either fail fast, or automatically route to a separate namespace/index for that embedder (prefer explicit behavior)
* Provide a deterministic convention:

  * e.g. collection name = `{base_collection}__{embedder_fingerprint_short}`
  * so experiments don’t trample each other

**Acceptance requirement:** Experiments with different embedders cannot accidentally query an index built with a different embedder without a loud error or automatic isolation.

---

### 6) Make swapping fast: CLI/env overrides + experiment hooks

Add a minimal override mechanism so testing doesn’t require editing files:

* Example:

  * `RAG_PROFILE=my_profile`
  * `RAG_EMBEDDER=provider:model` override
  * `RAG_RERANKER=provider:model` override
* Or CLI flags if this is a service:

  * `--rag-profile=...`
  * `--embedder=...`
  * `--reranker=...`

Also add a simple “experiment runner” entrypoint (script or command) that can run:

* A/B comparisons across combinations of embedders and rerankers
* On a small dataset/golden set already in the repo (or create a tiny synthetic one if none exists)
* Output metrics (even basic ones like recall@k or reranker changes) + configuration used

**Acceptance requirement:** I can swap embedder/reranker in seconds and run a repeatable experiment.

---

### 7) HTTP client consolidation (avoid per-component bespoke networking)

We currently have HTTP client wrappers in multiple places.

Refactor so that:

* There is a shared, configurable HTTP client module (timeouts, retries, base headers, tracing).
* Embedder and reranker adapters use it consistently.
* Lifecycle is managed properly (singleton per process or injected per request, depending on app architecture).
* If async is used, ensure correct async client reuse and closing semantics.

**Acceptance requirement:** One obvious place defines HTTP behavior; components don’t create random ad-hoc clients.

---

## Implementation Plan (step-by-step)

You (agentic coder) should do this in an ordered, low-risk way:

1. **Repo audit**

   * Find the current RAG pipeline entrypoint(s).
   * Identify where embedding and reranking are selected/configured today.
   * Locate all hardcoded model names and profile name checks.
   * Locate wrapper/client classes where identity mismatch can occur.
   * Produce a brief inventory: “these are the modules to refactor and why”.

2. **Define contracts + data types**

   * Add `Embedder` and `Reranker` interfaces + core `Document`/`ScoredDocument` types if missing.
   * Ensure minimal and stable surface area (do not leak provider-specific details).

3. **Create `ComponentSpec` + profile schema**

   * `ComponentSpec`: `type`, `provider`, `model`, `params`, `runtime_id`, `fingerprint()`
   * `RagProfile`: embedder spec, reranker spec, retrieval params, vector index params

4. **Build registry/factory**

   * Registry maps `(type, provider)` to an adapter constructor.
   * Factory reads `ComponentSpec` and instantiates correct adapter.
   * All adapters expose correct metadata and use shared HTTP client.

5. **Adapterize existing implementations**

   * Wrap existing embedders/rerankers behind the new interfaces without changing their internals too much.
   * If there are multiple existing wrapper layers, consolidate until there’s one adapter per provider.

6. **Wire pipeline using dependency injection**

   * Pipeline builder takes a `RagProfile` and returns a pipeline object with embedder + retriever + reranker.
   * Update call sites to use the pipeline builder.
   * Remove direct instantiation and hardcoded model names.

7. **Index fingerprint enforcement**

   * Implement embedder fingerprint.
   * Implement namespace/collection naming convention.
   * Add validation and clear errors.

8. **Overrides + experiment runner**

   * Add env/CLI overrides.
   * Add a small script/command that can run multiple combinations and report results.

9. **Testing**

   * Unit tests:

     * registry selects correct adapter
     * profile parsing + validation
     * fingerprint determinism
     * mismatch detection
   * Integration test:

     * build pipeline with one profile, ensure components are correctly instantiated
     * if possible, run a tiny retrieval + rerank pass with stubbed providers

10. **Docs + migration**

* Document:

  * how to add a new embedder or reranker
  * how to define a new profile
  * how to run experiments
* Add deprecation aliases for old names if necessary.

---

## Deliverables

* New/refactored modules implementing:

  * `Embedder` interface
  * `Reranker` interface
  * `ComponentSpec` + fingerprinting
  * `RagProfile` + profile loader
  * registry/factory
  * shared HTTP client module
  * pipeline builder that takes a profile and returns a ready-to-use pipeline
* Refactored application call sites: no hardcoded model names or per-module profile logic.
* Tests + minimal documentation.
* A short “How to swap embedders/rerankers” section with concrete examples.

---

## Definition of Done (acceptance checklist)

* [ ] Switching embedder or reranker requires changing only a profile name / env var / CLI flag — not code.
* [ ] No hardcoded embedder/reranker model strings remain outside config/profile definitions and adapter internals.
* [ ] Wrapper/model identity mismatch is prevented or fails fast with a clear error.
* [ ] Embedder fingerprint is recorded and checked against index metadata or collection naming.
* [ ] Pipeline construction is centralized (one factory/builder path).
* [ ] Unit tests cover registry selection, profile validation, fingerprint determinism, mismatch detection.
* [ ] Docs explain how to add new components and how to run an experiment matrix.

---

## Extra notes / pitfalls to handle

* Embedding dimensionality changes between models; do not assume dimension is constant.
* Some embedders need text normalization, truncation policies, or batch sizing; these must be part of the spec params.
* Rerankers may have strict max token limits; parameters should be configurable.
* Ensure logging/tracing prints canonical identifiers: profile name, embedder component_id, reranker component_id, fingerprints.
* If the app is multi-tenant: ensure profile selection and pipeline caching are safe and don’t mix tenants.

Importantly, we also MUST ensure that whatever we implement is strictly compliant with our codebase architecture - for instance, it must respect our atomic ingestion pipeline and our query / retrieval pipeline.

We must also accommodate locally deployed embedders and proprietary api based embedders.  For instance, we already support a locally deployed embedder (BGE-M3) and a proprietary API based embedder (jina-embedder-v3).  We must be prepared to support more of each kind.  Soon, I'll be integrating / adding support for snowflake-arctic-l-v2.0 (local embedding service) and voyage-3-large (proprietary api based)
