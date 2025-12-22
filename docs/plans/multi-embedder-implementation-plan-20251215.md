Research grounding (what enables this)

  - Jina v4: outputs 2048-D by default and supports Matryoshka dimension truncation (e.g., 1024) via a dimensions
    request parameter; also exposes task adapters like retrieval.query / retrieval.passage (Qdrant’s Jina embedding
    guide: https://qdrant.tech/documentation/embeddings/jina-embeddings/, Jina model card: https://jina.ai/models/jina-
    embeddings-v4/).
  - Qdrant: supports named vectors, sparse vectors, and multivectors (late interaction) as first-class citizens (Qdrant
    vectors concept: https://qdrant.tech/documentation/concepts/vectors/).
  - Multivectors best practice: use dense vectors for candidate recall and multivector (ColBERT-style MaxSim) for
    reranking; disable HNSW indexing for multivectors when they’re only used as a reranker (hnsw_config.m=0) (Qdrant
    multivector tutorial: https://qdrant.tech/documentation/advanced-tutorials/using-multivector-representations/).
  - Fusion: weighted fusion / weighted RRF is a standard way to combine multiple retrieval signals (Elastic weighted RRF
    overview: https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf).

  Brave search is currently rate-limited in this environment; I used Firecrawl to pull primary docs instead.

  ———

  ## Plan: Multi-Embedder “Embedding Stack” + TUI Configuration (no implementation)

  ### 1) Assess current codebase baseline (what we already have)

  - Single-provider embedding profiles already exist via config/embedding_profiles.yaml and EMBEDDINGS_PROFILE, applied
    by src/shared/config.py.
  - Providers already exist and are selectable by profile:
      - BGE-M3 service provider supports dense+sparse+ColBERT-like multivectors: src/providers/embeddings/
        bge_m3_service.py
      - Jina API provider supports dense: src/providers/embeddings/jina.py
  - Qdrant schema and validation are centralized:
      - Schema plan builder: src/shared/qdrant_schema.py (content, title, doc_title, optional late-interaction, sparse
        vectors)
      - Collection creation/reconcile: src/ingestion/build_graph.py (_ensure_qdrant_collection)
  - Retrieval already supports multi-vector fusion across named vector fields with weights:
      - src/query/hybrid_retrieval.py (QdrantMultiVectorRetriever, vector_fields weights)
  - The missing piece is: query and ingestion assume one embedder produces the same dense vector used across all dense
    fields, and one embedder produces sparse/colbert. We need a first-class way to route “vector field X comes from
    embedder Y”.

  ### 2) Introduce a new config concept: Embedding Stack (composition of profiles)

  Goal: let users choose:

  - Single embedder (today’s behavior)
  - Hybrid: dense from Jina v4 (1024 or 2048), sparse+colbert from BGE-M3
  - Ensemble: dense from both Jina v4 and BGE-M3, fused (plus sparse/colbert as desired)

  Config structure (additive; keep current embedding.profile working)

  - Add embedding.stack (optional) + stack definitions file, e.g. config/embedding_stacks.yaml.
  - Stack definition references existing profiles and defines which vector fields each produces.

  Example stack schema (conceptual):

  - stack_id: jina_v4_2048__bge_sparse_colbert
  - dense_models: list of {profile, fields} where fields default to ["content","title","doc_title"]
  - sparse_model: {profile, fields} for text-sparse, doc_title-sparse, title-sparse, entity-sparse
  - multivector_model: {profile, field_name} default late-interaction__<profile_slug> if dims differ from dense
  - primary_query_vector: computed: first dense model’s content field
  - vector_name_template: stable naming rule: "{field}__{profile}" for dense, late-interaction__{profile} for multivector

  Why this naming rule:

  - It prevents silent dimension collisions when multiple models are stored in one collection.
  - It works with Qdrant’s “named vectors per point” model (https://qdrant.tech/documentation/concepts/vectors/).

  Embedding versioning (defensive)

  - Define embedding_version payload as the stack fingerprint (stable string or hash of selected profiles + dims +
    versions).
  - This keeps existing embedding_version filtering, cache keys, and drift checks meaningful across mixed configurations:
      - Used in payload indexes in src/shared/qdrant_schema.py
      - Used in query filters in src/query/hybrid_retrieval.py and src/mcp_server/query_service.py

  ### 3) Extend embedding profiles to include Jina v4 presets

  Update config/embedding_profiles.yaml with (at minimum):

  - jina_v4_1024 (provider jina-ai, model jina-embeddings-v4, dims 1024, tasks retrieval.passage/retrieval.query)
  - jina_v4_2048 (same model, dims 2048)
    Grounding: Jina v4 default 2048 and supports dimensions override + task adapters (Qdrant Jina embeddings guide:
    https://qdrant.tech/documentation/embeddings/jina-embeddings/).

  Provider support plan:

  - Cloud: extend src/providers/embeddings/jina.py to send dimensions and task per call (it already tracks dims and task;
    formalize for v4).
  - Local: add a second provider type (plan-only) that targets a configurable base URL and optional auth (so users can
    point at a locally deployed v4 service without changing code).

  ### 4) Add a small “Embedding Stack Resolver” module (no big-module growth)

  Create a new cohesive module, e.g. src/providers/embedding_stack.py (~300–500 LoC target):

  - Loads stack definitions + profile definitions.
  - Validates:
      - required env vars per profile (e.g., JINA_API_KEY, BGE_M3_API_URL)
      - capability compatibility (don’t assign sparse role to a profile that lacks supports_sparse)
      - dimension expectations per vector field
  - Produces:
      - VectorFieldPlan: mapping vector_name -> {kind: dense|sparse|multivector, dims, provider_profile}
      - StackFingerprint: embedding_version string
      - PrimaryVectorName: used for search.vector.qdrant.query_vector_name

  This module becomes the single “source of truth” used by both ingestion and retrieval.

  ### 5) Implement a composite embedder (adapter) without breaking call sites

  Create src/providers/embeddings/composite.py (~300–500 LoC):

  - Holds multiple concrete providers built via existing ProviderFactory (src/providers/factory.py).
  - Exposes two layers:
      1. Compatibility surface (to minimize changes): embed_documents, embed_query, and (if configured) embed_sparse,
         embed_query_all-like outputs.
      2. New explicit surface (used by new code paths): embed_for_vector_fields(texts, field_plan) returning a dict of
         {vector_name: vectors} for dense + multivector, plus sparse vectors separately.

  Defensive behavior:

  - If a stack references a vector field but provider call fails:
      - ingestion: obey a strictness flag per vector kind (dense strict, sparse best-effort by default, multivector
        configurable)
      - retrieval: skip that channel and log structured degradation (don’t crash query path in dev)

  ### 6) Qdrant schema generation becomes “plan-driven”

  Add a new function alongside build_qdrant_schema() in src/shared/qdrant_schema.py (plan-only):

  - build_qdrant_schema_from_field_plan(field_plan) -> QdrantSchemaPlan
  - It must support:
      - multiple named dense vectors with different sizes (each VectorParams(size=...))
      - sparse vectors unchanged
      - multivectors with MultiVectorConfig(MAX_SIM) and HnswConfigDiff(m=0) (aligned with Qdrant’s multivector guidance:
        https://qdrant.tech/documentation/advanced-tutorials/using-multivector-representations/)

  Update validation similarly:

  - validate_qdrant_schema_from_field_plan(...) compares collection config to expected vector names + sizes, sparse
    fields, and multivector settings.

  Rollout safety:

  - If the collection exists but dims differ for an existing vector name, we must fail fast and require a new namespaced
    collection (Qdrant cannot “resize” an existing vector field safely).

  ### 7) Ingestion pipeline integration points (minimal edits, new helper module)

  Keep src/ingestion/atomic.py and src/ingestion/build_graph.py changes minimal by adding a new helper module, e.g.:

  - src/ingestion/embedding_write_plan.py (~300–500 LoC)
      - takes document, sections, entities, stack config
      - returns the Qdrant point vector payloads per section: dense vectors dict + multivector dict + sparse vectors dict
      - sets payload metadata: embedding_version (stack fingerprint), plus optional per-provider provenance fields

  Key adjacent integrations to account for:

  - src/ingestion/build_graph.py _ensure_qdrant_collection() should use the stack-driven schema plan.
  - src/ingestion/incremental.py and src/ingestion/reconcile.py need to understand multiple dense vector fields (they
    currently “clone” a base vector across a fixed list).
  - Any ingestion-time drift checks keyed on embedding_version must use stack fingerprint rather than single profile
    version.

  ### 8) Query/retrieval integration points (support per-field query vectors)

  Update retrieval to stop assuming “one dense vector fits all fields”:

  - Extend the query embedding bundle concept used in src/query/hybrid_retrieval.py to carry per-vector-name query
    vectors:
      - e.g. dense_by_name: Dict[str, List[float]]
      - and multivector_by_name similarly (if we ever support multiple multivectors)
  - _build_prefetch_entries() should use the correct query vector for each dense field.
  - Qdrant query API path can prefetch from multiple dense vectors (already structured as a list of prefetch entries);
    then rerank using the selected multivector field (Qdrant shows this pattern directly: https://qdrant.tech/
    documentation/advanced-tutorials/using-multivector-representations/).

  Fusion config:

  - Continue using search.hybrid.vector_fields weights, but now those keys can include model-qualified names like
    content__jina_v4_2048.
  - Document recommended defaults for weights using weighted RRF intuition (Elastic weighted RRF: https://www.elastic.co/
    search-labs/blog/weighted-reciprocal-rank-fusion-rrf).

  Adjacent systems to update:

  - src/services/cross_doc_linking.py currently hardcodes doc_title, doc_title-sparse, late-interaction. It should derive
    names from the active stack plan (or at least choose a single canonical dense doc_title field for linking).

  ### 9) User-friendly TUI: embedctl

  Add a dedicated CLI/TUI tool (do not extend the already-large ingestion CLI):

  - New entrypoint: scripts/embedctl.py or src/tools/embedctl.py
  - Reads:
      - config/embedding_profiles.yaml
      - config/embedding_stacks.yaml
  - UX:
      - “Pick a preset stack” (recommended for most users)
      - “Build a custom stack” (advanced): choose dense model(s), sparse model, multivector model, and dims (e.g., Jina
        v4 1024 vs 2048)
      - Shows warnings for missing env vars (e.g., JINA_API_KEY, BGE_M3_API_URL)
      - Writes a small, explicit config artifact:
          - safest: write a .env.embed snippet containing EMBEDDINGS_STACK=<id> plus any required settings; user can
            source it or copy into .env
          - optionally: write a dedicated file like config/embedding_runtime.yaml and point EMBEDDING_PROFILES_PATH/
            EMBEDDINGS_STACK_PATH to it (no in-place YAML mutation required)

  Dependency policy (defensive):

  - Prefer stdlib curses or plain interactive prompts to avoid adding heavy deps.
  - If you want richer UI, add an optional dependency (plan-only), not required by services.

  ### 10) Rollout strategy (avoid regressions)

  - Phase 1: additive config + stack resolver + schema generation; keep default stack equivalent to today (bge_m3
    everywhere).
  - Phase 2: enable Jina v4 dense-only stack in dev with a namespaced collection (reuse existing EMBEDDING_NAMESPACE_MODE
    behavior but extend it from “profile” to “stack”).
  - Phase 3: hybrid stack (Jina dense + BGE sparse+colbert); validate ingestion and retrieval end-to-end.
  - Phase 4: optional ensemble stack (Jina dense + BGE dense + BGE sparse+colbert), tune weights.

  ### 11) Testing & validation plan (targeted)

  - Unit tests:
      - stack resolution + fingerprint stability
      - schema plan correctness (vector names + dims + multivector config)
      - query vector routing (each vector name uses the right provider)
  - Integration tests (Qdrant):
      - create a collection with multiple named vectors of different sizes + sparse vectors
      - insert and query using correct using= vector name(s)
  - Regression tests:
      - ensure single-profile mode behaves exactly as today (no stack specified)

  ———

  ## Key decisions I need from you (so the plan is implementable)

  - Do you want multiple dense providers stored simultaneously in the same collection (ensemble), or only “one dense
    provider active at a time” plus optional sparse/colbert from another?
  - Should stacks always create a new Qdrant collection per stack (safer), or do you want a mode that adds new vector
    fields into the existing collection (works only when new vector names don’t collide and dims are compatible)?
  - For Jina v4 local: do you expect an OpenAI-style /v1/embeddings local service, or should we plan to add a dedicated
    local runtime (e.g., llama.cpp/GGUF) later?
