On the GPU node right now, the active retrieval stack is this.

  Authoritative runtime plan
  From retrieval_plan_resolved on weka-mcp-server:

  - profile: graph_assisted
  - use_weighted_fusion: true
  - use_signal_pool: true
  - signal_pool_before_colbert: true
  - use_colbert: true
  - use_related_to_expansion: true
  - use_related_to_blending: true
  - use_entity_graph_channel: true
  - use_graph_enrichment: false
  - use_structure_expansion: true
  - use_focused_rerank_text: true
  - use_specificity_adjustment: true
  - graph_garbage_filter_on: true
  - graph_score_normalized_on: true
  - use_graph_score_override: false

  Enabled

  1. Hybrid retrieval
      - search.hybrid.enabled: true
      - profile-driven control is active
  2. Qdrant multi-vector retrieval
      - dense enabled
      - sparse enabled
      - title sparse enabled
      - entity sparse enabled
      - query API enabled
      - query strategy: weighted
  3. Python-side fusion
      - multi-vector fusion method: rrf
      - weighted query API fusion is also on underneath because the plan requires it
  4. Signal pool
      - enabled
      - active before ColBERT
      - confirmed live by signal_pool_built logs
  5. ColBERT pre-rerank
      - enabled
      - confirmed live by colbert_rerank_complete
  6. Cross-encoder reranker
      - enabled
      - model: mixedbread-ai/mxbai-rerank-large-v2
      - confirmed live by rerank_complete
  7. Specificity adjustment
      - enabled via profile_overrides.use_specificity_adjustment: true
      - confirmed live in logs as specificity_applied: true
  8. RELATED_TO path
      - expansion enabled
      - blending enabled
      - confirmed live by related_to_blending_applied
  9. Entity graph channel
      - enabled
      - confirmed live by graph_augmentation_complete
  10. Structure-aware expansion

  - enabled

  11. Adaptive graph relationship selection

  - enabled via graph_adaptive_enabled: true

  12. Graph garbage filter

  - enabled

  13. Graph score normalization

  - enabled

  Disabled

  1. Graph enrichment
      - use_graph_enrichment: false
      - this is the “full graph enrichment” path, not the graph channel
  2. Graph score override / graph-as-reranker
      - use_graph_score_override: false
      - graph is assisting, not replacing the fused score model
  3. BM25
      - search.hybrid.bm25.enabled: false
      - lexical retrieval is coming from sparse vectors, not BM25
  4. Neo4j master disable
      - not disabled
      - neo4j_disabled: false
      - so graph paths are allowed to run

  Important nuance
  The YAML still contains deprecated legacy flags like:

  - graph_channel_enabled: false
  - graph_enrichment_enabled: false

  Those are not the truth anymore when profile is set. The resolved plan is the truth. Right now that means:

  - graph channel: on
  - graph enrichment: off

  What recent logs show in practice
  For recent metadata-shaped runs:

  - graph channel is active
  - it retrieved 5 graph candidates
  - 1 overlapped and blended with an existing result
  - 4 new graph candidates were added
  - ColBERT ran
  - reranker ran
  - specificity was applied

  So the short answer is:

  Currently on: dense+sparse Qdrant retrieval, weighted query API fusion, RRF, signal pool, ColBERT, cross-encoder reranking, specificity adjustment, RELATED_TO expansion/blending, entity graph channel,
  structure expansion, graph garbage filtering.

  Currently off: BM25, graph enrichment, graph score override.


   Diagram

  flowchart TD
      A[User Query] --> B[Normalize filters + build lexical_query]
      B --> C[Query analysis
      classify_query_intent
      entity disambiguation / boost terms]
      C --> D[Qdrant multi-vector retrieval
      dense: content/title/doc_title
      sparse: text/title/entity
      query_api_weighted]
      D --> E[Top-level fusion
      BM25 branch empty
      RRF path still runs]
      E --> F[Soft score adjustments
      entity boost
      structural boost]
      F --> G[RELATED_TO expansion
      Neo4j cross-doc edges]
      G --> H[RELATED_TO blending
      query-type lambda]
      H --> I[Entity graph channel
      Neo4j MENTIONS
      support gate
      sparse gate
      conservative merge]
      I --> J[Signal pool build
      active
      before ColBERT]
      J --> K[Optional pre-rerank structural candidates
      only when precision_mode = false]
      K --> L[ColBERT rerank
      full pool reordered]
      L --> M[Hydrate parent_path_norm]
      M --> N[Cross-encoder reranker
      mxbai-rerank-large-v2
      focused rerank text]
      N --> O[Specificity adjustment
      active]
      O --> P[Top-K seeds]
      P --> Q[Citation hydration]
      Q --> R[Expansion gate]
      R --> S[Bounded adjacency expansion
      only when triggered and not precision_mode]
      S --> T[Post-seed structure expansion
      only when not precision_mode]
      T --> U[Dedup + coverage annotation]
      U --> V[Graph enrichment branch
      disabled]
      V --> W[Doc continuity boost]
      W --> X[Context budget enforcement]
      X --> Y[Final retrieved chunks]

  Disabled branches right now

  - BM25: off
  - graph enrichment: off
  - graph score override / graph-as-reranker: off
  - microdoc extras: off

  Enabled branches right now

  - Qdrant dense + sparse retrieval
  - weighted query API path
  - RRF fusion path
  - entity boost
  - structural boost
  - RELATED_TO expansion
  - RELATED_TO blending
  - entity graph channel
  - signal pool
  - ColBERT
  - cross-encoder reranker
  - focused rerank text
  - specificity adjustment
  - structure-aware expansion logic

  Important nuance
  For many subsystem / precision queries like the metadata one you just ran, these branches often no-op:

  - pre-rerank structural candidate expansion
  - bounded adjacency expansion
  - post-seed structure expansion

  That happens because precision_mode suppresses some expansion logic to avoid polluting precise technical queries.

  What is likely doing the most work on your good metadata result

  1. Qdrant dense+sparse retrieval
  2. RELATED_TO expansion + blending
  3. graph channel merge
  4. signal pool
  5. ColBERT
  6. cross-encoder reranker
  7. specificity adjustment


   The claimed dataflow is fundamentally accurate — 22 of 24 steps validated in the correct order. The agent found one missing step (microdoc expansion), one incomplete specification (signal pool
  dual-ordering), and a few minor gating nuances. The overall flow shape is correct.
  ─────────────────────────────────────────────────

  The dataflow is verified with corrections. Here's the accurate ASCII diagram:

  ╔══════════════════════════════════════════════════════════════════════╗
  ║                    HybridRetriever.retrieve()                       ║
  ║                     src/query/hybrid_retrieval.py                   ║
  ╚══════════════════════════════════════════════════════════════════════╝

   ┌─────────────────────────────────────────────────────────────────┐
   │ 1. QUERY PREP                                          L798-830│
   │    normalize_filters()                                         │
   │    build lexical_query                                         │
   │    extract doc_tag                                             │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 2. QUERY ANALYSIS                                      L847-932│
   │    GLiNER entity disambiguation (disambiguator.process)        │
   │    classify_query_intent() → intent (precision_mode, anchors)  │
   │    extract boost_terms for entity/structural boosting          │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 3. VECTOR RETRIEVAL                          L934-950 → _vb    │
   │    vector_retriever.search()                                   │
   │    ├─ dense: content, title, doc_title                         │
   │    ├─ sparse: text-sparse, title-sparse, entity-sparse         │
   │    └─ query_api_weighted (adaptive RRF weights per query type) │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 4. FUSION                                    L983-998 → _fp    │
   │    RRF or weighted fusion                                      │
   │    (BM25 branch empty but code path runs)                      │
   │    sort by fused_score DESC                                    │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 5. SOFT SCORE ADJUSTMENTS                          L1033-1068  │
   │    _apply_entity_boost()           (always)                    │
   │    _apply_structural_boost()       (always)                    │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 6. RELATED_TO EXPANSION                     L1080-1090 → _gp  │
   │    gate: _plan.use_related_to_expansion AND NOT neo4j_disabled │
   │    _expand_from_related_docs()                                 │
   │    Neo4j RELATED_TO edges → fetch chunks from related docs     │
   │    annotate related_to_edge_score / prior / mutual / quality   │
   ├─────────────────────────────────────────────────────────────────┤
   │ 7. RELATED_TO BLENDING                      L1091-1092 → _gp  │
   │    gate: _plan.use_related_to_blending                         │
   │    _blend_related_to_scores()                                  │
   │    per-query-type lambda blend into fused_score                │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 8. ENTITY GRAPH CHANNEL                    L1100-1115 → _gp   │
   │    gate: _plan.use_entity_graph_channel AND NOT neo4j_disabled │
   │    _graph_retrieval_channel()                                  │
   │    ├─ GLiNER entities + precision_anchors → Cypher MATCH       │
   │    ├─ anchor aggregation per-chunk (WITH c, collect)           │
   │    ├─ support threshold: max(anchor_count, entity_count) >= 2  │
   │    └─ hard sparse gate (no fallback)                           │
   │    _merge_graph_channel_candidates()                           │
   │    ├─ overlap: bounded fused_score boost                       │
   │    └─ novel: score ceiling at position 60                      │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 9-11. SIGNAL POOL + ColBERT                       L1157-1320  │
   │                                                                │
   │  ┌── A1 path (pool_before_colbert = true) ──────────────────┐  │
   │  │ 9a.  pre-rerank structural expansion     (NOT precision) │  │
   │  │ 10a. build_signal_pool()                                 │  │
   │  │ 11a. ColBERT reranks ENTIRE pool                         │  │
   │  └──────────────────────────────────────────────────────────┘  │
   │                                                                │
   │  ┌── Legacy path (pool_before_colbert = false) ─────────────┐  │
   │  │ 9b.  ColBERT on truncated candidates                     │  │
   │  │ 10b. pre-rerank structural expansion     (NOT precision) │  │
   │  │ 11b. build_signal_pool() AFTER ColBERT                   │  │
   │  └──────────────────────────────────────────────────────────┘  │
   │                                                                │
   │  Production: A1 path (pool_before_colbert = true)              │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 12. HYDRATE parent_path_norm                           L1333  │
   │     Neo4j heading hierarchy for rerank candidates              │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 13. CROSS-ENCODER RERANKER                 L1331-1343 → _rp   │
   │     gate: _reranker_enabled AND candidates exist               │
   │     _apply_reranker()                                          │
   │     ├─ mxbai-rerank-large-v2 via local-reranker-service        │
   │     ├─ focused rerank text (precision_mode + anchors)          │
   │     └─ batched (32 docs / 4096 tokens per batch)               │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 14. SPECIFICITY ADJUSTMENT                 L1354-1379 → _rp   │
   │     gate: _plan.use_specificity_adjustment                     │
   │           AND precision_mode AND primary_anchors               │
   │           AND NOT has_cloud_cues                                │
   │     anchor bonus (+0.15) / deploy penalty (-0.10)              │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 15. TOP-K SEEDS                                        L1381  │
   │     seeds = ordered_candidates[:top_k]                         │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 16. CITATION HYDRATION                                 L1414  │
   │     _hydrate_missing_citations(seeds)                          │
   │     (if reranker inactive: re-sort by citation_labels)         │
   ├─────────────────────────────────────────────────────────────────┤
   │ 16b. MICRODOC EXPANSION                          L1440-1448   │
   │      gate: microdoc_enabled AND micro_max_neighbors > 0        │
   │      _expand_microdoc_results()                    → _ep       │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 17. EXPANSION GATE                                L1450-1458  │
   │     _should_expand() → (triggered, reason, delta, tokens)      │
   │     gate: ExpandWhen config (AUTO/NEVER/ALWAYS/QUERY_LENGTH)   │
   ├─────────────────────────────────────────────────────────────────┤
   │ 18. BOUNDED ADJACENCY EXPANSION               L1463-1498 → _ep│
   │     gate: triggered AND expansion_enabled AND NOT precision    │
   │     _bounded_expansion() via NEXT_CHUNK edges                  │
   │     sparse gating on expanded neighbors                        │
   ├─────────────────────────────────────────────────────────────────┤
   │ 19. POST-SEED STRUCTURE EXPANSION              L1500-1529 → _ep│
   │     gate: NOT precision_mode (runs regardless of trigger)      │
   │     _expand_with_structure()                                   │
   │     sibling + parent_section + shared_entity expansion         │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 20. DEDUP + COVERAGE                             L1535-1552   │
   │     _dedup_results()                                           │
   │     _hydrate_missing_citations(all_results)                    │
   │     _annotate_coverage()  (Neo4j connection/mention counts)    │
   ├─────────────────────────────────────────────────────────────────┤
   │ 21. GRAPH ENRICHMENT                        L1541-1550 → _gp  │
   │     _apply_graph_enrichment()                                  │
   │     gate: _plan.use_graph_enrichment (currently OFF)           │
   │     (called unconditionally; returns empty when disabled)      │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 22. DOC CONTINUITY BOOST                      L1591 → _fp     │
   │     _apply_doc_continuity_boost(alpha=0.12)                    │
   ├─────────────────────────────────────────────────────────────────┤
   │ 23. CONTEXT BUDGET                            L1596-1600 → _ep│
   │     _enforce_context_budget(primaries)                         │
   │     _enforce_context_budget(extras, starting_tokens=...)       │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 24. RETURN                                             L1702  │
   │     final_results = primaries + extras                         │
   │     return final_results, metrics                              │
   └─────────────────────────────────────────────────────────────────┘


   ═══════════════════════════════════════════════════════════════════
    CURRENT PRODUCTION STATE (graph_assisted profile)
   ═══════════════════════════════════════════════════════════════════

    ENABLED                          │  DISABLED / NO-OP
    ─────────────────────────────────┼──────────────────────────────
    Qdrant dense + sparse retrieval  │  BM25 (no fulltext index)
    Weighted query API path          │  Graph enrichment (plan OFF)
    RRF fusion                       │  Graph score override
    Entity boost                     │  Microdoc extras
    Structural boost                 │  Graph-as-reranker
    RELATED_TO expansion + blending  │
    Entity graph channel             │  OFTEN NO-OP (precision_mode)
    Signal pool (A1, before ColBERT) │  ───────────────────────────
    ColBERT late-interaction         │  Pre-rerank structural exp.
    Cross-encoder reranker           │  Bounded adjacency expansion
    Focused rerank text              │  Post-seed structure exp.
    Specificity adjustment           │
    Structure-aware expansion logic  │

  Corrections from the original claim

  1. Microdoc expansion was missing — it sits between citation hydration (step 16) and the expansion gate (step 17). Currently disabled in production.
  2. Signal pool has two orderings — the A1 path (production) runs pool → ColBERT. The legacy path runs ColBERT → pool. The claim only described A1.
  3. Graph enrichment gating — _apply_graph_enrichment() is called unconditionally at the orchestrator level; the disable check is inside the delegated function via _plan.use_graph_enrichment.
  4. Post-seed structure expansion runs regardless of the expansion gate trigger — it has its own NOT precision_mode gate independent of _should_expand().
