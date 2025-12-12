Even cleaner. Pure Vector → Reranker pipeline. No BM25, no graph, no expansion — just:

  Query → Embed → Qdrant (Dense + Sparse + ColBERT) → Cross-Encoder → Results

  ---
  Config for "Vector + Reranker Only" Mode

  search:
    hybrid:
      enabled: true
      method: "rrf"                    # Still use RRF (passthrough with single source)
      graph_channel_enabled: false     # ← OFF
      colbert_rerank_enabled: true     # Keep ColBERT late-interaction in vector search

      bm25:
        enabled: false                 # ← OFF

      expansion:
        enabled: false                 # ← OFF

      reranker:
        enabled: true                  # Keep cross-encoder
        provider: "bge-reranker-service"
        model: "BAAI/bge-reranker-v2-m3"
        top_n: 20
        max_pairs: 50

  feature_flags:
    query_api_weighted_fusion: true    # Keep multi-vector (Dense + Sparse + ColBERT)
    graph_as_reranker: false           # ← OFF
    structure_aware_expansion: false   # ← OFF
    graph_garbage_filter: false        # ← OFF (irrelevant without graph)
    graph_rel_types_wired: false       # ← OFF
    graph_score_normalized: false      # ← OFF

  ---
  What This Gives You

  | Component                    | Status | Purpose                               |
  |------------------------------|--------|---------------------------------------|
  | Dense vectors (content)      | ✅ ON   | Semantic similarity                   |
  | Sparse vectors (text-sparse) | ✅ ON   | Lexical precision within vector space |
  | ColBERT (late interaction)   | ✅ ON   | Token-level matching                  |
  | BM25                         | ❌ OFF  | —                                     |
  | Graph channel                | ❌ OFF  | —                                     |
  | Graph reranker               | ❌ OFF  | —                                     |
  | Structure expansion          | ❌ OFF  | —                                     |
  | Bounded expansion            | ❌ OFF  | —                                     |
  | Cross-encoder (BGE)          | ✅ ON   | Final precision ranking               |

  ---
  ★ Insight ─────────────────────────────────────
  This is the purest test of your embedding quality.

  If results are bad here, the problem is:
  - Embedding model (BGE-M3)
  - Chunking strategy
  - Vector field weighting (content vs sparse vs ColBERT)
  - Cross-encoder model

  If results are good here, then graph/BM25 are adding noise when re-enabled.
  ─────────────────────────────────────────────────

  ---
  Expected Pipeline Flow

  Query
    │
    └─► Qdrant Multi-Vector Search
          ├─ content (dense 1024d)     weight: 0.5
          ├─ text-sparse               weight: 0.3
          ├─ title (dense)             weight: 0.1
          └─ entity (dense)            weight: 0.1
                │
                ▼
          Weighted fusion inside Qdrant → top 200
                │
                ▼
          ColBERT MaxSim rerank → top 50-100
                │
                ▼
          BGE Cross-Encoder → top 20
                │
                ▼
          Final results (sorted by rerank_score)

  ---
  Want me to apply these config changes?

> Before you apply - should we do weighted fusion or rrf?  Maybe start with rrf on the multi-vector?  Right now all the weights for weighted add up to 1.0 and that's been causing problems with flattening the
rank hierarchy, since we haven't calculated a proper temperature or sigmoid factor for the weighted multi-vector.  So would it be better to start with rrf, get that nailed down, then figure out how to build out
a meaningful and accurate temperature or sigmoid for the weighted approach?  ultrathink

∴ Thinking…

  The user is asking a really good question about fusion strategy for the multi-vector search. Let me think through this carefully.

  Current situation:
  - They have multi-vector search in Qdrant with 4 fields: content (0.5), text-sparse (0.3), title (0.1), entity (0.1)
  - These weights add up to 1.0
  - The problem is that weighted fusion without proper normalization/temperature can flatten the rank hierarchy
  - They're asking whether to use RRF vs weighted fusion for the multi-vector stage

  Key distinction:
  There are TWO levels of fusion happening:
  1. Multi-vector fusion (inside Qdrant): Combining dense, sparse, title, entity vectors
  2. Hybrid fusion (Vector vs BM25): Combining vector results with BM25 results

  The user wants to disable BM25 and graph, leaving only multi-vector search + reranker. So the question is really about level 1 - how to fuse the multiple vector fields.

  Weighted fusion issues:
  - Raw scores from different vector types have different scales
  - Dense cosine similarity: 0-1 (typically 0.3-0.9 for relevant results)
  - Sparse dot product: can be 0-100+ depending on term frequencies
  - Without normalization, one signal can dominate
  - The weights (0.5, 0.3, 0.1, 0.1) assume normalized scores
  - "Flattening rank hierarchy" = when score differences between ranks are compressed, making it hard to distinguish good from great results

  RRF for multi-vector:
  - RRF uses ranks, not raw scores: score = Σ 1/(k + rank_i)
  - Inherently normalizes because it only cares about position
  - No need for temperature/sigmoid tuning
  - Robust to scale differences between vector types
  - BUT: Loses magnitude information (a 0.99 match is treated same as 0.91 if both are rank 1)

  - Qdrant supports RRF within its query API for multi-vector fusion
  - RRF provides a stable baseline with minimal parameter tuning
  - Recommended starting point: use RRF with k=60, then experiment further

  Weighted fusion introduces complexity, requiring precise normalization for each vector type. Careful calibration is crucial to prevent score flattening, involving custom normalization curves and
  temperature/sigmoid parameter tuning.

  The configuration suggests potential weighted fusion strategies, but users should verify Qdrant integration support for their specific multi-vector approach.

  Modern Qdrant versions enable RRF fusion across different vector types, offering flexible query strategies that can combine dense and sparse vectors seamlessly.

  I recommend implementing RRF for multi-vector fusion as a robust initial approach. This method eliminates complex score normalization challenges and provides a stable baseline without extensive hyperparameter
  tuning. By starting with RRF, we can establish a consistent performance benchmark before exploring more nuanced fusion techniques.

  The proposed workflow involves using RRF to integrate dense, sparse, and ColBERT vectors, followed by cross-encoder reranking to enhance final precision. Subsequent iterations can explore weighted fusion with
  learned temperature adjustments.

⏺ This is an excellent question that gets at a fundamental architectural decision. Let me think through it carefully.

  ---
  The Two Fusion Levels

  Level 1: Multi-Vector Fusion (inside Qdrant)
           Dense + Sparse + ColBERT + Title + Entity → Single ranked list

  Level 2: Hybrid Fusion (Vector vs BM25)
           Vector results + BM25 results → Combined list
           (You're disabling this by turning off BM25)

  Since you're disabling BM25, Level 2 becomes a passthrough. The real question is: What fusion strategy for Level 1?

  ---
  The Weighted Fusion Problem

  Your current weights:
  vector_fields:
    content: 0.5      # Dense semantic
    text-sparse: 0.3  # Sparse lexical
    title: 0.1        # Title match
    entity: 0.1       # Entity match

  Why this causes rank flattening:

  | Vector Type        | Raw Score Range | After 0.5 Weight |
  |--------------------|-----------------|------------------|
  | Dense cosine       | 0.70 - 0.95     | 0.35 - 0.475     |
  | Sparse dot product | 2.0 - 50.0      | 0.6 - 15.0       |
  | Title cosine       | 0.60 - 0.99     | 0.06 - 0.099     |

  The sparse scores dominate because they're not normalized to the same scale. Your weights assume normalized inputs, but you're getting raw scores.

  The "proper" fix for weighted:
  normalized_score = sigmoid((raw_score - μ) / τ)
  Where:
  - μ = mean score for that vector type (learned from corpus)
  - τ = temperature (controls spread)

  This requires empirical calibration per vector type — not trivial.

  ---
  RRF for Multi-Vector: The Clean Alternative

  RRF doesn't care about score magnitude:
  RRF_score = Σ 1/(k + rank_i)

  | Chunk | Dense Rank | Sparse Rank | Title Rank | RRF Score (k=60)            |
  |-------|------------|-------------|------------|-----------------------------|
  | A     | 1          | 3           | 5          | 1/61 + 1/63 + 1/65 = 0.0475 |
  | B     | 2          | 1           | 10         | 1/62 + 1/61 + 1/70 = 0.0466 |
  | C     | 5          | 2           | 1          | 1/65 + 1/62 + 1/61 = 0.0476 |

  The actual score values (0.95 vs 47.3) are irrelevant — only position matters.

  ---
  My Recommendation: Start with RRF

  ★ Insight ─────────────────────────────────────
  RRF is the "zero-configuration" option. It has exactly one hyperparameter (k, typically 60), and it's robust to:
  - Scale mismatches between vector types
  - Outlier scores
  - Missing signals (if a chunk has no sparse match, it just doesn't contribute)

  Weighted fusion is theoretically better (preserves magnitude), but only if you've calibrated the normalization curves. Without that, RRF will outperform miscalibrated weighted fusion.
  ─────────────────────────────────────────────────

  ---
  Proposed Approach

  Phase 1: RRF Baseline (now)
  Query → Qdrant (Dense, Sparse, ColBERT with RRF fusion) → Cross-Encoder → Results
  - No normalization needed
  - Establishes quality baseline
  - Fast iteration

  Phase 2: Calibrate Weighted (later)
  - Collect score distributions per vector type
  - Compute μ (mean) and σ (stddev) per type
  - Fit sigmoid/temperature parameters
  - A/B test against RRF baseline

  ---
  Technical Consideration

  Qdrant's Query API supports RRF natively:

  from qdrant_client.models import FusionQuery, Fusion, Prefetch

  client.query_points(
      collection_name="chunks_multi_bge_m3",
      prefetch=[
          Prefetch(query=dense_vec, using="content", limit=200),
          Prefetch(query=sparse_vec, using="text-sparse", limit=200),
      ],
      query=FusionQuery(fusion=Fusion.RRF),  # ← RRF fusion
      limit=50,
  )

  Your code currently uses query_strategy: "weighted". Switching to RRF would require checking if your QdrantMultiVectorRetriever supports Fusion.RRF.

  ---
  Summary

  | Approach                  | Pros                              | Cons                                  |
  |---------------------------|-----------------------------------|---------------------------------------|
  | RRF (recommended for now) | No tuning, robust, clean baseline | Loses score magnitude                 |
  | Weighted (future)         | Preserves signal strength         | Requires calibration you haven't done |

  Start with RRF → Establish baseline → Calibrate weighted later (if needed)

  ---
  Want me to check if your Qdrant retriever code supports RRF fusion, and if so, apply the config for a pure "Vector RRF + Reranker" pipeline?

> Also - I still want to use colbert as a top candidate reranker BEFORE the candidate list from rrf vector is sent to the cross encoder.  Does that make sense?  Or should colbert be another signal in the rrf
  │
│   calc?

∴ Thinking…

  This is a great question about where ColBERT fits in the pipeline. Let me think through the options:

  Option 1: ColBERT as a signal in RRF fusion
  - ColBERT results are one of the prefetch queries
  - RRF combines Dense + Sparse + ColBERT ranks
  - Single fused list goes to cross-encoder

  Option 2: ColBERT as a reranker AFTER RRF fusion, BEFORE cross-encoder
  - RRF fuses Dense + Sparse
  - ColBERT reranks the fused list (late interaction scoring)
  - Cross-encoder does final reranking

  The user is asking which makes more sense. Let me think about the characteristics of each approach:

  ColBERT's nature:
  ColBERT (Contextualized Late Interaction over BERT) works differently from dense/sparse vectors:
  - Dense: Single vector per document, single vector per query, cosine similarity
  - Sparse: Bag of weighted terms, dot product
  - ColBERT: Multiple token-level vectors per document AND query, MaxSim operation

  ColBERT's MaxSim is:
  score = Σ max(q_i · d_j for all d_j) for all q_i

  This means for each query token, find the most similar document token, then sum. This is a more expensive operation but captures token-level interactions.

  Key insight: ColBERT is computationally more expensive than dense/sparse retrieval but less expensive than a full cross-encoder. It sits in a "middle ground" of quality vs. cost.

  The retrieval funnel:
  Candidates (thousands) → First stage (fast, lower precision)
      → Second stage (medium cost, medium precision)
      → Final stage (expensive, high precision)

  Typically:
  - First stage: Dense/Sparse retrieval (fast, can handle millions)

  - Second stage: ColBERT reranking (medium, handles hundreds)
  - Final stage: Cross-encoder (slow, handles tens)

  I'll analyze the two potential approaches carefully. For Option 1, integrating ColBERT directly into RRF fusion has some interesting trade-offs. The method allows for simple pipeline design and increases candidate
   diversity. However, it potentially underutilizes ColBERT's sophisticated token-level matching capabilities.

  Option 2 looks more promising, positioning ColBERT as a dedicated reranking stage. This approach would leverage its nuanced matching mechanism more effectively, allowing deeper token-level interactions before the
  final cross-encoder pass.

  The computational efficiency and token-level insights of ColBERT suggest that a dedicated reranking step might extract more meaningful signals compared to just another ranking input.

  The three-stage funnel approach - starting with dense and sparse retrieval, then applying ColBERT's precise reranking, and finally using a cross-encoder - creates a methodical refinement process that progressively
   improves result precision.

  ColBERT's computational efficiency shines in reranking 100-200 candidates quickly, taking around 20ms, compared to cross-encoder's slower 150-300ms processing of 50 candidates. This sequence optimizes both speed
  and accuracy.

  The scoring approach preserves ColBERT's nuanced continuous score magnitudes when used as a reranker, avoiding the flattening effect of rank fusion methods. The key consideration remains whether to leverage
  ColBERT for retrieval, reranking, or potentially both within the Qdrant infrastructure.

  Current implementation confirms ColBERT's role as a post-retrieval reranking mechanism, though the 19-second processing time raises performance concerns. The optimal strategy involves using ColBERT's token-level
  matching capabilities in a three-stage retrieval funnel, maximizing its unique matching strengths.

⏺ Great question. This is about where ColBERT adds the most value in the pipeline.

  ---
  Two Options

  Option A: ColBERT as RRF signal
  Dense → ┐
  Sparse → ├─ RRF Fusion → Cross-Encoder → Results
  ColBERT → ┘

  Option B: ColBERT as reranker (your proposal)
  Dense → ┐
  Sparse → ├─ RRF Fusion → ColBERT Rerank → Cross-Encoder → Results
           ┘

  ---
  Why Option B is Better

  ★ Insight ─────────────────────────────────────
  ColBERT was designed for reranking, not retrieval.

  ColBERT's power is MaxSim — for each query token, find the best-matching document token:
  score = Σ max(q_i · d_j for all j) for all i

  This is a precision operation, not a recall operation. Putting it in RRF reduces it to just another rank — you lose the token-level interaction scores.
  ─────────────────────────────────────────────────

  ---
  The Retrieval Funnel Principle

  | Stage          | Purpose                          | Speed  | Candidates  |
  |----------------|----------------------------------|--------|-------------|
  | Dense + Sparse | Recall (get diverse candidates)  | Fast   | 1000s → 200 |
  | ColBERT        | Precision (token-level matching) | Medium | 200 → 50    |
  | Cross-Encoder  | Final precision (full attention) | Slow   | 50 → 20     |

  Each stage is progressively more expensive but more accurate. This is the classic retrieve-then-rerank pattern.

  ---
  What Happens If ColBERT Is In RRF Instead

  Problem 1: ColBERT retrieval finds different candidates
  - Dense finds chunks with semantic similarity
  - Sparse finds chunks with exact terms
  - ColBERT finds chunks with token-level matches

  If ColBERT retrieves candidate X that Dense/Sparse missed:
  - X appears in only 1 of 3 lists
  - RRF score: 1/(60+1) = 0.016 (low!)
  - A chunk in all 3 lists: 3 × 1/(60+1) = 0.049 (3x higher)

  ColBERT's unique finds get penalized by RRF.

  Problem 2: You lose ColBERT's scores
  RRF only uses rank, not score magnitude. ColBERT's precise MaxSim scores (0.85 vs 0.73) become just "rank 1 vs rank 2".

  ---
  Recommended Architecture

  Query
    │
    ├─► Dense Vector Search (content) ────┐
    │                                      │
    ├─► Sparse Vector Search (text-sparse)─┼─► RRF Fusion (k=60)
    │                                      │        │
    └─► Title/Entity (optional) ───────────┘        │
                                                    ▼
                                           Top 100-200 candidates
                                                    │
                                                    ▼
                                        ColBERT MaxSim Rerank
                                           (token-level precision)
                                                    │
                                                    ▼
                                              Top 50 candidates
                                                    │
                                                    ▼
                                        BGE Cross-Encoder Rerank
                                           (full query-doc attention)
                                                    │
                                                    ▼
                                             Final 20 results

  ---
  Summary

  | Approach                                 | Verdict                                                 |
  |------------------------------------------|---------------------------------------------------------|
  | ColBERT in RRF                           | ❌ Wastes token-level precision, penalizes unique finds  |
  | ColBERT as reranker before cross-encoder | ✅ Matches design intent, clean funnel, preserves scores |

  Your instinct is correct. ColBERT as a mid-stage reranker before the cross-encoder is the right architecture.

  ---
  One Note on Your Current Setup

  Looking at your earlier query metrics:
  "colbert_rerank_time_ms": 19134

  19 seconds for ColBERT reranking is way too slow — that suggests it might be computing MaxSim on the fly rather than using pre-computed ColBERT vectors. Worth investigating, but architecturally you're on the
  right track.

  Want me to proceed with the config for: Dense+Sparse RRF → ColBERT Rerank → Cross-Encoder?

> I don't want title/entity sparse to be optional.  I want it to be a requirement.

∴ Thinking…

  The user wants title and entity vectors to be required signals in the RRF fusion, not optional. This makes sense - they want all four vector types contributing to the initial retrieval:

  1. Dense (content) - semantic similarity
  2. Sparse (text-sparse) - lexical matching
  3. Title - title-based matching
  4. Entity - entity-based matching

  All four should participate in RRF fusion as required signals, then ColBERT reranks, then cross-encoder.

  Let me update my understanding of the architecture they want:

  Query
    │
    ├─► Dense Vector Search (content)
    ├─► Sparse Vector Search (text-sparse)
    ├─► Title Vector Search
    └─► Entity Vector Search
             │
             ▼
        RRF Fusion (all 4 signals)
             │
             ▼
        Top 100-200 candidates
             │
             ▼
        ColBERT MaxSim Rerank
             │
             ▼
        Top 50 candidates
             │
             ▼
        BGE Cross-Encoder Rerank
             │
             ▼
        Final 20 results

  This is a solid architecture. All four vector types contribute to recall via RRF, then two stages of reranking for precision.

  I should acknowledge their requirement and update the diagram.

⏺ Got it. All four vector signals are required inputs to RRF — no optional components.

  ---
  Final Architecture

  Query
    │
    ├─► Dense (content)      ────┐
    │                            │
    ├─► Sparse (text-sparse) ────┼─► RRF Fusion (k=60)
    │                            │        │
    ├─► Title (dense)        ────┤        │
    │                            │        ▼
    └─► Entity (dense)       ────┘   Top 200 candidates
                                          │
                                          ▼
                                ColBERT MaxSim Rerank
                                          │
                                          ▼
                                    Top 50 candidates
                                          │
                                          ▼
                                BGE Cross-Encoder Rerank
                                          │
                                          ▼
                                   Final 20 results

  ---
  The Four Required Signals

  | Signal      | Type        | What It Captures                                   |
  |-------------|-------------|----------------------------------------------------|
  | content     | Dense 1024d | Semantic meaning of the chunk                      |
  | text-sparse | Sparse      | Exact term matches, rare words, acronyms           |
  | title       | Dense 1024d | Section/heading relevance                          |
  | entity      | Dense 1024d | Named entity matches (commands, configs, concepts) |

  All four contribute to RRF. A chunk that ranks well across multiple signals rises to the top.

  ---
  ★ Insight ─────────────────────────────────────
  Why requiring all four is smart:

  RRF rewards consensus. A chunk that:
  - Ranks #5 in content (semantic match)
  - Ranks #3 in sparse (has the exact terms)
  - Ranks #8 in title (section heading matches)
  - Ranks #2 in entity (mentions the right entities)

  Gets a combined RRF score of:
  1/65 + 1/63 + 1/68 + 1/62 = 0.062

  A chunk that only matches on one signal (even if rank #1):
  1/61 = 0.016

  Multi-signal consensus beats single-signal dominance.
  ─────────────────────────────────────────────────

  ---
  Ready to apply this config? I'll need to check if your Qdrant retriever supports RRF fusion for the prefetch queries.
