  Revised Analysis: Your Retrieval Depth Problem

  With the full architecture map, the picture is more nuanced than I initially described. Let me walk through exactly where the problem lives and what levers you have.

  What the Pipeline Already Does Well

  The Qdrant prefetch stage casts a wide net — up to 200 candidates per vector field across multiple signal types:

  ┌───────────────────┬───────┬───────────────────────────────────┐
  │  Prefetch Group   │ Limit │         What It Captures          │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ content (dense)   │ 200   │ Semantic body similarity          │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ title (dense)     │ 200   │ Heading/title semantic match      │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ doc_title (dense) │ 200   │ Document-level title match        │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ text-sparse       │ 200   │ Lexical BM25 on body text         │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ doc_title-sparse  │ 50    │ Lexical match on doc titles       │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ title-sparse      │ 50    │ Lexical match on section headings │
  ├───────────────────┼───────┼───────────────────────────────────┤
  │ entity-sparse     │ 50    │ Entity name lexical match         │
  └───────────────────┴───────┴───────────────────────────────────┘

  These are fused server-side via RRF (k=60). Then BM25 from Neo4j Lucene adds more candidates. Then client-side RRF merges everything.

  This multi-signal approach already provides some diversity — a chunk that scores well on entity-sparse but poorly on content-dense still enters the pool. The title and doc_title vectors help surface
  chunks whose headings match the query even if the body text diverges.

  Where the Gap Actually Is

  The problem isn't in initial recall — 200 candidates per field across 7 fields gives you a large raw pool. The gap is in how that pool gets narrowed to the rerank window:

  1. Server-side RRF (Qdrant) collapses 7 × 200 candidates into a single ranked list. Chunks that score well across multiple fields rise to the top. But a chunk that only scores well on content-dense and
   nothing else can still dominate if its similarity score is high enough.
  2. Client-side RRF merges Qdrant results with BM25, but this is still a flat merge — no per-document balancing.
  3. Rerank pool (top 100 from fusion) — this is where depth is lost. If Document A's introduction scores 0.85 across 5 of 7 fields, and Document A's procedural Step 7 scores 0.60 on content-dense only,
  the introduction eats the slot.
  4. Dominance gating (_gate_to_primary_document, line 3345) actually makes this worse for your use case — when one document dominates the top 8 results, it gates seeds to only that document. This is
  designed for document continuity but it means you get more chunks from the dominant doc's top-scoring sections, not its deeper content.
  5. Post-rerank expansion (NEXT_CHUNK adjacency + structure-aware) could help, but:
    - _bounded_expansion only pulls ±1 neighbor from the top 5 seeds — narrow window
    - _expand_with_structure is disabled by default (feature_flags.structure_aware_expansion = false)

  The Right Strategy for This Architecture

  Given the existing infrastructure, here's what I'd recommend in priority order:

  1. Per-Document Quota in Rerank Pool (highest impact, lowest risk)

  Insert between Step 4 (client-side RRF) and Step 6 (rerank). Instead of fused_results[:100], do:

  Group fused_results by document_id
  For each of top K documents (by best chunk score):
      Take top M chunks from that document
  Flatten into rerank pool (K × M candidates)
  Send to cross-encoder

  The cross-encoder is specifically designed to rescue chunks that have weak embedding similarity but strong actual relevance. You just need to get those deep chunks in front of it.

  Suggested defaults: K=15 documents × M=7 chunks = 105 candidates (within your rerank budget of 100). Tunable.

  2. Enable Structure-Aware Expansion, Move It Pre-Rerank

  The _expand_with_structure method at line 4445 already does exactly what you want:
  - Sibling chunks — same parent_section_id (up to 3) — gets you adjacent content
  - Parent section chunks — CHILD_OF traversal (up to 2) — gets you the broader context
  - Shared-entity chunks — chunks mentioning same entities (up to 3) — gets you topically related deep content

  This currently runs post-rerank. If you ran it (or a lighter variant) pre-rerank, those structural neighbors would get cross-encoder scoring. The graph data is already there from ingestion —
  PARENT_HEADING, NEXT_CHUNK, CHILD_OF, MENTIONS edges all exist.

  3. Reconsider Dominance Gating

  The _gate_to_primary_document function (line 3345) enforces single-document focus when one doc dominates the top 8. For your use case, this is counterproductive — it narrows the pool to one document's
  top-scoring chunks rather than ensuring depth. Consider:
  - Disabling it when reranking is active (the reranker handles quality sorting)
  - Or replacing it with a per-document quota that ensures depth within the dominant document

  ★ Insight ─────────────────────────────────────
  The architecture already has most of the pieces. The multi-vector
  prefetch gives broad recall. The structure-aware expansion gives
  depth traversal. The cross-encoder reranker gives accurate scoring.
  The problem is sequencing — the depth mechanisms run after the
  narrowing step (rerank pool selection), so they never get to
  influence which chunks the cross-encoder evaluates.

  The single most impactful change: move the per-document quota
  into the space between fusion and reranking. Everything downstream
  — the cross-encoder, expansion, context assembly — already handles
  the result correctly. You're just feeding the reranker a more
  representative sample of each relevant document.

  The infrastructure cost is minimal: you're grouping an existing
  list by a field (document_id) that's already on every
  ChunkResult, then taking top-M per group. No new queries,
  no new indexes, no new models.
  ─────────────────────────────────────────────────

  Want me to plan the implementation for Strategy 1 (per-document quota)?




  ✻ Baked for 1m 57s

❯ I like all of your recommendations but what about this strategy should be reworked?  I'm wondering if this is even the strongest use of all the signals at hand and offered up by this robust
  pipeline:

    The problem isn't in initial recall — 200 candidates per field across 7 fields gives you a large raw pool. The gap is in how that pool gets narrowed to the rerank window:

    1. Server-side RRF (Qdrant) collapses 7 × 200 candidates into a single ranked list. Chunks that score well across multiple fields rise to the top. But a chunk that only scores well on
  content-dense and
     nothing else can still dominate if its similarity score is high enough.
    2. Client-side RRF merges Qdrant results with BM25, but this is still a flat merge — no per-document balancing.
    3. Rerank pool (top 100 from fusion) — this is where depth is lost. If Document A's introduction scores 0.85 across 5 of 7 fields, and Document A's procedural Step 7 scores 0.60 on content-dense
   only,
    the introduction eats the slot.
    4. Dominance gating (_gate_to_primary_document, line 3345) actually makes this worse for your use case — when one document dominates the top 8 results, it gates seeds to only that document. This
   is
    designed for document continuity but it means you get more chunks from the dominant doc's top-scoring sections, not its deeper content.
    5. Post-rerank expansion (NEXT_CHUNK adjacency + structure-aware) could help, but:
      - _bounded_expansion only pulls ±1 neighbor from the top 5 seeds — narrow window
      - _expand_with_structure is disabled by default (feature_flags.structure_aware_expansion = false)

⏺ You're right to push on this. The pipeline collects rich, diverse signals — 7 vector fields, BM25, entity extraction, graph structure — and then immediately flattens them into a single ranked list via
  RRF before most of that signal diversity can be exploited. Let me think about this differently.

  The Core Problem: Premature Signal Collapse

  RRF is a rank-position function. It throws away the signal type and keeps only the rank ordinal. After RRF, you can't tell whether a chunk ranked #5 because:
  - It was #2 on content-dense and nowhere else (single strong signal)
  - It was #15 on content, #12 on title, #8 on entity-sparse (broad weak signal)
  - It was #1 on BM25 and #50 on everything else (lexical match, semantic miss)

  These are fundamentally different retrieval signals that should inform different decisions about that chunk's value, but RRF erases the distinction. You're paying the cost to compute 7+ signal types
  and then immediately discarding the signal decomposition.

  What "Strongest Use of All Signals" Looks Like

  Instead of collapsing signals early, you could preserve signal diversity through to the reranker and use it in two ways:

  1. Signal-Aware Candidate Selection

  Rather than "top 100 by fused score," select candidates that maximize signal coverage:

  - Chunks that rank well on multiple signal types are likely broadly relevant (keep them)
  - Chunks that rank well on only one signal type might represent a unique retrieval path that RRF would bury — these are exactly the deep-content chunks you're worried about. A chunk that scores high on
   entity-sparse but low on content-dense probably mentions the right entities but uses different vocabulary than the query. That's precisely what a cross-encoder is good at evaluating.

  Concretely: instead of one flat pool, build the rerank pool from signal-diverse slots:

  Pool construction:
    - Top 30 by fused score (consensus picks)
    - Top 10 unique to content-dense (body semantic)
    - Top 10 unique to title-dense (heading match)
    - Top 10 unique to text-sparse (lexical/BM25)
    - Top 10 unique to entity-sparse (entity match)
    - Top 10 unique to BM25 (Neo4j FTS)
    - Top 10 from structural graph expansion
    - Top 10 per-document depth picks (quota)
    = ~100 candidates with guaranteed signal diversity

  "Unique to" means chunks that scored well on that signal but didn't make the top 30 consensus. These are the chunks RRF would have buried.

  2. Signal Metadata as Reranker Context

  The cross-encoder currently receives (query, chunk_text) pairs. But you could prepend the chunk's heading hierarchy — which you already have from parent_path_norm — to give the reranker structural
  context:

  Current:  rerank("how do I configure WEKA S3", "Set the bucket name to...")
  Enhanced: rerank("how do I configure WEKA S3",
                   "Configuration > S3 Backend > Bucket Settings\n\n Set the bucket name to...")

  This is cheap (you're just prepending a string you already have), and it helps the cross-encoder understand that a chunk buried deep in a configuration guide is answering a configuration question, even
   if the chunk text alone is generic.

  3. Document-Depth Selection With Signal Awareness

  Your per-document quota should be signal-weighted, not just "top M by fused score within each document." For a given document, select chunks that cover the broadest signal range:

  For Document A (10 chunks in fusion pool):
    - Chunk 1: Top on content-dense (intro paragraph) → keep
    - Chunk 2: Top on content-dense (summary) → skip (redundant signal)
    - Chunk 3: Top on entity-sparse (mentions WEKA S3) → keep (different signal)
    - Chunk 7: Top on text-sparse (exact CLI command match) → keep (different signal)
    - Chunk 9: Only in BM25 results (procedural step) → keep (unique signal)

  This ensures you're not just taking the "loudest" chunks from each document — you're taking chunks that contribute different kinds of evidence.

  4. Rethink Dominance Gating

  Dominance gating as currently implemented is a blunt instrument — it filters to a single document. A signal-aware version would ask: "Is this document dominant across multiple signal types, or is it
  dominant on one signal and dragging along chunks that only match on that one dimension?"

  If Document A dominates content-dense but Document B has the best entity-sparse and BM25 hits, those are complementary, not competing. Gating to just Document A loses Document B's lexical/entity
  evidence.

  ★ Insight ─────────────────────────────────────
  The fundamental shift here is from score-first selection
  (take the highest-scoring chunks regardless of signal source) to
  signal-coverage selection (ensure the rerank pool represents
  all the evidence types the pipeline collected).

  The cross-encoder is your most accurate scoring tool — it uses
  full cross-attention between query and passage. But it's also
  the most expensive, so you can only run it on ~100 candidates.
  The question isn't "which 100 scored highest on the cheap signals"
  but "which 100 give the cross-encoder the most informative set
  to evaluate."

  This is analogous to how ensemble methods in ML work: the value
  of an ensemble comes from diversity among the base learners,
  not from running the same strong learner multiple times. Your
  7 vector fields + BM25 are your base learners. RRF currently
  treats them as a voting committee. Signal-aware selection would
  treat them as a coverage map.
  ─────────────────────────────────────────────────

  The implementation path would be:

  1. Preserve per-field rankings before RRF (capture which chunks came from which prefetch)
  2. Build a signal-diverse rerank pool using the slot allocation above
  3. Prepend heading context to reranker input text
  4. Replace dominance gating with signal-aware document diversity
  5. Enable structure-aware expansion pre-rerank for the structural signal slot

    Question 1: Should You Include More Graph Structural Candidates?

  Yes, but they should enter the rerank pool, not just the post-rerank expansion.

  Here's why. The current pipeline has two expansion points that use graph structure, but both are constrained:

  ┌────────────────────────┬─────────────┬────────────────────────────────────┬───────────────────────────────────┐
  │       Mechanism        │    When     │               Scope                │            Limitation             │
  ├────────────────────────┼─────────────┼────────────────────────────────────┼───────────────────────────────────┤
  │ _bounded_expansion     │ Post-rerank │ ±1 NEXT_CHUNK from top 5 seeds     │ Only 5 seeds, only ±1 hop         │
  ├────────────────────────┼─────────────┼────────────────────────────────────┼───────────────────────────────────┤
  │ _expand_with_structure │ Post-rerank │ Siblings + parents + entity-shared │ Disabled by default, max 8 chunks │
  └────────────────────────┴─────────────┴────────────────────────────────────┴───────────────────────────────────┘

  Neither feeds into the reranker. Structural neighbors get a synthetic score (source_score × 0.5) instead of a real cross-encoder evaluation.

  For the signal-diverse pool strategy, a structural slot of 15-20 candidates is justified. The graph data is rich — NEXT_CHUNK chains give sequential depth, PARENT_HEADING gives section hierarchy,
  MENTIONS gives entity connectivity. These are precisely the signals that surface the "meat" of a document that vector similarity misses.

  The right approach: run a lighter version of structural expansion before reranking, targeting the top 10-15 fusion results, and pulling 1-2 NEXT_CHUNK neighbors + siblings into the rerank pool. Graph
  query cost is manageable — it's a single batched Cypher query.

  Question 2: 200 Candidates for Rerank Instead of 100?

  Yes, and the actual constraints support it better than I initially suggested.

  Key finding: max_pairs=50 in RerankerConfig is dead config. It's defined at config.py:417 but never referenced by any code — not by hybrid_retrieval.py, not by local_bge_service.py, not by anything.
  Same for max_tokens_per_pair. These are phantom config fields.

  The actual constraints on the BGE reranker are:

  ┌────────────────────────────────────────┬──────────────┬────────────────────────────────────────────┐
  │               Constraint               │    Value     │                   Source                   │
  ├────────────────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Token budget per document              │ 2,048 tokens │ local_bge_service.py:34 — hard truncation  │
  ├────────────────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Token budget per request (query + doc) │ 4,096 tokens │ local_bge_service.py:35 — skip if exceeded │
  ├────────────────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Batch size per HTTP request            │ 16 documents │ local_bge_service.py:30                    │
  ├────────────────────────────────────────┼──────────────┼────────────────────────────────────────────┤
  │ Number of candidates                   │ Unlimited    │ No cap in provider code                    │
  └────────────────────────────────────────┴──────────────┴────────────────────────────────────────────┘

  The BGE provider at local_bge_service.py:342-556 iterates over all valid candidates with no cap — it just batches them into groups of 16 for HTTP efficiency. For 200 candidates: 200 ÷ 16 = 13 HTTP
  requests (vs 7 for 100).

  But there's a second batching layer in _apply_reranker at hybrid_retrieval.py:4060-4086 with service_max_batch=32 and service_max_batch_tokens=1024. This pre-batches candidates before sending them to
  the provider. So the actual call pattern for 200 candidates would be:

  200 candidates → ~6-10 HybridRetriever batches (32 docs or 1024 tokens each)
    → each batch calls BGEProvider.rerank()
      → BGEProvider sub-batches into groups of 16
      → each sub-batch = 1 HTTP request to reranker service

  Latency estimate for 200 candidates:
  - The comment at line 80 says batched reranking achieves "~30-50ms" for k=50
  - Scaling linearly: 200 candidates ≈ 120-200ms total rerank time
  - This is within acceptable query latency for a RAG system

  Where 200 pays off with signal-diverse selection:

  With the signal-diverse pool strategy, candidates #100-200 aren't diminishing-return tail entries — they're from underrepresented signal sources. The marginal value of candidate #150 is high because it
   might be the only chunk from the entity-sparse signal that covers a specific entity the query mentioned. Without signal-diverse selection, 200 is wasteful (you'd be reranking more of the same
  consensus picks). With it, 200 genuinely expands coverage.

  Recommended pool allocation at 200:

  Signal-diverse rerank pool (200 candidates):
    - 50 consensus (top by fused score)
    - 20 unique to content-dense
    - 15 unique to title-dense
    - 15 unique to doc_title-dense
    - 20 unique to text-sparse / BM25
    - 15 unique to entity-sparse
    - 15 unique to title-sparse
    - 20 structural graph expansion (NEXT_CHUNK + siblings)
    - 30 per-document depth (top docs, deeper chunks)
    = 200 with guaranteed signal diversity

  ★ Insight ─────────────────────────────────────
  The discovery that max_pairs=50 is dead config is important.
  It was likely intended to limit rerank cost but was never wired
  in. The actual provider has no candidate cap — it processes
  everything you give it in batches of 16. This means the pool
  size decision is purely a latency/quality tradeoff, not a hard
  constraint.

  At 200 candidates with batched reranking (~120-200ms), you're
  adding maybe 60-100ms over the current 100-candidate baseline.
  For a system that already does multi-vector Qdrant queries +
  Neo4j BM25 + graph expansion, that's noise. The quality gain
  from signal diversity far outweighs the latency cost.

  The structural graph candidates are the single highest-value
  addition to the pool. Vector similarity tells you "this chunk
  talks about similar topics." Graph structure tells you "this
  chunk is the next step in the procedure" or "this chunk is in
  the same configuration section." Those are orthogonal signals
  that the cross-encoder can evaluate but can't discover on its
  own.
  ─────────────────────────────────────────────────

  So to answer directly: yes to both. 200 candidates with 15-20 structural slots and signal-diverse selection is the right design point. Ready to plan the implementation when you are.

  ⏺ Good — now I have the actual model specs. Here are the corrected hard limits:

  Model Stack: Actual Specifications

  ┌─────────────────┬────────────────────────┬──────────────────────────────────┬─────────────────────┬────────────────────────────────────────────────┐
  │      Role       │         Model          │          Context Window          │     Output Dims     │                 Key Constraint                 │
  ├─────────────────┼────────────────────────┼──────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────┤
  │ Dense embedding │ Qwen3-Embedding-0.6B   │ 32K tokens                       │ 1024 (MRL: 32–1024) │ Huge context; chunks will never hit this limit │
  ├─────────────────┼────────────────────────┼──────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────┤
  │ Reranker        │ Qwen3-Reranker-4B      │ 8K tokens (code), 32K (reported) │ n/a (yes/no logit)  │ query + document must fit in 8K                │
  ├─────────────────┼────────────────────────┼──────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────┤
  │ ColBERT         │ ColBERTv2 (colbert-ai) │ 512 tokens                       │ 128 per token       │ This is the bottleneck                         │
  ├─────────────────┼────────────────────────┼──────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────┤
  │ Sparse          │ SPLADEv3 (NAVER)       │ 512 tokens (BERT-based)          │ ~30K vocab sparse   │ Trained at 128-256, works up to 512            │
  └─────────────────┴────────────────────────┴──────────────────────────────────┴─────────────────────┴────────────────────────────────────────────────┘

  What Changes vs. Current Config

  Reranker: Much More Headroom Than Currently Configured

  The codebase currently has:
  MAX_TOKENS_PER_DOC = 2048   # local_bge_service.py:34
  MAX_TOKENS_TOTAL = 4096     # local_bge_service.py:35 (query + doc)

  Qwen3-Reranker-4B's actual context is 8,192 tokens (from the HuggingFace reference code: max_length = 8192). That means:

  - A typical query of ~50-100 tokens leaves ~8,000 tokens for the document
  - Your current MAX_TOKENS_PER_DOC = 2048 is artificially capping at 25% of capacity
  - You could rerank chunks up to ~8,000 tokens without truncation

  For the 200-candidate pool, this changes the math significantly. Most chunks in a RAG system are 400-500 tokens. At 8K context, the reranker can handle even combined/expanded chunks without truncation.
   The per-document token budget is not a constraint.

  Latency impact of 200 candidates: At batch size 16, that's 13 HTTP requests. Qwen3-Reranker-4B is a 4B-parameter model doing a single forward pass per batch (it outputs a yes/no logit, not a full
  generation). With GPU inference, each batch should be ~20-40ms. Total: ~250-500ms for 200 candidates — acceptable for a RAG query.

  ColBERT: The Real Bottleneck

  ColBERTv2 has a 512-token hard limit (BERT-based architecture). This means:

  - Any chunk longer than 512 tokens gets truncated for ColBERT scoring
  - Your current target chunk size of 400-500 tokens is right at the edge
  - Combined chunks or expanded chunks (heading + body) could easily exceed 512 tokens
  - The late-interaction vectors are 128-dim per token, so a 512-token chunk produces a 512 × 128 matrix

  For the signal-diverse pool: ColBERT candidates will represent chunks that score well on fine-grained token-level matching but are capped at 512 tokens of context. This means ColBERT inherently favors
  shorter, denser chunks — it literally can't see beyond the first ~500 tokens.

  SPLADEv3: Similar Constraint to ColBERT

  SPLADEv3 is BERT-based with a 512-token max (trained at 128-256, functional up to 512). The NAVER team confirmed this on HuggingFace. Same truncation concern as ColBERT — chunks beyond 512 tokens lose
  their tail content for sparse scoring.

  Dense Embedding: No Constraint

  Qwen3-Embedding-0.6B at 32K tokens is effectively unlimited for chunk-level operations. Your chunks will never approach this limit. MRL support (Matryoshka Representation Learning) means you can use
  1024-dim for storage and optionally truncate to smaller dims for faster ANN search without re-embedding.

  Revised Pool Allocation at 200 Candidates

  Given these constraints, the signal-diverse pool should account for the fact that ColBERT and SPLADE have a 512-token view while the dense embedder sees the full chunk and the reranker can evaluate up
  to 8K tokens:

  Signal-diverse rerank pool (200 candidates):
    50  consensus (top by fused score — multi-signal agreement)
    20  unique to dense (Qwen3-0.6B — full chunk semantics)
    15  unique to ColBERT (token-level precision, 512-token window)
    15  unique to SPLADE (learned sparse, 512-token window)
    15  unique to BM25 (Neo4j FTS — exact lexical match)
    15  unique to entity-sparse (entity name matching)
    20  structural graph expansion (NEXT_CHUNK + siblings)
    30  per-document depth (signal-weighted quota)
    20  buffer (overflow from any oversubscribed slot)
    = 200

  Should You Include More Structural Candidates?

  Yes — and the reranker headroom makes this even more compelling. With 8K tokens available, you can prepend the full heading path to each structural candidate:

  "Configuration > S3 Backend > Bucket Settings > Access Control\n\n
  Set the IAM role ARN in the weka_s3_iam_role field..."

  That heading path might add 20-30 tokens — negligible against an 8K budget. The cross-encoder gets both the structural position and the chunk content, so it can accurately judge whether "Step 7 of the
  S3 configuration procedure" is relevant to a query about S3 configuration even though the chunk text itself just says "Set the IAM role ARN."

  Graph structural candidates should get 20 slots (up from the 10 I initially suggested), because:
  - The reranker can evaluate them with full structural context (headings prepended)
  - They represent the only signal type that captures document position (depth/sequence)
  - They're the primary mechanism for surfacing "the meat" beyond introductions

  ★ Insight ─────────────────────────────────────
  The model stack creates an interesting asymmetry: your two
  retrieval-stage models (ColBERT at 512 tokens, SPLADE at 512
  tokens) have a narrow view, while your reranker (Qwen3-4B at
  8K tokens) has a wide view. This means chunks that were partially
  truncated during retrieval scoring can be evaluated in full by
  the reranker.

  This actually strengthens the case for a larger rerank pool.
  A long chunk might score poorly on ColBERT (which only saw its
  first 512 tokens) but score highly on the reranker (which sees
  all 8K tokens including the critical content in the second half).
  The signal-diverse pool ensures these "ColBERT-truncated" chunks
  still get a chance at full cross-encoder evaluation.

  The current MAX_TOKENS_PER_DOC = 2048 should be raised to at
  least 4096 (or 7500 with query overhead) to actually use the
  Qwen3-Reranker-4B's capacity. You're leaving 75% of the model's
  context window on the table.
  ─────────────────────────────────────────────────

  Two concrete config changes to make immediately (independent of the signal-diverse pool work):

  1. Raise MAX_TOKENS_PER_DOC from 2048 to 7500 in local_bge_service.py
  2. Raise MAX_TOKENS_TOTAL from 4096 to 8192 in local_bge_service.py

  These unlock the reranker's actual capacity with zero architectural changes. Want me to plan the full signal-diverse pool implementation now?




  ⏺ Now let me write the final plan file synthesizing all exploration and design work.

⏺ Updated plan
  ⎿  Wrote 234 lines to /Users/brennanconley/.claude/plans/cryptic-percolating-lynx.md
     # Signal-Diverse Rerank Pool Implementation Plan

     ## Context

     The current retrieval pipeline collects rich, diverse signals — 7 Qdrant vector fields, Neo4j BM25, entity extraction, graph structure — then immediately flattens them into a single ranked list
     via RRF. The reranker (Qwen3-Reranker-4B, 8K context) only sees the top 100 by fused score, which means deep document content and signal-diverse chunks get buried. Introduction/summary chunks
     dominate because they score well across multiple embeddings, while the substantive procedural content deeper in documents never enters the rerank pool.

     **Goal:** Replace the flat "top 100 by fused score" rerank input with a signal-diverse pool of 200 candidates that ensures the cross-encoder evaluates chunks from every retrieval signal source,
     with per-document depth guarantees and structural graph context.

     ---

     ## Implementation Phases

     ### Phase 0: Raise Reranker Token Limits

     Qwen3-Reranker-4B supports 8K context. Current limits are at 25% capacity.

     **File: `src/providers/rerank/local_bge_service.py`**
     - Line 34: `MAX_TOKENS_PER_DOC` 2048 → 7500
     - Line 35: `MAX_TOKENS_TOTAL` 4096 → 8192

     **File: `src/query/hybrid_retrieval.py`**
     - Line ~4062: `service_max_batch_tokens` 1024 → 4096 (in `_apply_reranker`)

     **Verify:** `pytest tests/unit/test_phase1_reranker_batching.py tests/query/test_reranker_integration.py -v`

     ---

     ### Phase 1: Add `parent_path_norm` to Retrieval Pipeline

     Currently `parent_path_norm` (full heading hierarchy path like "Configuration > S3 Backend > Bucket Settings") exists only in Neo4j, not on `ChunkResult`. Needed for reranker context enrichment.

     **File: `src/query/hybrid_retrieval.py`**

     1a. Add field to `ChunkResult` dataclass (after line ~181):
     ```python
     parent_path_norm: Optional[str] = None
     ```

     1b. Add `_hydrate_parent_paths()` method to `HybridRetriever`:
     - Batch Cypher: `UNWIND $ids AS cid MATCH (c:Chunk {id: cid}) WHERE c.parent_path_norm IS NOT NULL RETURN c.id, c.parent_path_norm`
     - One round-trip for ~200 IDs, ~5-15ms
     - Skip if `neo4j_disabled`

     1c. Modify text construction in `_apply_reranker` (line ~4028-4034):
     - Current: `f"{heading}\n\n{text_body}"`
     - New: `f"{parent_path_norm}\n\n{heading}\n\n{text_body}"` when parent_path_norm is available
     - Adds ~20-30 tokens of structural breadcrumb context to each reranker input

     ---

     ### Phase 2: Config Additions

     **File: `src/shared/config.py`**

     New `SignalPoolConfig` class (after `ExpansionStructureConfig` ~line 394):

     | Field | Type | Default | Purpose |
     |---|---|---|---|
     | `enabled` | `bool` | `False` | Master switch |
     | `pool_size` | `int` | `200` | Total rerank pool size |
     | `consensus_slots` | `int` | `50` | Top by fused_score |
     | `content_dense_slots` | `int` | `20` | Unique to content-dense vector |
     | `title_dense_slots` | `int` | `15` | Unique to title-dense vector |
     | `doc_title_dense_slots` | `int` | `15` | Unique to doc_title-dense vector |
     | `text_sparse_slots` | `int` | `20` | Unique to text-sparse/BM25 |
     | `entity_sparse_slots` | `int` | `15` | Unique to entity-sparse |
     | `title_sparse_slots` | `int` | `15` | Unique to title-sparse |
     | `structural_slots` | `int` | `20` | NEXT_CHUNK + sibling expansion |
     | `per_doc_depth_slots` | `int` | `30` | Per-document depth coverage |
     | `per_doc_k` | `int` | `5` | Top K documents for depth quota |
     | `per_doc_m` | `int` | `6` | Signal-diverse chunks per doc |
     | `fallback_to_provenance` | `bool` | `True` | Degrade to BM25/vector when per-field scores unavailable |

     Add to `HybridSearchConfig` (~line 528): `signal_pool: SignalPoolConfig`

     Add to `FeatureFlagsConfig` (~line 945): `signal_diverse_rerank_pool: bool = False`

     ---

     ### Phase 3: Signal-Diverse Pool Builder (NEW MODULE)

     **New file: `src/query/signal_pool.py`**

     Pure-function module — no IO, no Qdrant/Neo4j calls. Independently testable.

     **Algorithm:**
     1. Fill consensus slots (top 50 by `fused_score`)
     2. Fill per-signal slots: for each signal type, sort all candidates by that signal's score, take top N that aren't already in pool
     3. Fill structural slots from pre-rerank expansion results
     4. Fill per-document depth slots: for top K documents, ensure M signal-diverse chunks are represented
     5. Backfill remaining capacity from `fused_score` order

     **Graceful degradation:** Per-field scores (`title_vec_score`, `entity_vec_score`, etc.) are only populated when weighted fusion is active. When they're `None`, fall back to coarser BM25 vs
     vector provenance (chunk has `bm25_score` > 0 → came from BM25; has `vector_score` > 0 → came from Qdrant).

     **Key design:** "Unique to" means the chunk scored well on that specific signal but did NOT make it into the consensus top 50. These are the chunks RRF buried — they're strong on one signal but
     weak across others. The cross-encoder is specifically designed to evaluate whether they're truly relevant.

     **Output:** `SignalPoolResult` dataclass with `pool` (List[ChunkResult]), `slot_fills` (Dict[str, int]), `degraded` (bool).

     ---

     ### Phase 4: Pipeline Integration

     **File: `src/query/hybrid_retrieval.py`**

     This is the main integration point. Modifications to `retrieve()` method.

     **4a. Auto-enable weighted fusion when signal pool is active**

     In `__init__`, after reading signal pool config: if signal pool enabled but weighted fusion off, auto-enable it with a warning log. This ensures per-field scores are populated.

     **Prerequisite insight:** The weighted fusion path (`query_api_weighted_fusion`) does Stage 2 per-field re-scoring of Qdrant candidates. It adds ~5-7 filtered Qdrant queries (~40-70ms) but
     populates `title_vec_score`, `doc_title_vec_score`, `entity_vec_score`, `lexical_vec_score`, `doc_title_sparse_score` on every ChunkResult. Without this, the signal pool degrades to BM25/vector
     provenance only.

     **4b. Insert signal pool between fusion and rerank (lines ~3275-3316)**

     Replace the current `fused_results[:rerank_pool_size]` logic:

     ```
     If signal pool enabled:
         1. Run pre-rerank structural expansion on top 10 fused results
            (reuse _expand_with_structure with force=True)
         2. Call build_signal_pool(fused_results, structural_candidates, config)
         3. Hydrate parent_path_norm for pool candidates
         4. Pass 200-candidate pool to _apply_reranker
     Else:
         existing behavior (flat top-N by fused_score)
     ```

     **4c. Add `force` parameter to `_expand_with_structure` (line ~4445)**

     Currently gated by `feature_flags.structure_aware_expansion` (default False). Add a `force=False` keyword arg that bypasses the feature flag check. This allows pre-rerank structural expansion
     without enabling the post-rerank expansion globally.

     **4d. Log signal pool metrics**

     Add to the metrics dict: `signal_pool_enabled`, `signal_pool_size`, `signal_pool_slot_fills`, `signal_pool_degraded`.

     ---

     ### Phase 5: Remove Dead Dominance Gating

     **File: `src/query/hybrid_retrieval.py`**

     Delete the `_gate_to_primary_document` nested function (lines 3345-3375) and its associated comments. Confirmed dead code — defined but never called. The signal-diverse pool inherently provides
     document diversity through the per-document depth slot and signal coverage across documents.

     ---

     ### Phase 6: Tests

     **New file: `tests/query/test_signal_pool.py`**

     Test the pure-function pool builder. Key test cases:
     - Consensus fills first (top fused_score chunks always present)
     - Signal-unique slots pull in non-consensus chunks (high on one signal, low on fused)
     - Graceful degradation without per-field scores
     - Structural slot includes expanded chunks
     - Per-document depth ensures multi-doc representation
     - No duplicates in pool
     - Pool size never exceeds configured limit
     - Backfill fills remaining capacity

     **Pattern:** Use `ChunkResult` factory function to create test chunks with controlled per-field scores. No mocking needed — signal_pool.py is pure functions.

     **Existing test updates:** Extend `tests/query/test_reranker_integration.py` with a test for `parent_path_norm` prepending in reranker text.

     ---

     ## Execution Order (Dependency Graph)

     ```
     Phase 0 (token limits)  ──┐
     Phase 1 (parent_path)   ──┤── all independent, can parallel
     Phase 2 (config)        ──┘
                                 ↓
     Phase 3 (signal_pool.py) ── depends on Phase 2 (uses SignalPoolConfig)
                                 ↓
     Phase 4 (integration)   ── depends on Phase 1, 2, 3
                                 ↓
     Phase 5 (dead code)     ── depends on Phase 4
     Phase 6 (tests)         ── Phase 3 tests can start with Phase 3; integration tests after Phase 4
     ```

     **Recommended:** Phases 0+1+2 in parallel → Phase 3 → Phase 4 → Phase 5+6

     ---

     ## Files Summary

     | File | Action | Phase |
     |---|---|---|
     | `src/providers/rerank/local_bge_service.py` | Modify: raise token limits | 0 |
     | `src/query/hybrid_retrieval.py` | Modify: ChunkResult field, `_hydrate_parent_paths`, `_apply_reranker` text, `retrieve()` integration, `_expand_with_structure` force param, remove dead
     gating, raise batch token limit | 0,1,4,5 |
     | `src/shared/config.py` | Modify: add `SignalPoolConfig`, update `HybridSearchConfig`, `FeatureFlagsConfig` | 2 |
     | `src/query/signal_pool.py` | **NEW**: signal-diverse pool builder | 3 |
     | `tests/query/test_signal_pool.py` | **NEW**: pool builder unit tests | 6 |
     | `tests/query/test_reranker_integration.py` | Modify: add parent_path_norm test | 6 |

     ---

     ## Verification

     After all phases:

     ```bash
     # Unit tests
     pytest tests/query/test_signal_pool.py -v
     pytest tests/query/test_reranker_integration.py -v
     pytest tests/unit/test_phase1_reranker_batching.py -v

     # Entry point imports (no regressions)
     python -c "from src.mcp_server.main import app"
     python -c "from src.ingestion.worker import process_job"
     python -c "from src.ingestion.auto.service import app"

     # CI guard (no new ACTIVE→DEAD violations)
     python scripts/ci/check_dead_imports.py

     # Integration: enable in config and run a query
     # Set feature_flags.signal_diverse_rerank_pool=True
     # Set feature_flags.query_api_weighted_fusion=True
     # Verify metrics contain signal_pool_* keys
     # Verify reranker receives ~200 candidates
     ```

     ---

     ## Model Constraints Reference

     | Model | Role | Context | Output | Constraint |
     |---|---|---|---|---|
     | Qwen3-Embedding-0.6B | Dense embedding | 32K tokens | 1024 dims (MRL 32-1024) | No constraint for chunks |
     | Qwen3-Reranker-4B | Cross-encoder reranker | 8K tokens | yes/no logit | query + doc must fit 8K |
     | ColBERTv2 (colbert-ai) | Late interaction | 512 tokens | 128 dims per token | **Bottleneck** — truncates chunks > 512 tokens |
     | SPLADEv3 (NAVER) | Learned sparse | 512 tokens | ~30K vocab sparse | Trained at 128-256, works to 512 |

     The 512-token ColBERT/SPLADE limit means chunks that were partially truncated during retrieval scoring can be evaluated in full by the 8K reranker — a key reason the signal-diverse pool adds
     value.








⏺ The plan is written. Here's a quick summary of the key architectural decisions:

  ★ Insight ─────────────────────────────────────
  Why a pure-function module (signal_pool.py)? The pool builder
  takes a list of scored chunks and produces a signal-diverse subset.
  It has zero IO — no Qdrant, no Neo4j, no HTTP calls. This means
  it can be unit tested with synthetic ChunkResult objects in <1ms
  per test. The structural expansion and parent_path hydration happen
  outside the pool builder and are passed in as arguments, keeping
  the algorithm clean and the IO at the integration layer.

  Why auto-enable weighted fusion? The signal-diverse pool is
  useless without per-field scores. The weighted fusion path adds
  ~40-70ms of filtered Qdrant queries but populates all 6 per-field
  score fields on every ChunkResult. Without it, the pool degrades
  to coarse BM25-vs-vector provenance only. Auto-enabling it when
  the signal pool is active prevents silent degradation.

  Why force=True on _expand_with_structure? The existing
  method is gated by a feature flag that defaults to False. Rather
  than globally enabling structure-aware expansion (which would also
  run it post-rerank), we add a force parameter so the signal
  pool integration can call it pre-rerank without side effects.
  ─────────────────────────────────────────────────

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Signal-Diverse Rerank Pool Implementation Plan

 Context

 The current retrieval pipeline collects rich, diverse signals — 7 Qdrant vector fields, Neo4j BM25, entity extraction, graph structure — then immediately flattens them into a single ranked list via
 RRF. The reranker (Qwen3-Reranker-4B, 8K context) only sees the top 100 by fused score, which means deep document content and signal-diverse chunks get buried. Introduction/summary chunks dominate
 because they score well across multiple embeddings, while the substantive procedural content deeper in documents never enters the rerank pool.

 Goal: Replace the flat "top 100 by fused score" rerank input with a signal-diverse pool of 200 candidates that ensures the cross-encoder evaluates chunks from every retrieval signal source, with
 per-document depth guarantees and structural graph context.

 ---
 Implementation Phases

 Phase 0: Raise Reranker Token Limits

 Qwen3-Reranker-4B supports 8K context. Current limits are at 25% capacity.

 File: src/providers/rerank/local_bge_service.py
 - Line 34: MAX_TOKENS_PER_DOC 2048 → 7500
 - Line 35: MAX_TOKENS_TOTAL 4096 → 8192

 File: src/query/hybrid_retrieval.py
 - Line ~4062: service_max_batch_tokens 1024 → 4096 (in _apply_reranker)

 Verify: pytest tests/unit/test_phase1_reranker_batching.py tests/query/test_reranker_integration.py -v

 ---
 Phase 1: Add parent_path_norm to Retrieval Pipeline

 Currently parent_path_norm (full heading hierarchy path like "Configuration > S3 Backend > Bucket Settings") exists only in Neo4j, not on ChunkResult. Needed for reranker context enrichment.

 File: src/query/hybrid_retrieval.py

 1a. Add field to ChunkResult dataclass (after line ~181):
 parent_path_norm: Optional[str] = None

 1b. Add _hydrate_parent_paths() method to HybridRetriever:
 - Batch Cypher: UNWIND $ids AS cid MATCH (c:Chunk {id: cid}) WHERE c.parent_path_norm IS NOT NULL RETURN c.id, c.parent_path_norm
 - One round-trip for ~200 IDs, ~5-15ms
 - Skip if neo4j_disabled

 1c. Modify text construction in _apply_reranker (line ~4028-4034):
 - Current: f"{heading}\n\n{text_body}"
 - New: f"{parent_path_norm}\n\n{heading}\n\n{text_body}" when parent_path_norm is available
 - Adds ~20-30 tokens of structural breadcrumb context to each reranker input

 ---
 Phase 2: Config Additions

 File: src/shared/config.py

 New SignalPoolConfig class (after ExpansionStructureConfig ~line 394):

 ┌────────────────────────┬──────┬─────────┬──────────────────────────────────────────────────────────┐
 │         Field          │ Type │ Default │                         Purpose                          │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ enabled                │ bool │ False   │ Master switch                                            │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ pool_size              │ int  │ 200     │ Total rerank pool size                                   │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ consensus_slots        │ int  │ 50      │ Top by fused_score                                       │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ content_dense_slots    │ int  │ 20      │ Unique to content-dense vector                           │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ title_dense_slots      │ int  │ 15      │ Unique to title-dense vector                             │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ doc_title_dense_slots  │ int  │ 15      │ Unique to doc_title-dense vector                         │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ text_sparse_slots      │ int  │ 20      │ Unique to text-sparse/BM25                               │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ entity_sparse_slots    │ int  │ 15      │ Unique to entity-sparse                                  │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ title_sparse_slots     │ int  │ 15      │ Unique to title-sparse                                   │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ structural_slots       │ int  │ 20      │ NEXT_CHUNK + sibling expansion                           │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ per_doc_depth_slots    │ int  │ 30      │ Per-document depth coverage                              │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ per_doc_k              │ int  │ 5       │ Top K documents for depth quota                          │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ per_doc_m              │ int  │ 6       │ Signal-diverse chunks per doc                            │
 ├────────────────────────┼──────┼─────────┼──────────────────────────────────────────────────────────┤
 │ fallback_to_provenance │ bool │ True    │ Degrade to BM25/vector when per-field scores unavailable │
 └────────────────────────┴──────┴─────────┴──────────────────────────────────────────────────────────┘

 Add to HybridSearchConfig (~line 528): signal_pool: SignalPoolConfig

 Add to FeatureFlagsConfig (~line 945): signal_diverse_rerank_pool: bool = False

 ---
 Phase 3: Signal-Diverse Pool Builder (NEW MODULE)

 New file: src/query/signal_pool.py

 Pure-function module — no IO, no Qdrant/Neo4j calls. Independently testable.

 Algorithm:
 1. Fill consensus slots (top 50 by fused_score)
 2. Fill per-signal slots: for each signal type, sort all candidates by that signal's score, take top N that aren't already in pool
 3. Fill structural slots from pre-rerank expansion results
 4. Fill per-document depth slots: for top K documents, ensure M signal-diverse chunks are represented
 5. Backfill remaining capacity from fused_score order

 Graceful degradation: Per-field scores (title_vec_score, entity_vec_score, etc.) are only populated when weighted fusion is active. When they're None, fall back to coarser BM25 vs vector provenance
 (chunk has bm25_score > 0 → came from BM25; has vector_score > 0 → came from Qdrant).

 Key design: "Unique to" means the chunk scored well on that specific signal but did NOT make it into the consensus top 50. These are the chunks RRF buried — they're strong on one signal but weak across
  others. The cross-encoder is specifically designed to evaluate whether they're truly relevant.

 Output: SignalPoolResult dataclass with pool (List[ChunkResult]), slot_fills (Dict[str, int]), degraded (bool).

 ---
 Phase 4: Pipeline Integration

 File: src/query/hybrid_retrieval.py

 This is the main integration point. Modifications to retrieve() method.

 4a. Auto-enable weighted fusion when signal pool is active

 In __init__, after reading signal pool config: if signal pool enabled but weighted fusion off, auto-enable it with a warning log. This ensures per-field scores are populated.

 Prerequisite insight: The weighted fusion path (query_api_weighted_fusion) does Stage 2 per-field re-scoring of Qdrant candidates. It adds ~5-7 filtered Qdrant queries (~40-70ms) but populates
 title_vec_score, doc_title_vec_score, entity_vec_score, lexical_vec_score, doc_title_sparse_score on every ChunkResult. Without this, the signal pool degrades to BM25/vector provenance only.

 4b. Insert signal pool between fusion and rerank (lines ~3275-3316)

 Replace the current fused_results[:rerank_pool_size] logic:

 If signal pool enabled:
     1. Run pre-rerank structural expansion on top 10 fused results
        (reuse _expand_with_structure with force=True)
     2. Call build_signal_pool(fused_results, structural_candidates, config)
     3. Hydrate parent_path_norm for pool candidates
     4. Pass 200-candidate pool to _apply_reranker
 Else:
     existing behavior (flat top-N by fused_score)

 4c. Add force parameter to _expand_with_structure (line ~4445)

 Currently gated by feature_flags.structure_aware_expansion (default False). Add a force=False keyword arg that bypasses the feature flag check. This allows pre-rerank structural expansion without
 enabling the post-rerank expansion globally.

 4d. Log signal pool metrics

 Add to the metrics dict: signal_pool_enabled, signal_pool_size, signal_pool_slot_fills, signal_pool_degraded.

 ---
 Phase 5: Remove Dead Dominance Gating

 File: src/query/hybrid_retrieval.py

 Delete the _gate_to_primary_document nested function (lines 3345-3375) and its associated comments. Confirmed dead code — defined but never called. The signal-diverse pool inherently provides document
 diversity through the per-document depth slot and signal coverage across documents.

 ---
 Phase 6: Tests

 New file: tests/query/test_signal_pool.py

 Test the pure-function pool builder. Key test cases:
 - Consensus fills first (top fused_score chunks always present)
 - Signal-unique slots pull in non-consensus chunks (high on one signal, low on fused)
 - Graceful degradation without per-field scores
 - Structural slot includes expanded chunks
 - Per-document depth ensures multi-doc representation
 - No duplicates in pool
 - Pool size never exceeds configured limit
 - Backfill fills remaining capacity

 Pattern: Use ChunkResult factory function to create test chunks with controlled per-field scores. No mocking needed — signal_pool.py is pure functions.

 Existing test updates: Extend tests/query/test_reranker_integration.py with a test for parent_path_norm prepending in reranker text.

 ---
 Execution Order (Dependency Graph)

 Phase 0 (token limits)  ──┐
 Phase 1 (parent_path)   ──┤── all independent, can parallel
 Phase 2 (config)        ──┘
                             ↓
 Phase 3 (signal_pool.py) ── depends on Phase 2 (uses SignalPoolConfig)
                             ↓
 Phase 4 (integration)   ── depends on Phase 1, 2, 3
                             ↓
 Phase 5 (dead code)     ── depends on Phase 4
 Phase 6 (tests)         ── Phase 3 tests can start with Phase 3; integration tests after Phase 4

 Recommended: Phases 0+1+2 in parallel → Phase 3 → Phase 4 → Phase 5+6

 ---
 Files Summary

 ┌───────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────┐
 │                   File                    │                                                                    Action                                                                     │  Phase  │
 ├───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┤
 │ src/providers/rerank/local_bge_service.py │ Modify: raise token limits                                                                                                                    │ 0       │
 ├───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┤
 │ src/query/hybrid_retrieval.py             │ Modify: ChunkResult field, _hydrate_parent_paths, _apply_reranker text, retrieve() integration, _expand_with_structure force param, remove    │ 0,1,4,5 │
 │                                           │ dead gating, raise batch token limit                                                                                                          │         │
 ├───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┤
 │ src/shared/config.py                      │ Modify: add SignalPoolConfig, update HybridSearchConfig, FeatureFlagsConfig                                                                   │ 2       │
 ├───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┤
 │ src/query/signal_pool.py                  │ NEW: signal-diverse pool builder                                                                                                              │ 3       │
 ├───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┤
 │ tests/query/test_signal_pool.py           │ NEW: pool builder unit tests                                                                                                                  │ 6       │
 ├───────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┤
 │ tests/query/test_reranker_integration.py  │ Modify: add parent_path_norm test                                                                                                             │ 6       │
 └───────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────┘

 ---
 Verification

 After all phases:

 # Unit tests
 pytest tests/query/test_signal_pool.py -v
 pytest tests/query/test_reranker_integration.py -v
 pytest tests/unit/test_phase1_reranker_batching.py -v

 # Entry point imports (no regressions)
 python -c "from src.mcp_server.main import app"
 python -c "from src.ingestion.worker import process_job"
 python -c "from src.ingestion.auto.service import app"

 # CI guard (no new ACTIVE→DEAD violations)
 python scripts/ci/check_dead_imports.py

 # Integration: enable in config and run a query
 # Set feature_flags.signal_diverse_rerank_pool=True
 # Set feature_flags.query_api_weighted_fusion=True
 # Verify metrics contain signal_pool_* keys
 # Verify reranker receives ~200 candidates

 ---
 Model Constraints Reference

 ┌────────────────────────┬────────────────────────┬────────────┬─────────────────────────┬────────────────────────────────────────────┐
 │         Model          │          Role          │  Context   │         Output          │                 Constraint                 │
 ├────────────────────────┼────────────────────────┼────────────┼─────────────────────────┼────────────────────────────────────────────┤
 │ Qwen3-Embedding-0.6B   │ Dense embedding        │ 32K tokens │ 1024 dims (MRL 32-1024) │ No constraint for chunks                   │
 ├────────────────────────┼────────────────────────┼────────────┼─────────────────────────┼────────────────────────────────────────────┤
 │ Qwen3-Reranker-4B      │ Cross-encoder reranker │ 8K tokens  │ yes/no logit            │ query + doc must fit 8K                    │
 ├────────────────────────┼────────────────────────┼────────────┼─────────────────────────┼────────────────────────────────────────────┤
 │ ColBERTv2 (colbert-ai) │ Late interaction       │ 512 tokens │ 128 dims per token      │ Bottleneck — truncates chunks > 512 tokens │
 ├────────────────────────┼────────────────────────┼────────────┼─────────────────────────┼────────────────────────────────────────────┤
 │ SPLADEv3 (NAVER)       │ Learned sparse         │ 512 tokens │ ~30K vocab sparse       │ Trained at 128-256, works to 512           │
 └────────────────────────┴────────────────────────┴────────────┴─────────────────────────┴────────────────────────────────────────────┘

 The 512-token ColBERT/SPLADE limit means chunks that were partially truncated during retrieval scoring can be evaluated in full by the 8K reranker — a key reason the signal-diverse pool adds value.


 # Signal-Diverse Rerank Pool Implementation Plan

## Context

The current retrieval pipeline collects rich, diverse signals — 7 Qdrant vector fields, Neo4j BM25, entity extraction, graph structure — then immediately flattens them into a single ranked list via RRF. The reranker (Qwen3-Reranker-4B, 8K context) only sees the top 100 by fused score, which means deep document content and signal-diverse chunks get buried. Introduction/summary chunks dominate because they score well across multiple embeddings, while the substantive procedural content deeper in documents never enters the rerank pool.

**Goal:** Replace the flat "top 100 by fused score" rerank input with a signal-diverse pool of 200 candidates that ensures the cross-encoder evaluates chunks from every retrieval signal source, with per-document depth guarantees and structural graph context.

---

## Implementation Phases

### Phase 0: Raise Reranker Token Limits

Qwen3-Reranker-4B supports 8K context. Current limits are at 25% capacity.

**File: `src/providers/rerank/local_bge_service.py`**
- Line 34: `MAX_TOKENS_PER_DOC` 2048 → 7500
- Line 35: `MAX_TOKENS_TOTAL` 4096 → 8192

**File: `src/query/hybrid_retrieval.py`**
- Line ~4062: `service_max_batch_tokens` 1024 → 4096 (in `_apply_reranker`)

**Verify:** `pytest tests/unit/test_phase1_reranker_batching.py tests/query/test_reranker_integration.py -v`

---

### Phase 1: Add `parent_path_norm` to Retrieval Pipeline

Currently `parent_path_norm` (full heading hierarchy path like "Configuration > S3 Backend > Bucket Settings") exists only in Neo4j, not on `ChunkResult`. Needed for reranker context enrichment.

**File: `src/query/hybrid_retrieval.py`**

1a. Add field to `ChunkResult` dataclass (after line ~181):
```python
parent_path_norm: Optional[str] = None
```

1b. Add `_hydrate_parent_paths()` method to `HybridRetriever`:
- Batch Cypher: `UNWIND $ids AS cid MATCH (c:Chunk {id: cid}) WHERE c.parent_path_norm IS NOT NULL RETURN c.id, c.parent_path_norm`
- One round-trip for ~200 IDs, ~5-15ms
- Skip if `neo4j_disabled`

1c. Modify text construction in `_apply_reranker` (line ~4028-4034):
- Current: `f"{heading}\n\n{text_body}"`
- New: `f"{parent_path_norm}\n\n{heading}\n\n{text_body}"` when parent_path_norm is available
- Adds ~20-30 tokens of structural breadcrumb context to each reranker input

---

### Phase 2: Config Additions

**File: `src/shared/config.py`**

New `SignalPoolConfig` class (after `ExpansionStructureConfig` ~line 394):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch |
| `pool_size` | `int` | `200` | Total rerank pool size |
| `consensus_slots` | `int` | `50` | Top by fused_score |
| `content_dense_slots` | `int` | `20` | Unique to content-dense vector |
| `title_dense_slots` | `int` | `15` | Unique to title-dense vector |
| `doc_title_dense_slots` | `int` | `15` | Unique to doc_title-dense vector |
| `text_sparse_slots` | `int` | `20` | Unique to text-sparse/BM25 |
| `entity_sparse_slots` | `int` | `15` | Unique to entity-sparse |
| `title_sparse_slots` | `int` | `15` | Unique to title-sparse |
| `structural_slots` | `int` | `20` | NEXT_CHUNK + sibling expansion |
| `per_doc_depth_slots` | `int` | `30` | Per-document depth coverage |
| `per_doc_k` | `int` | `5` | Top K documents for depth quota |
| `per_doc_m` | `int` | `6` | Signal-diverse chunks per doc |
| `fallback_to_provenance` | `bool` | `True` | Degrade to BM25/vector when per-field scores unavailable |

Add to `HybridSearchConfig` (~line 528): `signal_pool: SignalPoolConfig`

Add to `FeatureFlagsConfig` (~line 945): `signal_diverse_rerank_pool: bool = False`

---

### Phase 3: Signal-Diverse Pool Builder (NEW MODULE)

**New file: `src/query/signal_pool.py`**

Pure-function module — no IO, no Qdrant/Neo4j calls. Independently testable.

**Algorithm:**
1. Fill consensus slots (top 50 by `fused_score`)
2. Fill per-signal slots: for each signal type, sort all candidates by that signal's score, take top N that aren't already in pool
3. Fill structural slots from pre-rerank expansion results
4. Fill per-document depth slots: for top K documents, ensure M signal-diverse chunks are represented
5. Backfill remaining capacity from `fused_score` order

**Graceful degradation:** Per-field scores (`title_vec_score`, `entity_vec_score`, etc.) are only populated when weighted fusion is active. When they're `None`, fall back to coarser BM25 vs vector provenance (chunk has `bm25_score` > 0 → came from BM25; has `vector_score` > 0 → came from Qdrant).

**Key design:** "Unique to" means the chunk scored well on that specific signal but did NOT make it into the consensus top 50. These are the chunks RRF buried — they're strong on one signal but weak across others. The cross-encoder is specifically designed to evaluate whether they're truly relevant.

**Output:** `SignalPoolResult` dataclass with `pool` (List[ChunkResult]), `slot_fills` (Dict[str, int]), `degraded` (bool).

---

### Phase 4: Pipeline Integration

**File: `src/query/hybrid_retrieval.py`**

This is the main integration point. Modifications to `retrieve()` method.

**4a. Auto-enable weighted fusion when signal pool is active**

In `__init__`, after reading signal pool config: if signal pool enabled but weighted fusion off, auto-enable it with a warning log. This ensures per-field scores are populated.

**Prerequisite insight:** The weighted fusion path (`query_api_weighted_fusion`) does Stage 2 per-field re-scoring of Qdrant candidates. It adds ~5-7 filtered Qdrant queries (~40-70ms) but populates `title_vec_score`, `doc_title_vec_score`, `entity_vec_score`, `lexical_vec_score`, `doc_title_sparse_score` on every ChunkResult. Without this, the signal pool degrades to BM25/vector provenance only.

**4b. Insert signal pool between fusion and rerank (lines ~3275-3316)**

Replace the current `fused_results[:rerank_pool_size]` logic:

```
If signal pool enabled:
    1. Run pre-rerank structural expansion on top 10 fused results
       (reuse _expand_with_structure with force=True)
    2. Call build_signal_pool(fused_results, structural_candidates, config)
    3. Hydrate parent_path_norm for pool candidates
    4. Pass 200-candidate pool to _apply_reranker
Else:
    existing behavior (flat top-N by fused_score)
```

**4c. Add `force` parameter to `_expand_with_structure` (line ~4445)**

Currently gated by `feature_flags.structure_aware_expansion` (default False). Add a `force=False` keyword arg that bypasses the feature flag check. This allows pre-rerank structural expansion without enabling the post-rerank expansion globally.

**4d. Log signal pool metrics**

Add to the metrics dict: `signal_pool_enabled`, `signal_pool_size`, `signal_pool_slot_fills`, `signal_pool_degraded`.

---

### Phase 5: Remove Dead Dominance Gating

**File: `src/query/hybrid_retrieval.py`**

Delete the `_gate_to_primary_document` nested function (lines 3345-3375) and its associated comments. Confirmed dead code — defined but never called. The signal-diverse pool inherently provides document diversity through the per-document depth slot and signal coverage across documents.

---

### Phase 6: Tests

**New file: `tests/query/test_signal_pool.py`**

Test the pure-function pool builder. Key test cases:
- Consensus fills first (top fused_score chunks always present)
- Signal-unique slots pull in non-consensus chunks (high on one signal, low on fused)
- Graceful degradation without per-field scores
- Structural slot includes expanded chunks
- Per-document depth ensures multi-doc representation
- No duplicates in pool
- Pool size never exceeds configured limit
- Backfill fills remaining capacity

**Pattern:** Use `ChunkResult` factory function to create test chunks with controlled per-field scores. No mocking needed — signal_pool.py is pure functions.

**Existing test updates:** Extend `tests/query/test_reranker_integration.py` with a test for `parent_path_norm` prepending in reranker text.

---

## Execution Order (Dependency Graph)

```
Phase 0 (token limits)  ──┐
Phase 1 (parent_path)   ──┤── all independent, can parallel
Phase 2 (config)        ──┘
                            ↓
Phase 3 (signal_pool.py) ── depends on Phase 2 (uses SignalPoolConfig)
                            ↓
Phase 4 (integration)   ── depends on Phase 1, 2, 3
                            ↓
Phase 5 (dead code)     ── depends on Phase 4
Phase 6 (tests)         ── Phase 3 tests can start with Phase 3; integration tests after Phase 4
```

**Recommended:** Phases 0+1+2 in parallel → Phase 3 → Phase 4 → Phase 5+6

---

## Files Summary

| File | Action | Phase |
|---|---|---|
| `src/providers/rerank/local_bge_service.py` | Modify: raise token limits | 0 |
| `src/query/hybrid_retrieval.py` | Modify: ChunkResult field, `_hydrate_parent_paths`, `_apply_reranker` text, `retrieve()` integration, `_expand_with_structure` force param, remove dead gating, raise batch token limit | 0,1,4,5 |
| `src/shared/config.py` | Modify: add `SignalPoolConfig`, update `HybridSearchConfig`, `FeatureFlagsConfig` | 2 |
| `src/query/signal_pool.py` | **NEW**: signal-diverse pool builder | 3 |
| `tests/query/test_signal_pool.py` | **NEW**: pool builder unit tests | 6 |
| `tests/query/test_reranker_integration.py` | Modify: add parent_path_norm test | 6 |

---

## Verification

After all phases:

```bash
# Unit tests
pytest tests/query/test_signal_pool.py -v
pytest tests/query/test_reranker_integration.py -v
pytest tests/unit/test_phase1_reranker_batching.py -v

# Entry point imports (no regressions)
python -c "from src.mcp_server.main import app"
python -c "from src.ingestion.worker import process_job"
python -c "from src.ingestion.auto.service import app"

# CI guard (no new ACTIVE→DEAD violations)
python scripts/ci/check_dead_imports.py

# Integration: enable in config and run a query
# Set feature_flags.signal_diverse_rerank_pool=True
# Set feature_flags.query_api_weighted_fusion=True
# Verify metrics contain signal_pool_* keys
# Verify reranker receives ~200 candidates
```

---

## Model Constraints Reference

| Model | Role | Context | Output | Constraint |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | Dense embedding | 32K tokens | 1024 dims (MRL 32-1024) | No constraint for chunks |
| Qwen3-Reranker-4B | Cross-encoder reranker | 8K tokens | yes/no logit | query + doc must fit 8K |
| ColBERTv2 (colbert-ai) | Late interaction | 512 tokens | 128 dims per token | **Bottleneck** — truncates chunks > 512 tokens |
| SPLADEv3 (NAVER) | Learned sparse | 512 tokens | ~30K vocab sparse | Trained at 128-256, works to 512 |

The 512-token ColBERT/SPLADE limit means chunks that were partially truncated during retrieval scoring can be evaluated in full by the 8K reranker — a key reason the signal-diverse pool adds value.


⏺ All green. Full end-to-end test:

⏺ Bash(echo "=== 1. DENSE EMBEDDINGS (Qwen3-0.6B) ==="
      curl -s -X POST http://10.25.0.50:8080/v1/embeddings \
        -H "Content-Type: application/json" \
        -d '{"input": "What is retrieval augmented generation?", "model": "qwen3-0.6b"}' | \
        python3 -c "import json,sys;d=json.load(sys.stdin);e=d['data'][0]['embedding'];print(f'dim={len(e)} first3={[round(x,5) for x in e[:3]]}')"

      echo ""
      echo "=== 2. SPARSE EMBEDDINGS (SPLADE v3) ==="
      curl -s -X POST http://10.25.0.50:8080/v1/embeddings/sparse \
        -H "Content-Type: application/json" \
        -d '{"input": "What is retrieval augmented generation?"}' | \
        python3 -c "import json,sys;d=json.load(sys.stdin);s=d['data'][0]['embedding'];print(f'non-zero terms: {len(s)}')"

      echo ""
      echo "=== 3. COLBERT ==="
      curl -s -X POST http://10.25.0.50:8080/v1/embeddings/colbert \
        -H "Content-Type: application/json" \
        -d '{"input": "What is retrieval augmented generation?", "input_type": "query"}' | \
        python3 -c "import json,sys;d=json.load(sys.stdin);i=d['data'][0];print(f'tokens={i[\"num_tokens\"]} dim={i[\"dim\"]}')"

      echo ""
      echo "=== 4. RERANKER (Qwen3-Reranker-4B) ==="
      curl -s -X POST http://10.25.0.50:8080/v1/rerank \
        -H "Content-Type: application/json" \
        -d '{
          "query": "What is RAG?",
          "documents": [
            "RAG combines retrieval with generation for grounded LLM responses.",
            "The weather is nice today in California."
          ]
        }' | python3 -c "import json,sys;d=json.load(sys.stdin);[print(f'  idx={r[\"index\"]} score={r[\"score\"]:.4f}') for r in d['results']]"

      echo ""
      echo "=== 5. NER (GLiNER) ==="
      curl -s -X POST http://10.25.0.50:8080/v1/extract \
        -H "Content-Type: application/json" \
        -d '{
          "texts": ["Dr. Sarah Chen at Stanford published a paper on RAG using NVIDIA GPUs in March 2025."],
          "labels": ["person", "organization", "technology", "date"]
        }' | python3 -c "import json,sys;d=json.load(sys.stdin);[print(f'  {e[\"text\"]:30s} → {e[\"label\"]:15s} ({e[\"score\"]:.3f})') for e in d['results'][0]['entities']]")
  ⎿  === 1. DENSE EMBEDDINGS (Qwen3-0.6B) ===
     dim=1024 first3=[-0.01376, -0.02621, -0.00628]

     === 2. SPARSE EMBEDDINGS (SPLADE v3) ===
     non-zero terms: 25

     === 3. COLBERT ===
     tokens=32 dim=128

     === 4. RERANKER (Qwen3-Reranker-4B) ===
       idx=0 score=0.9742
       idx=1 score=0.0003

     === 5. NER (GLiNER) ===
       Dr. Sarah Chen                 → person          (0.982)
       Stanford                       → organization    (0.973)
       RAG                            → technology      (0.861)
       NVIDIA GPUs                    → technology      (0.837)
       March 2025                     → date            (0.975)

⏺ All 5 endpoints working through the unified gateway on port 8080. Final stack:

  ┌───────────────────┬──────────────────────┬─────────────────────────────┬────────────────────────────┐
  │      Service      │        Model         │          Endpoint           │           Status           │
  ├───────────────────┼──────────────────────┼─────────────────────────────┼────────────────────────────┤
  │ Dense Embeddings  │ Qwen3-Embedding-0.6B │ :8080/v1/embeddings         │ 1024-dim vectors           │
  ├───────────────────┼──────────────────────┼─────────────────────────────┼────────────────────────────┤
  │ Sparse Embeddings │ SPLADE v3            │ :8080/v1/embeddings/sparse  │ 25 terms                   │
  ├───────────────────┼──────────────────────┼─────────────────────────────┼────────────────────────────┤
  │ ColBERT           │ ColBERT v2           │ :8080/v1/embeddings/colbert │ 32 tokens x 128d           │
  ├───────────────────┼──────────────────────┼─────────────────────────────┼────────────────────────────┤
  │ Reranker          │ Qwen3-Reranker-4B    │ :8080/v1/rerank             │ 8K context, P(yes) scoring │
  ├───────────────────┼──────────────────────┼─────────────────────────────┼────────────────────────────┤
  │ NER               │ GLiNER Medium v2.1   │ :8080/v1/extract            │ 5/5 entities               │
  └───────────────────┴──────────────────────┴─────────────────────────────┴────────────────────────────┘

  GPU: 12,875 MiB / 24,576 MiB (52%) — 11.2 GB headroom

  The two issues were:
  1. YAML profile placed outside profiles: block — fixed by rebuilding the config file properly
  2. curl not available in the Qwen3 runtime images (they use nvidia/cuda:*-runtime which is minimal) — fixed by switching healthchecks to Python-based urllib.request

  NIM reranker is fully replaced. The Qwen3-Reranker-4B gives you 8K token context (vs NIM's 512), yes/no probability scoring, and instruction-tunable queries — while using similar VRAM.
