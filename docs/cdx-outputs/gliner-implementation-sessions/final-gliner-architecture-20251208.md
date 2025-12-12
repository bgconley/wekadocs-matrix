Now I have the complete picture. Let me provide you with a comprehensive explanation:

  ---
  GLiNER Entity Embedding & Retrieval Flow

  Part 1: Ingestion (Document Time)

  GLiNER entities are embedded in 3 different ways during ingestion:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                    GLiNER ENTITY EMBEDDING PATHS                            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │  1. DENSE VECTORS (content, title, doc_title) ── Uses _embedding_text ──►  │
  │     ┌───────────────────────────────────────────────────────────────────┐  │
  │     │ Text: "WEKA setup guide"                                          │  │
  │     │ GLiNER extracts: [WEKA (weka_software_component)]                 │  │
  │     │ _embedding_text: "WEKA setup guide\n\n[Context: weka_software_    │  │
  │     │                   component: WEKA]"                               │  │
  │     │                                                                   │  │
  │     │ ──► BGE-M3 Dense Embedding (1024-D)                              │  │
  │     └───────────────────────────────────────────────────────────────────┘  │
  │                                                                             │
  │  2. COLBERT MULTI-VECTORS ────────────── Uses _embedding_text ──────────►  │
  │     ┌───────────────────────────────────────────────────────────────────┐  │
  │     │ Same _embedding_text: "WEKA setup...[Context: weka_software_...]" │  │
  │     │                                                                   │  │
  │     │ ──► BGE-M3 ColBERT Embedding (1024-D × N tokens)                 │  │
  │     │     Token-level embeddings include entity context tokens          │  │
  │     └───────────────────────────────────────────────────────────────────┘  │
  │                                                                             │
  │  3. ENTITY-SPARSE VECTOR ─────────────── Uses _mentions ────────────────►  │
  │     ┌───────────────────────────────────────────────────────────────────┐  │
  │     │ _mentions: [{name: "WEKA", type: "weka_software_component",       │  │
  │     │              source: "gliner", ...}]                              │  │
  │     │                                                                   │  │
  │     │ entity_text: "WEKA"  (concatenated entity names)                  │  │
  │     │                                                                   │  │
  │     │ ──► BGE-M3 Sparse Embedding (BM25-style term weights)            │  │
  │     └───────────────────────────────────────────────────────────────────┘  │
  │                                                                             │
  │  4. ENTITY_METADATA (Qdrant Payload) ─── Stored for query-time boosting ─► │
  │     ┌───────────────────────────────────────────────────────────────────┐  │
  │     │ entity_metadata: {                                                │  │
  │     │   entity_types: ["weka_software_component"],                      │  │
  │     │   entity_values: ["WEKA"],                                        │  │
  │     │   entity_values_normalized: ["weka"],  # For case-insensitive     │  │
  │     │   entity_count: 1                                                 │  │
  │     │ }                                                                 │  │
  │     └───────────────────────────────────────────────────────────────────┘  │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘

  Code References:

  | Embedding Path  | Source Field      | Code Location                                    |
  |-----------------|-------------------|--------------------------------------------------|
  | Dense (content) | _embedding_text   | atomic.py:1207-1209 → batch_content → embed()    |
  | Dense (title)   | title (unchanged) | atomic.py:1270 → embed()                         |
  | ColBERT         | _embedding_text   | atomic.py:1383 → embed_colbert(batch_content)    |
  | Entity-Sparse   | _mentions         | atomic.py:1489-1534 → embed_sparse(entity_names) |

  ---
  Part 2: Retrieval (Query Time)

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                        RETRIEVAL PIPELINE FLOW                              │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │  USER QUERY: "How do I mount WEKA on RHEL?"                                │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │ Step 0: GLiNER Query Disambiguation (hybrid_retrieval.py:2781-2793)  │  │
  │  │         Extract entities: [WEKA, RHEL]                               │  │
  │  │         boost_terms: ["weka", "rhel"]                                │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │ Step 1: Vector Search (6-signal RRF) (line 2811-2870)                │  │
  │  │                                                                      │  │
  │  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │  │
  │  │   │content-dense│  │ title-dense │  │doc_title-   │                 │  │
  │  │   │  (1024-D)   │  │  (1024-D)   │  │dense (1024) │                 │  │
  │  │   │  weight=1.0 │  │  weight=1.0 │  │  weight=1.0 │                 │  │
  │  │   └─────────────┘  └─────────────┘  └─────────────┘                 │  │
  │  │                                                                      │  │
  │  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │  │
  │  │   │ text-sparse │  │title-sparse │  │entity-sparse│ ◄── GLiNER     │  │
  │  │   │  (BM25)     │  │  (BM25)     │  │  (BM25)     │     entities!  │  │
  │  │   │  weight=1.0 │  │ weight=2.0  │  │ weight=1.5  │                 │  │
  │  │   └─────────────┘  └─────────────┘  └─────────────┘                 │  │
  │  │                                                                      │  │
  │  │   ──► RRF Fusion combines all 6 signals                             │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │ Step 2: RRF Fusion (line 2877-2922)                                  │  │
  │  │         Reciprocal Rank Fusion across all 6 signals                  │  │
  │  │         Output: fused_results (sorted by fused_score)                │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │ Step 3: GLiNER Entity Boosting (line 2927-2946)  ◄── POST-RETRIEVAL │  │
  │  │                                                                      │  │
  │  │   for each result:                                                   │  │
  │  │     entity_values = result.entity_metadata.entity_values_normalized  │  │
  │  │     matches = count(boost_terms ∩ entity_values)                     │  │
  │  │     if matches > 0:                                                  │  │
  │  │       boost = 1.0 + min(0.5, matches * 0.1)  # Max 50% boost        │  │
  │  │       result.fused_score *= boost                                    │  │
  │  │       result.entity_boost_applied = True                             │  │
  │  │                                                                      │  │
  │  │   Re-sort results after boosting                                     │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │ Step 4: ColBERT Reranking (optional) (line 2988-3043)               │  │
  │  │         Token-level MaxSim scoring                                   │  │
  │  │         Uses enriched embeddings (with entity context)               │  │
  │  │         Does NOT directly see entity_metadata                        │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ┌──────────────────────────────────────────────────────────────────────┐  │
  │  │ Step 5: BGE Cross-Encoder Reranking (optional) (line 3046-3090)     │  │
  │  │         Final authoritative scoring (rerank_score)                   │  │
  │  │         Uses original text, NOT enriched text                        │  │
  │  └──────────────────────────────────────────────────────────────────────┘  │
  │       │                                                                     │
  │       ▼                                                                     │
  │  FINAL RESULTS                                                              │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘

  ---
  Summary: Where GLiNER Entities Impact Retrieval

  | Stage                | GLiNER Impact | How                                                                     |
  |----------------------|---------------|-------------------------------------------------------------------------|
  | Dense Vectors        | ✅ YES         | _embedding_text includes [Context: entities]                            |
  | ColBERT Vectors      | ✅ YES         | Same _embedding_text used                                               |
  | Entity-Sparse Vector | ✅ YES         | _mentions → entity names → sparse embedding                             |
  | RRF Fusion           | ✅ Indirect    | Entity-sparse contributes with weight 1.5x                              |
  | Post-Retrieval Boost | ✅ YES         | entity_metadata.entity_values_normalized matched against query entities |
  | ColBERT Rerank       | ✅ Indirect    | Stored ColBERT vectors contain entity context tokens                    |
  | BGE Cross-Encoder    | ❌ NO          | Uses original text, not _embedding_text                                 |

  ★ Insight ─────────────────────────────────────
  Key Design Decision: GLiNER entities influence retrieval through 4 distinct channels:
  1. Semantic boost (dense embeddings see entity context)
  2. Token-level matching (ColBERT tokens include entity terms)
  3. Lexical matching (entity-sparse vector with 1.5x RRF weight)
  4. Post-retrieval re-scoring (entity_metadata payload matching)

  The cross-encoder reranker is intentionally NOT affected - it uses original clean text. This prevents "over-fitting" to entity matches and maintains cross-encoder's role as an independent final arbiter.
  ─────────────────────────────────────────────────

  ---
  The _embedding_text Pattern (Critical!)

  # From ner_gliner.py:109-114
  if title:
      chunk["_embedding_text"] = (
          f"{title}\n\n{base_text}\n\n[Context: {entity_context}]"
      )
  else:
      chunk["_embedding_text"] = f"{base_text}\n\n[Context: {entity_context}]"

  Example:
  - Original text: "Mount the filesystem using the mount command"
  - GLiNER entities: [("NFS", "network_or_storage_protocol"), ("mount", "cli_command")]
  - _embedding_text: "Mount the filesystem using the mount command\n\n[Context: network_or_storage_protocol: NFS; cli_command: mount]"

  This enriched text is what gets embedded into:
  - Dense content vector (1024-D)
  - ColBERT multi-vectors (1024-D × N tokens)

  The original chunk["text"] remains unchanged and is what gets stored in Qdrant payload and shown to users.
