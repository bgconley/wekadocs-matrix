# Hybrid RAG v2.2 Architecture

This document captures the working architecture for the v2.2 upgrade of the WekaDocs Matrix hybrid retrieval stack. It anchors implementation Sections 3–10 of the integration plan and serves as the canonical reference for engineers making changes across ingestion, retrieval, and supporting services.

---

## 1. Goals
- Adopt the Neo4j v2.2 schema (dual `document_id`/`doc_id`, additive indexes, richer chunk relationships).
- Move vector storage to Qdrant `chunks_multi` with named vectors (`content`, `title`, optional `entity`).
- Preserve the existing Phase-7E BM25 + vector fusion stack while enhancing it for multi-vector search and future reranking.
- Keep context-assembly/token budget guardrails intact (Jina v3 tokenizer, 4.5k-token LLM window).
- Lay groundwork for semantic chunking/NER-driven metadata and reranker integration without structural rewrites later.

---

## 2. System Overview

```
┌──────────────┐   parsed docs   ┌───────────────────┐   chunks + metadata   ┌──────────────┐
│  Parsers &   │───────────────▶│ Chunk Assembler   │───────────────────────▶│ GraphBuilder │
│  Extractors  │                │ (GreedyCombiner)  │                        │ (Neo4j +     │
└──────────────┘                └───────────────────┘                        │  Qdrant)     │
                                                                              └────┬────────┘
                                                                                   │
                                                                                   │ embeddings+payloads
                                                                                   ▼
                                                                             ┌──────────────┐
                                                                             │  Qdrant      │
                                                                             │  chunks_multi│
                                                                             └──────────────┘

┌──────────────┐   BM25 hits   ┌────────────────┐   weighted ANN hits   ┌────────────────────┐   context  ┌──────────┐
│ Neo4j FT     │──────────────▶│ HybridRetriever│◀──────────────────────│ QdrantWeightedSearcher│────────▶│Assembler │
│ (BM25)       │               │ (fusion, exp.) │                        │ (content/title/…)  │        │ (4.5k tok)│
└──────────────┘               └────────────────┘                        └────────────────────┘        └──────────┘
```

---

## 3. Key Components

### 3.1 Ingestion
- **Chunk Assembler (`src/ingestion/chunk_assembler.py`)**: Greedy combiner respecting section blocks, token budgets (Jina tokenizer), and provides hook for semantic chunking post-processing.
- The default assembler is now the **StructuredChunker**, which anchors on H1/H2 blocks, enforces deterministic ordering, splits deterministically when provider limits are exceeded, and emits `doc_is_microdoc` metadata (plus adjacency stubs when required) instead of collapsing small documents. This keeps expansion reliable while letting retrieval decide how to stitch microdocs.
- **GraphBuilder (`src/ingestion/build_graph.py`)**:
  - Writes Documents/Sections/Chunks into Neo4j with both `document_id` and `doc_id` set.
  - Executes schema v2.2 relationship builders (NEXT, PREV, SAME_HEADING, CHILD_OF, MENTIONS, PARENT_OF) post-ingestion.
  - Computes embeddings using Jina v3 provider and upserts to Qdrant `chunks_multi` using named vectors.
  - Records metadata (doc_tag, lang, version, hashes, tenant) for filters.
- **Reconciler (`src/ingestion/reconcile.py`)**: Ensures Qdrant payloads + vectors mirror Neo4j data by scrolling `chunks_multi` and deleting/reinserting drift.

### 3.2 Vector Storage
- **Qdrant `chunks_multi`**
  - Named vectors: `content` (full chunk text), `title` (headings/title-focused embedding), `entity` (reserved for future NER embeddings, may initially reuse content vector).
  - Payload includes canonical chunk metadata plus future-ready fields (`entities`, `topics`, `semantic_hash`).
  - Collection ensures tuned HNSW parameters (m=48, ef_construct=256, etc.) and payload indexes for filterable fields (document_id, doc_tag, tenant, etc.).

### 3.3 Retrieval
- **BM25 Retriever**: Existing Neo4j full-text search acts as lexical branch.
- **QdrantWeightedSearcher** (new integration): issues multi-field ANN searches, returns `Candidate` objects with per-vector scores.
- **HybridRetriever**: orchestrates BM25 + ANN fusion (RRF or weighted), optional dominance gating, graph expansion (NEXT_CHUNK now, typed relationships optional), microdoc stitching, and token-aware packing.
- **ContextAssembler**: preserves headings/provenance, enforces 4.5k token budget using Jina tokenizer.
- **MCP Query Service**: wraps ChunkResult outputs for downstream APIs.

### 3.4 Future Hooks
- **Semantic Chunking / NER**
  - Post-assembler hook (`chunk_assembler._enrich_chunk`) plus `src/ingestion/semantic.py` stub provider already attach `semantic_metadata` shells and log metrics; StructuredChunker ensures every chunk receives the hook before persistence.
  - `GraphBuilder._extract_semantic_metadata` forwards any enriched metadata into Neo4j/Qdrant payloads, and the `entity` vector slot is reserved for future NER embeddings.
  - **Next steps:** swap the stub for a real enricher, add entity vectors via Jina or HF models, and enable the feature by setting `chunk_assembly.semantic.enabled=true` in config.
- **Reranker (e.g., Jina reranker v3)**
  - `_apply_reranker` hook sits between fusion and expansion; config placeholders under `search.hybrid.reranker` control provider/model settings. Metrics already expose `reranker_applied` and `reranker_time_ms` (currently zero).
  - **Next steps:** implement the Jina reranker client (using `JINA_API_KEY`), flip the config flag, and extend tests/dashboards to track reranker throughput/latency.

---

## 4. Data Flow Summary
1. **Ingestion**
   - Parse document → assemble sections into chunks → assign canonical metadata (IDs, headings, tokens, doc_tag, etc.).
   - Upsert into Neo4j with schema-instrumented properties and relationships.
   - Compute embeddings (content/title[/entity]) and upsert into Qdrant `chunks_multi`.
   - Reconciler ensures vector store matches graph state; cleanup script enables resets.

2. **Retrieval**
   - Query triggers BM25 search + multi-vector ANN search.
   - QdrantWeightedSearcher returns candidates with per-field scores and payload metadata.
   - HybridRetriever fuses rankings, optionally reranks, applies graph expansion, microdoc extras, then enforces token budgets.
   - ContextAssembler produces ordered text blocks (with chunk IDs) for the LLM.

---

## 5. Deployment / Ops Considerations
- **Bootstrap**: run Neo4j schema script + Qdrant setup prior to ingest.
- **Cleanup**: updated `scripts/cleanup-databases.py` resets Neo4j/Qdrant/Redis while keeping schema metadata.
- **Metrics**: ingestion + retrieval metrics tagged with schema version, collection name, and reranker status.
- **Config**: all tunables (vector field weights, graph traversal mode, reranker settings) live under `config/search` with environment overrides.

---

This architecture document will be kept in sync with implementation progress. All subsequent sections of the integration plan reference the components detailed here.
