# Vector Pipeline Reform & Semantic Enrichment Master Plan

**Status:** Approved for Implementation
**Version:** 2.0 (Master Plan)
**Target:** Vector Pipeline Optimization (Graph/Neo4j work deferred)
**Date:** December 7, 2025

---

## 1. Executive Summary

This Master Plan outlines the comprehensive reform of the `wekadocs-matrix` RAG pipeline. It synthesizes extensive research into **Semantic Chunking**, **Metadata Enrichment**, and **Zero-Shot NER** to address current limitations in chunking quality and semantic understanding.

**Strategic Focus:**
The user has explicitly directed a focus **exclusively on making the vector pipeline more robust**. All Neo4j/graph-specific enhancements (graph traversal, microdoc stubs) are deferred. The goal is to get vector retrieval "absolutely singing" before addressing graph enhancements.

**Core Pillars:**
1.  **Semantic Chunking:** Replacing structural splitting with BGE-M3 embedding-similarity chunking to preserve semantic boundaries.
2.  **Semantic Enrichment (GLiNER):** Implementing zero-shot NER to tag technical entities (`inode`, `error 10054`) and boost them in retrieval.
3.  **Topic Fingerprinting (YAKE):** Unsupervised keyphrase extraction to capture domain-specific topics (`high availability`, `stripe width`) without training.
4.  **Multi-Vector Fusion:** Expanding the Qdrant schema to 9 vectors per chunk, utilizing weighted RRF to blend semantic and lexical signals.

---

## 2. Infrastructure & Environment Context

### 2.1 Container Architecture
| Container | Purpose | Key Ports | Persistent Data |
|-----------|---------|-----------|-----------------|
| `weka-qdrant` | Vector store | 6333, 6334 | Yes (volume) |
| `weka-mcp-server` | HTTP MCP + STDIO | 8000 | No (ro volume) |
| `weka-ingestion-worker` | RQ background jobs | None | No (ro volume) |
| `weka-neo4j` | Graph database (Bypassed for retrieval) | 7687, 7474 | Yes (volume) |

### 2.2 Embedding Services (Host Access)
*   **BGE-M3 Service:** `http://127.0.0.1:9000` (Dense, Sparse, ColBERT)
*   **Reranker Service:** `http://127.0.0.1:9001` (BAAI/bge-reranker-v2-m3)

### 2.3 Current Vector Schema (Targeting 9 Vectors)

| Vector Name | Type | Dimensions | Purpose | Status |
|-------------|------|------------|---------|--------|
| `content` | Dense | 1024 | Main semantic content | Existing |
| `title` | Dense | 1024 | Section heading semantic | Existing |
| `doc_title` | Dense | 1024 | Document title semantic | Existing |
| `text-sparse` | Sparse | Variable | BM25-style lexical content | Existing |
| `title-sparse` | Sparse | Variable | Section heading lexical | Existing |
| `doc_title-sparse` | Sparse | Variable | Document title lexical | Existing |
| `entity-sparse` | Sparse | Variable | Entity name lexical (GLiNER) | **Refining** |
| `keyphrase-sparse` | Sparse | Variable | Topic/Keyword lexical (YAKE) | **NEW** |
| `late-interaction` | Multi | 1024×N | ColBERT MaxSim | Existing |

---

## 3. Architecture & Data Flow

### The "Enriched Vector" Pipeline

```
Document
    │
    ▼
┌─────────────────────────────────────────────────┐
│  SEMANTIC CHUNKING (BGE-M3)                     │
│  ├── Sentence tokenization                      │
│  ├── Context window (±1 sentence)               │
│  ├── Dense embedding per sentence               │
│  ├── Cosine distance calculation                │
│  └── Percentile-based breakpoint detection      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  METADATA ENRICHMENT (atomic.py Hook)           │
│  │                                              │
│  ├── [GLiNER Service] (Zero-Shot NER)           │
│  │   ├── Extract: "error 10054", "inode"        │
│  │   ├── Transient Injection: "_embedding_text" │
│  │   └── Output: entity-sparse vector           │
│  │                                              │
│  ├── [YAKE Service] (Keyphrase Extraction)      │
│  │   ├── Extract: "performance tuning"          │
│  │   └── Output: keyphrase-sparse vector        │
│  │                                              │
│  └── [Context Prepend]                          │
│      └── Prepend doc_title to chunk text        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  MULTI-VECTOR EMBEDDING (BGE-M3)                │
│  │                                              │
│  ├── Dense Vectors (Content, Title)             │
│  ├── Sparse Vectors (Text, Entity, Keyphrase)   │
│  └── ColBERT Vectors                            │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STORAGE & INDEXING                             │
│  ├── Qdrant: 9 Vectors + Metadata Payload       │
│  └── Neo4j: Graph Nodes (Filtered GLiNER)       │
└─────────────────────────────────────────────────┘
```

---

## 4. Core Component: Semantic Chunking

**Objective:** Replace arbitrary token/header splitting with semantic boundaries to ensure chunks represent complete thoughts.

### 4.1 Research Basis
Benchmarks (Superlinked, Firecrawl) show embedding-similarity chunking achieves **88-91% recall**, significantly outperforming hierarchical clustering and approaching LLM-based chunking without the latency cost.

### 4.2 Algorithm (Implementation Detail)
We will implement `SemanticChunker` in `src/ingestion/chunking/semantic.py`.

```python
def semantic_chunk(text: str, model: BGE_M3, percentile: int = 80) -> list[str]:
    # 1. Tokenize into sentences
    sentences = sent_tokenize(text)

    # 2. Add context window (±1 sentence) to smooth transitions
    combined = [" ".join(sentences[max(0, i-1):min(len(sentences), i+2)]) for i in range(len(sentences))]

    # 3. Generate dense embeddings
    embeddings = model.encode(combined)

    # 4. Calculate cosine distance between consecutive sentences
    distances = [1 - cosine_similarity(e[i], e[i+1]) for i in range(len(embeddings)-1)]

    # 5. Determine breakpoints (peaks in distance)
    threshold = np.percentile(distances, percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    # 6. Group sentences into chunks based on breakpoints
    return create_chunks(sentences, breakpoints)
```

**Configurable Parameters:**
*   `percentile_threshold` (Default: 80): Controls granularity.
*   `min_chunk_tokens` (Default: 100): Prevents fragmentation.
*   `max_chunk_tokens` (Default: 1000): Enforces upper limit.

---

## 5. Core Component: GLiNER Entity Enrichment

**Objective:** Inject domain-specific semantic understanding ("This is an error code", "This is a filesystem object") into the vector space.

### 5.1 Service Architecture
Implemented as a Singleton service in `src/providers/ner/gliner_service.py` with:
*   **Auto-Device Detection:** `mps` (Apple), `cuda` (NVIDIA), or `cpu`.
*   **Circuit Breaker:** Graceful degradation if model fails.
*   **Caching:** LRU cache for query extraction to minimize latency.
*   **Observability:** Prometheus metrics for extraction time and entity counts.

### 5.2 Safe Integration Strategy (The "Atomic Hook")
We hook into `src/ingestion/atomic.py` to enrich chunks *just before* embedding.

**Critical Safety Mechanisms:**
1.  **Transient Text Injection:** We DO NOT modify `chunk["text"]` (which is stored/displayed). We create a transient `_embedding_text`:
    `Original content... [Context: error_code: 10054; component: inode]`
    `atomic.py` uses this for embedding generation, then discards it.
2.  **Graph Hygiene:** GLiNER entities are added to `_mentions` with `source="gliner"`. The Neo4j writer filters these out to prevent "ghost nodes" in the graph.
3.  **Additive Sparse:** GLiNER entities contribute to the `entity-sparse` vector, providing a "semantic keyword index".

### 5.3 Configuration (`config/development.yaml`)
We use a comprehensive label set tailored to high-performance storage:

```yaml
ner:
  enabled: false # Gated
  model_name: "urchade/gliner_medium-v2.1"
  device: "auto"
  labels:
    - "software"
    - "hardware"
    - "filesystem_object"       # inode, snapshot, dentry
    - "architecture_component"  # backend, frontend, failure domain
    - "cloud_provider"          # AWS, Azure
    - "cloud_service"           # S3, EC2
    - "command"
    - "parameter"
    - "protocol"                # NFS, SMB
    - "error_message"
    - "metric"
    - "concept"
```

---

## 6. Core Component: Keyphrase Extraction (YAKE)

**Objective:** Capture significant topics ("high availability", "stripe width") that aren't strictly entities but are critical for retrieval.

### 6.1 Tool Selection: Why YAKE?
Research identified `pytextrank` (spaCy-based) vs `YAKE`.
*   **Decision:** **YAKE** (Yet Another Keyword Extractor).
*   **Reasoning:** Unsupervised, statistical, lightweight, and strictly domain-agnostic. It does not require training or heavy linguistic models (like spaCy), making it robust for technical jargon.

### 6.2 Implementation
*   **Helper:** `src/ingestion/extract/keyphrase_yake.py`
*   **Process:** Extract top-N keyphrases from chunk text.
*   **Vectorization:** Convert list to string `keyphrase1 keyphrase2...` and generate sparse embedding via BGE-M3.
*   **Storage:** Stored in new `keyphrase-sparse` vector in Qdrant.

---

## 7. Retrieval Strategy: Post-Retrieval Boosting

**Problem:** Qdrant filters are strict (`must`). We want to *boost* documents that match entities, not exclude those that don't.

**Solution:** Python-side Rescoring in `HybridRetriever`.

1.  **Query Analysis:** Run GLiNER on user query to find entities (e.g., "10054").
2.  **Retrieval:** Fetch top-K candidates (e.g., 40).
3.  **Rescoring:** Iterate through candidates. Check if their metadata payload contains "10054".
4.  **Boost:** Apply a multiplier (e.g., 1.1x) for matches.
5.  **Sort & Rerank:** Re-sort list and pass top results to Cross-Encoder.

---

## 8. Implementation Plan

### Phase 1: Core Infrastructure
1.  **Dependencies:** Update `requirements.txt` (gliner, yake, nltk).
2.  **Config:** Update `config/development.yaml` and `src/shared/config.py`.
3.  **Services:** Implement `GLiNERService` (with observability) and `KeyphraseService`.

### Phase 2: Ingestion Logic
1.  **Semantic Chunker:** Implement `SemanticChunker` logic.
2.  **Enrichment Hooks:** Create `src/ingestion/extract/ner_gliner.py`.
3.  **Atomic Integration:** Modify `src/ingestion/atomic.py`:
    *   Inject `_embedding_text`.
    *   Populate `_mentions` with `source="gliner"`.
    *   Filter `source="gliner"` from Neo4j edges.

### Phase 3: Vector Schema
1.  **Schema Update:** Add `keyphrase-sparse` to `src/shared/qdrant_schema.py`.
2.  **RRF Weights:** Configure weights in `development.yaml` (keyphrase-sparse: 1.5).

### Phase 4: Retrieval Logic
1.  **Query Disambiguator:** Update to use GLiNER (cached).
2.  **Hybrid Retriever:** Implement `_apply_entity_boost` method for rescoring.

### Phase 5: Execution (Clean Re-ingestion)
**Strategy:** "Clean Slate".
Because Semantic Chunking changes chunk boundaries, we cannot update in place.
1.  **Wipe:** Delete Qdrant collection and Neo4j database.
2.  **Re-ingest:** Process the full corpus through the new pipeline.

---

## 9. Known Issues & Mitigations (From Research)

| Issue | Status | Mitigation Strategy |
|-------|--------|---------------------|
| **Microdoc Stubs Empty** | Deferred | These are graph artifacts; ignored by vector search. Acceptable for Vector-Only phase. |
| **doc_fallback Pollution** | Pending | Set `doc_fallback_enabled: false` in config to prevent merging unrelated headers. |
| **build_graph.py Bug** | Deferred | We are using `atomic.py` path for enrichment, bypassing this legacy bug. |
| **Latency** | Risk | Semantic Chunking + GLiNER is heavy. Monitor ingestion times; increase timeouts if needed. |

---

## 10. Success Metrics

1.  **Recall:** Significant improvement for "concept" queries ("how does weka manage inodes").
2.  **Precision:** Reduction in generic answers for specific error codes or parameters.
3.  **Stability:** Pipeline handles full corpus re-ingestion without OOM.

---
*Based on Session Context research and refined GLiNER implementation architecture.*
