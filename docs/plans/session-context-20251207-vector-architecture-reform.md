# Session Context & Master Plan: Vector Architecture Reform

**Session Date:** 2025-12-07
**Status:** Approved for Implementation (Master Plan v2.1)
**Focus:** Vector pipeline optimization, Semantic Chunking, GLiNER NER, Metadata Enrichment
**Target:** Vector retrieval robustness (Graph/Neo4j work deferred)

---

## 1. Executive Summary

This document serves as the **Master Plan** for reforming the `wekadocs-matrix` vector pipeline. It synthesizes the research findings from the Dec 7th session with specific, code-level implementation plans for **Semantic Chunking**, **GLiNER Entity Enrichment**, and **YAKE Keyphrase Extraction**.

**Strategic Decision:**
The focus is exclusively on **vector retrieval robustness**. All Neo4j/graph-specific enhancements (graph traversal, microdoc stubs) are deferred. The goal is to maximize retrieval precision by enriching the vector space with semantic concepts and ensuring chunks represent complete thoughts.

**Core Pillars:**
1.  **Semantic Chunking:** Replacing structural splitting with BGE-M3 embedding-similarity chunking.
2.  **Semantic Enrichment (GLiNER):** Zero-shot NER to tag technical entities (`inode`, `error 10054`) using a "safe injection" pattern (`_embedding_text`).
3.  **Topic Fingerprinting (YAKE):** Unsupervised keyphrase extraction for domain topics (`high availability`) stored in a new `keyphrase-sparse` vector.
4.  **Safe Integration:** All enrichment is filtered from Neo4j to prevent "ghost node" pollution.

---

## 2. Infrastructure Access & Architecture

### 2.1 Container Architecture (Reference)

| Container | Purpose | Key Ports | Persistent Data | Mount Pattern |
|-----------|---------|-----------|-----------------|---------------|
| `weka-qdrant` | Vector store | 6333, 6334 | Yes (volume) | N/A |
| `weka-mcp-server` | HTTP MCP + STDIO | 8000 | No | `./src:/app/src:ro` |
| `weka-ingestion-worker` | RQ background jobs | None | No | `./src:/app/src:ro` |
| `weka-neo4j` | Graph database | 7687, 7474 | Yes (volume) | N/A |

### 2.2 Embedding Services (Host Access)

*   **BGE-M3 Service:** `http://127.0.0.1:9000` (Dense, Sparse, ColBERT)
    *   *Critical:* Used for both Semantic Chunking and Vector Generation.
*   **Reranker Service:** `http://127.0.0.1:9001` (BAAI/bge-reranker-v2-m3)

### 2.3 Vector Schema (Target: 9 Vectors)

| Vector Name | Type | Purpose | Source |
|-------------|------|---------|--------|
| `content` | Dense | Main semantic meaning | BGE-M3 (Enriched text) |
| `title` | Dense | Heading semantic meaning | BGE-M3 |
| `doc_title` | Dense | Document context | BGE-M3 |
| `text-sparse` | Sparse | BM25-style content match | BGE-M3 |
| `title-sparse` | Sparse | Heading lexical match | BGE-M3 |
| `doc_title-sparse` | Sparse | Doc title lexical match | BGE-M3 |
| `entity-sparse` | Sparse | Entity name match | **GLiNER** |
| `keyphrase-sparse` | Sparse | Topic/Keyword match | **YAKE** (New) |
| `late-interaction` | Multi | Fine-grained token match | ColBERT |

---

## 3. Implementation Plan: Core Components

### 3.1 Semantic Chunking (BGE-M3)

**Objective:** Ensure chunks represent complete semantic thoughts/topics, replacing arbitrary token splitting.

**Implementation:** `src/ingestion/chunking/semantic.py`

```python
def semantic_chunk(text: str, model: BGE_M3, percentile: int = 80) -> list[str]:
    # 1. Tokenize into sentences
    sentences = sent_tokenize(text)

    # 2. Add context window (±1 sentence)
    combined = [" ".join(sentences[max(0, i-1):min(len(sentences), i+2)]) for i in range(len(sentences))]

    # 3. Generate dense embeddings
    embeddings = model.encode(combined)

    # 4. Calculate cosine distance between consecutive sentences
    distances = [1 - cosine_similarity(e[i], e[i+1]) for i in range(len(embeddings)-1)]

    # 5. Determine breakpoints (peaks > 80th percentile)
    threshold = np.percentile(distances, percentile)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    # 6. Create chunks
    return create_chunks(sentences, breakpoints)
```

---

### 3.2 Semantic Enrichment (GLiNER)

**Objective:** Inject domain-specific understanding (`inode` is a filesystem object, `10054` is an error code) into the vector space.

**Service:** `src/providers/ner/gliner_service.py` (Singleton, Auto-Device, Circuit Breaker).

**Configuration:** `config/development.yaml`
```yaml
ner:
  enabled: false # Gated
  model_name: "urchade/gliner_medium-v2.1"
  device: "auto" # mps/cuda/cpu
  labels:
    - "weka_software_component (e.g. backend, frontend, agent, client)"
    - "operating_system (e.g. RHEL, Ubuntu, Rocky Linux)"
    - "hardware_component (e.g. NVMe, NIC, GPU, switch)"
    - "filesystem_object (e.g. inode, snapshot, file, directory)"
    - "cloud_provider_or_service (e.g. AWS, S3, Azure, EC2)"
    - "cli_command (e.g. weka fs, mount, systemctl)"
    - "configuration_parameter (e.g. --net-apply, stripe-width)"
    - "network_or_storage_protocol (e.g. NFS, SMB, S3, POSIX, TCP)"
    - "error_message_or_code (e.g. 10054, Connection refused)"
    - "performance_metric (e.g. IOPS, latency, throughput)"
    - "file_system_path (e.g. /mnt/weka, /etc/fstab)"
```

**Safe Integration Hook:** `src/ingestion/extract/ner_gliner.py` -> `atomic.py`

1.  **Transient Text Injection:**
    *   Create `_embedding_text` = `Original Text... [Context: error_code: 10054; component: inode]`.
    *   `atomic.py` uses this for embedding, then discards it. **Stored text remains clean.**
2.  **Graph Hygiene:**
    *   Add entities to `_mentions` with `source="gliner"`.
    *   `atomic.py` **filters these out** when creating Neo4j `MENTIONS` edges to prevent ghost nodes.
3.  **Entity-Sparse Vector:**
    *   GLiNER entities contribute to the `entity-sparse` vector generation.

---

### 3.3 Topic Fingerprinting (YAKE)

**Objective:** Capture significant topics ("high availability", "stripe width") that aren't strictly entities but are critical for retrieval.

**Tool:** **YAKE** (Yet Another Keyword Extractor).
*   **Why?** Unsupervised, statistical, lightweight, and strictly domain-agnostic. No spaCy dependency (avoids the "news data" bias).

**Implementation:** `src/ingestion/extract/keyphrase_yake.py`
*   Extract top-N keyphrases.
*   Join into a string.
*   Generate `keyphrase-sparse` vector via BGE-M3.

---

## 4. Retrieval & Ranking Strategy

### 4.1 Post-Retrieval Boosting (Soft Filter)

**Problem:** Qdrant filters are strict (`must`).
**Solution:** Python-side Rescoring in `HybridRetriever`.

1.  **Query Analysis:** Run GLiNER on user query to find entities (cached).
2.  **Retrieval:** Fetch top-K candidates (e.g., 40).
3.  **Rescoring:** Iterate through candidates.
    *   Check metadata payload: `if query_entity in doc_entity_metadata`:
    *   Apply boost multiplier (e.g., 1.1x per match, capped).
4.  **Sort & Rerank:** Re-sort list and pass top 10 to Cross-Encoder.

---

## 5. Migration Strategy

**Strategy:** **Clean Re-ingestion**.

Because **Semantic Chunking** changes the fundamental boundaries of text chunks (based on meaning rather than headers/tokens), "updating in place" is impossible. Old chunks would have different IDs and content boundaries than new ones.

**Procedure:**
1.  **Delete:** Wipe Qdrant collection (`chunks_multi_bge_m3`) and Neo4j database.
2.  **Re-ingest:** Run the ingestion worker on the full corpus.
    *   Semantic Chunking applied.
    *   GLiNER & YAKE enrichment applied.
    *   9 vectors generated per chunk.

---

## 6. Known Issues & Architecture Notes

### 6.1 Architectural Issues (from Research)
*   **Microdoc Stubs:** 50 empty chunks exist in Neo4j (graph artifacts). **Decision:** Ignore/Defer. They don't affect vector search.
*   **doc_fallback Pollution:** Merging unrelated H1s. **Fix:** Set `doc_fallback_enabled: false` in config.
*   **build_graph.py Bug:** Mentions attachment logic is flawed. **Fix:** Irrelevant; we are using the `atomic.py` hook which bypasses this path.

### 6.2 Design Note: atomic.py vs build_graph.py
We hook into `atomic.py` because GLiNER's role here is **vector enrichment**, not graph structure definition. By hooking at the transactional layer (just before embedding), we keep the "Semantic Layer" (GLiNER) separate from the "Structural Layer" (Regex/Markdown) until they merge at the vector level.

---

## 7. Implementation Checklist

### Phase 1: Infrastructure
- [ ] Add dependencies (`gliner>=0.2.24`, `yake`, `nltk`, `scipy`).
- [ ] Create `GLiNERService` (Singleton, Auto-Device, Observability).
- [ ] Update `config/development.yaml` (NER config, Labels).

### Phase 2: Ingestion
- [ ] Implement `SemanticChunker` logic.
- [ ] Implement `enrich_chunks_with_entities` (GLiNER hook).
- [ ] Implement `extract_keyphrases` (YAKE hook).
- [ ] Modify `atomic.py`:
    - [ ] Call enrichers.
    - [ ] Consume `_embedding_text`.
    - [ ] Filter `source="gliner"` from Neo4j.

### Phase 3: Schema
- [ ] Update `qdrant_schema.py` (add `keyphrase-sparse`).
- [ ] Update RRF weights in config.

### Phase 4: Retrieval
- [ ] Update `QueryDisambiguator` (GLiNER extraction).
- [ ] Update `HybridRetriever` (Post-Retrieval Boosting).

### Phase 5: Execution
- [ ] Clean Re-ingestion.

---

## 8. Coding Standards & Modularity Strategy

**Objective:** Prevent code bloat and maintain maintainability, particularly given the existing large files (`atomic.py`: ~3.6k lines, `hybrid_retrieval.py`: ~5k lines).

### 8.1 File Size Guidelines
*   **300-500 lines:** Soft ceiling. Consider splitting.
*   **1000 lines:** Hard review trigger. File is likely doing too much.
*   **Exception:** Complex, cohesive algorithms (e.g., a self-contained graph algorithm) or legacy files we are actively refactoring *out* of.

### 8.2 Anti-Monolith Strategy
We will strictly avoid expanding existing monolithic files.
*   **`atomic.py` strategy:** Do NOT add implementation logic here.
    *   **Bad:** Writing the GLiNER extraction loop inside `atomic.py`.
    *   **Good:** `from src.ingestion.extract.ner_gliner import enrich_chunks`. The hook in `atomic.py` should be 1-3 lines max.
*   **`hybrid_retrieval.py` strategy:** Do NOT add complex NLP logic here.
    *   **Bad:** Implementing GLiNER query analysis inside `retrieve()`.
    *   **Good:** Delegate to `QueryDisambiguator` class in a separate module.

### 8.3 Module Structure
*   `src/ingestion/extract/ner_gliner.py`: Dedicated module for GLiNER logic.
*   `src/ingestion/extract/keyphrase_yake.py`: Dedicated module for YAKE logic.
*   `src/ingestion/chunking/semantic.py`: Dedicated module for Semantic Chunking logic.
*   `src/providers/ner/gliner_service.py`: Dedicated service class.

This ensures new features are isolated, testable, and do not contribute to the technical debt of existing large files.

---
*Based on Session Context research and GLiNER Implementation Plan v1.5.*
