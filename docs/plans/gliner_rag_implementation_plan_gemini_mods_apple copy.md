# GLiNER Integration Implementation Plan
## Entity-Enhanced RAG with Qdrant + BGE-M3 Multi-Vector Embeddings

**Version:** 1.5 (Final Polish & Code Snippets)
**Target:** Agentic Coder with Codebase Context
**Last Updated:** December 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Dependencies & Model Selection](#3-dependencies--model-selection)
4. [Phase 1: Core Infrastructure](#4-phase-1-core-infrastructure)
5. [Phase 2: Document Ingestion Pipeline](#5-phase-2-document-ingestion-pipeline)
6. [Phase 4: Qdrant Collection Schema](#6-phase-4-qdrant-collection-schema)
7. [Phase 5: Hybrid Search Implementation](#7-phase-5-hybrid-search-implementation)
8. [Phase 6: Performance Optimization](#8-phase-6-performance-optimization)
9. [Phase 7: Migration Strategy](#9-phase-7-migration-strategy)
10. [Configuration Reference](#10-configuration-reference)
11. [Testing & Validation](#11-testing--validation)

---

## 1. Executive Summary

### Objective
Integrate GLiNER (Generalist Lightweight Named Entity Recognition) into an existing Qdrant + BGE-M3 RAG pipeline to enhance retrieval quality.

**Key Features (v1.5):**
*   **Non-Destructive Text Injection:** Uses transient `_embedding_text`.
*   **Graph Consistency:** Filters GLiNER entities from Neo4j edges (Vector-only enrichment).
*   **Post-Retrieval Boosting:** Soft filtering via Python-side rescoring.
*   **Observability:** Full integration with Prometheus metrics and OpenTelemetry tracing.
*   **Safety:** Feature flags, circuit breakers, and auto-device detection.

### Expected Outcomes
- **Improved recall** for entity-centric queries.
- **Better disambiguation** for homonyms.
- **Clean Data:** No visible pollution of stored text.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ Raw Docs │───▶│ Entity-Aware  │───▶│   GLiNER     │───▶│  BGE-M3     │ │
│  │          │    │   Chunker     │    │  Extraction  │    │  Embedding  │ │
│  └──────────┘    └───────────────┘    └──────────────┘    └─────────────┘ │
│                                              │                    │        │
│                                              ▼                    ▼        │
│                                    ┌─────────────────────────────────────┐ │
│                                    │         QDRANT COLLECTION          │ │
│                                    │  ┌─────────┐ ┌────────┐ ┌────────┐ │ │
│                                    │  │ Dense   │ │ Sparse │ │ColBERT │ │ │
│                                    │  │ Vector  │ │ Vector │ │Vectors │ │ │
│                                    │  └─────────┘ └────────┘ └────────┘ │ │
│                                    │  ┌─────────────────────────────────┐ │ │
│                                    │  │     Entity Metadata Payload    │ │ │
│                                    │  │  - entity_types: [str]         │ │ │
│                                    │  │  - entity_values: [str]        │ │ │
│                                    │  │  - entity_map: {text: type}    │ │ │
│                                    │  └─────────────────────────────────┘ │ │
│                                    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUERY PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │  User    │───▶│   GLiNER      │───▶│   Query      │───▶│  BGE-M3     │ │
│  │  Query   │    │  Extraction   │    │ Augmentation │    │  Embedding  │ │
│  └──────────┘    └───────────────┘    └──────────────┘    └─────────────┘ │
│                         │                                        │         │
│                         ▼                                        ▼         │
│               ┌─────────────────┐                    ┌─────────────────┐  │
│               │ Entity Boosting │                    │  Multi-Vector   │  │
│               │ (Re-scoring)    │                    │     Query       │  │
│               └────────┬────────┘                    └────────┬────────┘  │
│                        │                                      │           │
│                        └──────────────┬───────────────────────┘           │
│                                       ▼                                    │
│                            ┌─────────────────────┐                        │
│                            │   Qdrant Hybrid     │                        │
│                            │   Search + Rerank   │                        │
│                            └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dependencies & Model Selection

### Required Packages

```python
# requirements.txt additions
gliner>=0.2.24
```

### Model Selection Guide

| Model | Size | Use Case | HuggingFace ID |
|---|---|---|---|
| **GLiNER Medium v2.1** | ~209M | Balanced (Recommended) | `urchade/gliner_medium-v2.1` |
| **GLiNER Large v2.5** | ~340M | Maximum Accuracy | `gliner-community/gliner_large-v2.5` |

**Recommendation:** Start with `gliner_medium-v2.1`.

---

## 4. Phase 1: Core Infrastructure

### 4.1 GLiNER Provider Service

**File:** `src/providers/ner/gliner_service.py`

*   **Auto-Device Detection:** Use `torch.backends.mps.is_available()` check.
*   **Circuit Breaker:** Wrap initialization in try/except.
*   **Caching:** Add LRU cache for query extraction.
*   **Observability:** Wraps calls with spans and metrics.

```python
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import torch
from gliner import GLiNER
from src.shared.observability import get_logger, metrics, trace
from src.shared.config import get_config

logger = get_logger(__name__)

# Metrics
GLINER_EXTRACTION_DURATION = metrics.histogram(
    "gliner_extraction_duration_seconds", "Time spent extracting entities"
)
GLINER_ENTITY_COUNT = metrics.counter(
    "gliner_entities_extracted_total", "Total entities extracted", ["label"]
)

@dataclass(frozen=True) # Frozen for hashing/caching
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float

class GLiNERService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GLiNERService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        config = get_config()
        self.config = config.ner
        self.model_name = self.config.model_name
        self.threshold = self.config.threshold
        self.default_batch_size = self.config.batch_size

        # Auto-detect device
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        self._model: Optional[GLiNER] = None
        self._initialized = True

    def initialize(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading GLiNER model: {self.model_name} on {self.device}")
        try:
            self._model = GLiNER.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            logger.info("GLiNER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            self._model = None

    @lru_cache(maxsize=1000)
    def extract_entities_cached(self, text: str, labels_tuple: tuple) -> List[Entity]:
        """Cached wrapper for single-text extraction (queries)."""
        return self._extract_impl(text, list(labels_tuple))

    def extract_entities(self, text: str, labels: List[str]) -> List[Entity]:
        if len(text) < 200:
            return self.extract_entities_cached(text, tuple(sorted(labels)))
        return self._extract_impl(text, labels)

    @trace.instrument("gliner_extract")
    def _extract_impl(self, text: str, labels: List[str]) -> List[Entity]:
        if self._model is None:
            self.initialize()
        if self._model is None:
            return []

        try:
            with GLINER_EXTRACTION_DURATION.time():
                raw = self._model.predict_entities(text, labels, threshold=self.threshold)

            entities = [Entity(e["text"], e["label"], e["start"], e["end"], e.get("score", 0.0)) for e in raw]

            for e in entities:
                GLINER_ENTITY_COUNT.labels(label=e.label).inc()

            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def batch_extract_entities(
        self,
        texts: List[str],
        labels: List[str],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts efficiently.
        """
        if self._model is None:
            self.initialize()
        if self._model is None:
            return [[] for _ in texts] # Fallback to empty lists

        threshold = threshold or self.threshold
        batch_size = batch_size or self.default_batch_size
        all_entities = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Thread Safety: GLiNER inference is generally thread-safe.
                batch_results = self._model.batch_predict_entities(
                    batch,
                    labels,
                    threshold=threshold
                )

                for raw_entities in batch_results:
                    entities = [
                        Entity(
                            text=e["text"],
                            label=e["label"],
                            start=e["start"],
                            end=e["end"],
                            score=e.get("score", threshold)
                        )
                        for e in raw_entities
                    ]
                    all_entities.append(entities)
            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                all_entities.extend([[] for _ in batch])

        return all_entities
```

### 4.2 Configuration

**File:** `src/shared/config.py`

```python
class NERConfig(BaseModel):
    enabled: bool = False
    model_name: str = "urchade/gliner_medium-v2.1"
    threshold: float = 0.45
    device: str = "auto"
    batch_size: int = 32
    labels: List[str] = ["software", "hardware", "error_code"]
```

---

## 5. Phase 2: Document Ingestion Pipeline

**Strategy:** Hook into `src/ingestion/atomic.py`.
**Design Note:** We hook into `atomic.py` (embedding prep) rather than `build_graph.py` (structure extraction) because GLiNER's primary role here is **vector enrichment**, not graph structure definition. This keeps "Semantic Layer" (GLiNER) separate from "Structural Layer" (Regex/Markdown) until they merge at the vector/sparse level.

### 5.1 Extraction Helper

**File:** `src/ingestion/extract/ner_gliner.py`

```python
import hashlib
from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import get_default_labels

def enrich_chunks_with_entities(chunks: List[Dict[str, Any]]) -> None:
    service = GLiNERService()
    labels = get_default_labels()

    texts = [c.get("text", "") for c in chunks]
    entities_list = service.batch_extract_entities(texts, labels)

    for chunk, entities in zip(chunks, entities_list):
        if not entities:
            continue

        # 1. Metadata (Payload)
        chunk["entity_metadata"] = {
            "entity_types": list(set(e.label for e in entities)),
            "entity_values": [e.text for e in entities],
            "entity_values_normalized": [e.text.lower().strip() for e in entities],
            "entity_count": len(entities)
        }

        # 2. Text Injection (Transient)
        # Create _embedding_text for vector generation only.
        entity_str = "; ".join([f"{e.label}: {e.text}" for e in entities])
        chunk["_embedding_text"] = f"{chunk['text']}\n\n[Context: {entity_str}]"

        # 3. Sparse Vector Enrichment (_mentions)
        existing_mentions = chunk.get("_mentions", [])
        seen_keys = {(m.get("name", "").lower(), m.get("type", "").lower()) for m in existing_mentions}

        new_mentions = []
        for e in entities:
            key = (e.text.lower(), e.label.lower())
            if key not in seen_keys:
                eid_hash = hashlib.sha256(e.text.encode()).hexdigest()[:16]
                new_mentions.append({
                    "name": e.text,
                    "type": e.label,
                    "entity_id": f"gliner:{e.label}:{eid_hash}",
                    "source": "gliner", # Marker for Neo4j filtering
                    "confidence": e.score
                })
                seen_keys.add(key)

        chunk["_mentions"] = existing_mentions + new_mentions
```

### 5.2 Hook into Atomic Ingestion

**File:** `src/ingestion/atomic.py`

```python
from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

def _prepare_ingestion(self, document_id, sections, ...):
    # ...

    # NEW: Enrich sections/chunks with GLiNER entities (Gated)
    if self.config.ner.enabled:
        try:
            enrich_chunks_with_entities(sections)
        except Exception as e:
            logger.warning(f"GLiNER enrichment failed (non-blocking): {e}")

    # ... proceed to embedding generation ...

def _compute_embeddings(self, ...):
    # ...
    # NEW: Consume transient text for embedding
    for section in sections:
        # Prefer transient enriched text if available
        content_text = section.get("_embedding_text") or builder._build_section_text_for_embedding(section)
        # ... generate embeddings ...

def _neo4j_create_mentions(self, tx, mentions):
    # NEW: Filter out GLiNER entities to prevent "ghost" nodes in the graph
    mentions_to_create = [m for m in mentions if m.get("source") != "gliner"]
    # ... existing creation logic using mentions_to_create ...
```

---

## 6. Phase 4: Qdrant Collection Schema

**Strategy:** Update existing schema definitions to include new payload indexes.

**File:** `src/shared/qdrant_schema.py`

```python
def build_qdrant_schema(...):
    # ...
    payload_indexes.extend([
        ("entity_types", PayloadSchemaType.KEYWORD),
        ("entity_values", PayloadSchemaType.KEYWORD),
        ("entity_values_normalized", PayloadSchemaType.KEYWORD),
        ("entity_count", PayloadSchemaType.INTEGER),
    ])
```

---

## 7. Phase 5: Hybrid Search Implementation

**Strategy:** **Post-Retrieval Rescoring** (Soft Filtering).

### 7.1 Query Disambiguation Helper

**File:** `src/query/processing/disambiguation.py`

```python
class QueryDisambiguator:
    # ...
    def process(self, query: str) -> Dict[str, Any]:
        entities = self.service.extract_entities(query, self.labels)
        return {
            "entities": entities,
            "boost_terms": [e.text.lower().strip() for e in entities]
        }
```

### 7.2 Modify Hybrid Retriever

**File:** `src/query/hybrid_retrieval.py`

```python
class HybridRetriever:
    def retrieve(self, query: str, top_k: int = 20):
        # 1. Analyze Query (Check enabled flag)
        boost_terms = []
        if self.config.ner.enabled:
             analysis = self.disambiguator.process(query)
             boost_terms = analysis.get("boost_terms", [])

        # 2. Retrieve (fetch extra candidates for re-ranking)
        fetch_k = top_k * 2 if boost_terms else top_k
        results = self.vector_retriever.search(query, top_k=fetch_k)

        # 3. Post-Retrieval Boosting (Soft Filter)
        if boost_terms:
            for res in results:
                doc_entities = res.payload.get("entity_metadata", {}).get("entity_values_normalized", [])
                matches = sum(1 for term in boost_terms if term in doc_entities)
                if matches > 0:
                    boost_factor = 1.0 + min(0.5, matches * 0.1) # Max 50% boost
                    res.score *= boost_factor
                    res.boosted = True

            results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]
```

---

## 9. Phase 7: Migration Strategy

Given the "atomic" nature of the system (documents are ingested units), we need a strategy for existing collections.

**Option 1: Full Re-ingestion (Recommended for Prod)**
*   **Why:** `_embedding_text` affects dense vector geometry. To get full benefit, vectors must be recomputed.
*   **Action:** Trigger full corpus re-ingestion.

**Option 2: Payload-Only Backfill (Rejected)**
*   **Why:** Since `_embedding_text` modifies the content used for dense vector generation, simply updating the payload without re-embedding would result in a mismatch between the vector space and the metadata, leading to inconsistent retrieval behavior.

**Option 3: Graceful Degradation (Recommended for Dev/Rollout)**
*   **Why:** Immediate impact without downtime.
*   **Action:**
    *   `HybridRetriever` checks for `entity_metadata` presence in payload.
    *   If missing, it treats the document as having 0 matches (no boost).
    *   No data corruption occurs; old docs simply don't get boosted.

---

## 10. Configuration Reference

**File:** `config/development.yaml`

```yaml
ner:
  enabled: false # Default OFF
  model_name: "urchade/gliner_medium-v2.1"
  device: "auto"
  batch_size: 32
  labels:
    - "software"
    - "hardware"
    - "filesystem_object"
    - "architecture_component"
    - "cloud_provider"
    - "cloud_service"
    - "command"
    - "parameter"
    - "protocol"
    - "error_message"
    - "metric"
    - "concept"
```

---

## 11. Testing & Validation

1.  **Integration Test:** `tests/integration/test_gliner_flow.py`
2.  **Unit Test:** `tests/unit/test_atomic_ingestion.py`

```

```
