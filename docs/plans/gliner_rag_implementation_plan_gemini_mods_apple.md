# GLiNER Integration Implementation Plan
## Entity-Enhanced RAG with Qdrant + BGE-M3 Multi-Vector Embeddings

**Version:** 1.7 (Simplified Structure)
**Target:** Agentic Coder with Codebase Context
**Last Updated:** December 2025

---

## Table of Contents

### Overview
- [Executive Summary](#executive-summary)
- [Architecture Overview](#architecture-overview)
- [Dependencies & Model Selection](#dependencies--model-selection)

### Implementation Phases
- [Phase 1: Core Infrastructure](#phase-1-core-infrastructure)
- [Phase 2: Document Ingestion Pipeline](#phase-2-document-ingestion-pipeline)
- [Phase 3: Qdrant Collection Schema](#phase-3-qdrant-collection-schema)
- [Phase 4: Hybrid Search Implementation](#phase-4-hybrid-search-implementation)
- [Phase 5: Performance Optimization](#phase-5-performance-optimization)

### Reference
- [Configuration Reference](#configuration-reference)
- [Testing & Validation](#testing--validation)

---

## Executive Summary

### Objective
Integrate GLiNER (Generalist Lightweight Named Entity Recognition) into an existing Qdrant + BGE-M3 RAG pipeline to enhance retrieval quality.

**Key Features:**
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

## Architecture Overview

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

## Dependencies & Model Selection

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

## Phase 1: Core Infrastructure

### GLiNER Provider Service

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

### Configuration Schema

**File:** `src/shared/config.py`

```python
class NERConfig(BaseModel):
    enabled: bool = False
    model_name: str = "urchade/gliner_medium-v2.1"
    threshold: float = 0.45
    device: str = "auto"
    batch_size: int = 32
    labels: List[str] = []  # See config/development.yaml for domain-specific labels
```

---

## Phase 2: Document Ingestion Pipeline

**Strategy:** Hook into `src/ingestion/atomic.py`.

**Design Note:** We hook into `atomic.py` (embedding prep) rather than `build_graph.py` (structure extraction) because GLiNER's primary role here is **vector enrichment**, not graph structure definition. This keeps "Semantic Layer" (GLiNER) separate from "Structural Layer" (Regex/Markdown) until they merge at the vector/sparse level.

### Extraction Helper

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

### Hook into Atomic Ingestion

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

## Phase 3: Qdrant Collection Schema

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

## Phase 4: Hybrid Search Implementation

**Strategy:** **Post-Retrieval Rescoring** (Soft Filtering).

### Query Disambiguation Helper

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

### Modify Hybrid Retriever

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

## Phase 5: Performance Optimization

### Apple Silicon (M1/M2/M3) Specifics

*   **Device:** `mps` (Metal Performance Shaders) is critical. `device="auto"` handles this, but verify logs show `Loading GLiNER model... on mps`.
*   **Batch Size:** **32** is the sweet spot.
    *   `< 16`: Underutilizes the GPU.
    *   `> 64`: Risks OOM on 16GB/32GB RAM machines when running alongside Docker/Qdrant.
*   **Quantization:** **Avoid `bitsandbytes`**. It is CUDA-optimized and often fails or runs slowly on MPS. Use standard FP16 (PyTorch default on MPS) or FP32.

### Production Tuning

*   **Thread Safety:** The singleton uses `_instance` locking, but for high-concurrency (e.g., 4+ workers), consider moving the model to a separate process or using `gunicorn` with preloaded app.
*   **Timeout:** Ingestion of large documents with GLiNER enabled will be slower. Increase `timeout_seconds` in `ingestion.config` to **600** (from 300) to prevent worker timeouts.

---

## Configuration Reference

**File:** `config/development.yaml`

```yaml
ner:
  enabled: false # Default OFF
  model_name: "urchade/gliner_medium-v2.1"
  device: "auto"
  batch_size: 32
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

---

## Testing & Validation

1.  **Integration Test:** `tests/integration/test_gliner_flow.py`
    *   Verify `_embedding_text` is generated but `text` remains clean.
    *   Verify `_mentions` contains GLiNER entities with `source="gliner"`.
    *   Verify `QueryDisambiguator` returns boost terms.

2.  **Unit Test:** `tests/unit/test_atomic_ingestion.py`
    *   Verify `_neo4j_create_mentions` filters out gliner sources.
    *   Verify embedding uses `_embedding_text`.
