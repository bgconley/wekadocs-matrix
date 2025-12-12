# GLiNER Integration Implementation Plan
## Entity-Enhanced RAG with Qdrant + BGE-M3 Multi-Vector Embeddings

**Version:** 1.2 (Integrated with Codebase Architecture)
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
9. [Configuration Reference](#9-configuration-reference)
10. [Testing & Validation](#10-testing--validation)

---

## 1. Executive Summary

### Objective
Integrate GLiNER (Generalist Lightweight Named Entity Recognition) into an existing Qdrant + BGE-M3 RAG pipeline to enhance retrieval quality through:

1. **Entity-enriched embeddings** - Append extracted entities to chunks before embedding to ensure dense vector semantic alignment.
2. **Integration with `entity-sparse`** - Populate the existing `_mentions` structure so `atomic.py` automatically generates sparse vectors for extracted entities.
3. **Query disambiguation** - Resolve polysemy/homonyms in user queries.
4. **Boosting (Soft Filtering)** - Use extracted entities to boost relevant documents rather than hard-filtering (avoiding zero recall).

### Why GLiNER?
- Zero-shot NER without fine-tuning
- Parallel entity extraction (not sequential like LLMs)
- 140x smaller than comparable LLMs with similar performance
- Apache 2.0 license
- ONNX export for production deployment

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
│               │ (Soft Filter)   │                    │     Query       │  │
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

**Recommendation:** Start with `gliner_medium-v2.1` for development.

**Hardware Note:** Target architecture is Apple M1 Max. Use `mps` device.

---

## 4. Phase 1: Core Infrastructure

### 4.1 GLiNER Provider Service

Create a modular provider service that encapsulates model loading and inference.

**File:** `src/providers/ner/gliner_service.py`

```python
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch
from gliner import GLiNER
from src.shared.observability import get_logger
from src.shared.config import get_config

logger = get_logger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    score: float

class GLiNERService:
    """
    GLiNER-based Named Entity Recognition service.
    Singleton pattern to avoid reloading model.
    """

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
        self.model_name = config.ner.model_name
        self.device = config.ner.device
        self.threshold = config.ner.threshold
        self.default_batch_size = config.ner.batch_size
        self._model: Optional[GLiNER] = None
        self._initialized = True

    def initialize(self) -> None:
        """
        Initialize the GLiNER model.
        """
        if self._model is not None:
            return

        logger.info(f"Loading GLiNER model: {self.model_name} on {self.device}")
        try:
            self._model = GLiNER.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            logger.info("GLiNER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise

    def extract_entities(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None
    ) -> List[Entity]:
        """
        Extract entities from a single text.
        """
        if self._model is None:
            self.initialize()

        threshold = threshold or self.threshold

        if not text or not text.strip():
            return []

        try:
            raw_entities = self._model.predict_entities(
                text,
                labels,
                threshold=threshold
            )

            return [
                Entity(
                    text=e["text"],
                    label=e["label"],
                    start=e["start"],
                    end=e["end"],
                    score=e.get("score", threshold)
                )
                for e in raw_entities
            ]
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

### 4.2 Domain-Specific Entity Labels

Create a configuration file for labels to avoid hardcoding.

**File:** `src/providers/ner/labels.py`

```python
from src.shared.config import get_config

def get_default_labels() -> list:
    """
    Get labels from configuration to allow easy tuning.
    """
    config = get_config()
    return config.ner.labels
```

---

## 5. Phase 2: Document Ingestion Pipeline

**Strategy:** Hook into `src/ingestion/atomic.py`. **CRITICAL:** Use existing `entity-sparse` architecture via `_mentions`.

### 5.1 Extraction Helper

**File:** `src/ingestion/extract/ner_gliner.py`

```python
from typing import List, Dict, Any
from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import get_default_labels
from src.shared.observability import get_logger

logger = get_logger(__name__)

def extract_entities_for_chunks(chunks: List[Dict[str, Any]]) -> List[List[Any]]:
    service = GLiNERService()
    labels = get_default_labels()

    texts = [c.get("text", "") for c in chunks]
    # M1 Max optimal batch size: ~32
    return service.batch_extract_entities(texts, labels)

def enrich_chunks_with_entities(chunks: List[Dict[str, Any]]) -> None:
    """
    In-place enrichment of chunks with:
    1. Entity Metadata (Payload)
    2. Text Injection (Dense Vector Context)
    3. _mentions (Entity-Sparse Vector Context)
    """
    entities_list = extract_entities_for_chunks(chunks)

    for chunk, entities in zip(chunks, entities_list):
        # 1. Metadata for filtering
        entity_types = list(set(e.label for e in entities))
        entity_values = [e.text for e in entities]
        entity_values_normalized = [v.lower().strip() for v in entity_values]

        chunk["entity_metadata"] = {
            "entity_types": entity_types,
            "entity_values": entity_values,
            "entity_values_normalized": entity_values_normalized,
            "entity_count": len(entities)
        }

        # 2. Text Injection: CRITICAL for BGE-M3
        # We append context so the embedding model sees the "semantic tags"
        if entities:
            entity_str = "; ".join([f"{e.label}: {e.text}" for e in entities])
            chunk["text"] = f"{chunk['text']}\n\n[Context: {entity_str}]"

        # 3. Integration with Existing "entity-sparse" (atomic.py)
        # atomic.py uses `_mentions` to generate sparse vectors for entities.
        # We append GLiNER entities to this list (Additive Strategy).
        existing_mentions = chunk.get("_mentions", [])

        # Deduplication key: (text, label)
        seen = {(m.get("name", "").lower(), m.get("label", "").lower()) for m in existing_mentions}

        new_mentions = []
        for e in entities:
            key = (e.text.lower(), e.label.lower())
            if key not in seen:
                new_mentions.append({
                    "name": e.text,
                    "label": e.label,
                    "entity_id": f"gliner:{e.label}:{e.text}", # Synthetic ID
                    "source": "gliner"
                })
                seen.add(key)

        chunk["_mentions"] = existing_mentions + new_mentions
```

### 5.2 Hook into Atomic Ingestion

**File:** `src/ingestion/atomic.py`

Modify `_prepare_ingestion`:

```python
from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

def _prepare_ingestion(self, document_id, sections, ...):
    # ... existing logic ...

    # NEW: Enrich sections/chunks with GLiNER entities
    # Must happen BEFORE embedding generation.
    try:
        enrich_chunks_with_entities(sections)
    except Exception as e:
        logger.warning(f"GLiNER enrichment failed (non-blocking): {e}")

    # ... proceed to embedding generation ...
```

---

## 6. Phase 4: Qdrant Collection Schema

**Strategy:** Update existing schema definitions.

**File:** `src/shared/qdrant_schema.py`

Update `build_qdrant_schema` to include new payload indexes:

```python
def build_qdrant_schema(...):
    # ... existing code ...

    # Add Entity Metadata Indexes
    payload_indexes.extend([
        ("entity_types", PayloadSchemaType.KEYWORD),
        ("entity_values", PayloadSchemaType.KEYWORD),
        ("entity_values_normalized", PayloadSchemaType.KEYWORD),
        ("entity_count", PayloadSchemaType.INTEGER),
    ])

    # ... return schema ...
```

---

## 7. Phase 5: Hybrid Search Implementation

**Strategy:** Use "Boosting" (Should) instead of "Hard Filtering" (Must).
**Constraint:** `QdrantMultiVectorRetriever._build_filter` creates strictly `must` filters. We must extend `HybridRetriever` and `QdrantMultiVectorRetriever` to support boosting.

### 7.1 Query Disambiguation Helper

**File:** `src/query/processing/disambiguation.py`

```python
from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import get_default_labels
from typing import Dict, List, Any

class QueryDisambiguator:
    def __init__(self):
        self.service = GLiNERService()
        self.labels = get_default_labels()

    def process(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return conditions for BOOSTING.
        """
        entities = self.service.extract_entities(query, self.labels, threshold=0.3)

        # Prepare boosting conditions
        boosting_conditions = []
        if entities:
            boosting_conditions = [
                {
                    "key": "entity_values_normalized",
                    "match": {"value": e.text.lower().strip()}
                }
                for e in entities
            ]

        return {
            "original_query": query,
            "entities": entities,
            "boosting_conditions": boosting_conditions
        }
```

### 7.2 Hook into Retrieve Method & Extend Vector Retriever

**File:** `src/query/hybrid_retrieval.py`

1.  **Update `QdrantMultiVectorRetriever.search`** to accept `should` argument.
2.  **Update `HybridRetriever.retrieve`** to utilize `should` conditions.

```python
from src.query.processing.disambiguation import QueryDisambiguator
from qdrant_client.http import models as rest

class HybridRetriever:
    def __init__(self, ...):
        # ... existing init ...
        self.disambiguator = QueryDisambiguator()

    def retrieve(self, query: str, ...):
        # 1. Process Query
        processed = self.disambiguator.process(query)
        boosting_conditions = processed.get("boosting_conditions", [])

        # 2. Convert to Qdrant Should Conditions
        should_clauses = []
        for cond in boosting_conditions:
            should_clauses.append(
                rest.FieldCondition(
                    key=cond["key"],
                    match=rest.MatchValue(value=cond["match"]["value"])
                )
            )

        # 3. Pass to vector retriever (Requires updating search signature)
        # self.vector_retriever.search(..., should=should_clauses)
```

---

## 8. Phase 6: Performance Optimization (Apple Silicon)

### 8.1 M1 Max Specifics

*   **Device:** Use `mps` (Metal Performance Shaders).
*   **Batch Size:** Configured to `32` (sweet spot for M1 Max).
*   **Quantization:** Avoid `bitsandbytes`. Use standard PyTorch MPS execution.

---

## 9. Configuration Reference

**File:** `src/shared/config.py`

```python
class NERConfig(BaseModel):
    enabled: bool = False
    model_name: str = "urchade/gliner_medium-v2.1"
    threshold: float = 0.45
    device: str = "mps"
    batch_size: int = 32
    labels: List[str] = ["software", "hardware", "error_code"] # Default

class Config(WekaBaseModel):
    # ...
    ner: NERConfig = Field(default_factory=NERConfig)
```

**File:** `config/development.yaml`

```yaml
ner:
  enabled: true
  model_name: "urchade/gliner_medium-v2.1"
  device: "mps"
  batch_size: 32
  labels:
    - "software"
    - "hardware"
    - "api"
    - "function"
    - "error code"
    - "version"
    - "command"
```

---

## 10. Testing & Validation

1.  **Integration Test:** `tests/integration/test_gliner_flow.py`
    *   Verify `GLiNERService` singleton initialization on `mps`.
    *   Verify `enrich_chunks_with_entities` populates `_mentions` correctly (additive).
    *   Verify `QueryDisambiguator` returns boosting conditions.
2.  **Performance Test:** Measure ingestion throughput with batch size 32 vs 8.
