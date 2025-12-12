# GLiNER Integration Implementation Plan
## Entity-Enhanced RAG with Qdrant + BGE-M3 Multi-Vector Embeddings

**Version:** 1.1
**Target:** Agentic Coder with Codebase Context
**Last Updated:** December 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Dependencies & Model Selection](#3-dependencies--model-selection)
4. [Phase 1: Core Infrastructure](#4-phase-1-core-infrastructure)
5. [Phase 2: Document Ingestion Pipeline](#5-phase-2-document-ingestion-pipeline)
6. [Phase 3: Query Processing Pipeline](#6-phase-3-query-processing-pipeline)
7. [Phase 4: Qdrant Collection Schema](#7-phase-4-qdrant-collection-schema)
8. [Phase 5: Hybrid Search Implementation](#8-phase-5-hybrid-search-implementation)
9. [Phase 6: Performance Optimization](#9-phase-6-performance-optimization)
10. [Phase 7: Testing & Validation](#10-phase-7-testing--validation)
11. [Configuration Reference](#11-configuration-reference)
12. [Known Issues & Gotchas](#12-known-issues--gotchas)

---

## 1. Executive Summary

### Objective
Integrate GLiNER (Generalist Lightweight Named Entity Recognition) into an existing Qdrant + BGE-M3 RAG pipeline to enhance retrieval quality through:

1. **Entity-enriched embeddings** - Append extracted entities to chunks before embedding
2. **Query disambiguation** - Resolve polysemy/homonyms in user queries
3. **Metadata-based filtering** - Use extracted entities for pre-filtering vector search
4. **Entity-aware chunking** - Respect entity boundaries during text splitting

### Why GLiNER?
- Zero-shot NER without fine-tuning
- Parallel entity extraction (not sequential like LLMs)
- 140x smaller than comparable LLMs with similar performance
- Apache 2.0 license
- ONNX export for production deployment
- Outperforms ChatGPT on zero-shot NER benchmarks

### Expected Outcomes
- **Improved recall** for entity-centric queries (15-30% based on research)
- **Better disambiguation** for homonyms/polysemous terms
- **Faster filtering** through entity metadata pre-filtering
- **More coherent chunks** that preserve entity context

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
│               │ Entity Filters  │                    │  Multi-Vector   │  │
│               │ for Pre-filter  │                    │     Query       │  │
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

# Core GLiNER
gliner>=0.2.24

# BGE-M3 Embeddings (Already present)
# FlagEmbedding>=1.2.0

# Qdrant Client (Already present)
# qdrant-client>=1.7.0

# For ONNX optimization (optional but recommended for production)
# onnxruntime>=1.16.0
# onnxruntime-gpu>=1.16.0  # Use strictly for CUDA environments
# onnxruntime-silicon>=1.16.0 # Use for Apple Silicon environments

# Utilities
# torch>=2.0.0 (Already present)
# transformers>=4.36.0 (Already present)
```

### Model Selection Guide

| Model | Size | Use Case | HuggingFace ID |
|---|---|---|---|
| **GLiNER Small v2.1** | ~110M | Development/Testing | `urchade/gliner_small-v2.1` |
| **GLiNER Medium v2.1** | ~209M | Balanced (Recommended) | `urchade/gliner_medium-v2.1` |
| **GLiNER Large v2.5** | ~340M | Maximum Accuracy | `gliner-community/gliner_large-v2.5` |
| **GLiNER XXL v2.5** | ~570M | Highest Performance | `gliner-community/gliner_xxl-v2.5` |
| **GLiNER Multitask** | ~340M | NER + Relations | `knowledgator/gliner-multitask-large-v0.5` |

**Recommendation:** Start with `gliner_medium-v2.1` for development, benchmark against `gliner_large-v2.5` for production decision.

**Hardware Note:** Supports cross-platform acceleration:
- **Apple Silicon:** Uses `mps`
- **NVIDIA:** Uses `cuda`
- **Fallback:** Uses `cpu`

---

## 4. Phase 1: Core Infrastructure

### 4.1 GLiNER Provider Service

Create a modular provider service that encapsulates model loading and inference with robust cross-platform device detection.

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
    Singleton pattern recommended to avoid reloading model.
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
        self.threshold = config.ner.threshold
        self.device = self._get_optimal_device(config.ner.device)
        self._model: Optional[GLiNER] = None
        self._initialized = True

    def _get_optimal_device(self, configured_device: str) -> str:
        """
        Resolve the best available hardware accelerator.
        Prioritizes: Configured > CUDA > MPS > CPU
        """
        if configured_device and configured_device != "auto":
            return configured_device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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

        # Guard against empty text
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
            logger.warning(f"Entity extraction failed for text sample: {e}")
            return []

    def batch_extract_entities(
        self,
        texts: List[str],
        labels: List[str],
        threshold: Optional[float] = None,
        batch_size: int = 8
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts efficiently.
        """
        if self._model is None:
            self.initialize()

        threshold = threshold or self.threshold
        all_entities = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
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
                # Fallback: append empty lists for failed batch to maintain index alignment
                all_entities.extend([[] for _ in batch])

        return all_entities
```

### 4.2 Domain-Specific Entity Labels

Create a configuration file for labels.

**File:** `src/providers/ner/labels.py`

```python
# Technical documentation labels (Default)
TECHNICAL_LABELS = [
    "software",
    "hardware",
    "api",
    "function",
    "parameter",
    "error code",
    "version",
    "protocol",
    "file format",
    "configuration",
    "metric",
    "specification",
    "command",
    "service"
]

def get_default_labels() -> list:
    return TECHNICAL_LABELS
```

---

## 5. Phase 2: Document Ingestion Pipeline

**Strategy:** Do NOT create a new `IngestionService`. Hook into `src/ingestion/atomic.py`.

### 5.1 Extraction Helper

Create a helper module to bridge `atomic.py` and the GLiNER service.

**File:** `src/ingestion/extract/ner_gliner.py`

```python
from typing import List, Dict, Any
from src.providers.ner.gliner_service import GLiNERService
from src.providers.ner.labels import get_default_labels
from src.shared.observability import get_logger

logger = get_logger(__name__)

def extract_entities_for_chunks(chunks: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    Batch extract entities for a list of chunks (dictionaries).
    Returns a list of entity lists corresponding to input chunks.
    """
    service = GLiNERService()
    labels = get_default_labels()

    texts = [c.get("text", "") for c in chunks]
    logger.info(f"Extracting entities for {len(texts)} chunks using GLiNER")

    return service.batch_extract_entities(texts, labels)

def enrich_chunks_with_entities(chunks: List[Dict[str, Any]]) -> None:
    """
    In-place enrichment of chunks with entity metadata.
    Adds 'entity_metadata' field to each chunk.
    """
    entities_list = extract_entities_for_chunks(chunks)

    for chunk, entities in zip(chunks, entities_list):
        # Format metadata for Qdrant payload
        entity_types = list(set(e.label for e in entities))
        entity_values = [e.text for e in entities]
        # Normalized for case-insensitive filtering
        entity_values_normalized = [v.lower().strip() for v in entity_values]

        chunk["entity_metadata"] = {
            "entity_types": entity_types,
            "entity_values": entity_values,
            "entity_values_normalized": entity_values_normalized,
            "entity_count": len(entities)
        }

        # Optional: Enrich text for embedding (append entities)
        # chunk["text"] = chunk["text"] + f"\nEntities: {', '.join(entity_values)}"
```

### 5.2 Hook into Atomic Ingestion

**File:** `src/ingestion/atomic.py`

Modify `_prepare_ingestion` (or equivalent preparation phase method):

```python
# Import at top
from src.ingestion.extract.ner_gliner import enrich_chunks_with_entities

# ... inside class/method ...
def _prepare_ingestion(self, document_id, sections, ...):
    # ... existing logic ...

    # NEW: Enrich sections/chunks with GLiNER entities
    # This happens before vector embedding generation
    try:
        enrich_chunks_with_entities(sections)
    except Exception as e:
        logger.warning(f"GLiNER enrichment failed (non-blocking): {e}")

    # ... proceed to embedding generation ...
```

### 5.3 Entity-Aware Chunking (Optional/Future)

Implement `EntityAwareChunker` as a strategy in `src/ingestion/chunk_assembler.py` only if strict boundary preservation is required. For now, enrichment of existing chunks is the priority.

---

## 6. Phase 3: Query Processing Pipeline

**Strategy:** Do NOT create `QueryProcessor`. Create a helper and hook into `HybridRetriever`.

### 6.1 Query Disambiguation Helper

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
        Analyze query, extract entities, and provide disambiguation/filters.
        """
        entities = self.service.extract_entities(query, self.labels, threshold=0.3)

        # Create Qdrant-compatible filters
        filters = {}
        if entities:
            filters["entity_values_normalized"] = [e.text.lower().strip() for e in entities]
            filters["entity_types"] = list(set(e.label for e in entities))

        # Disambiguated text (inject labels)
        # e.g., "apple" -> "apple [organization]"
        # This is a naive implementation; improved logic requires offset handling
        disambiguated_text = query
        # (Implementation details for offset-based replacement omitted for brevity)

        return {
            "original_query": query,
            "entities": entities,
            "filters": filters,
            # "disambiguated_text": disambiguated_text
        }
```

---

## 7. Phase 4: Qdrant Collection Schema

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

## 8. Phase 5: Hybrid Search Implementation

**Strategy:** Integrate into `src/query/hybrid_retrieval.py`.

### 8.1 Hook into Retrieve Method

**File:** `src/query/hybrid_retrieval.py`

```python
# Import
from src.query.processing.disambiguation import QueryDisambiguator

class HybridRetriever:
    def __init__(self, ...):
        # ... existing init ...
        self.disambiguator = QueryDisambiguator()

    def retrieve(self, query: str, ...):
        # ... start ...

        # 1. Process Query with GLiNER
        processed = self.disambiguator.process(query)

        # 2. Extract Filters
        entity_filters = processed.get("filters", {})

        # 3. Merge with existing filters (if any)
        if entity_filters:
            # Logic to merge entity_filters['entity_values_normalized']
            # into the qdrant filter passed to vector_retriever
            pass

        # ... proceed with search ...
```

---

## 9. Phase 6: Performance Optimization (Cross-Platform)

### 9.1 Hardware Acceleration Strategy

*   **Auto Detection:** Use `_get_optimal_device()` to select best accelerator.
*   **M1 Max:** Uses `mps` (Metal Performance Shaders).
*   **x86_64 + NVIDIA:** Uses `cuda`.
*   **Fallback:** Uses `cpu`.
*   **Batch Size:** Start small (8-16) and test. GLiNER is relatively efficient.

---

## 10. Configuration

**File:** `src/shared/config.py`

Add Pydantic settings for NER. **CRITICAL:** Add as a top-level field in `Config` (NOT `AppConfig`).

```python
class NERConfig(BaseModel):
    enabled: bool = False
    model_name: str = "urchade/gliner_medium-v2.1"
    threshold: float = 0.45
    device: str = "auto" # Auto-detect best hardware (cuda/mps/cpu)

class Config(WekaBaseModel):
    # ...
    # Add new field here:
    ner: NERConfig = Field(default_factory=NERConfig)
```

**File:** `config/development.yaml`

```yaml
ner:
  enabled: true
  model_name: "urchade/gliner_medium-v2.1"
  device: "auto"
```

---

## 11. Testing

Create `tests/integration/test_gliner_flow.py` to verify:
1.  Service singleton loads model on detected hardware (mps/cuda/cpu).
2.  Extraction returns expected entities for sample text.
3.  Ingestion hook successfully populates `entity_metadata` in a mock chunk.
4.  Retrieval hook successfully extracts entities from a mock query.

---

## 12. Implementation Checklist

1.  [ ] **Dependencies:** Add `gliner` to `requirements.txt`.
2.  **Schema:** Update `src/shared/qdrant_schema.py` (indexes).
3.  **Config:** Update `src/shared/config.py` and `config/development.yaml`.
4.  **Core Service:** Create `src/providers/ner/gliner_service.py` & labels.
5.  **Ingestion:** Create `src/ingestion/extract/ner_gliner.py` and hook into `atomic.py`.
6.  **Query:** Create `src/query/processing/disambiguation.py` and hook into `hybrid_retrieval.py`.
7.  **Verify:** Run integration tests.

```
