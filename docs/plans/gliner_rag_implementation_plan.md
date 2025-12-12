# GLiNER Integration Implementation Plan
## Entity-Enhanced RAG with Qdrant + BGE-M3 Multi-Vector Embeddings

**Version:** 1.0
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

# BGE-M3 Embeddings
FlagEmbedding>=1.2.0

# Qdrant Client
qdrant-client>=1.7.0

# For ONNX optimization (optional but recommended for production)
onnxruntime>=1.16.0
onnxruntime-gpu>=1.16.0  # If using GPU

# Utilities
torch>=2.0.0
transformers>=4.36.0
```

### Model Selection Guide

| Model | Size | Use Case | HuggingFace ID |
|-------|------|----------|----------------|
| **GLiNER Small v2.1** | ~110M | Development/Testing | `urchade/gliner_small-v2.1` |
| **GLiNER Medium v2.1** | ~209M | Balanced (Recommended) | `urchade/gliner_medium-v2.1` |
| **GLiNER Large v2.5** | ~340M | Maximum Accuracy | `gliner-community/gliner_large-v2.5` |
| **GLiNER XXL v2.5** | ~570M | Highest Performance | `gliner-community/gliner_xxl-v2.5` |
| **GLiNER Multitask** | ~340M | NER + Relations | `knowledgator/gliner-multitask-large-v0.5` |

**Recommendation:** Start with `gliner_medium-v2.1` for development, benchmark against `gliner_large-v2.5` for production decision.

---

## 4. Phase 1: Core Infrastructure

### 4.1 GLiNER Service Class

```python
# services/ner_service.py

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import torch
from gliner import GLiNER

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    score: float

@dataclass
class NERConfig:
    """Configuration for NER service."""
    model_name: str = "urchade/gliner_medium-v2.1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.45  # Lower for better recall in RAG
    batch_size: int = 8
    max_length: int = 512  # Max tokens per text
    use_onnx: bool = False
    onnx_path: Optional[str] = None

class GLiNERService:
    """
    GLiNER-based Named Entity Recognition service.

    Designed for RAG enhancement with entity extraction,
    query disambiguation, and metadata generation.
    """

    def __init__(self, config: NERConfig):
        self.config = config
        self._model: Optional[GLiNER] = None
        self._default_labels: List[str] = []

    def initialize(self, entity_labels: List[str]) -> None:
        """
        Initialize the GLiNER model.

        IMPORTANT: Call model.to('cuda') NOT model.cuda() for proper GPU usage.
        See: https://github.com/urchade/GLiNER/issues/88

        Args:
            entity_labels: Default entity types to extract
        """
        self._default_labels = entity_labels

        if self.config.use_onnx and self.config.onnx_path:
            self._model = GLiNER.from_pretrained(
                self.config.onnx_path,
                load_onnx_model=True,
                load_tokenizer=True
            )
        else:
            self._model = GLiNER.from_pretrained(self.config.model_name)

        # CRITICAL: Use .to('cuda') not .cuda() for proper GPU utilization
        if self.config.device == "cuda":
            self._model = self._model.to('cuda')

    def extract_entities(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[Entity]:
        """
        Extract entities from a single text.

        Args:
            text: Input text
            labels: Entity types to extract (uses defaults if None)
            threshold: Confidence threshold (uses config if None)

        Returns:
            List of Entity objects
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        labels = labels or self._default_labels
        threshold = threshold or self.config.threshold

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

    def batch_extract_entities(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts efficiently.

        Uses batching to improve throughput on GPU.

        Args:
            texts: List of input texts
            labels: Entity types to extract
            threshold: Confidence threshold

        Returns:
            List of entity lists, one per input text
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        labels = labels or self._default_labels
        threshold = threshold or self.config.threshold

        all_entities = []

        # Process in batches to avoid OOM
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

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

            # Clear CUDA cache between batches for large datasets
            if self.config.device == "cuda" and len(texts) > 1000:
                torch.cuda.empty_cache()

        return all_entities

    def generate_entity_metadata(
        self,
        entities: List[Entity]
    ) -> Dict[str, Any]:
        """
        Generate metadata payload for Qdrant from extracted entities.

        Returns:
            Dict with entity_types, entity_values, and entity_map
        """
        entity_types = list(set(e.label for e in entities))
        entity_values = [e.text for e in entities]
        entity_map = {e.text: e.label for e in entities}

        # Normalized versions for better matching
        entity_values_normalized = [v.lower().strip() for v in entity_values]

        return {
            "entity_types": entity_types,
            "entity_values": entity_values,
            "entity_values_normalized": entity_values_normalized,
            "entity_map": entity_map,
            "entity_count": len(entities)
        }
```

### 4.2 Domain-Specific Entity Labels

```python
# config/entity_labels.py

"""
Entity labels configuration.

Customize these based on your domain. GLiNER works best with:
- Lowercase or Title Case labels
- Clear, descriptive label names
- 10-30 labels per extraction call (performance degrades with too many)
"""

# Generic RAG labels - good starting point
GENERIC_LABELS = [
    "person",
    "organization",
    "location",
    "date",
    "time",
    "money",
    "percentage",
    "product",
    "event",
    "technology",
    "concept"
]

# Technical documentation labels
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
    "specification"
]

# Business/Enterprise labels
BUSINESS_LABELS = [
    "company",
    "person",
    "product",
    "service",
    "contract",
    "regulation",
    "policy",
    "department",
    "role",
    "project",
    "metric",
    "currency amount"
]

# Scientific/Research labels
SCIENTIFIC_LABELS = [
    "chemical compound",
    "gene",
    "protein",
    "disease",
    "drug",
    "organism",
    "measurement",
    "methodology",
    "institution",
    "researcher"
]

# Storage/AI/ML domain labels (relevant for parallel file systems)
STORAGE_AI_LABELS = [
    "file system",
    "storage system",
    "model",
    "dataset",
    "framework",
    "hardware",
    "benchmark",
    "metric",
    "configuration",
    "workload type",
    "organization",
    "technology",
    "protocol",
    "performance metric"
]

def get_labels_for_domain(domain: str) -> list:
    """Get appropriate labels for a domain."""
    domain_map = {
        "generic": GENERIC_LABELS,
        "technical": TECHNICAL_LABELS,
        "business": BUSINESS_LABELS,
        "scientific": SCIENTIFIC_LABELS,
        "storage_ai": STORAGE_AI_LABELS
    }
    return domain_map.get(domain, GENERIC_LABELS)
```

---

## 5. Phase 2: Document Ingestion Pipeline

### 5.1 Entity-Aware Text Chunker

```python
# services/chunking_service.py

from typing import List, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    respect_entity_boundaries: bool = True
    min_chunk_size: int = 100

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    source_doc_id: str

class EntityAwareChunker:
    """
    Text chunker that respects entity boundaries.

    Ensures entities are not split across chunk boundaries
    when possible, improving entity context preservation.
    """

    def __init__(self, config: ChunkConfig, ner_service: 'GLiNERService'):
        self.config = config
        self.ner_service = ner_service

        # Sentence boundary pattern
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+'
        )

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        entity_labels: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Split document into chunks respecting entity boundaries.

        Algorithm:
        1. Split into sentences
        2. Extract entities from full text
        3. Group sentences into chunks of ~chunk_size
        4. Adjust boundaries to not split entities
        5. Add overlap for context continuity
        """
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Step 2: Extract entities (for boundary awareness)
        entities = []
        if self.config.respect_entity_boundaries and entity_labels:
            entities = self.ner_service.extract_entities(text, entity_labels)

        # Step 3-4: Group sentences into chunks
        chunks = self._create_chunks(text, sentences, entities, doc_id)

        return chunks

    def _split_into_sentences(self, text: str) -> List[Tuple[int, int, str]]:
        """Split text into sentences with position tracking."""
        sentences = []
        last_end = 0

        for match in self._sentence_pattern.finditer(text):
            sent_text = text[last_end:match.start()].strip()
            if sent_text:
                sentences.append((last_end, match.start(), sent_text))
            last_end = match.end()

        # Don't forget the last sentence
        if last_end < len(text):
            sent_text = text[last_end:].strip()
            if sent_text:
                sentences.append((last_end, len(text), sent_text))

        return sentences

    def _create_chunks(
        self,
        full_text: str,
        sentences: List[Tuple[int, int, str]],
        entities: List['Entity'],
        doc_id: str
    ) -> List[Chunk]:
        """Create chunks respecting entity boundaries."""
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_start = 0
        chunk_index = 0

        for sent_start, sent_end, sent_text in sentences:
            sent_length = len(sent_text)

            # Check if adding this sentence exceeds chunk size
            if current_length + sent_length > self.config.chunk_size and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text, chunk_end = self._finalize_chunk(
                    full_text,
                    current_chunk_sentences,
                    chunk_start,
                    entities
                )

                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    chunk_index=chunk_index,
                    source_doc_id=doc_id
                ))
                chunk_index += 1

                # Calculate overlap start
                overlap_start = max(0, len(current_chunk_sentences) - 2)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                current_length = sum(len(s[2]) for s in current_chunk_sentences)

                if current_chunk_sentences:
                    chunk_start = current_chunk_sentences[0][0]
                else:
                    chunk_start = sent_start

            current_chunk_sentences.append((sent_start, sent_end, sent_text))
            current_length += sent_length

        # Don't forget final chunk
        if current_chunk_sentences:
            chunk_text, chunk_end = self._finalize_chunk(
                full_text,
                current_chunk_sentences,
                chunk_start,
                entities
            )
            chunks.append(Chunk(
                text=chunk_text,
                start_char=chunk_start,
                end_char=chunk_end,
                chunk_index=chunk_index,
                source_doc_id=doc_id
            ))

        return chunks

    def _finalize_chunk(
        self,
        full_text: str,
        sentences: List[Tuple[int, int, str]],
        chunk_start: int,
        entities: List['Entity']
    ) -> Tuple[str, int]:
        """
        Finalize chunk boundaries, adjusting to not split entities.

        Returns:
            Tuple of (chunk_text, chunk_end_position)
        """
        if not sentences:
            return "", chunk_start

        chunk_end = sentences[-1][1]

        # Check if any entity spans the boundary
        if self.config.respect_entity_boundaries and entities:
            for entity in entities:
                # Entity starts before boundary but ends after
                if entity.start < chunk_end <= entity.end:
                    # Extend chunk to include full entity
                    chunk_end = entity.end
                    break
                # Entity starts at boundary
                elif entity.start == chunk_end:
                    # Include entity in this chunk if it's not too long
                    if entity.end - chunk_start < self.config.chunk_size * 1.5:
                        chunk_end = entity.end

        chunk_text = full_text[chunk_start:chunk_end].strip()
        return chunk_text, chunk_end
```

### 5.2 Document Ingestion Orchestrator

```python
# services/ingestion_service.py

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""
    collection_name: str
    batch_size: int = 32
    enrich_with_entities: bool = True
    entity_enrichment_template: str = "\n\nKey Entities: {entities}"

@dataclass
class DocumentRecord:
    """Processed document ready for vector storage."""
    id: str
    text: str
    enriched_text: str
    dense_vector: List[float]
    sparse_vector: Dict[int, float]
    colbert_vectors: List[List[float]]
    metadata: Dict[str, Any]

class IngestionService:
    """
    Orchestrates document ingestion with NER enrichment.

    Pipeline:
    1. Chunk documents (entity-aware)
    2. Extract entities per chunk
    3. Enrich chunks with entity context
    4. Generate multi-vector embeddings
    5. Prepare metadata payload
    6. Batch upsert to Qdrant
    """

    def __init__(
        self,
        config: IngestionConfig,
        ner_service: 'GLiNERService',
        embedding_service: 'BGEEmbeddingService',
        chunker: 'EntityAwareChunker',
        qdrant_client: 'QdrantClient'
    ):
        self.config = config
        self.ner_service = ner_service
        self.embedding_service = embedding_service
        self.chunker = chunker
        self.qdrant_client = qdrant_client

    def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        entity_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Ingest documents with NER enrichment.

        Args:
            documents: List of dicts with 'id', 'text', and optional metadata
            entity_labels: Entity types to extract

        Returns:
            Ingestion statistics
        """
        stats = {
            "total_documents": len(documents),
            "total_chunks": 0,
            "total_entities": 0,
            "start_time": datetime.now().isoformat(),
            "errors": []
        }

        all_records = []

        for doc in documents:
            try:
                records = self._process_document(doc, entity_labels)
                all_records.extend(records)
                stats["total_chunks"] += len(records)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id')}: {e}")
                stats["errors"].append({
                    "doc_id": doc.get('id'),
                    "error": str(e)
                })

        # Batch upsert to Qdrant
        self._batch_upsert(all_records)

        stats["end_time"] = datetime.now().isoformat()
        return stats

    def _process_document(
        self,
        document: Dict[str, Any],
        entity_labels: List[str]
    ) -> List[DocumentRecord]:
        """Process single document into records."""
        doc_id = document.get('id', self._generate_doc_id(document['text']))
        doc_text = document['text']
        doc_metadata = document.get('metadata', {})

        # Step 1: Chunk the document
        chunks = self.chunker.chunk_document(
            text=doc_text,
            doc_id=doc_id,
            entity_labels=entity_labels
        )

        if not chunks:
            return []

        # Step 2: Batch extract entities from all chunks
        chunk_texts = [c.text for c in chunks]
        all_entities = self.ner_service.batch_extract_entities(
            chunk_texts,
            entity_labels
        )

        # Step 3: Enrich chunks and generate embeddings
        records = []
        enriched_texts = []

        for chunk, entities in zip(chunks, all_entities):
            enriched_text = self._enrich_chunk(chunk.text, entities)
            enriched_texts.append(enriched_text)

        # Step 4: Batch embed enriched texts
        embeddings = self.embedding_service.batch_encode(enriched_texts)

        # Step 5: Create records
        for i, (chunk, entities, embedding) in enumerate(zip(chunks, all_entities, embeddings)):
            entity_metadata = self.ner_service.generate_entity_metadata(entities)

            record = DocumentRecord(
                id=f"{doc_id}_chunk_{chunk.chunk_index}",
                text=chunk.text,
                enriched_text=enriched_texts[i],
                dense_vector=embedding['dense'],
                sparse_vector=embedding['sparse'],
                colbert_vectors=embedding['colbert'],
                metadata={
                    **doc_metadata,
                    **entity_metadata,
                    "source_doc_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.start_char,
                    "char_end": chunk.end_char
                }
            )
            records.append(record)

        return records

    def _enrich_chunk(self, text: str, entities: List['Entity']) -> str:
        """
        Enrich chunk text with entity annotations.

        This creates an augmented representation that includes
        entity context, improving embedding quality for entity-centric queries.
        """
        if not self.config.enrich_with_entities or not entities:
            return text

        # Format: "EntityText (EntityType)"
        entity_strings = [f"{e.text} ({e.label})" for e in entities]
        unique_entities = list(dict.fromkeys(entity_strings))  # Preserve order, remove dups

        if unique_entities:
            entities_str = ", ".join(unique_entities[:10])  # Limit to top 10
            enriched = text + self.config.entity_enrichment_template.format(
                entities=entities_str
            )
            return enriched

        return text

    def _generate_doc_id(self, text: str) -> str:
        """Generate deterministic document ID from content."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _batch_upsert(self, records: List[DocumentRecord]) -> None:
        """Batch upsert records to Qdrant."""
        # Implementation depends on your Qdrant setup
        # See Phase 4 for collection schema
        pass
```

---

## 6. Phase 3: Query Processing Pipeline

### 6.1 Query Processor with NER Disambiguation

```python
# services/query_service.py

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class ProcessedQuery:
    """Processed query with NER enrichment."""
    original_query: str
    disambiguated_query: str
    entities: List['Entity']
    entity_filters: Dict[str, List[str]]
    dense_vector: List[float]
    sparse_vector: Dict[int, float]
    colbert_vectors: List[List[float]]

class QueryProcessor:
    """
    Processes user queries with NER-based disambiguation.

    Key capabilities:
    1. Extract entities from query
    2. Disambiguate homonyms/polysemous terms
    3. Generate entity-aware embeddings
    4. Create metadata filters for pre-filtering
    """

    def __init__(
        self,
        ner_service: 'GLiNERService',
        embedding_service: 'BGEEmbeddingService',
        entity_labels: List[str]
    ):
        self.ner_service = ner_service
        self.embedding_service = embedding_service
        self.entity_labels = entity_labels

    def process_query(
        self,
        query: str,
        enable_disambiguation: bool = True,
        generate_filters: bool = True
    ) -> ProcessedQuery:
        """
        Process a user query for retrieval.

        Args:
            query: Raw user query
            enable_disambiguation: Whether to augment query with entity types
            generate_filters: Whether to create entity-based filters

        Returns:
            ProcessedQuery with all components for hybrid search
        """
        # Step 1: Extract entities from query
        entities = self.ner_service.extract_entities(
            query,
            self.entity_labels,
            threshold=0.4  # Lower threshold for queries - better recall
        )

        # Step 2: Disambiguate query
        disambiguated_query = query
        if enable_disambiguation and entities:
            disambiguated_query = self._disambiguate_query(query, entities)

        # Step 3: Generate embeddings from disambiguated query
        embeddings = self.embedding_service.encode(disambiguated_query)

        # Step 4: Create entity filters
        entity_filters = {}
        if generate_filters and entities:
            entity_filters = self._create_entity_filters(entities)

        return ProcessedQuery(
            original_query=query,
            disambiguated_query=disambiguated_query,
            entities=entities,
            entity_filters=entity_filters,
            dense_vector=embeddings['dense'],
            sparse_vector=embeddings['sparse'],
            colbert_vectors=embeddings['colbert']
        )

    def _disambiguate_query(
        self,
        query: str,
        entities: List['Entity']
    ) -> str:
        """
        Inject entity type annotations into query for disambiguation.

        Example:
            Input: "How does Jordan's work affect the Jordan Valley?"
            Output: "How does [Jordan: person]'s work affect the [Jordan Valley: location]?"

        This helps the embedding model understand context and reduces
        ambiguity in the vector representation.
        """
        # Sort by position (reverse) to maintain correct offsets during replacement
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        disambiguated = query
        for entity in sorted_entities:
            annotation = f"[{entity.text}: {entity.label}]"
            disambiguated = (
                disambiguated[:entity.start] +
                annotation +
                disambiguated[entity.end:]
            )

        return disambiguated

    def _create_entity_filters(
        self,
        entities: List['Entity']
    ) -> Dict[str, List[str]]:
        """
        Create filter conditions for Qdrant pre-filtering.

        Returns filter structure compatible with Qdrant query API.
        """
        entity_types = list(set(e.label for e in entities))
        entity_values = [e.text for e in entities]
        entity_values_normalized = [v.lower().strip() for v in entity_values]

        return {
            "entity_types": entity_types,
            "entity_values": entity_values,
            "entity_values_normalized": entity_values_normalized
        }

    def extract_search_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine search strategy.

        Returns hints about:
        - Whether query is entity-centric
        - Recommended filter strictness
        - Whether to prioritize sparse vs dense
        """
        entities = self.ner_service.extract_entities(query, self.entity_labels)

        # Calculate entity density
        entity_char_coverage = sum(e.end - e.start for e in entities)
        coverage_ratio = entity_char_coverage / len(query) if query else 0

        return {
            "is_entity_centric": coverage_ratio > 0.3,
            "entity_count": len(entities),
            "coverage_ratio": coverage_ratio,
            "recommend_entity_filter": len(entities) >= 1,
            "recommend_sparse_boost": coverage_ratio > 0.4,
            "detected_entity_types": list(set(e.label for e in entities))
        }
```

---

## 7. Phase 4: Qdrant Collection Schema

### 7.1 Collection Configuration

```python
# services/qdrant_service.py

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    PayloadSchemaType
)
from typing import Optional

class QdrantCollectionManager:
    """
    Manages Qdrant collection for multi-vector + entity metadata storage.
    """

    # BGE-M3 vector dimensions
    DENSE_DIM = 1024
    COLBERT_DIM = 1024  # Per-token dimension

    def __init__(self, client: QdrantClient):
        self.client = client

    def create_collection(
        self,
        collection_name: str,
        on_disk_payload: bool = True,
        replication_factor: int = 1
    ) -> None:
        """
        Create Qdrant collection optimized for multi-vector + entity metadata.

        Schema:
        - Named vector: "dense" (1024-dim dense from BGE-M3)
        - Named vector: "colbert" (multivector for late interaction)
        - Sparse vector: "sparse" (lexical weights from BGE-M3)
        - Payload: entity metadata + document metadata
        """

        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"Collection {collection_name} already exists")
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # Dense vector for semantic search
                "dense": VectorParams(
                    size=self.DENSE_DIM,
                    distance=Distance.COSINE,
                    on_disk=True,  # Recommended for large collections
                ),
                # ColBERT multivector for late interaction reranking
                "colbert": VectorParams(
                    size=self.COLBERT_DIM,
                    distance=Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    on_disk=True,
                ),
            },
            sparse_vectors_config={
                # Sparse vector for lexical matching
                "sparse": SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                    )
                )
            },
            on_disk_payload=on_disk_payload,
            replication_factor=replication_factor,
        )

        # Create payload indexes for entity filtering
        self._create_payload_indexes(collection_name)

    def _create_payload_indexes(self, collection_name: str) -> None:
        """Create indexes on entity metadata fields for efficient filtering."""

        # Index for entity_types array (keyword matching)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="entity_types",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for entity_values array (keyword matching)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="entity_values",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for normalized entity values (for case-insensitive matching)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="entity_values_normalized",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for source document ID (useful for document-level operations)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="source_doc_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for entity count (for scoring/filtering)
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="entity_count",
            field_schema=PayloadSchemaType.INTEGER
        )
```

### 7.2 Point Structure

```python
# Example point structure for upsert

def create_qdrant_point(record: 'DocumentRecord') -> models.PointStruct:
    """
    Create Qdrant point from document record.

    Note: ColBERT vectors are stored as multivector (list of vectors).
    """
    return models.PointStruct(
        id=record.id,  # Can be string UUID or integer
        vector={
            "dense": record.dense_vector,
            "colbert": record.colbert_vectors,  # List[List[float]]
            "sparse": models.SparseVector(
                indices=list(record.sparse_vector.keys()),
                values=list(record.sparse_vector.values())
            )
        },
        payload={
            # Original and enriched text
            "text": record.text,
            "enriched_text": record.enriched_text,

            # Entity metadata (indexed)
            "entity_types": record.metadata.get("entity_types", []),
            "entity_values": record.metadata.get("entity_values", []),
            "entity_values_normalized": record.metadata.get("entity_values_normalized", []),
            "entity_map": record.metadata.get("entity_map", {}),
            "entity_count": record.metadata.get("entity_count", 0),

            # Document metadata
            "source_doc_id": record.metadata.get("source_doc_id"),
            "chunk_index": record.metadata.get("chunk_index"),
            "char_start": record.metadata.get("char_start"),
            "char_end": record.metadata.get("char_end"),

            # Additional metadata from source document
            **{k: v for k, v in record.metadata.items()
               if k not in ["entity_types", "entity_values", "entity_values_normalized",
                           "entity_map", "entity_count", "source_doc_id",
                           "chunk_index", "char_start", "char_end"]}
        }
    )
```

---

## 8. Phase 5: Hybrid Search Implementation

### 8.1 Search Service

```python
# services/search_service.py

from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    collection_name: str
    dense_limit: int = 50      # Prefetch limit for dense
    sparse_limit: int = 50     # Prefetch limit for sparse
    final_limit: int = 20      # Final results after fusion
    use_entity_filter: bool = True
    filter_mode: str = "should"  # "should" (OR) or "must" (AND)
    enable_colbert_rerank: bool = True
    colbert_rerank_limit: int = 100

@dataclass
class SearchResult:
    """Individual search result."""
    id: str
    score: float
    text: str
    entities: Dict[str, str]
    metadata: Dict[str, Any]

class HybridSearchService:
    """
    Hybrid search with entity-based filtering and ColBERT reranking.

    Search flow:
    1. Generate entity filter from query entities
    2. Prefetch with dense vectors (filtered)
    3. Prefetch with sparse vectors (filtered)
    4. Fuse results with RRF
    5. Rerank with ColBERT (optional)
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        query_processor: 'QueryProcessor',
        config: SearchConfig
    ):
        self.client = qdrant_client
        self.query_processor = query_processor
        self.config = config

    def search(
        self,
        query: str,
        filter_override: Optional[models.Filter] = None,
        limit_override: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Execute hybrid search with entity filtering.

        Args:
            query: User search query
            filter_override: Optional custom filter (overrides entity filter)
            limit_override: Optional limit override

        Returns:
            List of SearchResult objects
        """
        # Process query
        processed = self.query_processor.process_query(
            query,
            enable_disambiguation=True,
            generate_filters=self.config.use_entity_filter
        )

        # Build entity filter
        entity_filter = None
        if self.config.use_entity_filter and processed.entity_filters:
            entity_filter = self._build_entity_filter(processed.entity_filters)

        # Use override if provided
        search_filter = filter_override or entity_filter

        final_limit = limit_override or self.config.final_limit

        # Execute hybrid search
        if self.config.enable_colbert_rerank:
            results = self._search_with_colbert_rerank(processed, search_filter, final_limit)
        else:
            results = self._search_with_fusion(processed, search_filter, final_limit)

        return results

    def _build_entity_filter(
        self,
        entity_filters: Dict[str, List[str]]
    ) -> Optional[models.Filter]:
        """
        Build Qdrant filter from extracted entities.

        Uses "should" (OR) by default - matches if ANY entity type matches.
        This improves recall while still boosting relevant results.
        """
        conditions = []

        # Filter by entity types
        if entity_filters.get("entity_types"):
            conditions.append(
                models.FieldCondition(
                    key="entity_types",
                    match=models.MatchAny(any=entity_filters["entity_types"])
                )
            )

        # Filter by entity values (normalized for case-insensitivity)
        if entity_filters.get("entity_values_normalized"):
            conditions.append(
                models.FieldCondition(
                    key="entity_values_normalized",
                    match=models.MatchAny(any=entity_filters["entity_values_normalized"])
                )
            )

        if not conditions:
            return None

        if self.config.filter_mode == "must":
            return models.Filter(must=conditions)
        else:
            return models.Filter(should=conditions)

    def _search_with_fusion(
        self,
        processed: 'ProcessedQuery',
        search_filter: Optional[models.Filter],
        limit: int
    ) -> List[SearchResult]:
        """Execute hybrid search with RRF fusion."""

        response = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                # Dense vector prefetch
                models.Prefetch(
                    query=processed.dense_vector,
                    using="dense",
                    limit=self.config.dense_limit,
                    filter=search_filter
                ),
                # Sparse vector prefetch
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(processed.sparse_vector.keys()),
                        values=list(processed.sparse_vector.values())
                    ),
                    using="sparse",
                    limit=self.config.sparse_limit,
                    filter=search_filter
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True
        )

        return self._format_results(response.points)

    def _search_with_colbert_rerank(
        self,
        processed: 'ProcessedQuery',
        search_filter: Optional[models.Filter],
        limit: int
    ) -> List[SearchResult]:
        """
        Execute hybrid search with ColBERT reranking.

        Flow:
        1. Prefetch candidates with dense + sparse (fused)
        2. Rerank top candidates with ColBERT late interaction
        """

        response = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                # First stage: hybrid prefetch
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=processed.dense_vector,
                            using="dense",
                            limit=self.config.dense_limit,
                            filter=search_filter
                        ),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=list(processed.sparse_vector.keys()),
                                values=list(processed.sparse_vector.values())
                            ),
                            using="sparse",
                            limit=self.config.sparse_limit,
                            filter=search_filter
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self.config.colbert_rerank_limit
                )
            ],
            # Second stage: ColBERT reranking
            query=processed.colbert_vectors,
            using="colbert",
            limit=limit,
            with_payload=True
        )

        return self._format_results(response.points)

    def _format_results(
        self,
        points: List[models.ScoredPoint]
    ) -> List[SearchResult]:
        """Format Qdrant response into SearchResult objects."""
        results = []
        for point in points:
            payload = point.payload or {}
            results.append(SearchResult(
                id=str(point.id),
                score=point.score,
                text=payload.get("text", ""),
                entities=payload.get("entity_map", {}),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["text", "enriched_text", "entity_map"]
                }
            ))
        return results
```

---

## 9. Phase 6: Performance Optimization

### 9.1 GPU Optimization for GLiNER

```python
# utils/gpu_optimization.py

import torch
from contextlib import contextmanager

@contextmanager
def optimized_inference(model, use_amp: bool = True):
    """
    Context manager for optimized inference.

    Note: AMP (Automatic Mixed Precision) may not significantly speed up
    GLiNER inference based on community reports, but it reduces memory.
    """
    model.eval()

    with torch.no_grad():
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                yield
        else:
            yield

def setup_model_for_inference(model, device: str = "cuda"):
    """
    Prepare GLiNER model for inference.

    CRITICAL: Use model.to('cuda') NOT model.cuda()
    See: https://github.com/urchade/GLiNER/issues/88
    """
    if device == "cuda" and torch.cuda.is_available():
        model = model.to('cuda')
        # Optionally enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
    return model

def batch_generator(items: list, batch_size: int):
    """Generate batches for processing."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

class InferenceOptimizer:
    """
    Handles batching and memory management for large-scale inference.
    """

    def __init__(self, batch_size: int = 8, clear_cache_threshold: int = 100):
        self.batch_size = batch_size
        self.clear_cache_threshold = clear_cache_threshold
        self._processed_count = 0

    def process_batches(self, items: list, process_fn, **kwargs):
        """Process items in optimized batches."""
        results = []

        for batch in batch_generator(items, self.batch_size):
            batch_results = process_fn(batch, **kwargs)
            results.extend(batch_results)

            self._processed_count += len(batch)

            # Periodic cache clearing for long-running jobs
            if (self._processed_count % self.clear_cache_threshold == 0
                and torch.cuda.is_available()):
                torch.cuda.empty_cache()

        return results
```

### 9.2 ONNX Conversion (Optional)

```python
# scripts/convert_to_onnx.py

"""
Convert GLiNER model to ONNX for production deployment.

Run: python convert_to_onnx.py --model_path urchade/gliner_medium-v2.1 \
                                --save_path ./gliner_onnx/ \
                                --quantize True

Note: ONNX may not always be faster than PyTorch for GLiNER.
Benchmark before deploying.
"""

import argparse
from gliner import GLiNER

def convert_gliner_to_onnx(
    model_path: str,
    save_path: str,
    quantize: bool = True
):
    """
    Convert GLiNER to ONNX format.

    Args:
        model_path: HuggingFace model ID or local path
        save_path: Directory to save ONNX model
        quantize: Whether to apply quantization
    """
    # Load model
    model = GLiNER.from_pretrained(model_path)

    # Convert to ONNX
    # Note: GLiNER provides built-in conversion
    # Check examples/convert_to_onnx.ipynb in GLiNER repo
    model.to_onnx(save_path, quantize=quantize)

    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--quantize", type=bool, default=True)
    args = parser.parse_args()

    convert_gliner_to_onnx(args.model_path, args.save_path, args.quantize)
```

### 9.3 Caching Strategy

```python
# utils/caching.py

from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import hashlib
import json

class EntityCache:
    """
    Cache for entity extraction results.

    Useful for:
    - Repeated queries
    - Re-processing documents
    - Development/testing
    """

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size

    def _make_key(self, text: str, labels: tuple) -> str:
        """Create cache key from text and labels."""
        content = f"{text}:{json.dumps(sorted(labels))}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, labels: tuple) -> Optional[Any]:
        """Get cached result."""
        key = self._make_key(text, labels)
        return self._cache.get(key)

    def set(self, text: str, labels: tuple, result: Any) -> None:
        """Cache result."""
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove oldest 10%
            keys_to_remove = list(self._cache.keys())[:self._max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]

        key = self._make_key(text, labels)
        self._cache[key] = result

    def cached_extract(
        self,
        extract_fn: Callable,
        text: str,
        labels: tuple,
        **kwargs
    ) -> Any:
        """Extract with caching."""
        cached = self.get(text, labels)
        if cached is not None:
            return cached

        result = extract_fn(text, list(labels), **kwargs)
        self.set(text, labels, result)
        return result
```

---

## 10. Phase 7: Testing & Validation

### 10.1 Test Cases

```python
# tests/test_ner_rag.py

import pytest
from typing import List

class TestGLiNERIntegration:
    """Test suite for GLiNER RAG integration."""

    @pytest.fixture
    def ner_service(self):
        """Initialize NER service for testing."""
        from services.ner_service import GLiNERService, NERConfig
        config = NERConfig(model_name="urchade/gliner_small-v2.1")
        service = GLiNERService(config)
        service.initialize(["person", "organization", "location", "product"])
        return service

    def test_entity_extraction_basic(self, ner_service):
        """Test basic entity extraction."""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino."
        entities = ner_service.extract_entities(text)

        entity_texts = [e.text for e in entities]
        assert "Apple Inc." in entity_texts or "Apple" in entity_texts
        assert "Steve Jobs" in entity_texts
        assert "Cupertino" in entity_texts

    def test_disambiguation_homonyms(self, ner_service):
        """Test disambiguation of homonymous terms."""
        # "Apple" as company vs fruit
        text1 = "Apple released a new iPhone today."
        text2 = "I ate an apple for breakfast."

        entities1 = ner_service.extract_entities(text1)
        entities2 = ner_service.extract_entities(text2)

        # In text1, Apple should be organization
        apple_entity1 = next((e for e in entities1 if "apple" in e.text.lower()), None)
        assert apple_entity1 is not None
        assert apple_entity1.label in ["organization", "company", "product"]

        # In text2, apple might not be extracted or be different type
        # (GLiNER may or may not extract common nouns depending on training)

    def test_batch_extraction_performance(self, ner_service):
        """Test batch extraction is faster than sequential."""
        import time

        texts = ["Sample text with entities like Google and Microsoft."] * 100

        # Batch extraction
        start = time.time()
        batch_results = ner_service.batch_extract_entities(texts)
        batch_time = time.time() - start

        # Sequential extraction
        start = time.time()
        seq_results = [ner_service.extract_entities(t) for t in texts]
        seq_time = time.time() - start

        # Batch should be significantly faster
        assert batch_time < seq_time * 0.8  # At least 20% faster

    def test_metadata_generation(self, ner_service):
        """Test entity metadata generation for Qdrant."""
        text = "Microsoft and Google are competing in AI."
        entities = ner_service.extract_entities(text)
        metadata = ner_service.generate_entity_metadata(entities)

        assert "entity_types" in metadata
        assert "entity_values" in metadata
        assert "entity_map" in metadata
        assert isinstance(metadata["entity_types"], list)
        assert isinstance(metadata["entity_values"], list)

class TestQueryDisambiguation:
    """Test query processing and disambiguation."""

    @pytest.fixture
    def query_processor(self):
        """Initialize query processor."""
        # Setup would depend on your implementation
        pass

    def test_polysemy_resolution(self, query_processor):
        """Test resolution of polysemous terms."""
        # "bank" as financial institution vs river bank
        query1 = "What are the interest rates at Chase bank?"
        query2 = "What fish live near the river bank?"

        processed1 = query_processor.process_query(query1)
        processed2 = query_processor.process_query(query2)

        # Check that disambiguation adds context
        assert "bank" in processed1.disambiguated_query.lower()
        assert any(e.label in ["organization", "company"]
                   for e in processed1.entities if "bank" in e.text.lower())

class TestHybridSearch:
    """Test hybrid search functionality."""

    def test_entity_filter_creation(self):
        """Test entity filter generation."""
        from services.search_service import HybridSearchService

        entity_filters = {
            "entity_types": ["person", "organization"],
            "entity_values": ["Elon Musk", "Tesla"],
            "entity_values_normalized": ["elon musk", "tesla"]
        }

        # Verify filter structure
        # Implementation-specific test
        pass

    def test_search_with_entity_filter_improves_relevance(self):
        """Test that entity filtering improves search relevance."""
        # This would require a test collection with known content
        # Compare results with and without entity filtering
        pass
```

### 10.2 Benchmarking Script

```python
# scripts/benchmark_ner.py

"""
Benchmark GLiNER performance for production sizing.

Metrics:
- Throughput (texts/second)
- Latency (p50, p95, p99)
- Memory usage
- GPU utilization
"""

import time
import statistics
from typing import List, Dict
import torch

def benchmark_ner_throughput(
    ner_service,
    texts: List[str],
    labels: List[str],
    num_iterations: int = 3
) -> Dict[str, float]:
    """Benchmark NER throughput."""

    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = ner_service.batch_extract_entities(texts, labels)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    total_texts = len(texts) * num_iterations
    total_time = sum(latencies)

    return {
        "throughput_texts_per_sec": total_texts / total_time,
        "avg_latency_ms": (statistics.mean(latencies) / len(texts)) * 1000,
        "p50_latency_ms": (statistics.median(latencies) / len(texts)) * 1000,
        "p95_latency_ms": (sorted(latencies)[int(0.95 * len(latencies))] / len(texts)) * 1000,
        "total_texts": total_texts,
        "total_time_sec": total_time
    }

def benchmark_memory_usage(ner_service, texts: List[str], labels: List[str]):
    """Benchmark GPU memory usage."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    _ = ner_service.batch_extract_entities(texts, labels)

    return {
        "gpu_available": True,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "current_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024
    }

if __name__ == "__main__":
    # Example benchmark run
    from services.ner_service import GLiNERService, NERConfig

    config = NERConfig(model_name="urchade/gliner_medium-v2.1", batch_size=16)
    service = GLiNERService(config)

    labels = ["person", "organization", "location", "product", "technology"]
    service.initialize(labels)

    # Generate test data
    test_texts = [
        "Microsoft and Google are competing in AI. Sundar Pichai leads Google."
    ] * 1000

    print("Running throughput benchmark...")
    throughput_results = benchmark_ner_throughput(service, test_texts, labels)
    print(f"Throughput: {throughput_results['throughput_texts_per_sec']:.2f} texts/sec")
    print(f"Avg latency: {throughput_results['avg_latency_ms']:.2f} ms/text")

    print("\nRunning memory benchmark...")
    memory_results = benchmark_memory_usage(service, test_texts[:100], labels)
    if memory_results["gpu_available"]:
        print(f"Peak GPU memory: {memory_results['peak_memory_mb']:.2f} MB")
```

---

## 11. Configuration Reference

### 11.1 Environment Variables

```bash
# .env.example

# GLiNER Configuration
GLINER_MODEL_NAME=urchade/gliner_medium-v2.1
GLINER_DEVICE=cuda
GLINER_THRESHOLD=0.45
GLINER_BATCH_SIZE=8
GLINER_USE_ONNX=false
GLINER_ONNX_PATH=

# BGE-M3 Configuration
BGE_MODEL_NAME=BAAI/bge-m3
BGE_DEVICE=cuda
BGE_BATCH_SIZE=32

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents

# Search Configuration
SEARCH_DENSE_LIMIT=50
SEARCH_SPARSE_LIMIT=50
SEARCH_FINAL_LIMIT=20
SEARCH_USE_ENTITY_FILTER=true
SEARCH_ENABLE_COLBERT_RERANK=true
```

### 11.2 Full Configuration Class

```python
# config/settings.py

from pydantic import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    """Application settings with validation."""

    # GLiNER
    gliner_model_name: str = "urchade/gliner_medium-v2.1"
    gliner_device: str = "cuda"
    gliner_threshold: float = 0.45
    gliner_batch_size: int = 8
    gliner_use_onnx: bool = False
    gliner_onnx_path: Optional[str] = None

    # Entity labels (comma-separated in env)
    entity_labels: List[str] = [
        "person", "organization", "location", "product",
        "technology", "date", "event", "concept"
    ]

    # BGE-M3
    bge_model_name: str = "BAAI/bge-m3"
    bge_device: str = "cuda"
    bge_batch_size: int = 32

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "documents"

    # Search
    search_dense_limit: int = 50
    search_sparse_limit: int = 50
    search_final_limit: int = 20
    search_use_entity_filter: bool = True
    search_enable_colbert_rerank: bool = True

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    respect_entity_boundaries: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## 12. Known Issues & Gotchas

### 12.1 Critical Issues

| Issue | Solution |
|-------|----------|
| **GPU not utilized** | Use `model.to('cuda')` NOT `model.cuda()`. See [GitHub Issue #88](https://github.com/urchade/GLiNER/issues/88) |
| **OOM on large batches** | Reduce batch size and call `torch.cuda.empty_cache()` between batches |
| **Slow inference** | Batch processing is critical. Single-text inference is much slower per text |
| **ONNX slower than PyTorch** | Benchmark before deploying ONNX. Sometimes PyTorch is faster |
| **Multi-GPU not supported** | GLiNER doesn't natively support `nn.DataParallel`. Use multiple processes instead |

### 12.2 Best Practices

1. **Entity Label Design**
   - Use lowercase or Title Case labels
   - Keep labels descriptive but concise
   - Limit to 10-30 labels per extraction call
   - Domain-specific labels often work better than generic ones

2. **Threshold Tuning**
   - Lower threshold (0.3-0.4) for queries = better recall
   - Higher threshold (0.5-0.6) for documents = better precision
   - Tune based on your precision/recall requirements

3. **Text Length**
   - Keep text under 512 tokens for best results
   - Longer texts may need chunking before NER extraction
   - GLiNER handles up to ~8KB per call, but quality degrades

4. **Caching**
   - Cache entity extraction results for repeated texts
   - Pre-compute entities during ingestion, not at query time
   - Consider Redis/Memcached for production caching

5. **Monitoring**
   - Track entity extraction latency
   - Monitor GPU memory usage
   - Log entity distribution for drift detection

### 12.3 Common Mistakes

```python
# ❌ WRONG: Using model.cuda()
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
model.cuda()  # GPU won't be properly utilized!

# ✅ CORRECT: Using model.to('cuda')
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
model = model.to('cuda')  # Proper GPU utilization

# ❌ WRONG: Processing texts one at a time
for text in texts:
    entities = model.predict_entities(text, labels)  # Very slow!

# ✅ CORRECT: Batch processing
all_entities = model.batch_predict_entities(texts, labels)  # Much faster

# ❌ WRONG: Not clearing cache for large jobs
for batch in batches:
    results = process(batch)  # Will eventually OOM

# ✅ CORRECT: Periodic cache clearing
for i, batch in enumerate(batches):
    results = process(batch)
    if i % 100 == 0:
        torch.cuda.empty_cache()
```

---

## Appendix A: Quick Start Checklist

- [ ] Install dependencies (`pip install gliner FlagEmbedding qdrant-client`)
- [ ] Define entity labels for your domain
- [ ] Initialize GLiNER with `model.to('cuda')`
- [ ] Create Qdrant collection with multi-vector + entity metadata schema
- [ ] Implement document ingestion with entity enrichment
- [ ] Implement query processing with disambiguation
- [ ] Configure hybrid search with entity filtering
- [ ] Benchmark and tune thresholds
- [ ] Add monitoring and logging
- [ ] Document domain-specific configurations

---

## Appendix B: Reference Links

- [GLiNER GitHub Repository](https://github.com/urchade/GLiNER)
- [GLiNER Paper (NAACL 2024)](https://arxiv.org/abs/2311.08526)
- [BGE-M3 HuggingFace](https://huggingface.co/BAAI/bge-m3)
- [Qdrant Hybrid Search Documentation](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [Qdrant Multi-Vector Tutorial](https://qdrant.tech/documentation/advanced-tutorials/using-multivector-representations/)
- [BGE-M3 + Qdrant Sample](https://github.com/yuniko-software/bge-m3-qdrant-sample)

---

*Document prepared for agentic coder integration. Adapt paths, class names, and configurations to match existing codebase conventions.*
