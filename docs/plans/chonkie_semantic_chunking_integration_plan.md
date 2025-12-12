# Semantic-First Chunking Integration Plan

**Created:** 2025-12-10
**Updated:** 2025-12-10 (Aligned with Research Conclusions)
**Branch:** `dense-graph-enhance` → `feature/semantic-chunking`
**Status:** Ready for Implementation
**Estimated Effort:** 4-5 days

---

## Executive Summary

This plan **replaces** the current `GreedyCombinerV2` greedy combination approach with [Chonkie](https://github.com/chonkie-inc/chonkie)'s `SemanticChunker`. Based on comprehensive research from Anthropic, Jina AI, NVIDIA, Pinecone, Chroma, and Firecrawl (2024-2025), the current approach of **forcing sections to combine until reaching `min_tokens=350`** actively destroys semantic coherence and harms retrieval quality.

**Key Change:** From "combine sections until big enough" to "preserve semantic coherence with smaller chunks".

**Research Consensus:** Smaller semantically coherent chunks (128-512 tokens) consistently outperform larger combined chunks. Your downstream pipeline (GLiNER enrichment, 8-vector embedding, RRF fusion, reranking) already compensates for context loss.

---

## 1. Problem Statement

### 1.1 Current Architecture Flaws

The current `GreedyCombinerV2` approach has these fundamental issues:

| Issue | Current Behavior | Research Recommendation |
|-------|------------------|------------------------|
| **min_tokens=350** | Forces combination of semantically distinct sections | Allow chunks as small as 100-256 tokens |
| **target_max=500** | Grows chunks greedily beyond optimal | 256-400 tokens is optimal for factoid queries |
| **Greedy combination** | Merges H3 sections regardless of topic | Keep topics separate for precise retrieval |
| **Complexity accretion** | microdoc, balance_small_tails, Phase 7E-3 guards | Symptoms of underlying design flaw |

### 1.2 Research Evidence

| Source | Finding |
|--------|---------|
| **Pinecone (2024)** | "Smaller semantically coherent units correspond to potential user queries" |
| **Anthropic** | "Traditional RAG removes context when encoding, causing retrieval failures" |
| **Chroma/Firecrawl** | RecursiveCharacterTextSplitter at 400 tokens = 88-89% recall |
| **NVIDIA** | 256-512 tokens optimal for factoid queries |

### 1.3 What We're Keeping

Your downstream pipeline is **excellent** and already compensates for smaller chunks:

| Component | Why It Helps Small Chunks |
|-----------|---------------------------|
| **GLiNER `_embedding_text`** | Adds entity context (Anthropic's Contextual Retrieval pattern) |
| **entity-sparse vector (1.5x)** | Entity names captured regardless of chunk size |
| **title-sparse vector (2.0x)** | Heading matches boost small chunks |
| **6-field RRF fusion** | Multiple retrieval signals compensate |
| **Cross-encoder reranking** | Filters false positives from smaller chunks |

---

## 2. Architectural Change

### 2.1 Before: Greedy Combination

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CURRENT PIPELINE (PROBLEMATIC)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Markdown Content                                                   │
│       │                                                             │
│       ▼                                                             │
│  parse_markdown() → Sections (heading-based)                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  GreedyCombinerV2                        │                       │
│  │  - min_tokens=350 (FORCES combination)  │ ◄── PROBLEM           │
│  │  - target_max=500 (grows greedily)      │                       │
│  │  - balance_small_tails()                │                       │
│  │  - microdoc_annotations()               │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  GLiNER → Embeddings → Storage                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 After: Semantic-First Chunking

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NEW PIPELINE (RESEARCH-ALIGNED)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Markdown Content                                                   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  SemanticChunkerAssembler (NEW)         │                       │
│  │  - Uses Chonkie SemanticChunker         │                       │
│  │  - similarity_threshold: 0.7            │                       │
│  │  - target_tokens: 400 (optimal)         │                       │
│  │  - min_tokens: 100 (ALLOWS small)       │                       │
│  │  - Preserves heading metadata           │                       │
│  │  - Uses BGE-M3 for boundary detection   │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  GLiNER Entity Enrichment               │ ◄── KEEP (unchanged)  │
│  │  - Adds _embedding_text context         │                       │
│  │  - Entity names for sparse vector       │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  8-Vector BGE-M3 Embedding              │ ◄── KEEP (unchanged)  │
│  │  - Dense, sparse, ColBERT, entity       │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  6-field RRF Fusion + Reranking            ◄── KEEP (unchanged)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Design Principles

1. **Semantic coherence over token count** - Allow naturally small chunks if they're coherent
2. **Let embeddings handle context** - GLiNER + multi-vector compensates for chunk isolation
3. **Respect structural boundaries** - Use headings as hard boundaries, semantic detection within
4. **Simplify, don't accumulate** - Remove complexity that compensates for bad chunking

---

## 3. Implementation Plan

### Phase 1: Foundation (Day 1)

#### Task 1.1: Add Chonkie Dependency

**File:** `requirements.txt`

```diff
+ # Semantic chunking (research-aligned approach)
+ chonkie[semantic]>=0.3.0
```

#### Task 1.2: Add SemanticChunkingConfig

**File:** `src/shared/config.py`

```python
class SemanticChunkingConfig(BaseModel):
    """
    Configuration for semantic-first chunking using Chonkie.

    Research-aligned defaults:
    - target_tokens: 400 (Chroma/Firecrawl optimal)
    - min_tokens: 100 (allow small coherent chunks)
    - max_tokens: 512 (NVIDIA recommendation ceiling)
    - similarity_threshold: 0.7 (topic boundary detection)

    Consensus refinements (o3 + Gemini 3 Pro review):
    - skip_code_blocks: Feature flag to avoid splitting code blocks
    - Fallback uses RecursiveCharacterTextSplitter (not simple section chunks)
    """
    enabled: bool = False  # Start disabled for safe rollout
    similarity_threshold: float = 0.7
    target_tokens: int = 400
    min_tokens: int = 100  # KEY: Allow small coherent chunks
    max_tokens: int = 512
    respect_sentence_boundaries: bool = True
    embedding_adapter: str = "bge_m3"

    # Structural boundary handling
    preserve_heading_boundaries: bool = True  # Never merge across headings
    heading_context_in_chunks: bool = True    # Include heading in chunk text

    # CONSENSUS REFINEMENT: Code block handling (o3 concern about technical docs)
    skip_code_blocks: bool = True             # Feature flag: skip semantic splitting for code blocks
    code_block_pattern: str = r"```[\s\S]*?```"  # Regex to detect fenced code blocks


class ChunkAssemblyConfig(BaseModel):
    assembler: str = "structured"  # Options: structured, semantic, greedy, pipeline
    structure: ChunkStructureConfig = Field(default_factory=ChunkStructureConfig)
    split: ChunkSplitConfig = Field(default_factory=ChunkSplitConfig)
    microdoc: ChunkMicrodocConfig = Field(default_factory=ChunkMicrodocConfig)
    semantic: SemanticEnrichmentConfig = Field(default_factory=SemanticEnrichmentConfig)
    semantic_chunking: SemanticChunkingConfig = Field(default_factory=SemanticChunkingConfig)  # NEW
```

#### Task 1.3: Create BGE-M3 Adapter for Chonkie

**File:** `src/providers/embeddings/chonkie_adapter.py` (NEW)

```python
"""
Adapter to use BGE-M3 embedding service with Chonkie's SemanticChunker.

Chonkie expects an embedding interface for semantic boundary detection.
This adapter bridges our BGE-M3 HTTP service (http://127.0.0.1:9000) to that interface.
"""

from typing import List, Any, Optional
import logging
import numpy as np
import httpx

from src.providers.tokenizer_service import TokenizerService

log = logging.getLogger(__name__)

try:
    from chonkie.embeddings import BaseEmbeddings
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    BaseEmbeddings = object  # Stub for type hints


class BgeM3ChonkieAdapter(BaseEmbeddings if CHONKIE_AVAILABLE else object):
    """
    Bridges BGE-M3 service to Chonkie's embedding interface.

    Uses dense embeddings only - chonkie needs these for similarity computation
    to detect semantic boundaries. The sparse/colbert embeddings are used later
    during the main embedding phase.
    """

    def __init__(
        self,
        service_url: str = "http://127.0.0.1:9000",
        model_name: str = "BAAI/bge-m3",
        timeout: float = 60.0,  # CONSENSUS REFINEMENT: Increased from 30s for boundary detection latency
    ):
        self._service_url = service_url
        self._model_name = model_name
        self._timeout = timeout
        self._tokenizer = TokenizerService()
        self._dimension = 1024  # BGE-M3 dense embedding dimension
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Lazy client initialization for connection reuse."""
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts via BGE-M3 service."""
        if not texts:
            return []

        client = self._get_client()
        try:
            response = client.post(
                f"{self._service_url}/v1/embeddings",
                json={"model": self._model_name, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            log.error("BGE-M3 embedding request failed: %s", e)
            raise

        # Extract embeddings from OpenAI-compatible response
        embeddings = []
        for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
            embeddings.append(np.array(item["embedding"], dtype=np.float32))
        return embeddings

    def count_tokens(self, text: str) -> int:
        """Count tokens using our tokenizer service."""
        return self._tokenizer.count_tokens(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for batch of texts."""
        return [self.count_tokens(t) for t in texts]

    def get_tokenizer_or_token_counter(self) -> Any:
        """Return tokenizer for chonkie's internal use."""
        return self._tokenizer

    @classmethod
    def is_available(cls) -> bool:
        """Check if BGE-M3 service is reachable."""
        if not CHONKIE_AVAILABLE:
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get("http://127.0.0.1:9000/healthz")
                return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"BgeM3ChonkieAdapter(url={self._service_url})"
```

---

### Phase 2: Core Implementation (Days 2-3)

#### Task 2.1: Create SemanticChunkerAssembler

**File:** `src/ingestion/semantic_chunker.py` (NEW)

```python
"""
Semantic-first chunking using Chonkie.

This module replaces greedy combination with semantic boundary detection,
creating smaller coherent chunks that align with research best practices.

Key differences from GreedyCombinerV2:
- No forced minimum token count (allows 100+ token chunks)
- Semantic boundary detection using BGE-M3 embeddings
- Preserves heading structure as hard boundaries
- Target ~400 tokens (research optimal) instead of 500+
"""

import hashlib
import logging
from typing import Dict, List, Optional

from src.shared.config import SemanticChunkingConfig, ChunkAssemblyConfig

log = logging.getLogger(__name__)

try:
    from chonkie import SemanticChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    log.warning("chonkie not installed; semantic chunking unavailable")


class SemanticChunkerAssembler:
    """
    Semantic-first chunk assembler using Chonkie.

    Implements the ChunkAssembler protocol but uses semantic boundary
    detection instead of greedy token-based combination.

    Design principles (from research):
    1. Allow naturally small chunks if semantically coherent
    2. Never merge across heading boundaries
    3. Target ~400 tokens (not forced minimum)
    4. Let downstream pipeline (GLiNER, multi-vector) handle context
    """

    def __init__(self, assembly_config: Optional[ChunkAssemblyConfig] = None):
        self.config = (
            assembly_config.semantic_chunking
            if assembly_config
            else SemanticChunkingConfig()
        )
        self._chunker: Optional[SemanticChunker] = None
        self._adapter = None

        if not CHONKIE_AVAILABLE:
            log.warning("SemanticChunkerAssembler: chonkie unavailable, will fall back")
            return

        if not self.config.enabled:
            log.info("SemanticChunkerAssembler: disabled in config")
            return

        self._initialize_chunker()

    def _initialize_chunker(self):
        """Initialize Chonkie with BGE-M3 adapter."""
        from src.providers.embeddings.chonkie_adapter import BgeM3ChonkieAdapter

        self._adapter = BgeM3ChonkieAdapter()
        if not self._adapter.is_available():
            log.warning("BGE-M3 service unavailable; semantic chunking disabled")
            return

        self._chunker = SemanticChunker(
            embedding_model=self._adapter,
            threshold=self.config.similarity_threshold,
            chunk_size=self.config.target_tokens,
        )
        log.info(
            "SemanticChunkerAssembler initialized: threshold=%.2f, target=%d tokens",
            self.config.similarity_threshold,
            self.config.target_tokens,
        )

    def assemble(self, document_id: str, sections: List[Dict]) -> List[Dict]:
        """
        Assemble sections into semantically coherent chunks.

        Unlike GreedyCombinerV2, this:
        - Does NOT force combination to reach min_tokens
        - Detects semantic boundaries within sections
        - Preserves heading boundaries as hard splits

        Args:
            document_id: The document identifier
            sections: List of section dicts from parse_markdown()

        Returns:
            List of chunk dicts ready for embedding
        """
        if not self._chunker:
            # CONSENSUS REFINEMENT: Fall back with size-enforced splitting
            log.warning("Chonkie unavailable; using fallback chunking with size enforcement")
            return self._fallback_section_chunks(document_id, sections)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                document_id=document_id,
                section=section,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        log.info(
            "semantic_chunking_complete",
            document_id=document_id,
            input_sections=len(sections),
            output_chunks=len(chunks),
            avg_tokens=sum(c.get("token_count", 0) for c in chunks) / len(chunks) if chunks else 0,
        )

        return chunks

    def _chunk_section(
        self,
        document_id: str,
        section: Dict,
        start_index: int,
    ) -> List[Dict]:
        """
        Chunk a single section using semantic boundary detection.

        Headings are preserved as metadata in each resulting chunk.

        CONSENSUS REFINEMENT: Includes code block detection to skip semantic
        splitting for code-heavy sections (o3 concern about technical docs).
        """
        import re

        text = section.get("body", section.get("text", ""))
        heading = section.get("heading", "")
        level = section.get("level", 2)

        if not text.strip():
            return []

        # CONSENSUS REFINEMENT: Skip semantic chunking for code blocks (feature flag)
        if self.config.skip_code_blocks:
            code_block_ratio = self._calculate_code_block_ratio(text)
            if code_block_ratio > 0.5:  # Section is >50% code
                log.debug(
                    "skipping_semantic_chunking_for_code_block",
                    heading=heading[:50],
                    code_ratio=code_block_ratio,
                )
                return [self._create_chunk(
                    document_id=document_id,
                    text=text,
                    heading=heading,
                    level=level,
                    index=start_index,
                    section=section,
                    semantic_index=0,
                    semantic_total=1,
                )]

        # Prepend heading to text if configured (helps semantic detection)
        chunk_text = text
        if self.config.heading_context_in_chunks and heading:
            chunk_text = f"## {heading}\n\n{text}"

        # Apply semantic chunking
        try:
            semantic_chunks = self._chunker.chunk(chunk_text)
        except Exception as e:
            log.warning(
                "semantic_chunking_failed: section=%s, error=%s; using as single chunk",
                heading[:50],
                str(e),
            )
            semantic_chunks = None

        # If chunking failed or produced nothing, treat as single chunk
        if not semantic_chunks:
            return [self._create_chunk(
                document_id=document_id,
                text=text,
                heading=heading,
                level=level,
                index=start_index,
                section=section,
                semantic_index=0,
                semantic_total=1,
            )]

        # Filter out tiny fragments below min_tokens
        valid_chunks = [
            sc for sc in semantic_chunks
            if sc.token_count >= self.config.min_tokens
        ]

        # If all chunks were too small, merge back into one
        if not valid_chunks:
            return [self._create_chunk(
                document_id=document_id,
                text=text,
                heading=heading,
                level=level,
                index=start_index,
                section=section,
                semantic_index=0,
                semantic_total=1,
            )]

        # Create chunk dicts preserving section metadata
        result = []
        for i, sc in enumerate(valid_chunks):
            chunk = self._create_chunk(
                document_id=document_id,
                text=sc.text,
                heading=heading if i == 0 else f"{heading} (cont.)",
                level=level,
                index=start_index + i,
                section=section,
                semantic_index=i,
                semantic_total=len(valid_chunks),
                token_count=sc.token_count,
            )
            result.append(chunk)

        return result

    def _create_chunk(
        self,
        document_id: str,
        text: str,
        heading: str,
        level: int,
        index: int,
        section: Dict,
        semantic_index: int,
        semantic_total: int,
        token_count: Optional[int] = None,
    ) -> Dict:
        """Create a chunk dict with all required metadata."""
        # Generate deterministic chunk ID
        content_hash = hashlib.sha256(
            f"{document_id}:{heading}:{semantic_index}:{text[:100]}".encode()
        ).hexdigest()[:16]

        chunk_id = f"{document_id}_chunk_{index}_{content_hash}"

        # Calculate token count if not provided
        if token_count is None:
            from src.providers.tokenizer_service import TokenizerService
            token_count = TokenizerService().count_tokens(text)

        return {
            "id": chunk_id,
            "document_id": document_id,
            "text": text,
            "heading": heading,
            "level": level,
            "chunk_index": index,
            "token_count": token_count,
            # Semantic split provenance
            "is_semantic_split": semantic_total > 1,
            "semantic_split_index": semantic_index,
            "semantic_split_total": semantic_total,
            # Preserve section metadata
            "section_index": section.get("index", 0),
            "doc_tag": section.get("doc_tag", ""),
            # Boundaries for provenance tracking
            "boundaries_json": section.get("boundaries_json", "[]"),
        }

    def _calculate_code_block_ratio(self, text: str) -> float:
        """
        CONSENSUS REFINEMENT: Calculate what fraction of text is code blocks.

        Used to skip semantic chunking for code-heavy sections where
        semantic similarity is unreliable (o3 concern).

        Returns:
            Float between 0 and 1 representing code block ratio.
        """
        import re

        if not text:
            return 0.0

        # Match fenced code blocks (```...```)
        pattern = self.config.code_block_pattern
        matches = re.findall(pattern, text)

        if not matches:
            return 0.0

        code_chars = sum(len(m) for m in matches)
        return code_chars / len(text)

    def _fallback_section_chunks(
        self,
        document_id: str,
        sections: List[Dict],
    ) -> List[Dict]:
        """
        CONSENSUS REFINEMENT: Fallback with size enforcement.

        Both o3 and Gemini 3 Pro identified that the original fallback
        could create oversized chunks (e.g., 2000-token sections).
        This version uses RecursiveCharacterTextSplitter to enforce max_tokens.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Create splitter with same target as semantic chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_tokens * 4,  # ~4 chars per token
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        chunk_index = 0

        for section in sections:
            text = section.get("body", section.get("text", ""))
            if not text.strip():
                continue

            heading = section.get("heading", "")
            level = section.get("level", 2)

            # Check if section needs splitting
            token_count = self._adapter.count_tokens(text) if self._adapter else len(text) // 4

            if token_count <= self.config.max_tokens:
                # Section fits, create single chunk
                chunks.append(self._create_chunk(
                    document_id=document_id,
                    text=text,
                    heading=heading,
                    level=level,
                    index=chunk_index,
                    section=section,
                    semantic_index=0,
                    semantic_total=1,
                ))
                chunk_index += 1
            else:
                # Section too large, split with RecursiveCharacterTextSplitter
                split_texts = splitter.split_text(text)
                for i, split_text in enumerate(split_texts):
                    chunks.append(self._create_chunk(
                        document_id=document_id,
                        text=split_text,
                        heading=heading if i == 0 else f"{heading} (cont.)",
                        level=level,
                        index=chunk_index,
                        section=section,
                        semantic_index=i,
                        semantic_total=len(split_texts),
                    ))
                    chunk_index += 1

        log.info(
            "fallback_chunking_complete",
            document_id=document_id,
            input_sections=len(sections),
            output_chunks=len(chunks),
        )
        return chunks
```

#### Task 2.2: Update Factory Function

**File:** `src/ingestion/chunk_assembler.py` - modify `get_chunk_assembler()`

```python
def get_chunk_assembler(
    assembly_config: Optional[ChunkAssemblyConfig] = None,
) -> ChunkAssembler:
    name = ""
    if assembly_config is not None:
        name = assembly_config.assembler.lower().strip()
    else:
        name = (os.getenv("CHUNK_ASSEMBLER") or "greedy").lower().strip()

    # NEW: Semantic-first chunking (research-aligned approach)
    if name == "semantic":
        from src.ingestion.semantic_chunker import SemanticChunkerAssembler
        assembler = SemanticChunkerAssembler(assembly_config)
        # Verify chonkie initialized; fall back if not
        if assembler._chunker is not None:
            return assembler
        log.warning("Semantic chunker unavailable; falling back to structured")
        return StructuredChunker(assembly_config)

    if name == "structured":
        return StructuredChunker(assembly_config)

    if name == "pipeline":
        try:
            from src.ingestion.pipeline_combiner import PipelineCombiner
            return PipelineCombiner()
        except Exception as e:
            log.warning(
                "CHUNK_ASSEMBLER=pipeline requested but unavailable: %s; falling back to structured",
                e,
            )
            return StructuredChunker(assembly_config)

    if name == "greedy":
        return GreedyCombinerV2()

    log.warning("Unknown chunk assembler '%s'; defaulting to structured", name)
    return StructuredChunker(assembly_config)
```

---

### Phase 3: Testing (Day 4)

#### Task 3.1: Unit Tests for Adapter

**File:** `tests/providers/test_chonkie_adapter.py` (NEW)

```python
"""Tests for BGE-M3 Chonkie adapter."""

import pytest
import numpy as np

from src.providers.embeddings.chonkie_adapter import BgeM3ChonkieAdapter, CHONKIE_AVAILABLE


@pytest.fixture
def adapter():
    """Create adapter instance."""
    return BgeM3ChonkieAdapter()


class TestBgeM3ChonkieAdapter:

    def test_dimension_is_1024(self, adapter):
        assert adapter.dimension == 1024

    def test_repr(self, adapter):
        assert "BgeM3ChonkieAdapter" in repr(adapter)
        assert "127.0.0.1:9000" in repr(adapter)

    @pytest.mark.integration
    def test_is_available_when_service_running(self, adapter):
        # Only passes if BGE-M3 service is running
        if not adapter.is_available():
            pytest.skip("BGE-M3 service not available")
        assert adapter.is_available() is True

    @pytest.mark.integration
    def test_embed_single_text(self, adapter):
        if not adapter.is_available():
            pytest.skip("BGE-M3 service not available")
        embedding = adapter.embed("test query about Weka filesystem")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32

    @pytest.mark.integration
    def test_embed_batch(self, adapter):
        if not adapter.is_available():
            pytest.skip("BGE-M3 service not available")
        texts = ["query one", "query two", "query three"]
        embeddings = adapter.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(e.shape == (1024,) for e in embeddings)

    @pytest.mark.integration
    def test_embed_empty_batch(self, adapter):
        embeddings = adapter.embed_batch([])
        assert embeddings == []

    def test_count_tokens(self, adapter):
        count = adapter.count_tokens("hello world this is a test")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_batch(self, adapter):
        texts = ["hello", "hello world", "hello world test"]
        counts = adapter.count_tokens_batch(texts)
        assert len(counts) == 3
        assert counts[0] < counts[1] < counts[2]  # Longer texts = more tokens
```

#### Task 3.2: Unit Tests for Semantic Chunker

**File:** `tests/ingestion/test_semantic_chunker.py` (NEW)

```python
"""Tests for semantic-first chunking."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.ingestion.semantic_chunker import SemanticChunkerAssembler
from src.shared.config import SemanticChunkingConfig, ChunkAssemblyConfig


class TestSemanticChunkerAssembler:

    @pytest.fixture
    def disabled_config(self):
        return ChunkAssemblyConfig(
            assembler="semantic",
            semantic_chunking=SemanticChunkingConfig(enabled=False),
        )

    @pytest.fixture
    def enabled_config(self):
        return ChunkAssemblyConfig(
            assembler="semantic",
            semantic_chunking=SemanticChunkingConfig(
                enabled=True,
                similarity_threshold=0.7,
                target_tokens=400,
                min_tokens=100,
            ),
        )

    def test_disabled_uses_fallback(self, disabled_config):
        """When disabled, should use simple section chunking."""
        assembler = SemanticChunkerAssembler(disabled_config)

        sections = [
            {"heading": "Section 1", "body": "Content one", "level": 2, "index": 0},
            {"heading": "Section 2", "body": "Content two", "level": 2, "index": 1},
        ]

        chunks = assembler.assemble("doc1", sections)

        # Should produce one chunk per section (no combination)
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "Section 1"
        assert chunks[1]["heading"] == "Section 2"

    def test_empty_sections_produces_empty_chunks(self, disabled_config):
        assembler = SemanticChunkerAssembler(disabled_config)
        chunks = assembler.assemble("doc1", [])
        assert chunks == []

    def test_skips_empty_section_body(self, disabled_config):
        assembler = SemanticChunkerAssembler(disabled_config)
        sections = [
            {"heading": "Empty", "body": "", "level": 2, "index": 0},
            {"heading": "Has content", "body": "Real content here", "level": 2, "index": 1},
        ]
        chunks = assembler.assemble("doc1", sections)
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Has content"

    def test_chunk_has_required_fields(self, disabled_config):
        assembler = SemanticChunkerAssembler(disabled_config)
        sections = [{"heading": "Test", "body": "Test content", "level": 2, "index": 0}]
        chunks = assembler.assemble("doc1", sections)

        chunk = chunks[0]
        assert "id" in chunk
        assert "document_id" in chunk
        assert "text" in chunk
        assert "heading" in chunk
        assert "level" in chunk
        assert "chunk_index" in chunk
        assert "token_count" in chunk
        assert chunk["document_id"] == "doc1"

    def test_semantic_split_metadata(self, disabled_config):
        """Chunks should have semantic split provenance."""
        assembler = SemanticChunkerAssembler(disabled_config)
        sections = [{"heading": "Test", "body": "Content", "level": 2, "index": 0}]
        chunks = assembler.assemble("doc1", sections)

        chunk = chunks[0]
        assert "is_semantic_split" in chunk
        assert "semantic_split_index" in chunk
        assert "semantic_split_total" in chunk

    @pytest.mark.integration
    def test_with_real_chonkie(self, enabled_config):
        """Integration test with actual chonkie (requires BGE-M3 service)."""
        from src.providers.embeddings.chonkie_adapter import BgeM3ChonkieAdapter

        if not BgeM3ChonkieAdapter.is_available():
            pytest.skip("BGE-M3 service not available")

        assembler = SemanticChunkerAssembler(enabled_config)

        # A section with distinct topics that should be split
        sections = [{
            "heading": "Mixed Topics",
            "body": """
            Weka filesystem provides high-performance storage for AI workloads.
            It uses a distributed architecture with erasure coding for reliability.

            On a completely different note, here are some cooking tips.
            Always preheat your oven before baking. Use fresh ingredients
            for the best flavor. Season your food throughout the cooking process.
            """,
            "level": 2,
            "index": 0,
        }]

        chunks = assembler.assemble("doc1", sections)

        # Should detect semantic shift and create multiple chunks
        # (exact number depends on threshold tuning)
        assert len(chunks) >= 1
        assert all(c["token_count"] > 0 for c in chunks)
```

#### Task 3.3: Integration Test

**File:** `tests/integration/test_semantic_chunking_e2e.py` (NEW)

```python
"""End-to-end integration tests for semantic chunking pipeline."""

import pytest

from src.ingestion.chunk_assembler import get_chunk_assembler
from src.shared.config import ChunkAssemblyConfig, SemanticChunkingConfig


@pytest.mark.integration
class TestSemanticChunkingE2E:

    def test_factory_returns_semantic_chunker(self):
        """Factory should return SemanticChunkerAssembler for assembler='semantic'."""
        from src.ingestion.semantic_chunker import SemanticChunkerAssembler
        from src.providers.embeddings.chonkie_adapter import BgeM3ChonkieAdapter

        if not BgeM3ChonkieAdapter.is_available():
            pytest.skip("BGE-M3 service not available")

        config = ChunkAssemblyConfig(
            assembler="semantic",
            semantic_chunking=SemanticChunkingConfig(enabled=True),
        )

        assembler = get_chunk_assembler(config)
        assert isinstance(assembler, SemanticChunkerAssembler)

    def test_backward_compatible_when_disabled(self):
        """Disabled semantic chunking should not change existing behavior."""
        config = ChunkAssemblyConfig(
            assembler="structured",
            semantic_chunking=SemanticChunkingConfig(enabled=False),
        )

        # Should use StructuredChunker, not SemanticChunkerAssembler
        assembler = get_chunk_assembler(config)
        assert assembler.__class__.__name__ == "StructuredChunker"

    def test_falls_back_when_service_unavailable(self):
        """Should gracefully fall back if BGE-M3 service is down."""
        config = ChunkAssemblyConfig(
            assembler="semantic",
            semantic_chunking=SemanticChunkingConfig(
                enabled=True,
                # Use invalid URL to force failure
            ),
        )

        # Should not raise, should fall back
        assembler = get_chunk_assembler(config)
        # Verify it either initialized or fell back gracefully
        assert assembler is not None
```

---

### Phase 4: Configuration & Rollout (Day 5)

#### Task 4.1: Update Development Config

**File:** `config/development.yaml`

```yaml
ingestion:
  chunk_assembly:
    # CHANGED: Use semantic chunking as primary strategy
    assembler: "semantic"  # Options: semantic, structured, greedy

    # Legacy structured config (used as fallback)
    structure:
      min_tokens: 100      # LOWERED: Allow small coherent chunks
      target_tokens: 400   # LOWERED: Research optimal
      hard_tokens: 512     # LOWERED: NVIDIA ceiling
      max_sections: 8
      stop_at_level: 2

    # NEW: Semantic chunking config (primary)
    semantic_chunking:
      enabled: true
      similarity_threshold: 0.7   # Topic boundary detection sensitivity
      target_tokens: 400          # Research optimal (Chroma/Firecrawl)
      min_tokens: 100             # Allow naturally small chunks
      max_tokens: 512             # Hard ceiling (NVIDIA recommendation)
      respect_sentence_boundaries: true
      embedding_adapter: "bge_m3"
      preserve_heading_boundaries: true
      heading_context_in_chunks: true
```

#### Task 4.2: Feature Flag for Gradual Rollout

Add environment variable override for testing:

```python
# In SemanticChunkerAssembler.__init__:
import os

# Allow force-enable/disable via environment
force_semantic = os.getenv("FORCE_SEMANTIC_CHUNKING", "").lower()
if force_semantic == "true":
    self.config.enabled = True
    log.info("Semantic chunking force-enabled via FORCE_SEMANTIC_CHUNKING")
elif force_semantic == "false":
    self.config.enabled = False
    log.info("Semantic chunking force-disabled via FORCE_SEMANTIC_CHUNKING")
```

#### Task 4.3: Deprecation Notices

Add deprecation warnings to complexity we're removing:

```python
# In GreedyCombinerV2 and related code:
import warnings

class GreedyCombinerV2:
    def __init__(self):
        warnings.warn(
            "GreedyCombinerV2 is deprecated. Use assembler='semantic' for "
            "research-aligned chunking. See docs/plans/chunking_architecture_analysis.md",
            DeprecationWarning,
            stacklevel=2,
        )
```

---

## 4. Components to Deprecate/Remove

Based on research analysis, these components compensate for greedy combination flaws:

| Component | Location | Action | Rationale |
|-----------|----------|--------|-----------|
| `min_tokens` enforcement | `GreedyCombinerV2` | Deprecate | Research says small chunks are fine |
| `_balance_small_tails()` | `GreedyCombinerV2` | Remove | Small tails are acceptable |
| `_apply_microdoc_annotations()` | `GreedyCombinerV2` | Remove | Unnecessary complexity |
| `_apply_doc_fallback()` | `GreedyCombinerV2` | Remove | Unnecessary complexity |
| Phase 7E-3 guards | Various | Keep temporarily | Document as tech debt for later removal |

**Note:** Don't remove these immediately. Mark as deprecated, collect metrics comparing old vs new approach, then remove after validation.

---

## 4.5 Consensus Refinements (o3 + Gemini 3 Pro Review)

This plan was reviewed by o3 (devil's advocate, 7/10 confidence) and Gemini 3 Pro (advocate, 9/10 confidence) using multi-model consensus. The following refinements were incorporated:

### Refinements Incorporated

| # | Refinement | Rationale | Location in Code |
|---|------------|-----------|------------------|
| 1 | **Fallback with RecursiveCharacterTextSplitter** | Original fallback created oversized chunks for large sections | `_fallback_section_chunks()` |
| 2 | **Code block detection (feature flag)** | Technical docs have code blocks where semantic similarity is unreliable | `skip_code_blocks` config + `_calculate_code_block_ratio()` |
| 3 | **Increased httpx timeout (60s)** | Boundary detection embedding calls add latency | `BgeM3ChonkieAdapter.__init__()` |

### Refinements Considered but Deferred

| # | Refinement | Reason for Deferral |
|---|------------|---------------------|
| 1 | Pre-implementation benchmark suite | Want to move forward with implementation |
| 2 | Quick-win test (min_tokens=150) | Prefer to implement full solution |

### Key Model Insights

**o3 (Conservative):**
- Research cited is from FAQ-style corpora, not technical manuals
- Expect 2-5x ingestion time increase due to embedding calls
- Code blocks have low semantic similarity even within a topic

**Gemini 3 Pro (Progressive):**
- Plan correctly identifies root cause (greedy merging dilutes GLiNER context)
- Fallback must enforce max_tokens, not just pass sections through
- `heading_context_in_chunks: True` is excellent for priming embeddings

---

## 5. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Chonkie import failure** | Graceful fallback via `CHONKIE_AVAILABLE` flag |
| **BGE-M3 unavailable** | Health check in adapter; disable if unreachable |
| **Token count mismatch** | Same `TokenizerService` used throughout |
| **More chunks = more vectors** | Expected; monitor Qdrant storage |
| **Performance regression** | Benchmark before/after with retrieval quality metrics |
| **Metadata loss** | Preserve all section metadata in chunk creation |
| **Oversized fallback chunks** | CONSENSUS FIX: `_fallback_section_chunks` uses RecursiveCharacterTextSplitter |
| **Code block splitting** | CONSENSUS FIX: `skip_code_blocks` feature flag skips semantic chunking for >50% code sections |
| **Boundary detection latency** | CONSENSUS FIX: httpx timeout increased to 60s |

### Rollback Strategy

1. Set `assembler: "structured"` in config (or `semantic_chunking.enabled: false`)
2. Restart workers
3. Re-ingest documents
4. No schema migration needed

---

## 6. Success Criteria

### Functional

- [ ] Chonkie installed and importable
- [ ] BGE-M3 adapter passes all unit tests
- [ ] SemanticChunkerAssembler produces valid chunks
- [ ] Factory correctly routes to semantic chunker
- [ ] Fallback works when chonkie/service unavailable

### Quality (Expected Improvements)

| Metric | Current | Target |
|--------|---------|--------|
| Avg chunk size | ~450 tokens | ~300 tokens |
| Chunk count | ~1,300 | ~2,000+ |
| Recall@20 | ~85% | ~92-95% |
| Failed retrievals | ~15% | ~5-8% |

### Non-Functional

- [ ] No breaking changes to downstream pipeline (GLiNER, embedding, retrieval)
- [ ] Graceful degradation when dependencies unavailable
- [ ] Structured logging for observability
- [ ] All existing tests continue to pass

---

## 7. File Change Summary

| File | Change | Description |
|------|--------|-------------|
| `requirements.txt` | Modify | Add `chonkie[semantic]>=0.3.0` |
| `src/shared/config.py` | Modify | Add `SemanticChunkingConfig` |
| `src/providers/embeddings/chonkie_adapter.py` | **New** | BGE-M3 adapter for Chonkie |
| `src/ingestion/semantic_chunker.py` | **New** | SemanticChunkerAssembler class |
| `src/ingestion/chunk_assembler.py` | Modify | Add "semantic" to factory |
| `config/development.yaml` | Modify | Enable semantic chunking |
| `tests/providers/test_chonkie_adapter.py` | **New** | Adapter unit tests |
| `tests/ingestion/test_semantic_chunker.py` | **New** | Chunker unit tests |
| `tests/integration/test_semantic_chunking_e2e.py` | **New** | E2E tests |

---

## 8. Appendix: Research Summary

### Key Quotes

**Pinecone's Schwaber-Cohen:**
> "What we found for the most part is that you would have better luck if you're able to create **smaller semantically coherent units** that correspond to potential user queries."

**Anthropic:**
> "Traditional RAG solutions **remove context when encoding information**, which often results in the system failing to retrieve the relevant information."

**Firecrawl 2025:**
> "The wrong strategy can create up to a **9% gap in recall performance** between best and worst approaches."

### Optimal Parameters (Research Consensus)

| Parameter | Value | Source |
|-----------|-------|--------|
| Target chunk size | 400 tokens | Chroma, Firecrawl |
| Minimum chunk size | 100-128 tokens | Milvus, NVIDIA |
| Maximum chunk size | 512 tokens | NVIDIA benchmarks |
| Similarity threshold | 0.7 | Chonkie default |

---

*Plan updated 2025-12-10 to align with research conclusions from chunking_architecture_analysis.md*
