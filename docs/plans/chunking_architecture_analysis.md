# Chunking Architecture Analysis: Current State vs. Research Best Practices

**Date:** 2025-12-10
**Status:** Critical Assessment
**Verdict:** Current greedy combination approach is fundamentally misaligned with modern RAG research

---

## Executive Summary

After comprehensive research across Anthropic, Jina AI, NVIDIA, Pinecone, Chroma, and Firecrawl's 2025 benchmarks, **the current `GreedyCombinerV2` architecture is doing the opposite of what research recommends**. The system greedily combines sections to reach token minimums, destroying semantic coherence in the process.

**Key Finding:** Smaller, semantically coherent chunks (128-512 tokens) consistently outperform larger combined chunks in retrieval quality. Your `min_tokens=350` threshold that *forces* combination is actively harming retrieval.

---

## Research Findings Summary

### 1. Optimal Chunk Size: Smaller is Better

| Source | Recommended Size | Key Quote |
|--------|------------------|-----------|
| **Pinecone (Stack Overflow 2024)** | "Smaller semantically coherent units" | "You would have better luck if you're able to create smaller semantically coherent units that correspond to potential user queries" |
| **Milvus** | 128-512 tokens | "Smaller chunks (128-256 tokens) are better for fact-based queries needing precise keyword matching" |
| **Firecrawl/NVIDIA 2024** | 256-512 tokens for factoid queries | "RecursiveCharacterTextSplitter at 400 tokens = 88-89% recall" |
| **Chroma Research** | 400 tokens optimal | "Performance varied by up to 9% in recall across methods" |
| **Anthropic** | "Few hundred tokens" | "Break down the knowledge base into smaller chunks of text, usually no more than a few hundred tokens" |

### 2. The Context Problem (and Solutions)

**Problem Identified by Both Anthropic and Jina:**
```
Original chunk: "The company's revenue grew by 3% over the previous quarter."
Problem: What company? What quarter? Context is lost.
```

**Anthropic's Solution - Contextual Retrieval:**
```
Contextualized chunk: "This chunk is from an SEC filing on ACME corp's
performance in Q2 2023; the previous quarter's revenue was $314 million.
The company's revenue grew by 3% over the previous quarter."
```
- Prepend chunk-specific context BEFORE embedding
- Reduced retrieval failures by **49%** (with BM25)
- With reranking: **67%** reduction

**Jina's Solution - Late Chunking:**
- Run transformer on ENTIRE document first (using 8K context)
- Then chunk and mean-pool
- Each chunk embedding is "conditioned on" surrounding context
- **nDCG@10 improvements across all BeIR datasets**

### 3. What NOT to Do

From Pinecone's Roie Schwaber-Cohen:
> "If I embedded a full chapter of content instead of just a page or paragraph, the vector database is going to find some semantic similarity between the query and that chapter. Now, is all that chapter relevant? **Probably not.**"

From Firecrawl 2025:
> "Smaller chunks match queries more precisely but lose surrounding context. Larger chunks preserve relationships between ideas but **dilute relevance in your embeddings.**"

---

## Current Architecture Analysis

### What `GreedyCombinerV2.assemble()` Does

```python
# Current problematic flow:
min_tokens = 350      # Forces combination to reach this
target_max = 500      # Growth target
hard_max = 750        # Maximum ceiling

# The greedy loop:
while g_tokens < self.target_max:  # Keep combining until 500+ tokens
    # Add next section...

# Force minimum:
while g_tokens < self.min_tokens:  # FORCE combination even below target
    # Keep adding sections to reach 350 minimum
```

### Problems Identified

| Issue | Current Behavior | Research Recommendation |
|-------|------------------|------------------------|
| **min_tokens=350** | Forces combination of distinct sections | Allow chunks as small as 100-256 tokens |
| **target_max=500** | Grows chunks greedily | 256-400 tokens is optimal |
| **Greedy combination** | Merges semantically distinct H3 sections | Keep topics separate |
| **Microdoc logic** | Complex special-casing | Unnecessary with proper semantic chunking |
| **Phase 7E-3 guards** | Bandaid on broken approach | Symptom of underlying problem |

### What You're Doing RIGHT

1. **GLiNER entity enrichment via `_embedding_text`** - This IS Anthropic's contextual retrieval pattern!
2. **8-vector multi-embedding** - Dense + sparse + entity vectors
3. **RRF fusion** - Combining multiple signals
4. **Reranking** - Final stage quality filter
5. **Heading structure preservation** - Good boundary cues

### The Core Contradiction

Your `_embedding_text` pattern already implements Anthropic's insight:
```python
# Current pattern in atomic.py:
chunk["_embedding_text"] = f"""Section: {heading}

{text}

[Context: {entity_names}]
"""
```

But then `GreedyCombinerV2` **destroys this benefit** by combining multiple topics into single chunks, diluting the embedding quality that the entity enrichment provides.

---

## Why Smaller Chunks Work With Your Architecture

Your system has **all the components** to make small chunks work brilliantly:

### 1. Entity-Sparse Vector Compensates for Small Chunks
Small chunks might miss keyword matches, but your `entity-sparse` vector (weight 1.5x) ensures entity names are captured regardless of chunk size.

### 2. Title-Sparse Vector Boosts Heading Matches
Your `title-sparse` vector (weight 2.0x) ensures queries matching section headings rank highly, even for small chunks.

### 3. RRF Fusion Aggregates Multiple Signals
6 vectors per chunk means small chunks still have rich retrieval signals.

### 4. Reranking Handles Noise
Cross-encoder reranking filters out false positives that small chunks might produce.

### 5. GLiNER Adds Context Back
Your `_embedding_text` pattern adds entity context that compensates for lost document context.

---

## Proposed Architecture: Semantic-First Chunking

### New Philosophy

**Old:** "Combine sections until we reach token minimum"
**New:** "Keep sections semantically coherent; let embeddings handle context"

### Simplified Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PROPOSED PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Markdown Content                                                   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  Chonkie SemanticChunker                │                       │
│  │  - threshold: 0.7                       │                       │
│  │  - chunk_size: 400 tokens               │                       │
│  │  - Uses BGE-M3 8K context               │                       │
│  │  - Respects sentence boundaries         │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  GLiNER Entity Enrichment               │ ← Keep this!          │
│  │  - Adds _embedding_text context         │                       │
│  │  - Entity names for sparse vector       │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │  8-Vector BGE-M3 Embedding              │ ← Keep this!          │
│  │  - Uses _embedding_text for context     │                       │
│  └─────────────────────────────────────────┘                       │
│       │                                                             │
│       ▼                                                             │
│  6-field RRF Fusion + Reranking            ← Keep this!            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration Changes

```yaml
# OLD (problematic):
chunk_assembly:
  assembler: "structured"
  structure:
    min_tokens: 350    # Forces combination
    target_tokens: 500
    hard_tokens: 750

# NEW (research-aligned):
chunk_assembly:
  assembler: "semantic"  # Use chonkie
  semantic:
    enabled: true
    similarity_threshold: 0.7
    target_tokens: 400        # Optimal per research
    max_tokens: 512           # Hard ceiling
    min_tokens: 100           # Allow small coherent chunks
    respect_sentence_boundaries: true
    embedding_adapter: "bge_m3"
```

### Code to Remove/Simplify

| Component | Action | Rationale |
|-----------|--------|-----------|
| `min_tokens` enforcement | Remove | Research says small chunks are fine |
| `target_max` growth loop | Simplify | Don't greedily combine |
| `_balance_small_tails()` | Remove | Small chunks are acceptable |
| `_apply_microdoc_annotations()` | Remove | Unnecessary complexity |
| `_apply_doc_fallback()` | Remove | Unnecessary complexity |
| Phase 7E-3 guards | Keep temporarily | Document these as technical debt |

### What to Keep

| Component | Reason |
|-----------|--------|
| GLiNER enrichment | Implements Anthropic's contextual retrieval |
| `_embedding_text` pattern | Adds document context to embeddings |
| 8-vector embedding | Rich retrieval signals |
| RRF fusion | Combines multiple signals effectively |
| Reranking | Quality filter for final results |

---

## Implementation Options

### Option A: Chonkie SemanticChunker (Recommended)

**Pros:**
- Purpose-built for semantic chunking
- Detects topic shifts automatically
- Well-maintained, active development
- Integrates with custom embeddings (your BGE-M3)

**Cons:**
- New dependency
- Requires embedding calls during chunking

```python
from chonkie import SemanticChunker

chunker = SemanticChunker(
    embedding_model=BgeM3ChonkieAdapter(),
    threshold=0.7,
    chunk_size=400
)

chunks = chunker.chunk(document_text)
```

### Option B: Late Chunking with BGE-M3

**Pros:**
- Uses your existing 8K context model
- No chunking-time embeddings needed
- Chunks are contextually conditioned

**Cons:**
- More complex implementation
- Requires full document fit in context

```python
# 1. Encode full document through transformer
token_embeddings = bge_m3.encode_tokens(full_document)

# 2. Define chunk boundaries (sentence-level)
boundaries = segment_by_sentences(full_document)

# 3. Mean-pool each chunk's token embeddings
chunk_embeddings = [
    mean_pool(token_embeddings[start:end])
    for start, end in boundaries
]
```

### Option C: RecursiveCharacterTextSplitter + Entity Context

**Pros:**
- Simple, well-understood
- No embedding overhead at chunking time
- Your GLiNER context adds back what's lost

**Cons:**
- Doesn't detect semantic boundaries
- May split mid-topic

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

---

## Recommended Implementation Plan

### Phase 1: Measure Baseline (Day 1)
1. Create retrieval quality benchmark with 20-30 realistic queries
2. Measure current recall@10, recall@20, MRR
3. Document results

### Phase 2: Simplify Current System (Day 2)
1. Lower `min_tokens` from 350 to 100 (allow small chunks)
2. Lower `target_tokens` from 500 to 400
3. Disable `microdoc` and `doc_fallback`
4. Re-run benchmark

### Phase 3: Integrate Chonkie (Days 3-4)
1. Add chonkie dependency
2. Create `SemanticChunker` assembler option
3. Implement BGE-M3 adapter
4. Keep GLiNER enrichment pipeline
5. Re-run benchmark

### Phase 4: Evaluate and Tune (Day 5)
1. Compare all three approaches
2. Tune similarity threshold
3. Document optimal configuration

---

## Expected Outcomes

Based on research findings:

| Metric | Current (Expected) | After Changes (Expected) |
|--------|-------------------|--------------------------|
| Recall@20 | ~85% | ~92-95% |
| Chunk count | ~1,300 | ~2,000-2,500 (smaller chunks) |
| Avg chunk size | 450 tokens | 250-350 tokens |
| Failed retrievals | ~15% | ~5-8% |

---

## Key Quotes to Remember

**Pinecone's Schwaber-Cohen:**
> "What we found for the most part is that you would have better luck if you're able to create **smaller semantically coherent units** that correspond to potential user queries."

**Anthropic:**
> "Traditional RAG solutions **remove context when encoding information**, which often results in the system failing to retrieve the relevant information."

**Jina AI:**
> "Semantic chunking is overrated. Especially when you write a super regex that leverages all possible boundary cues and heuristics to segment text accurately **without the need for complex language models**."

**Firecrawl 2025:**
> "The wrong strategy can create up to a **9% gap in recall performance** between best and worst approaches."

---

## Conclusion

Your current architecture has excellent downstream components (multi-vector, RRF, reranking, entity enrichment) but the **chunking layer is actively working against them**. The greedy combination approach:

1. Destroys semantic coherence by merging distinct topics
2. Creates oversized chunks that dilute embedding quality
3. Adds complexity (microdoc, balance_small_tails, Phase 7E-3 guards) to compensate for fundamental design flaws

**Recommendation:** Replace `GreedyCombinerV2` with chonkie's `SemanticChunker` while keeping your excellent enrichment and retrieval pipeline intact.

---

*Analysis based on research from Anthropic, Jina AI, Pinecone, NVIDIA, Chroma, and Firecrawl (2024-2025)*
