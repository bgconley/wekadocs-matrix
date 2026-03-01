# Snowflake Arctic Qdrant Collection Design

**Date:** 2026-01-19
**Status:** Approved
**Author:** Claude + Brennan

## Overview

Create the `chunks_multi_snowflake_arctic_v2l` Qdrant collection to support Snowflake Arctic as the primary dense embedder in the multi-embedder pipeline.

## Context

- The config (`embedding_profiles.yaml`) specifies `snowflake_arctic_v2l` as the dense embedder
- The MCP server health check fails because the collection doesn't exist
- This blocks Claude Desktop from connecting to the wekadocs-matrix MCP server

## Design Decisions

### 1. Schema Only (Empty Collection)
Create the collection structure without data. Data will be populated via ingestion pipeline later.

### 2. Include Sparse Vectors
Even though Snowflake Arctic is dense-only, include sparse vector configs for pipeline compatibility. The pipeline embeds sparse vectors (via BGE-M3) alongside each dense collection.

### 3. Match Existing Schema
Schema matches `chunks_multi_bge_m3` and `chunks_multi_voyage_context_3` exactly for consistency.

## Collection Schema

### Dense Vectors (1024 dims, cosine)
- `content` - Main chunk text embedding
- `title` - Section title embedding
- `doc_title` - Document title embedding
- `late-interaction` - ColBERT multi-vector (from BGE-M3)

### Sparse Vectors (on-disk)
- `text-sparse`
- `title-sparse`
- `doc_title-sparse`
- `entity-sparse`

### Payload Indexes (36 fields)
See implementation script for full list. Matches existing collections exactly.

## Implementation

Script: `scripts/qdrant_setup_snowflake_arctic.py`

```bash
python scripts/qdrant_setup_snowflake_arctic.py
```

## Verification

```bash
curl -s http://localhost:6333/collections/chunks_multi_snowflake_arctic_v2l | jq .result.status
# Expected: "green"
```
