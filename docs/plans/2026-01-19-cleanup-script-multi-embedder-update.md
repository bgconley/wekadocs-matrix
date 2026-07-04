# Cleanup Script Multi-Embedder Update

**Date:** 2026-01-19
**Status:** Implemented

## Summary

Updated `scripts/cleanup-databases.py` to support full multi-embedder reset and ensure all WEKA-specific Neo4j node types are properly cleaned.

## Problem Statement

The cleanup script had two gaps preventing a clean database reset:

1. **Neo4j:** 6 node types existed in the database but weren't in `DATA_LABELS`, causing them to be orphaned after cleanup
2. **Qdrant:** Only the configured primary collection was targeted, missing other embedding profile collections (`bge_m3`, `voyage_context_3`)

## Changes Made

### Neo4j DATA_LABELS Additions

Added 6 domain-specific entity types discovered in WEKA documentation:

```python
"CapacityMetric",   # Storage capacity metrics
"CloudProvider",    # AWS, Azure, GCP references
"ProcedureStep",    # Sub-steps within Procedure nodes
"Protocol",         # Network/storage protocols (NFS, SMB, etc.)
"StorageConcept",   # WEKA-specific storage concepts
"Version",          # Software version references
```

### Qdrant Multi-Embedder Pattern Matching

Replaced static collection targeting with dynamic pattern matching:

**Before:** Only cleaned configured collection + hardcoded bases
**After:** Pattern-matches ALL `chunks_multi*` collections at runtime

New logic:
```python
# Prefixes to match (chunks_multi, chunks_multi_*, chunks)
self.qdrant_collection_prefixes = ["chunks_multi", "chunks"]
# Skip test collections
self.qdrant_skip_prefixes = ["test_"]

def _should_clean_collection(self, name: str) -> bool:
    # Skip test_* collections
    # Include chunks, chunks_multi, chunks_multi_*
```

## Behavior

| Collection | Before | After |
|------------|--------|-------|
| `chunks_multi` | Cleaned | Cleaned |
| `chunks_multi_snowflake_arctic_v2l` | Cleaned (if configured) | Cleaned |
| `chunks_multi_bge_m3` | **Skipped** | Cleaned |
| `chunks_multi_voyage_context_3` | **Skipped** | Cleaned |
| `test_*` | Skipped | Skipped |

## Verification

Dry-run output confirms:
- 4 target collections found and would be cleaned
- 4 test collections correctly skipped
- 3,099 Neo4j nodes would be deleted (including new types)
- SchemaVersion preserved, schema intact (25 constraints, 96 indexes)

## Usage

```bash
# Dry run (recommended first)
PYTHONPATH=. python scripts/cleanup-databases.py --dry-run --skip-redis

# Full cleanup
PYTHONPATH=. NEO4J_PASSWORD=xxx REDIS_PASSWORD=xxx python scripts/cleanup-databases.py
```

## Files Modified

- `scripts/cleanup-databases.py`
