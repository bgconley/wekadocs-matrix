# Phase 4-5: Atomic Ingestion Observability & Feature Parity

**Decision Date:** 2025-11-28
**Status:** Implemented
**Branch:** `dense-graph-enhance`
**Affected Files:** `src/ingestion/atomic.py`, `src/shared/config.py`, `src/shared/observability/metrics.py`

---

## Executive Summary

Phase 4-5 adds production observability and feature parity to the Atomic Ingestion Coordinator, aligning it with GraphBuilder capabilities while providing Prometheus metrics for SLO monitoring.

---

## Phase 4: Observability

### 4.1 Dimension Validation via `upsert_validated()`

**Problem:** The atomic path used raw `qdrant_client.upsert()` with no dimension validation, allowing schema mismatches to propagate silently.

**Solution:** Switch to `upsert_validated()` which:
- Validates all vector dimensions BEFORE write
- Raises `ValueError` on mismatch (fail-fast)
- Records Prometheus metrics automatically

**Implementation:**
```python
# atomic.py:1677-1683
self.qdrant_client.upsert_validated(
    collection_name=collection,
    points=points,
    expected_dim=expected_dim,  # Dict[str, int]: {"content": 1024, "title": 1024}
    wait=True,
)
```

### 4.2 Expected Dimensions Parameter

**Problem:** Multi-vector Qdrant collections require dimension validation per named vector.

**Solution:** Wire `expected_dim: Dict[str, int]` through the retry wrapper:
```python
expected_dim = {
    "content": builder.embedding_settings.dimensions,
    "title": builder.embedding_settings.dimensions,
}
if self.include_entity_vector:
    expected_dim["entity"] = builder.embedding_settings.dimensions
```

### 4.3 Delete Telemetry

**Problem:** Compensation phase deletes had no observability, making SLO monitoring incomplete.

**Solution:** Added Prometheus metrics to `_qdrant_delete_with_retry`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `qdrant_delete_total` | Counter | `collection_name`, `status` | Delete operation count |
| `qdrant_operation_latency_ms` | Histogram | `collection_name`, `operation="delete"` | Delete latency |

---

## Phase 5: Advanced Features

### 5.1 Entity Vector Support

**Problem:** GraphBuilder includes an "entity" vector slot; atomic path did not.

**Solution:** Added optional entity vector (copy of content embedding):

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QDRANT_INCLUDE_ENTITY_VECTOR` | `true` | Include entity vector for GraphBuilder schema parity |

**Storage Impact:**
- ON (default): 3 dense vectors per point (content, title, entity)
- OFF: 2 dense vectors per point (~33% storage savings)

**Implementation:**
```python
# atomic.py:1616-1619
if self.include_entity_vector:
    vectors["entity"] = section_embeddings["content"]
```

### 5.2 Unified Strict Mode Configuration

**Problem:** Strict mode was a constructor parameter in atomic path but config-driven in GraphBuilder.

**Solution:** Added `VALIDATION_STRICT_MODE` to unified config:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VALIDATION_STRICT_MODE` | `false` | Fail on validation warnings (not just errors) |

**Backward Compatibility:**
- Explicit `strict_mode=True/False` in constructor overrides config
- `strict_mode=None` (default) reads from config

---

## Strict Mode Reference

All three strict mode flags for reference:

| Flag | Location | Default | Controls |
|------|----------|---------|----------|
| `VALIDATION_STRICT_MODE` | Settings | `false` | Pre-commit validation warnings fail ingestion |
| `EMBEDDING_STRICT_MODE` | Settings | `true` | Embedding profile drift blocks ingestion |
| `sparse_strict_mode` | QdrantVectorConfig | `false` | Sparse embedding failures block ingestion |

---

## Prometheus Metrics Summary

### Upsert Metrics (via `upsert_validated`)
- `qdrant_upsert_total{collection_name, status}` - Counter
- `qdrant_operation_latency_ms{collection_name, operation="upsert"}` - Histogram

### Delete Metrics (Phase 4.3)
- `qdrant_delete_total{collection_name, status}` - Counter
- `qdrant_operation_latency_ms{collection_name, operation="delete"}` - Histogram

### Status Labels
- `success` - Operation completed successfully
- `error` - Operation failed (after retries)

---

## Configuration Summary

```bash
# Entity Vector (default ON)
export QDRANT_INCLUDE_ENTITY_VECTOR=true   # Include entity vector
export QDRANT_INCLUDE_ENTITY_VECTOR=false  # Disable for 33% storage savings

# Strict Mode (default OFF)
export VALIDATION_STRICT_MODE=false  # Warn on validation issues (default)
export VALIDATION_STRICT_MODE=true   # Fail on validation warnings

# Embedding Strict Mode (default ON)
export EMBEDDING_STRICT_MODE=true    # Block on profile drift (default)
export EMBEDDING_STRICT_MODE=false   # Warn only on profile drift
```

---

## Testing

All changes validated with existing test suite:
```
============================== 43 passed in 3.35s ==============================
```

---

## Migration Notes

### For Existing Deployments
- No action required - defaults maintain backward compatibility
- Entity vectors now included by default (was not present before)
- Dimension validation now active (may surface pre-existing schema issues)

### For New Deployments
- Review `QDRANT_INCLUDE_ENTITY_VECTOR` setting based on storage budget
- Consider enabling `VALIDATION_STRICT_MODE=true` for stricter data quality

---

*Document generated: 2025-11-28*
