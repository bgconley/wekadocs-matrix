# CircuitBreaker Unification Analysis

> **Phase**: 5c (from CLEANUP-PLAN.md)
> **Goal**: 3 CircuitBreaker implementations → 1 canonical implementation in `shared/resilience/`
> **Date**: 2026-06-03

---

## 1. Implementation Inventory

### Implementation A: `src/shared/resilience/circuit_breaker.py` (331 LOC)
- **Status**: ACTIVE — the canonical "keeper" per CLEANUP-PLAN
- **Classes**: `CircuitBreaker`, `CircuitState(Enum)`
- **Also exports**: `_safe_parse_int()`, `_safe_parse_float()` helpers

### Implementation B: `src/connectors/circuit_breaker.py` (167 LOC)
- **Status**: ACTIVE — slated for deletion per CLEANUP-PLAN
- **Classes**: `CircuitBreaker`, `CircuitBreakerState(str, Enum)`
- **Module scope**: External connector resilience (GitHub, Notion, Confluence)

### Implementation C: `src/providers/embeddings/jina.py` lines 101–148 (~48 LOC)
- **Status**: ACTIVE — inline class, slated for deletion per CLEANUP-PLAN
- **Classes**: `CircuitBreaker` (no enum — uses bare strings)
- **Module scope**: Jina AI API resilience (embeddings + reranker)

---

## 2. Detailed Comparison Table

| Feature | A: `shared/resilience` | B: `connectors` | C: `jina` (inline) |
|---|---|---|---|
| **File** | `src/shared/resilience/circuit_breaker.py` | `src/connectors/circuit_breaker.py` | `src/providers/embeddings/jina.py:101-148` |
| **Class name** | `CircuitBreaker` | `CircuitBreaker` | `CircuitBreaker` |
| **State enum** | `CircuitState(Enum)` | `CircuitBreakerState(str, Enum)` | None (bare strings) |
| **Thread safety** | ✅ `threading.Lock()` | ✅ `threading.Lock()` | ❌ None |
| **LOC (class only)** | ~210 (lines 123–331) | ~150 (lines 27–167) | ~48 (lines 101–148) |
| **Full file LOC** | 331 | 167 | 748 (shared with provider) |
| **Env var config** | ✅ `CIRCUIT_BREAKER_FAILURE_THRESHOLD`, `CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | ❌ | ❌ |
| **Safe env parsers** | ✅ `_safe_parse_int()`, `_safe_parse_float()` | ❌ | ❌ |
| **Structured logging** | ✅ JSON-style `extra={}` | ❌ f-strings | ❌ f-strings |

### Constructor Parameters

| Parameter | A: `shared/resilience` | B: `connectors` | C: `jina` (inline) |
|---|---|---|---|
| Name/identifier | `name: str` (required) | — | — |
| Failure threshold | `failure_threshold: int = 5` (env-configurable) | `failure_threshold: int = 5` | `failure_threshold: int = 5` |
| Recovery timeout | `recovery_timeout: float = 30.0` (env-configurable) | `timeout_seconds: int = 60` | `timeout: int = 300` |
| Half-open max calls | — (implicit: 1) | `half_open_max_calls: int = 3` | — (implicit: 1) |

### Public Methods

| Method | A: `shared/resilience` | B: `connectors` | C: `jina` (inline) |
|---|---|---|---|
| Check if request allowed | `allow_request() -> bool` | `can_proceed() -> bool` | `can_attempt() -> bool` |
| Record success | `record_success() -> None` | `record_success() -> None` | `record_success()` |
| Record failure | `record_failure() -> None` | `record_failure() -> None` | `record_failure()` |
| Manual reset | `reset() -> None` | `reset() -> None` | — |
| State getter | `state` (property) | `get_state() -> CircuitBreakerState` | `self.state` (bare attribute) |
| Failure count getter | `failure_count` (property) | via `get_stats()` | `self.failures` (bare attribute) |
| Quick state checks | `is_open() -> bool`, `is_closed() -> bool` | — | — |
| Statistics | — | `get_stats() -> dict` | — |
| `__repr__` | ✅ | ❌ | ❌ |

### Private Methods

| Method | A: `shared/resilience` | B: `connectors` | C: `jina` (inline) |
|---|---|---|---|
| State transitions | Inline in `allow_request()`/`record_*()` | `_transition_to_closed()`, `_transition_to_open()`, `_transition_to_half_open()` | Inline |
| Timeout check | Inline in `allow_request()` | `_should_attempt_reset() -> bool` | Inline in `can_attempt()` |

---

## 3. State Machine Comparison

All three implementations follow the same **CLOSED → OPEN → HALF_OPEN** state machine:

```
          failure >= threshold
CLOSED ────────────────────────► OPEN
  ▲                                │
  │ success in                     │ timeout elapsed
  │ HALF_OPEN                      ▼
  └──────────────────────── HALF_OPEN
         failure in           │
         HALF_OPEN ───────────► OPEN
```

### Behavioral Differences

| Behavior | A: `shared/resilience` | B: `connectors` | C: `jina` (inline) |
|---|---|---|---|
| HALF_OPEN allows | 1 request (implicit) | Up to `half_open_max_calls` requests | 1 request (implicit) |
| Success in CLOSED | Resets failure count | Resets failure count (if > 0) | Resets failure count |
| Failure tracking | Consecutive count | Consecutive count | Consecutive count |
| Timeout unit | `float` (seconds) | `int` (seconds) | `int` (seconds) |
| Default timeout | 30s | 60s | 300s |

---

## 4. Caller Map

### Implementation A (`shared/resilience`) — 4 callers

| File | How it uses CircuitBreaker |
|---|---|
| `src/providers/rerank/local_reranker_service.py:136,384,510,535` | `CircuitBreaker(name=...)`, `allow_request()`, `record_failure()`, `record_success()`, `state.value` |
| `src/shared/resilience/__init__.py` | Re-exports `CircuitBreaker`, `CircuitState` |
| `tests/unit/test_phase1_reranker_batching.py` | `CircuitBreaker(name="test")`, assigns to `provider._circuit_breaker` |
| `tests/integration/test_phase1_entity_edges.py` | Same pattern as above |

### Implementation B (`connectors`) — 5 callers

| File | How it uses CircuitBreaker |
|---|---|
| `src/connectors/__init__.py` | Re-exports `CircuitBreaker`, `CircuitBreakerState` |
| `src/connectors/base.py:72,76,115,142,162,240` | `can_proceed()`, `record_success()`, `record_failure()`, `get_state().value` |
| `src/connectors/manager.py:18,88` | Creates `CircuitBreaker(failure_threshold=..., timeout_seconds=...)` |
| `src/connectors/github.py:31-32` | Inherits circuit breaker from `BaseConnector` |
| `tests/p5_t1_test.py:18,52+` | Tests state transitions via `get_state()`, `can_proceed()`, etc. |

### Implementation C (`jina` inline) — 3 callers

| File | How it uses CircuitBreaker |
|---|---|
| `src/providers/embeddings/jina.py:238,301,338,351` | `CircuitBreaker(failure_threshold=5, timeout=300)`, `can_attempt()`, `record_success()`, `record_failure()` |
| `src/providers/rerank/jina.py:86,88,145,165,179` | `from src.providers.embeddings.jina import CircuitBreaker`, `CircuitBreaker(failure_threshold=5, timeout=300)`, `can_attempt()`, `record_success()`, `record_failure()` |
| `tests/test_jina_adaptive_batching.py:20,319+` | Tests `CircuitBreaker(failure_threshold=..., timeout=...)` |

---

## 5. Unique Features Per Implementation

### A: `shared/resilience` (KEEP — canonical)
- ✅ `name` parameter for structured logging/metrics
- ✅ Environment variable configuration with safe parsing
- ✅ Thread-safe with `threading.Lock()`
- ✅ Rich structured logging (`extra={}` dicts)
- ✅ `is_open()` / `is_closed()` convenience methods
- ✅ `__repr__` for debugging
- ❌ No `half_open_max_calls` (hardcoded to 1)
- ❌ No `get_stats()` method

### B: `connectors` (DELETE — migrate callers)
- ✅ `half_open_max_calls` — allows multiple test requests in HALF_OPEN
- ✅ `get_stats()` — returns state dict for monitoring/diagnostics
- ✅ Private transition methods — cleaner state machine code
- ✅ `CircuitBreakerState(str, Enum)` — backward-compatible string comparison
- ❌ No `name` parameter
- ❌ No env var config
- ❌ f-string logging (not structured)

### C: `jina` inline (DELETE — migrate callers)
- ✅ Simplest/smallest (~48 LOC) — minimal footprint
- ❌ No thread safety
- ❌ No enum (bare strings)
- ❌ No `reset()` method
- ❌ No structured logging
- ❌ Bundled in provider file (wrong location)

---

## 6. Consolidation Recommendation

### Approach: Enhance A → Replace B and C

The CLEANUP-PLAN already specifies the right approach:
> Keep `src/shared/resilience/circuit_breaker.py` (the thread-safe one). Delete `connectors/circuit_breaker.py` and `providers/embeddings/jina.py`'s inline `CircuitBreaker`. Wire `connectors` and `jina.py` to import from `shared/resilience`.

### Required Enhancements to Implementation A

Before B and C can be replaced, Implementation A needs **two features** from B:

| Enhancement | From | Effort | Notes |
|---|---|---|---|
| `half_open_max_calls` parameter | B | ~15 LOC | Add param to constructor + counter in `allow_request()` |
| `get_stats() -> dict` method | B | ~10 LOC | Trivial addition |

### API Adapter Layer (Method Name Mapping)

Callers use different method names. Two migration strategies:

**Option 1: Add aliases to A (simplest, zero-risk)**
```python
class CircuitBreaker:
    # Canonical names
    def allow_request(self) -> bool: ...

    # Aliases for backward compat
    can_proceed = allow_request  # For connectors callers
    can_attempt = allow_request  # For jina callers
```

**Option 2: Update all callers (cleaner, more PR churn)**
- Rename `can_proceed()` → `allow_request()` in `connectors/base.py`
- Rename `can_attempt()` → `allow_request()` in `jina.py` (embedding + rerank)
- Rename `get_state()` → `state` property in `connectors/base.py`

**Recommended**: Option 2 (update callers). The caller count is low (12 total), and the rename is mechanical.

### Migration Steps

1. **Enhance A** with `half_open_max_calls` and `get_stats()`
2. **Migrate C** (jina inline — simplest):
   - Remove `CircuitBreaker` class from `src/providers/embeddings/jina.py`
   - Add `from src.shared.resilience import CircuitBreaker` in both jina files
   - Update method calls: `can_attempt()` → `allow_request()`
   - Update test imports in `tests/test_jina_adaptive_batching.py`
   - Pass `name="jina"` to constructor
3. **Migrate B** (connectors):
   - Replace `src/connectors/circuit_breaker.py` with thin re-export:
     ```python
     from src.shared.resilience.circuit_breaker import CircuitBreaker
     # Alias for backward compat
     CircuitBreakerState = CircuitState  # or update callers
     ```
   - OR delete file and update all 5 import sites
   - Update `connectors/base.py`: `can_proceed()` → `allow_request()`, `get_state()` → `state`
   - Update `connectors/manager.py`: `timeout_seconds` → `recovery_timeout`
   - Update `tests/p5_t1_test.py` to use new API

---

## 7. LOC Impact Estimate

### Lines Eliminated

| Source | Lines Removed | Notes |
|---|---|---|
| `src/connectors/circuit_breaker.py` (full file) | 167 | Replaced by thin re-export or deleted |
| `src/providers/embeddings/jina.py` CircuitBreaker class | 48 | Removed from file |
| **Total removed** | **~215** | |

### Lines Added

| Target | Lines Added | Notes |
|---|---|---|
| `shared/resilience/circuit_breaker.py` enhancements | ~25 | `half_open_max_calls` + `get_stats()` |
| Migration shim in `connectors/__init__.py` | ~5 | Thin re-export (if keeping compat) |
| Caller updates (method renames) | ~15 | Across 8 files |
| Test updates | ~15 | Updated imports and method calls |
| **Total added** | **~60** | |

### Net Savings: ~155 LOC

This aligns with the CLEANUP-PLAN's broader estimate of −1,500 to −2,000 LOC for the full Phase 5 (which includes Chonkie adapters and HTTP clients).

---

## 8. Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| **API incompatibility** | 🟡 Medium | Different method names (`can_proceed` vs `allow_request`). Low caller count (12 files) makes mechanical rename feasible. |
| **`half_open_max_calls` behavior change** | 🟡 Medium | Connectors currently allow 3 test calls in HALF_OPEN; shared allows 1. Must add this parameter to shared implementation to preserve connector behavior. |
| **Thread safety improvement for Jina** | 🟢 Low | Jina's inline version has NO thread safety. Migrating to shared is a strict improvement. No behavior risk. |
| **State enum rename** | 🟡 Medium | `CircuitBreakerState` (str, Enum) → `CircuitState` (Enum). Callers comparing with `.value` work fine. The `str` mixin in B was for serialization compat. |
| **Default timeout mismatch** | 🟡 Medium | Jina uses 300s, connectors use 60s, shared uses 30s. Must pass explicit values at construction — callers already do this. |
| **Test breakage** | 🟡 Medium | 3 test files import from different paths. All must be updated. `tests/p5_t1_test.py` is the most complex (tests state machine transitions). |
| **Import cycle risk** | 🟢 Low | `shared/resilience` has no upstream dependencies — safe for all modules to import. |

### Overall Risk: 🟢 LOW

The consolidation is straightforward because:
1. All three implement the same core pattern (CLOSED → OPEN → HALF_OPEN)
2. Caller count is small (12 files total)
3. Implementation A is the strict superset in features (thread safety, env config, structured logging)
4. Only 2 features from B need porting (`half_open_max_calls`, `get_stats()`)
5. Implementation C is strictly inferior (no thread safety) — migration is a pure upgrade

---

## 9. Summary

```
Current: 3 CircuitBreaker classes (546 LOC of circuit breaker code across 3 files)
Target:  1 CircuitBreaker class (~356 LOC) + thin re-exports

Net savings: ~155 LOC
Caller updates: 12 files
New tests needed: 0 (existing tests cover all 3 implementations)
Unique features to preserve: 2 (half_open_max_calls, get_stats)
Estimated effort: ~2 hours of focused work
```
