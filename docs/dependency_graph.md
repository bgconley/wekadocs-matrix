# Codebase Dependency Graph & Cleanup Analysis

**Date:** December 3, 2025
**Scope:** Retrieval, Ranking, and Ingestion Pipelines
**Status:** Analysis Complete

---

## 1. Executive Summary

The codebase currently contains two parallel implementations for core functionality: a **Legacy Path** (`hybrid_search.py`) and a **Current Path** (`hybrid_retrieval.py` and `atomic.py`).

*   **Retrieval:** The system has fully migrated to `HybridRetriever`. The old `HybridSearchEngine` and its `Ranker` are orphaned and safe to delete.
*   **Ingestion:** The system has migrated to a Transactional/Saga pattern via `atomic.py`. The old `GraphBuilder` is still *active*, but it has been demoted to a helper class used by `atomic.py`. Direct usage of `GraphBuilder` (bypassing `atomic.py`) is considered legacy/unsafe.

---

## 2. Retrieval Pipeline

### A. Current Path (Active)
This is the live path used by the MCP server (`stdio_server.py`) and the Query Service.

*   **Entry Point:** `src/mcp_server/stdio_server.py`
    *   `↓` (instantiates)
*   **Orchestrator:** `src/query/hybrid_retrieval.py` -> Class: `HybridRetriever`
    *   **Logic:** Handles Fusion (RRF/Weighted), Expansion, and Ranking internally.
    *   **Dependencies:**
        *   `class QdrantMultiVectorRetriever` (Internal to file, handles Dense/Sparse/ColBERT)
        *   `class BM25Retriever` (Internal to file, handles Neo4j Fulltext)
        *   `src/providers/rerank/local_bge_service.py` (`BGERerankerServiceProvider`)

### B. Legacy Path (Dead Code)
These files are no longer instantiated by the main application but may be referenced by old tests.

*   **Orchestrator:** `src/query/hybrid_search.py` -> Class: `HybridSearchEngine`
    *   **Status:** **ORPHANED**. No callers in `src/mcp_server`.
    *   **Dependencies:**
        *   `class QdrantVectorStore` (Legacy wrapper)
        *   `class Neo4jVectorStore` (Legacy wrapper)
*   **Ranking:** `src/query/ranking.py` -> Class: `Ranker`
    *   **Status:** **ORPHANED**. Only used by `HybridSearchEngine`. `HybridRetriever` implements its own feature extraction and ranking logic.

---

## 3. Ingestion Pipeline

### A. Current Path (Active)
This path implements the "Saga Pattern" to ensure atomicity between Neo4j and Qdrant.

*   **Entry Point:** `src/ingestion/atomic.py`
    *   **Role:** Transaction Coordinator / Saga Manager.
    *   **Logic:** Prepare -> Write Neo4j -> Write Qdrant -> Commit/Rollback.
    *   `↓` (calls)
*   **Worker:** `src/ingestion/build_graph.py` -> Class: `GraphBuilder`
    *   **Role:** Implementation detail for Neo4j Cypher operations.
    *   **Status:** **ACTIVE**. Essential library code.

### B. Legacy Usage (Unsafe)
*   **Direct Call:** Calling `GraphBuilder.upsert_document()` directly.
    *   **Risk:** Writes to Neo4j without coordinating Qdrant writes or handling rollback.
    *   **Remediation:** All ingestion scripts should be updated to use `atomic.py`.

---

## 4. Component Inventory & Status

| Component / File | Status | Notes |
| :--- | :--- | :--- |
| `src/mcp_server/stdio_server.py` | **Active** | Main Entry Point |
| `src/mcp_server/query_service.py` | **Active** | Service Layer |
| `src/query/hybrid_retrieval.py` | **Active** | **Primary Retrieval Engine**. Contains `HybridRetriever`. |
| `src/ingestion/atomic.py` | **Active** | **Primary Ingestion Engine**. Coordinates transactions. |
| `src/ingestion/build_graph.py` | **Active** | **Shared Lib**. Used by `atomic.py` for graph ops. |
| `src/providers/rerank/*.py` | **Active** | Reranking providers used by `HybridRetriever`. |
| `src/query/hybrid_search.py` | 🔴 **Legacy** | Old engine. **Safe to Delete** after test cleanup. |
| `src/query/ranking.py` | 🔴 **Legacy** | Old ranker. **Safe to Delete** after test cleanup. |
| `tests/p2_t3_test.py` | ⚠️ **Legacy** | Tests the *Old Engine*. Needs migration or deletion. |
| `tests/test_integration_prephase7.py` | ⚠️ **Legacy** | Tests the *Old Ranker*. Needs migration. |

---

## 5. Cleanup Recommendations

1.  **Refactor Tests:** `tests/p2_t3_test.py` and `tests/test_integration_prephase7.py` are keeping the legacy code alive. These tests should be updated to test `HybridRetriever` instead of `HybridSearchEngine`.
2.  **Delete Legacy Files:** Once tests are migrated, `src/query/hybrid_search.py` and `src/query/ranking.py` can be deleted to reduce cognitive load and prevent accidental usage.
3.  **Audit Ingestion Scripts:** Ensure no scripts are importing `GraphBuilder` directly for execution. All ingestion entry points (CLI, Workers) must route through `atomic.py`.
