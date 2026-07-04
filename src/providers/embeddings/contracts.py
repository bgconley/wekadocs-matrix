# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SparseEmbedding:
    """Sparse/token-weight representation produced by models like BGE-M3."""

    indices: List[int]
    values: List[float]


@dataclass(frozen=True)
class MultiVectorEmbedding:
    """ColBERT-style per-token vectors."""

    vectors: List[List[float]]


@dataclass(frozen=True)
class QueryEmbeddingBundle:
    """Full embedding bundle for a query string."""

    dense: List[float]
    sparse: Optional[SparseEmbedding] = None
    multivector: Optional[MultiVectorEmbedding] = None
