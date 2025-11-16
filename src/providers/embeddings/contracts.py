from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SparseEmbedding:
    """Sparse/token-weight representation produced by models like BGE-M3."""

    indices: List[int]
    values: List[float]

    def is_empty(self) -> bool:
        return not self.indices or not self.values


@dataclass(frozen=True)
class MultiVectorEmbedding:
    """ColBERT-style per-token vectors."""

    vectors: List[List[float]]

    def is_empty(self) -> bool:
        return not self.vectors


@dataclass(frozen=True)
class DocumentEmbeddingBundle:
    """Full embedding bundle for a document/section text."""

    dense: List[float]
    sparse: Optional[SparseEmbedding] = None
    multivector: Optional[MultiVectorEmbedding] = None


@dataclass(frozen=True)
class QueryEmbeddingBundle:
    """Full embedding bundle for a query string."""

    dense: List[float]
    sparse: Optional[SparseEmbedding] = None
    multivector: Optional[MultiVectorEmbedding] = None
