"""
Shared utilities for vector operations.

This module provides common utilities used across ingestion modules
to ensure consistent vector handling and validation.
"""

from typing import Optional


def vector_expected_dim(vector: object) -> Optional[int]:
    """
    Determine the expected dimensionality of a vector.

    This function handles various vector formats:
    - None -> None (no dimension)
    - Sparse vectors (with indices/values attributes) -> None (variable length)
    - List of floats -> length of list
    - List of lists (multi-vectors like ColBERT) -> length of inner list
    - Other sized objects -> attempt to get length

    Args:
        vector: A vector object in various formats

    Returns:
        The expected dimension as an int, or None if not applicable
        (e.g., for sparse vectors or None input)

    Examples:
        >>> vector_expected_dim([1.0, 2.0, 3.0])
        3
        >>> vector_expected_dim([[1.0, 2.0], [3.0, 4.0]])  # ColBERT multi-vector
        2
        >>> vector_expected_dim(None)
        None
    """
    if vector is None:
        return None

    # Sparse vectors have indices and values - no fixed dimension
    if hasattr(vector, "indices") and hasattr(vector, "values"):
        return None

    if isinstance(vector, list):
        if not vector:
            return None
        first = vector[0]
        # Handle multi-vectors (list of lists, like ColBERT)
        if isinstance(first, list):
            return len(first)
        # Regular dense vector
        return len(vector)

    # Handle numpy arrays and similar sized objects
    if hasattr(vector, "__len__") and not isinstance(vector, (str, bytes)):
        try:
            return len(vector)
        except TypeError:
            return None

    return None
