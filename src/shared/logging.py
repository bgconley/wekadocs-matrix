"""
Compatibility logging module.

Bridges older imports (src.shared.logging) to the current observability system
so modules like chunk_assembler can call get_logger without import errors.
"""

from src.shared.observability import get_logger  # re-export for compatibility

__all__ = ["get_logger"]
