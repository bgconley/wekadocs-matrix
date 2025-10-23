# Implements Phase 1, Task 1.2 (MCP server foundation)
# See: /docs/spec.md §7 (Observability)
# Structured logging with correlation IDs

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog

# Context variable for correlation ID
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> str:
    """Get or create correlation ID for the current context"""
    corr_id = correlation_id_ctx.get()
    if corr_id is None:
        corr_id = str(uuid.uuid4())
        correlation_id_ctx.set(corr_id)
    return corr_id


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for the current context"""
    correlation_id_ctx.set(corr_id)


def add_correlation_id(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add correlation ID to log event"""
    corr_id = correlation_id_ctx.get()
    if corr_id:
        event_dict["correlation_id"] = corr_id
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup structured logging with JSON output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Note:
        When WEKADOCS_STDIO_MODE=1, skips stdout logging configuration.
        This allows STDIO MCP server to maintain clean stdout for JSON-RPC.
    """
    # Skip stdout logging config when running in STDIO mode
    # STDIO server handles its own stderr-only logging in bootstrap
    if not os.environ.get("WEKADOCS_STDIO_MODE"):
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level.upper()),
        )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            add_correlation_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerAdapter:
    """Adapter for structured logging with common fields"""

    def __init__(self, logger: structlog.BoundLogger, **default_fields: Any):
        self.logger = logger
        self.default_fields = default_fields

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Internal log method with default fields"""
        fields = {**self.default_fields, **kwargs}
        getattr(self.logger, level)(event, **fields)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("error", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log("critical", event, **kwargs)

    def bind(self, **new_fields: Any) -> "LoggerAdapter":
        """Create a new adapter with additional bound fields"""
        fields = {**self.default_fields, **new_fields}
        return LoggerAdapter(self.logger, **fields)
