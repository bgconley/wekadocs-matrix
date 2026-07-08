"""Structured logging for docpipe.

Deliberately dependency-free (stdlib ``logging`` only). Emits either
human-readable key=value lines (default) or single-line JSON (``DOCPIPE_LOG_JSON=1``)
so a run can be grepped or piped into a log processor. All pipeline stages log
through :func:`get_logger` so the end-of-run report and live progress share one
stream.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_CONFIGURED = False


class _KeyValueFormatter(logging.Formatter):
    """Render ``logger.info("msg", extra={"fields": {...}})`` as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        base = f"{ts} {record.levelname:<5} {record.name} {record.getMessage()}"
        fields = getattr(record, "fields", None)
        if fields:
            kv = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
            return f"{base} {kv}"
        return base


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        return json.dumps(payload, default=str)


def _fmt(value: Any) -> str:
    raw = str(value)
    s = raw.replace("\\", "\\\\").replace("\n", "\\n")
    return f'"{s}"' if any(c.isspace() for c in raw) or "=" in raw else s


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root ``docpipe`` logger once. Idempotent."""

    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    if os.environ.get("DOCPIPE_LOG_JSON") == "1":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_KeyValueFormatter())
    root = logging.getLogger("docpipe")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``docpipe`` namespace."""

    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"docpipe.{name}")


def log(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Convenience: ``log(lg, logging.INFO, "converted", page=3, endpoint="bb")``."""

    logger.log(level, msg, extra={"fields": fields})
