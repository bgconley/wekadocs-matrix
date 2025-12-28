"""
STDIO MCP Server for Claude Desktop.
Runs the shared low-level MCP server over STDIO transport.
"""

from __future__ import annotations

import builtins
import logging

# --- STDIO-SAFE BOOTSTRAP: Must be at the very top ---
import os
import sys
import warnings

# Unbuffered, UTF-8 (avoids partial writes / encoding surprises)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# 1) Route all Python logging to STDERR (never STDOUT)
root = logging.getLogger()
# Remove any handlers installed during module import
for h in list(root.handlers):
    root.removeHandler(h)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(
    logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
)
root.addHandler(stderr_handler)
root.setLevel(logging.INFO)

# Silence noisy libraries that might spam
for noisy in ("httpx", "urllib3", "botocore", "boto3", "opentelemetry", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# 2) Redirect accidental print() calls to STDERR
_builtin_print = builtins.print


def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    return _builtin_print(*args, **kwargs)


builtins.print = _stderr_print

# 3) If you use Rich or Loguru, force them to STDERR (examples below):
try:
    from rich.console import Console
    from rich.logging import RichHandler

    root.handlers.clear()
    root.addHandler(
        RichHandler(console=Console(stderr=True), markup=False, show_path=False)
    )
except Exception:
    pass

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=False, diagnose=False)
except Exception:
    pass

# Optional: demote Deprecation/Runtime warnings to STDERR (default)
warnings.simplefilter("default")

# 4) Configure structlog to use STDERR (our app uses structlog via get_logger)
try:
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),  # Human-readable for stderr
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
except Exception:
    pass
# --- END STDIO-SAFE BOOTSTRAP ---

import anyio  # noqa: E402
from mcp.server.lowlevel.server import NotificationOptions  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402

from src.mcp_server.mcp_app import build_mcp_server  # noqa: E402
from src.shared.observability import get_logger  # noqa: E402

logger = get_logger(__name__)


async def run_stdio_server() -> None:
    server = build_mcp_server()
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions()
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def main() -> None:
    logger.info("Starting wekadocs MCP server with STDIO transport")
    anyio.run(run_stdio_server)


if __name__ == "__main__":
    main()
