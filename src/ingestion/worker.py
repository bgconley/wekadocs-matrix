# Ingestion worker stub for Phase 1
# Full implementation in Phase 3
# See: /docs/implementation-plan.md → Phase 3

import asyncio
import signal
import sys

from src.shared import init_config
from src.shared.observability import get_logger, setup_logging

# Initialize config and logging
config, settings = init_config()
setup_logging(config.app.log_level)
logger = get_logger(__name__)

# Shutdown flag
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received", signal=signum)
    shutdown_event.set()


async def main():
    """Main worker loop - stub for Phase 1"""
    logger.info("Ingestion worker starting (Phase 1 stub)")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Ingestion worker ready (waiting for Phase 3 implementation)")

    # Wait for shutdown signal
    await shutdown_event.wait()

    logger.info("Ingestion worker shutting down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ingestion worker stopped by user")
    except Exception as e:
        logger.error("Ingestion worker failed", error=str(e))
        sys.exit(1)
