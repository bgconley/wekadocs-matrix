"""
Parser module for WekaDocs ingestion pipeline.

This module provides a unified interface for document parsing with
support for multiple parser backends and feature-flag-controlled routing.

Parser Engines:
    - "legacy": Original markdown + BeautifulSoup parser
    - "markdown-it-py": New AST-based parser with source line mapping

Usage:
    from src.ingestion.parsers import parse_markdown

    # Automatically uses the configured parser engine
    result = parse_markdown(source_uri, raw_text)
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

from src.ingestion.parsers.shadow_comparison import ShadowModeError

logger = structlog.get_logger(__name__)

# Parser engine constants
ENGINE_LEGACY = "legacy"
ENGINE_MARKDOWN_IT_PY = "markdown-it-py"
DEFAULT_ENGINE = ENGINE_MARKDOWN_IT_PY


def get_parser_engine() -> str:
    """
    Get the configured parser engine from settings.

    Returns:
        Parser engine name: "legacy" or "markdown-it-py"
    """
    try:
        from src.shared.config import get_config

        config = get_config()
        engine = (
            config.get("ingestion", {}).get("parser", {}).get("engine", DEFAULT_ENGINE)
        )
        return engine
    except Exception as e:
        logger.warning(
            "Failed to get parser config, using default",
            error=str(e),
            default=DEFAULT_ENGINE,
        )
        return DEFAULT_ENGINE


def get_shadow_mode() -> bool:
    """
    Check if shadow mode is enabled (run both parsers, compare results).

    Returns:
        True if shadow mode is enabled
    """
    try:
        from src.shared.config import get_config

        config = get_config()
        return config.get("ingestion", {}).get("parser", {}).get("shadow_mode", False)
    except Exception:
        return False


def get_fail_on_mismatch() -> bool:
    """
    Check if fail_on_mismatch is enabled (raise error when parsers differ).

    Only meaningful when shadow_mode is also enabled. When True, the pipeline
    will raise ShadowModeError if the parsers produce different results.

    Use cases:
        - CI/CD validation: Catch regressions before merge
        - Migration testing: Ensure equivalence on test corpus

    Returns:
        True if fail_on_mismatch is enabled
    """
    try:
        from src.shared.config import get_config

        config = get_config()
        return (
            config.get("ingestion", {}).get("parser", {}).get("fail_on_mismatch", False)
        )
    except Exception:
        return False


def parse_markdown(source_uri: str, raw_text: str) -> Dict[str, Any]:
    """
    Parse Markdown document using the configured parser engine.

    This is the main entry point for markdown parsing. It routes to the
    appropriate parser backend based on the configuration:
    - "legacy": Uses markdown + BeautifulSoup (original implementation)
    - "markdown-it-py": Uses AST-based parsing with source line mapping

    When shadow_mode is enabled, both parsers run and differences are logged.

    Args:
        source_uri: Source URI/path of the document
        raw_text: Raw markdown text

    Returns:
        Dict with 'Document' and 'Sections' keys

    Raises:
        ImportError: If the configured parser is not available
    """
    engine = get_parser_engine()
    shadow_mode = get_shadow_mode()

    if shadow_mode:
        return _parse_with_shadow_comparison(source_uri, raw_text, engine)

    return _parse_with_engine(source_uri, raw_text, engine)


def _parse_with_engine(source_uri: str, raw_text: str, engine: str) -> Dict[str, Any]:
    """
    Parse using a specific engine.

    Args:
        source_uri: Source URI/path of the document
        raw_text: Raw markdown text
        engine: Parser engine name

    Returns:
        Parsed document dict
    """
    if engine == ENGINE_MARKDOWN_IT_PY:
        try:
            from src.ingestion.parsers.markdown_it_parser import (
                parse_markdown as parse_mit,
            )

            return parse_mit(source_uri, raw_text)
        except ImportError as e:
            logger.warning(
                "markdown-it-py parser not available, falling back to legacy",
                error=str(e),
            )
            engine = ENGINE_LEGACY

    if engine == ENGINE_LEGACY:
        from src.ingestion.parsers.markdown import parse_markdown as parse_legacy

        return parse_legacy(source_uri, raw_text)

    # Unknown engine - fall back to legacy with warning
    logger.warning(
        "Unknown parser engine, falling back to legacy",
        engine=engine,
        valid_engines=[ENGINE_LEGACY, ENGINE_MARKDOWN_IT_PY],
    )
    from src.ingestion.parsers.markdown import parse_markdown as parse_legacy

    return parse_legacy(source_uri, raw_text)


def _parse_with_shadow_comparison(
    source_uri: str, raw_text: str, primary_engine: str
) -> Dict[str, Any]:
    """
    Parse with both engines and log differences (shadow mode).

    The primary engine's result is returned, but both are run for comparison.
    Differences are logged for analysis. If fail_on_mismatch is enabled,
    raises ShadowModeError when parsers produce different results.

    Args:
        source_uri: Source URI/path of the document
        raw_text: Raw markdown text
        primary_engine: The engine whose result should be returned

    Returns:
        Result from primary engine

    Raises:
        ShadowModeError: If fail_on_mismatch is enabled and parsers differ
    """
    from src.ingestion.parsers.markdown import parse_markdown as parse_legacy
    from src.ingestion.parsers.shadow_comparison import (
        ShadowModeError,
        compare_parser_results,
        log_comparison_result,
    )

    # Try to import markdown-it-py parser
    try:
        from src.ingestion.parsers.markdown_it_parser import parse_markdown as parse_mit

        mit_available = True
    except ImportError:
        mit_available = False
        parse_mit = None

    # Run legacy parser
    legacy_result = parse_legacy(source_uri, raw_text)

    # Run markdown-it-py parser if available
    if mit_available and parse_mit is not None:
        mit_result = parse_mit(source_uri, raw_text)

        # Compare results using structured comparison
        comparison = compare_parser_results(source_uri, legacy_result, mit_result)

        # Log comparison result
        log_comparison_result(comparison)

        # Check fail_on_mismatch
        if comparison.has_differences and get_fail_on_mismatch():
            raise ShadowModeError(comparison)

        # Return based on primary engine
        if primary_engine == ENGINE_MARKDOWN_IT_PY:
            return mit_result
    else:
        logger.warning(
            "Shadow mode enabled but markdown-it-py not available",
            source_uri=source_uri,
        )

    return legacy_result


# Convenience re-exports for direct imports
# Note: ShadowModeError is imported at the top of the module

__all__ = [
    "parse_markdown",
    "get_parser_engine",
    "get_shadow_mode",
    "get_fail_on_mismatch",
    "ShadowModeError",
    "ENGINE_LEGACY",
    "ENGINE_MARKDOWN_IT_PY",
]
