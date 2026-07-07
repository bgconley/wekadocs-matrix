"""Guard: no orphaned shadow-module copies re-appear, and nothing imports them."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The real contract: no active surface imports OR otherwise depends on the shadow
# copies, including non-Python references such as a deploy script that copies them.
FORBIDDEN = re.compile(
    r"(from|import)\s+patched\b|"
    r"atomic_patched|"
    r"chonkie_adapter_patched|"
    r"semantic_chunker_patched|"
    r"(^|[^A-Za-z0-9_])patched/"
)
SCAN_DIRS = ["src", "tests", "scripts", "deploy"]
SCAN_ROOT_FILES = ["Makefile", "docker-compose.yml", "pyproject.toml"]
SCAN_SUFFIXES = {
    ".py",
    ".sh",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".txt",
    ".env",
    "",
}


def _iter_active_files():
    guard = Path(__file__).resolve()
    for directory in SCAN_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if (
                path.is_file()
                and path.suffix in SCAN_SUFFIXES
                and path.resolve() != guard
            ):
                yield path
    for name in SCAN_ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            yield path


def test_no_patched_shadow_directory():
    assert not (
        ROOT / "patched"
    ).exists(), "the orphaned patched/ shadow dir must stay deleted"


def test_nothing_references_shadow_modules():
    offenders = [
        str(path.relative_to(ROOT))
        for path in _iter_active_files()
        if FORBIDDEN.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == [], f"legacy shadow-module references found: {offenders}"
