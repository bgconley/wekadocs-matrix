"""Static guard for runtime-connected legacy vendor residue."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_SURFACE_PATHS = [
    "src",
    "config",
    "data/ingest/nutanix",
    "docker",
    "deploy",
    "scripts",
    "services",
    "tools",
    ".github",
    "docker-compose.yml",
    "Makefile",
    ".env.example",
    "tests",
]

EXCLUDED_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    "reports/",
    "docs/archive/",
    "docs/superpowers/findings/",
    "docs/superpowers/plans/",
    "docs/cdx-outputs/",
    "tests/e2e_v22_prod/artifacts/",
    "repo-analysis-artifacts/",
    "claude-raw/",
)

SKIP_DIR_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
SKIP_SUFFIXES = {".bak", ".pyc", ".pyo", ".so", ".dylib"}
LEGACY_FRAGMENT = "we" + "ka"
LEGACY_PATTERN = re.compile(re.escape(LEGACY_FRAGMENT), re.IGNORECASE)


def _relative_posix(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _should_scan(path: Path) -> bool:
    rel = _relative_posix(path)
    if rel.startswith("tests/eval/") and rel != "tests/eval/queries.yaml":
        return False
    if SKIP_DIR_PARTS & set(Path(rel).parts):
        return False
    if path.suffix in SKIP_SUFFIXES:
        return False
    return not any(
        rel == prefix.rstrip("/") or rel.startswith(prefix)
        for prefix in EXCLUDED_PREFIXES
    )


def _iter_active_files() -> list[Path]:
    files: list[Path] = []
    for rel_path in ACTIVE_SURFACE_PATHS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        if path.is_file():
            if _should_scan(path):
                files.append(path)
            continue
        files.extend(
            child
            for child in path.rglob("*")
            if child.is_file() and _should_scan(child)
        )
    return sorted(files)


def test_generated_and_historical_paths_are_out_of_scope():
    legacy_lower = "we" + "ka"
    excluded_examples = [
        PROJECT_ROOT / "reports/phase-7/queries/neo4j-test-results.txt",
        PROJECT_ROOT / "docs/archive/legacy-vendor/example.md",
        PROJECT_ROOT
        / f"docs/superpowers/findings/2026-07-04-{legacy_lower}-residue-surface-map.md",
        PROJECT_ROOT
        / f"docs/superpowers/plans/2026-07-05-finish-live-{legacy_lower}-residue-rename.md",
    ]

    for path in excluded_examples:
        assert _should_scan(path) is False


def test_active_runtime_config_and_tests_reject_old_vendor_residue():
    offenders: list[str] = []

    for file_path in _iter_active_files():
        rel = _relative_posix(file_path)
        if LEGACY_PATTERN.search(rel):
            offenders.append(rel)
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LEGACY_PATTERN.search(text):
            offenders.append(rel)

    assert offenders == []
