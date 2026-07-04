"""Static guard for runtime-connected WEKA residue."""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = [
    "src",
    "config",
    "data/ingest/nutanix",
    "docker",
    "deploy",
    "scripts",
    "services",
    ".github",
    "docker-compose.yml",
    "Makefile",
]
LEGACY_PATTERN = re.compile(r"\b(?:W[E]KA|WekaDocs|wekadocs|weka-docs|weka-)\b")
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules"}
SKIP_SUFFIXES = {".bak", ".pyc", ".pyo", ".so", ".dylib"}


def test_runtime_connected_surfaces_do_not_reference_weka():
    offenders: list[str] = []

    for rel_path in RUNTIME_PATHS:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        if path.is_file():
            files = [path]
        else:
            files = [
                child
                for child in path.rglob("*")
                if child.is_file()
                and not (SKIP_DIRS & set(child.parts))
                and child.suffix not in SKIP_SUFFIXES
            ]
        for file_path in files:
            if LEGACY_PATTERN.search(str(file_path.relative_to(PROJECT_ROOT))):
                offenders.append(str(file_path.relative_to(PROJECT_ROOT)))
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if LEGACY_PATTERN.search(text):
                offenders.append(str(file_path.relative_to(PROJECT_ROOT)))

    assert offenders == []
