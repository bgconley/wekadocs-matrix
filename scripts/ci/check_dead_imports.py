#!/usr/bin/env python3
"""
CI guard: detect ACTIVE/MIXED/STANDALONE modules importing DEAD/PHANTOM modules.

Scans all @status annotations in src/ and checks that no live module imports
from a dead or phantom module. Exits non-zero if violations are found.

Usage:
    python scripts/ci/check_dead_imports.py
    python scripts/ci/check_dead_imports.py --verbose
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Statuses that should NOT import from DEAD/PHANTOM
LIVE_STATUSES = {"ACTIVE", "MIXED", "STANDALONE", "DORMANT", "TEST_ONLY"}

# Statuses that are considered "dead" — live modules must not import from these
DEAD_STATUSES = {"DEAD", "PHANTOM"}

STATUS_PATTERN = re.compile(r"#\s*@status:\s*(\S+)")

# Known violations that are intentional and scheduled for cleanup.
# Format: (source_relative_path, target_relative_path)
# Remove entries as they are resolved in later phases.
ALLOWLIST = {
    # MIXED module — dead import is inside dead method, cleaned in Phase E
    ("src/ingestion/build_graph.py", "src/ingestion/reconcile.py"),
    # Lazy import inside config-gated _parse_with_shadow_comparison()
    # shadow_comparison only loads when shadow_mode=true (never in prod)
    ("src/ingestion/parsers/__init__.py", "src/ingestion/parsers/shadow_comparison.py"),
}


def get_module_status(filepath: Path) -> str | None:
    """Extract the module-level @status from a Python file's header."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Only scan first 15 lines for module-level status
            for i, line in enumerate(f):
                if i >= 15:
                    break
                match = STATUS_PATTERN.search(line)
                if match:
                    return match.group(1).split()[0].rstrip(",")
    except (OSError, UnicodeDecodeError):
        pass
    return None


def get_imports_from_file(filepath: Path) -> list[tuple[int, str]]:
    """Extract all 'from src.X import Y' lines with line numbers."""
    imports = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("from src.") and "import" in stripped:
                    imports.append((lineno, stripped))
    except (OSError, UnicodeDecodeError):
        pass
    return imports


def module_path_from_import(import_line: str) -> str | None:
    """Extract the module path from an import statement.

    'from src.neo.contract_checks import foo' -> 'src/neo/contract_checks.py'
    'from src.learning import bar' -> 'src/learning/__init__.py'
    """
    match = re.match(r"from\s+(src\.[^\s]+)\s+import", import_line)
    if not match:
        return None
    dotted = match.group(1)
    parts = dotted.split(".")
    # Try as direct module file first
    as_file = SRC_DIR.parent / Path(*parts).with_suffix(".py")
    if as_file.exists():
        return str(as_file.relative_to(PROJECT_ROOT))
    # Try as package __init__.py
    as_package = SRC_DIR.parent / Path(*parts) / "__init__.py"
    if as_package.exists():
        return str(as_package.relative_to(PROJECT_ROOT))
    return None


def check_violations(verbose: bool = False) -> list[dict]:
    """Scan for ACTIVE→DEAD import violations."""
    # Build status map
    status_map: dict[str, str] = {}
    for py_file in SRC_DIR.rglob("*.py"):
        rel = str(py_file.relative_to(PROJECT_ROOT))
        status = get_module_status(py_file)
        if status:
            status_map[rel] = status

    violations = []

    for filepath, source_status in status_map.items():
        if source_status not in LIVE_STATUSES:
            continue

        full_path = PROJECT_ROOT / filepath
        imports = get_imports_from_file(full_path)

        for lineno, import_line in imports:
            target_path = module_path_from_import(import_line)
            if not target_path:
                continue

            target_status = status_map.get(target_path)
            if target_status and target_status in DEAD_STATUSES:
                if (filepath, target_path) in ALLOWLIST:
                    continue
                violations.append(
                    {
                        "source": filepath,
                        "source_status": source_status,
                        "target": target_path,
                        "target_status": target_status,
                        "line": lineno,
                        "import": import_line,
                    }
                )

    return violations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all scanned files"
    )
    args = parser.parse_args()

    violations = check_violations(verbose=args.verbose)

    if not violations:
        print("OK: No ACTIVE->DEAD import violations found in src/")
        return 0

    print(f"FAIL: {len(violations)} import violation(s) found:\n")
    for v in violations:
        print(f"  {v['source']}:{v['line']}")
        print(f"    [{v['source_status']}] imports [{v['target_status']}]")
        print(f"    {v['import']}")
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
