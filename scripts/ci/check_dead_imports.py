#!/usr/bin/env python3
"""
CI guard: detect ACTIVE/MIXED/STANDALONE modules importing DEAD/PHANTOM modules.

Scans all @status annotations in src/ and checks that no live module imports
from a dead or phantom module. Exits non-zero if violations are found.

Also enforces allowlist hygiene: stale allowlist entries (that no longer
correspond to real violations) cause a failure to prevent accumulation.

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
# IMPORTANT: Stale entries (no longer matching real violations) will fail CI.
ALLOWLIST = {
    # MIXED module — dead import is inside dead method, cleaned in Phase E
    ("src/ingestion/build_graph.py", "src/ingestion/reconcile.py"),
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
    """Extract all src.* import lines with line numbers.

    Handles both forms:
        from src.foo.bar import baz
        import src.foo.bar
    """
    imports = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("from src.") and "import" in stripped:
                    imports.append((lineno, stripped))
                elif stripped.startswith("import src."):
                    imports.append((lineno, stripped))
    except (OSError, UnicodeDecodeError):
        pass
    return imports


def module_path_from_import(import_line: str) -> str | None:
    """Extract the module path from an import statement.

    Handles:
        'from src.neo.contract_checks import foo' -> 'src/neo/contract_checks.py'
        'from src.learning import bar' -> 'src/learning/__init__.py'
        'import src.neo.contract_checks' -> 'src/neo/contract_checks.py'
        'import src.learning' -> 'src/learning/__init__.py'
    """
    # Try 'from X import Y' form first
    match = re.match(r"from\s+(src\.[^\s]+)\s+import", import_line)
    if not match:
        # Try 'import X' form
        match = re.match(r"import\s+(src\.[^\s,]+)", import_line)
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


def check_violations(verbose: bool = False) -> tuple[list[dict], set[tuple]]:
    """Scan for ACTIVE→DEAD import violations.

    Returns:
        (violations, used_allowlist_entries)
    """
    # Build status map
    status_map: dict[str, str] = {}
    for py_file in SRC_DIR.rglob("*.py"):
        rel = str(py_file.relative_to(PROJECT_ROOT))
        status = get_module_status(py_file)
        if status:
            status_map[rel] = status

    violations = []
    used_allowlist: set[tuple] = set()

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
                key = (filepath, target_path)
                if key in ALLOWLIST:
                    used_allowlist.add(key)
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

    return violations, used_allowlist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all scanned files"
    )
    args = parser.parse_args()

    violations, used_allowlist = check_violations(verbose=args.verbose)

    exit_code = 0

    # Check for stale allowlist entries
    stale = ALLOWLIST - used_allowlist
    if stale:
        print(f"FAIL: {len(stale)} stale allowlist entry/entries:\n")
        for src, tgt in sorted(stale):
            print(f"  {src} -> {tgt}")
            print("  (no longer a real violation — remove from ALLOWLIST)")
            print()
        exit_code = 1

    # Check for violations
    if violations:
        print(f"FAIL: {len(violations)} import violation(s) found:\n")
        for v in violations:
            print(f"  {v['source']}:{v['line']}")
            print(f"    [{v['source_status']}] imports [{v['target_status']}]")
            print(f"    {v['import']}")
            print()
        exit_code = 1

    if exit_code == 0:
        allowed = len(used_allowlist)
        suffix = f" ({allowed} allowlisted)" if allowed else ""
        print(f"OK: No ACTIVE->DEAD import violations found in src/{suffix}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
