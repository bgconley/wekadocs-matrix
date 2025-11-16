from __future__ import annotations

import re
from pathlib import Path

import pytest

BANNED_EXTENSIONS = {".py", ".ts", ".cypher"}
PATTERN = re.compile(r"\bproperties\s*\(")


def test_no_properties_function_calls() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    violations: list[str] = []
    for path in src_root.rglob("*"):
        if not path.is_file() or path.suffix not in BANNED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(f"{path.relative_to(repo_root)}:{line}")
    if violations:
        pytest.fail(
            "Found forbidden properties(...) projections in src: "
            + ", ".join(violations)
        )
