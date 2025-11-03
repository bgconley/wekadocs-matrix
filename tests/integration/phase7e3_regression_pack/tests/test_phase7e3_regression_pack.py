"""
Phase 7E-3 Regression Pack
- Ingest a variety of technical-doc patterns (normal + corner cases)
- Verify stitched context ordering and citation numbering
- Emit a comprehensive JSON report for offline review
"""

import inspect
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.ingestion.build_graph import ingest_document
from src.mcp_server.query_service import QueryService

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REG_RESULTS: List[Dict[str, Any]] = []

RPK_VERBOSITY = os.getenv("RPK_VERBOSITY", "graph")
RPK_TOP_K = int(os.getenv("RPK_TOP_K", "5"))
RPK_EMBEDDING_VERSION = os.getenv("RPK_EMBEDDING_VERSION", "jina-embeddings-v3")

_CITATION_LINE = re.compile(r"^\[\d+\]\s+")


def _read(path: str) -> str:
    return (DOCS_DIR / path).read_text(encoding="utf-8")


def _get_answer_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp

    if hasattr(resp, "answer") and getattr(resp, "answer"):
        return str(getattr(resp, "answer"))

    if hasattr(resp, "answer_markdown") and getattr(resp, "answer_markdown"):
        return str(getattr(resp, "answer_markdown"))

    if hasattr(resp, "to_dict"):
        try:
            data = resp.to_dict()
            for key in ("answer", "answer_markdown", "markdown", "text"):
                value = data.get(key)
                if value:
                    return str(value)
        except Exception:
            pass

    if isinstance(resp, dict):
        for key in ("answer", "answer_markdown", "markdown", "text"):
            value = resp.get(key)
            if value:
                return str(value)

    return ""


def _parse_citations(answer_text: str) -> List[str]:
    return [
        line.strip()
        for line in (answer_text or "").splitlines()
        if _CITATION_LINE.match(line.strip())
    ]


def _invoke_query(qs: QueryService, query: str):
    sig = inspect.signature(qs.search)
    kwargs: Dict[str, Any] = {}

    for name in sig.parameters:
        if name == "query":
            kwargs[name] = query
        elif name in {"top_k", "max_results"}:
            kwargs[name] = RPK_TOP_K
        elif name == "verbosity":
            kwargs[name] = RPK_VERBOSITY
        elif name == "filters":
            kwargs[name] = {"embedding_version": RPK_EMBEDDING_VERSION}
        elif name == "expand_graph":
            kwargs[name] = True

    return qs.search(**kwargs)


def _assert_citations_and_order(
    resp: Any,
    expect_first: Optional[str] = None,
    expect_second: Optional[str] = None,
) -> Dict[str, Any]:
    answer_text = _get_answer_text(resp)
    citations = _parse_citations(answer_text)

    assert citations, "No citations found in response."

    if expect_first:
        assert (
            expect_first in citations[0]
        ), f"First citation should reference {expect_first!r}, got {citations[0]!r}"

    if expect_second:
        assert (
            expect_second in citations[1]
        ), f"Second citation should reference {expect_second!r}, got {citations[1]!r}"

    return {"answer": answer_text, "citations": citations}


# Test catalog: (docfile, query, expected first two citation substrings in order)
CASES = [
    (
        "case01_steps_basic.md",
        "Detailed instructions for completing step 2 installation REGPACK-01",
        ["Step 1: Prepare Environment", "Step 2: Complete Installation"],
    ),
    (
        "case02_deep_nesting.md",
        "How do I set parameter X for the pipeline REGPACK-02",
        ["Step 1: Provision Resources", "Detail A: Set Parameter X"],
    ),
    (
        "case03_api_endpoints.md",
        "Create a new user via API REGPACK-03",
        ["Endpoint: GET /v1/users", "Endpoint: POST /v1/users"],
    ),
    (
        "case04_break_keywords.md",
        "Apply a rolling update REGPACK-04",
        ["Step 1: Drain Nodes", "Step 2: Apply Update"],
    ),
    (
        "case05_empty_parent.md",
        "Install services REGPACK-05",
        ["Step 1: Bootstrap", "Step 2: Install Services"],
    ),
    (
        "case06_duplicate_headings.md",
        "Overview of the Payments module REGPACK-06",
        ["Overview"],
    ),
    (
        "case07_unicode_and_code.md",
        "Procédure d’installation REGPACK-07",
        ["Step 1: Préparer l’environnement", "Step 2: Installer"],
    ),
    (
        "case08_setext_headers.md",
        "Execute step 2 REGPACK-08",
        ["Step 1: Prepare", "Step 2: Execute"],
    ),
    (
        "case09_tables_and_long.md",
        "Tune timeout REGPACK-09",
        ["Step 1: Set Threads", "Step 2: Tune Timeout"],
    ),
    (
        "case10_glossary_appendix.md",
        "Assign roles REGPACK-10",
        ["Step 1: Create Tenant", "Step 2: Assign Roles"],
    ),
]


@pytest.mark.parametrize("docfile,query,expected", CASES)
def test_regression_case(docfile, query, expected):
    content = _read(docfile)
    src_uri = f"tests://regression_pack/{docfile}"
    stats = ingest_document(source_uri=src_uri, content=content, format="markdown")

    qs = QueryService()
    response = _invoke_query(qs, query)

    record: Dict[str, Any] = {
        "docfile": docfile,
        "query": query,
        "ingestion": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "reason": "",
        "citations": [],
        "answer_preview": "",
    }

    try:
        parsed = _assert_citations_and_order(
            response,
            expect_first=expected[0] if expected else None,
            expect_second=expected[1] if len(expected) > 1 else None,
        )
        record["citations"] = parsed["citations"]
        record["answer_preview"] = parsed["answer"][:800]

        if "ORDER_STEP_ONE" in content and "ORDER_STEP_TWO" in content:
            answer_text = parsed["answer"]
            idx_one = answer_text.find("ORDER_STEP_ONE")
            idx_two = answer_text.find("ORDER_STEP_TWO")
            assert (
                idx_one == -1 or idx_two == -1 or idx_one < idx_two
            ), "ORDER markers appear out of order in stitched context"

    except AssertionError as exc:
        record["passed"] = False
        record["reason"] = str(exc)
        raise
    finally:
        REG_RESULTS.append(record)


def pytest_sessionfinish(session, exitstatus):
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exitstatus": exitstatus,
        "env": {
            "RPK_VERBOSITY": RPK_VERBOSITY,
            "RPK_TOP_K": RPK_TOP_K,
            "RPK_EMBEDDING_VERSION": RPK_EMBEDDING_VERSION,
        },
        "cases": REG_RESULTS,
    }

    out_path = ARTIFACTS_DIR / "phase7e3_regression_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRegression report written to: {out_path}")
