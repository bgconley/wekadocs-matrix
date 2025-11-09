"""
Phase 7E-3 Production Docs Pack (No Filename Prefixes)
- Preserves your original filenames exactly.
- Uses PRODPACK-01/02 tags ONLY in source_uri/query for test-time scoping.
- Emits a machine-readable JSON report.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.ingestion.build_graph import ingest_document
from src.mcp_server.query_service import QueryService
from src.shared.connections import get_connection_manager

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

VERBOSITY = os.getenv("PRODPACK_VERBOSITY", "graph")
TOP_K = int(os.getenv("PRODPACK_TOP_K", "5"))
EMBED_VER = os.getenv(
    "PRODPACK_EMBED_VERSION", os.getenv("EMBEDDING_VERSION", "jina-embeddings-v3")
)
CLEAN_BEFORE = os.getenv("PRODPACK_CLEAN", "1") == "1"

_CITE_RE = re.compile(r"^\[\d+\]\s", re.M)


def _get_answer_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "answer") and getattr(resp, "answer"):
        return str(getattr(resp, "answer"))
    if hasattr(resp, "answer_markdown") and getattr(resp, "answer_markdown"):
        return str(getattr(resp, "answer_markdown"))
    if hasattr(resp, "to_dict"):
        try:
            d = resp.to_dict()
            for k in ("answer", "answer_markdown", "markdown", "text"):
                if d.get(k):
                    return str(d[k])
        except Exception:
            pass
    if isinstance(resp, dict):
        for k in ("answer", "answer_markdown", "markdown", "text"):
            if resp.get(k):
                return str(resp[k])
    return ""


def _parse_citations(answer_text: str) -> List[str]:
    return [
        ln.strip()
        for ln in (answer_text or "").splitlines()
        if _CITE_RE.match(ln.strip())
    ]


def _cleanup_prodpack_docs():
    cm = get_connection_manager()
    driver = cm.get_neo4j_driver()
    cypher = """
    MATCH (d:Document)
    WHERE coalesce(d.source_uri,d.uri,d.document_uri,'') STARTS WITH 'tests://prodpack/'
    WITH d
    OPTIONAL MATCH (d)-[*..5]->(n)
    WHERE n.document_id = d.id OR coalesce(n.source_uri,'') STARTS WITH 'tests://prodpack/'
    WITH collect(DISTINCT n)+[d] AS ns
    UNWIND ns AS n
    DETACH DELETE n;
    """
    with driver.session() as sess:
        sess.run(cypher)


def _run_query(qs: QueryService, query: str) -> Any:
    import inspect

    sig = inspect.signature(qs.search)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name == "query":
            kwargs[name] = query
        elif name in {"top_k", "max_results"}:
            kwargs[name] = TOP_K
        elif name == "verbosity":
            kwargs[name] = VERBOSITY
        elif name == "filters":
            kwargs[name] = {"embedding_version": EMBED_VER}
        elif name == "expand_graph":
            kwargs[name] = True
    return qs.search(**kwargs)


def _assert_sequential(citations: List[str]):
    assert citations, "No citations found in response."
    nums = []
    for c in citations[:6]:
        m = re.match(r"^\[(\d+)\]\s", c)
        if m:
            nums.append(int(m.group(1)))
    assert nums == sorted(nums) and nums == list(
        range(1, len(nums) + 1)
    ), "Citations are not sequentially numbered."


def _write_report(results: List[Dict[str, Any]], exitstatus: int):
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exitstatus": exitstatus,
        "env": {
            "VERBOSITY": VERBOSITY,
            "TOP_K": TOP_K,
            "EMBED_VER": EMBED_VER,
        },
        "results": results,
    }
    out = ARTIFACTS_DIR / "phase7e3_prod_docs_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[prod-docs-pack] wrote report → {out}\n")


CASES = [
    (
        "tests://prodpack/PRODPACK-01/additional-protocols_s3_s3-information-lifecycle-management_s3-information-lifecycle-management.md",
        "additional-protocols_s3_s3-information-lifecycle-management_s3-information-lifecycle-management.md",
        "PRODPACK-01",
        "How do I configure or verify S3 lifecycle management policies PRODPACK-01",
        ["lifecycle", "management", "policy", "s3", "ilm"],
    ),
    (
        "tests://prodpack/PRODPACK-02/weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md",
        "weka-filesystems-and-object-stores_attaching-detaching-object-stores-to-from-filesystems_attaching-detaching-object-stores-to-from-filesystems-1.md",
        "PRODPACK-02",
        "How do I attach or detach an object store from a filesystem PRODPACK-02",
        [
            "attach",
            "attaching",
            "detach",
            "detaching",
            "object",
            "store",
            "filesystem",
            "bucket",
        ],
    ),
]

RESULTS: List[Dict[str, Any]] = []


def setup_module():
    if CLEAN_BEFORE:
        _cleanup_prodpack_docs()


@pytest.mark.parametrize("src_uri,filename,tag,query,keywords", CASES)
def test_prod_doc(
    src_uri: str, filename: str, tag: str, query: str, keywords: List[str]
):
    path = DOCS_DIR / filename
    content = path.read_text(encoding="utf-8")
    stats = ingest_document(source_uri=src_uri, content=content, format="markdown")

    qs = QueryService()
    resp = _run_query(qs, query)

    answer = _get_answer_text(resp)
    citations = _parse_citations(answer)

    rec = {
        "tag": tag,
        "src_uri": src_uri,
        "filename": filename,
        "query": query,
        "ingestion": stats,
        "citations": citations[:6],
        "answer_preview": answer[:1200],
        "passed": True,
        "reason": "",
    }

    try:
        assert len(answer) > 0, "Empty answer text."
        assert len(citations) >= 2, "Expected at least 2 citations."
        _assert_sequential(citations)
        first_ci = citations[0].lower()
        assert any(
            k in first_ci for k in keywords
        ), f"First citation doesn't look relevant for {tag}. Got: {citations[0]}"
    except AssertionError as e:
        rec["passed"] = False
        rec["reason"] = str(e)
        raise
    finally:
        RESULTS.append(rec)


def teardown_module():
    _write_report(RESULTS, exitstatus=0 if all(r.get("passed") for r in RESULTS) else 1)
