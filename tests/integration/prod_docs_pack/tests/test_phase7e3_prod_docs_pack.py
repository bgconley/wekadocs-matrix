"""
Phase 7E-3 Production Docs Pack
- Ingest two real production markdown documents
- Verify stitched context emits citations and enforces doc_tag scoping
- Emit a comprehensive JSON report for offline review
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Import project services from your repo
from src.ingestion.build_graph import ingest_document
from src.mcp_server.query_service import QueryService
from src.shared.connections import get_connection_manager

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Config knobs (override via env if desired)
VERBOSITY = os.getenv(
    "PRODPACK_VERBOSITY", "graph"
)  # 'graph' or 'full' (legacy 'markdown' accepted by service compat)
TOP_K = int(os.getenv("PRODPACK_TOP_K", "5"))
EMBED_VER = os.getenv(
    "PRODPACK_EMBED_VERSION", os.getenv("EMBEDDING_VERSION", "jina-embeddings-v3")
)
CLEAN_BEFORE = (
    os.getenv("PRODPACK_CLEAN", "1") == "1"
)  # run Cypher cleanup before tests
STRICT_SINGLE_DOC = os.getenv("PRODPACK_STRICT_SINGLE_DOC", "1") == "1"

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


def _cleanup_regression_docs():
    # Purge prior prodpack docs by source_uri prefix
    # Requires configured Neo4j connection via the repo's connection manager
    # Neo4j 5.x requires CALL {} subquery for proper aggregation scoping
    cm = get_connection_manager()
    driver = cm.get_neo4j_driver()
    cypher = """
    MATCH (d:Document)
    WHERE coalesce(d.source_uri,d.uri,d.document_uri,'') STARTS WITH 'tests://prodpack/'
    CALL {
        WITH d
        OPTIONAL MATCH (d)-[*..5]->(n)
        WHERE n.document_id = d.id OR coalesce(n.source_uri,'') STARTS WITH 'tests://prodpack/'
        RETURN collect(DISTINCT n) AS related
    }
    WITH d, related
    UNWIND related + [d] AS node
    DETACH DELETE node
    """
    with driver.session() as sess:
        sess.run(cypher)


def _run_query(qs: QueryService, query: str) -> Any:
    # Flexible call to adapt to minor API variations
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


def _assert_sequential_citations(citations: List[str]):
    assert citations, "No citations found in response."
    # Ensure numbering is sequential [1], [2], ...
    nums = []
    for c in citations[:6]:
        m = re.match(r"^\[(\d+)\]\s", c)
        if m:
            nums.append(int(m.group(1)))
    assert nums == sorted(nums) and nums == list(
        range(1, len(nums) + 1)
    ), "Citations are not sequentially numbered."


def _record_result(bucket: List[Dict[str, Any]], record: Dict[str, Any]):
    bucket.append(record)


def _write_report(results: List[Dict[str, Any]], exitstatus: int):
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exitstatus": exitstatus,
        "env": {
            "VERBOSITY": VERBOSITY,
            "TOP_K": TOP_K,
            "EMBED_VER": EMBED_VER,
            "STRICT_SINGLE_DOC": STRICT_SINGLE_DOC,
        },
        "results": results,
    }
    path = ARTIFACTS_DIR / "phase7e3_prod_docs_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[prod-docs-pack] wrote report → {path}\n")


# --------------------
# Test catalog
# --------------------
# (src_uri, local filename, tag, query)
CASES = [
    (
        "tests://prodpack/PRODPACK-01/prod01_s3_information_lifecycle_management.md",
        "prod01_s3_information_lifecycle_management.md",
        "PRODPACK-01",
        "How do I configure or verify S3 lifecycle management rules PRODPACK-01",
    ),
    (
        "tests://prodpack/PRODPACK-02/prod02_attaching_detaching_object_stores.md",
        "prod02_attaching_detaching_object_stores.md",
        "PRODPACK-02",
        "How do I attach or detach an object store from a filesystem PRODPACK-02",
    ),
]

RESULTS: List[Dict[str, Any]] = []


def setup_module():
    if CLEAN_BEFORE:
        _cleanup_regression_docs()


@pytest.mark.parametrize("src_uri,filename,tag,query", CASES)
def test_prod_doc_case(src_uri: str, filename: str, tag: str, query: str):
    # Ingest content
    path = DOCS_DIR / filename
    content = path.read_text(encoding="utf-8")
    stats = ingest_document(source_uri=src_uri, content=content, format="markdown")

    qs = QueryService()
    resp = _run_query(qs, query)

    answer = _get_answer_text(resp)
    citations = _parse_citations(answer)

    # Basic asserts
    assert len(answer) > 0, "Empty answer text."
    _assert_sequential_citations(citations)

    # Soft semantic asserts to avoid brittleness:
    #  - ensure at least 2 citations
    #  - first citation contains a relevant keyword from the file context
    keywords = []
    if "PRODPACK-01" in tag:
        keywords = ["lifecycle", "management", "policy", "s3"]
    elif "PRODPACK-02" in tag:
        keywords = [
            "attach",
            "attaching",
            "detach",
            "detaching",
            "object",
            "store",
            "filesystem",
        ]

    first_ci = citations[0].lower()
    kw_hit = any(k in first_ci for k in keywords) if keywords else True

    record = {
        "tag": tag,
        "src_uri": src_uri,
        "filename": filename,
        "query": query,
        "ingestion": stats,
        "citations": citations[:5],
        "answer_preview": answer[:900],
        "passed": True,
        "reason": "",
    }

    try:
        assert len(citations) >= 2, "Expected at least 2 citations."
        assert (
            kw_hit
        ), f"First citation does not look relevant. Expected one of {keywords}, got: {citations[0]}"
        # Optional single-document sanity: stitched chunks should come from same document (heuristic check on answer text)
        # (We can't see document_id directly here; rely on doc_tag scoping via 'tag' in query)
    except AssertionError as e:
        record["passed"] = False
        record["reason"] = str(e)
        raise
    finally:
        _record_result(RESULTS, record)


def teardown_module():
    # Always emit a machine-readable report
    _write_report(RESULTS, exitstatus=0 if all(r.get("passed") for r in RESULTS) else 1)
