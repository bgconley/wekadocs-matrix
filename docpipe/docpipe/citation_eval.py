"""Downstream citation evaluation for docpipe corpus builds."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_EVIDENCE_STATUS_RE = re.compile(r"\Akb_retrieve_evidence returned \d+ quotes?\.\Z")


@dataclass(frozen=True)
class SourceExpectation:
    source_pdf: str
    slug: str
    must_cite_text: str


@dataclass(frozen=True)
class CitationQuestion:
    id: str
    question: str
    expected_sources: list[SourceExpectation]
    must_answer_contain: list[str]


@dataclass
class CitationCaseResult:
    id: str
    question: str
    passed: bool
    missing_answer_terms: list[str] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)
    quote_count: int = 0
    answer_source: str = "answer"


@dataclass
class CitationReport:
    cases: list[CitationCaseResult]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def pass_fraction(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        cases = [
            {
                "id": case.id,
                "question": case.question,
                "passed": case.passed,
                "missing_answer_terms": case.missing_answer_terms,
                "missing_sources": case.missing_sources,
                "quote_count": case.quote_count,
                "answer_source": case.answer_source,
            }
            for case in self.cases
        ]
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.total - self.passed,
            "pass_fraction": self.pass_fraction,
            "cases": cases,
        }


def tool_arguments(
    *,
    tool_name: str,
    question: str,
    top_k: int,
    max_quotes: int,
    retrieval_depth: int,
) -> dict[str, Any]:
    if tool_name == "search_documentation":
        return {
            "query": question,
            "top_k": top_k,
            "verbosity": "graph",
        }
    return {
        "question": question,
        "top_k": top_k,
        "max_quotes": max_quotes,
        "retrieval_depth": retrieval_depth,
    }


def load_questions(path: Path) -> list[CitationQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("citation spec must contain a 'questions' list")

    questions: list[CitationQuestion] = []
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            raise ValueError(f"question {index} must be an object")
        raw_sources = raw.get("expected_sources")
        if not isinstance(raw_sources, list):
            raise ValueError(f"question {index} must contain expected_sources")
        sources: list[SourceExpectation] = []
        for source_index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, dict):
                raise ValueError(
                    f"question {index} source {source_index} must be an object"
                )
            sources.append(
                SourceExpectation(
                    source_pdf=_required_str(
                        raw_source, "source_pdf", f"question {index} source"
                    ),
                    slug=_required_str(raw_source, "slug", f"question {index} source"),
                    must_cite_text=_required_str(
                        raw_source, "must_cite_text", f"question {index} source"
                    ),
                )
            )
        raw_terms = raw.get("must_answer_contain")
        if not isinstance(raw_terms, list) or not all(
            isinstance(term, str) and term.strip() for term in raw_terms
        ):
            raise ValueError(f"question {index} must contain must_answer_contain")
        questions.append(
            CitationQuestion(
                id=_required_str(raw, "id", f"question {index}"),
                question=_required_str(raw, "question", f"question {index}"),
                expected_sources=sources,
                must_answer_contain=[term.strip() for term in raw_terms],
            )
        )
    return questions


def evaluate_payload(
    question: CitationQuestion, payload: dict[str, Any]
) -> CitationCaseResult:
    quotes = _extract_quotes(payload)
    answer_text, answer_source = _extract_answer_text(payload, quotes)
    answer_haystack = _norm(answer_text)
    missing_answer_terms = [
        term
        for term in question.must_answer_contain
        if _norm(term) not in answer_haystack
    ]
    missing_sources = [
        f"{source.slug}:{source.must_cite_text}"
        for source in question.expected_sources
        if not _source_is_cited(source, quotes)
    ]
    return CitationCaseResult(
        id=question.id,
        question=question.question,
        passed=not missing_answer_terms and not missing_sources,
        missing_answer_terms=missing_answer_terms,
        missing_sources=missing_sources,
        quote_count=len(quotes),
        answer_source=answer_source,
    )


def _required_str(raw: dict[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must contain non-empty {key}")
    return value.strip()


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _extract_structured(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("structuredContent"), dict):
        return payload["structuredContent"]
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    return payload


def _content_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("content")
    if not isinstance(raw, list):
        result = payload.get("result")
        raw = result.get("content") if isinstance(result, dict) else None
    return [item for item in raw or [] if isinstance(item, dict)]


def _extract_quotes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    structured = _extract_structured(payload)
    raw_quotes = structured.get("quotes")
    if not isinstance(raw_quotes, list):
        raw_quotes = structured.get("evidence")
    quotes = [quote for quote in raw_quotes or [] if isinstance(quote, dict)]

    for item in _content_items(payload):
        if item.get("type") != "json" or not isinstance(item.get("json"), dict):
            continue
        answer_json = item["json"]
        evidence = answer_json.get("evidence")
        if isinstance(evidence, list):
            quotes.extend(quote for quote in evidence if isinstance(quote, dict))
    return quotes


def _extract_answer_text(
    payload: dict[str, Any], quotes: list[dict[str, Any]]
) -> tuple[str, str]:
    parts: list[str] = []
    structured = _extract_structured(payload)
    draft = structured.get("answer_draft")
    if isinstance(draft, dict):
        markdown = draft.get("markdown")
        if isinstance(markdown, str):
            parts.append(markdown)
        claims = draft.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, dict) and isinstance(claim.get("text"), str):
                    parts.append(claim["text"])
    for key in ("answer_markdown", "answer"):
        value = structured.get(key)
        if isinstance(value, str):
            parts.append(value)

    for item in _content_items(payload):
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            text = item["text"].strip()
            if text and not _is_evidence_status_text(text):
                parts.append(text)
        if item.get("type") == "json" and isinstance(item.get("json"), dict):
            answer_json = item["json"]
            for key in ("answer", "answer_markdown"):
                value = answer_json.get(key)
                if isinstance(value, str):
                    parts.append(value)

    if parts:
        return "\n".join(parts), "answer"
    return "\n".join(_quote_text(quote) for quote in quotes), "quotes"


def _is_evidence_status_text(text: str) -> bool:
    return bool(_EVIDENCE_STATUS_RE.match(text))


def _quote_text(quote: dict[str, Any]) -> str:
    for key in ("quote", "text", "excerpt", "snippet"):
        value = quote.get(key)
        if isinstance(value, str):
            return value
    return ""


def _quote_source_text(quote: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("doc_tag", "uri", "source_uri", "title", "section_id", "passage_id"):
        value = quote.get(key)
        if isinstance(value, str):
            values.append(value)
    parent = quote.get("parent_path")
    if isinstance(parent, str):
        values.append(parent)
    return " ".join(values)


def _source_is_cited(source: SourceExpectation, quotes: list[dict[str, Any]]) -> bool:
    expected_text = _norm(source.must_cite_text)
    slug = _norm(source.slug)
    source_pdf = _norm(Path(source.source_pdf).stem)
    for quote in quotes:
        quote_text = _norm(_quote_text(quote))
        source_text = _norm(_quote_source_text(quote))
        if expected_text not in quote_text:
            continue
        if slug in source_text or source_pdf in source_text:
            return True
    return False


def create_connection(base_url: str) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    mcp_path = f"{base_path}/_mcp/" if base_path else "/_mcp/"
    if scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            host, port, timeout=120
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=120)
    return conn, mcp_path


def mcp_tool_call(
    *,
    base_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    conn, mcp_path = create_connection(base_url)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "docpipe-citation-eval", "version": "1.0"},
        },
    }
    conn.request("POST", mcp_path, json.dumps(initialize), headers)
    init_resp = conn.getresponse()
    session_id = init_resp.getheader("mcp-session-id")
    init_body = init_resp.read().decode()
    if init_resp.status >= 400:
        raise RuntimeError(
            f"MCP initialize failed HTTP {init_resp.status}: {init_body}"
        )
    if session_id:
        headers["mcp-session-id"] = session_id

    conn.request(
        "POST",
        mcp_path,
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        headers,
    )
    conn.getresponse().read()

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    conn.request("POST", mcp_path, json.dumps(payload), headers)
    response = conn.getresponse()
    body = response.read().decode()
    if response.status >= 400:
        raise RuntimeError(f"MCP tool call failed HTTP {response.status}: {body}")
    parsed = _parse_mcp_response(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("MCP tool call returned a non-object JSON response")
    if parsed.get("error"):
        raise RuntimeError(f"MCP tool call returned error: {parsed['error']}")
    result = parsed.get("result")
    payload = result if isinstance(result, dict) else parsed
    _raise_for_tool_error(payload)
    return payload


def _parse_mcp_response(body: str) -> Any:
    stripped = body.strip()
    if not stripped:
        return {}
    if stripped.startswith("data:"):
        data_lines = []
        for line in stripped.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        stripped = "\n".join(data_lines)
    return json.loads(stripped)


def run_evaluation(
    questions: list[CitationQuestion],
    *,
    base_url: str,
    tool_name: str,
    top_k: int,
    max_quotes: int,
    retrieval_depth: int,
) -> CitationReport:
    cases: list[CitationCaseResult] = []
    for question in questions:
        payload = mcp_tool_call(
            base_url=base_url,
            tool_name=tool_name,
            arguments=tool_arguments(
                tool_name=tool_name,
                question=question.question,
                top_k=top_k,
                max_quotes=max_quotes,
                retrieval_depth=retrieval_depth,
            ),
        )
        _raise_for_tool_error(payload)
        cases.append(evaluate_payload(question, payload))
    return CitationReport(cases)


def _raise_for_tool_error(payload: dict[str, Any]) -> None:
    if not payload.get("isError"):
        return
    messages: list[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                messages.append(item["text"])
    detail = "; ".join(messages) if messages else json.dumps(payload, sort_keys=True)
    raise RuntimeError(f"MCP tool returned error: {detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run downstream citation checks against an MCP RAG gateway."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/eval/citation_questions.json"),
        help="Citation question JSON spec.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="MCP server base URL.",
    )
    parser.add_argument(
        "--tool-name",
        default="kb_retrieve_evidence",
        help="MCP tool to call.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-quotes", type=int, default=8)
    parser.add_argument("--retrieval-depth", type=int, default=80)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec and print question ids without calling the gateway.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        questions = load_questions(args.spec)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"citation-eval spec error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"questions={len(questions)}")
        for question in questions:
            print(f"{question.id}: {question.question}")
        return 0

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        report = run_evaluation(
            questions,
            base_url=args.base_url,
            tool_name=args.tool_name,
            top_k=args.top_k,
            max_quotes=args.max_quotes,
            retrieval_depth=args.retrieval_depth,
        )
    except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"citation-eval gateway error: {exc}", file=sys.stderr)
        return 2

    data = report.to_dict()
    data.update(
        {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": (time.monotonic() - started) * 1000.0,
            "base_url": args.base_url,
            "tool_name": args.tool_name,
            "spec": str(args.spec),
        }
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"report={args.report}")
    print(
        f"passed={report.passed}/{report.total} pass_fraction={report.pass_fraction:.3f}"
    )
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
