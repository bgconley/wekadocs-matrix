"""
Retrieval diagnostics persistence for MCP retrieval tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.shared.observability import get_logger

logger = get_logger(__name__)

_DATE_FORMAT = "%Y-%m-%d"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class RetrievalDiagnosticEmitter:
    def __init__(self, *, base_dir: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parents[3]
        default_dir = root / "reports" / "retrieval_diagnostics"
        resolved = base_dir or os.getenv("RETRIEVAL_DIAGNOSTICS_DIR")
        if resolved:
            self.base_dir = Path(resolved)
        else:
            self.base_dir = default_dir
        self.sample_rate = _float_env("RETRIEVAL_DIAGNOSTICS_SAMPLE_RATE", 0.01)
        self.retention_days = _int_env("RETRIEVAL_DIAGNOSTICS_RETENTION_DAYS", 14)
        self.store_query_text = _bool_env(
            "RETRIEVAL_DIAGNOSTICS_STORE_QUERY_TEXT", "false"
        )
        self.force = _bool_env("RETRIEVAL_DIAGNOSTICS_FORCE", "false")
        self._lock = asyncio.Lock()
        self._last_cleanup = 0.0

    def _should_emit(self, *, force: bool, error: bool) -> bool:
        if force or self.force or error:
            return True
        if self.sample_rate <= 0:
            return False
        return random.random() < self.sample_rate

    def _maybe_cleanup(self) -> None:
        if self.retention_days <= 0:
            return
        now = time.time()
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now
        cutoff = datetime.now(timezone.utc).timestamp() - (self.retention_days * 86400)
        try:
            for path in self.base_dir.iterdir():
                if not path.is_dir():
                    continue
                try:
                    day = datetime.strptime(path.name, "%Y-%m-%d")
                except ValueError:
                    continue
                if day.replace(tzinfo=timezone.utc).timestamp() < cutoff:
                    for child in path.glob("*"):
                        with contextlib.suppress(Exception):
                            child.unlink()
                    with contextlib.suppress(Exception):
                        path.rmdir()
        except Exception:
            logger.warning("retrieval_diagnostics_cleanup_failed")

    def cleanup(self) -> None:
        """Best-effort retention cleanup; safe to call at startup."""
        self._maybe_cleanup()

    def resolve_markdown_path(self, *, date: str, diagnostic_id: str) -> Path:
        if not date:
            raise ValueError("date is required")
        try:
            datetime.strptime(date, _DATE_FORMAT)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc
        if not diagnostic_id or not _SAFE_ID_RE.match(diagnostic_id):
            raise ValueError("diagnostic_id contains invalid characters")
        return self.base_dir / date / f"{diagnostic_id}.md"

    def read_markdown(self, *, date: str, diagnostic_id: str) -> str:
        path = self.resolve_markdown_path(date=date, diagnostic_id=diagnostic_id)
        return path.read_text(encoding="utf-8")

    def _redact_query(self, record: dict[str, Any]) -> None:
        query = record.get("query") or {}
        raw = query.get("raw") or ""
        if not self.store_query_text:
            query["query_hash"] = (
                hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else None
            )
            query["query_length"] = len(raw) if raw else 0
            query["raw"] = None
            query["normalized"] = None
        record["query"] = query

    def _render_markdown(self, record: dict[str, Any]) -> str:
        diag_id = record.get("diagnostic_id", "")
        timestamp = record.get("timestamp", "")
        transport = record.get("transport", "")
        tool = record.get("tool", "")
        session_id = record.get("session_id", "")
        otel = record.get("otel", {}) or {}
        scope = record.get("scope", {}) or {}
        timing = record.get("timing_ms", {}) or {}
        counts = record.get("counts", {}) or {}
        budgets = record.get("budgets", {}) or {}

        lines = [f"# Retrieval Diagnostics — {diag_id}", ""]
        lines.append(f"- timestamp: {timestamp}")
        lines.append(f"- transport: {transport}")
        lines.append(f"- tool: {tool}")
        lines.append(f"- session_id: {session_id}")
        lines.append(f"- trace: {otel.get('trace_id')}/{otel.get('span_id')}")
        lines.append(
            "- scope: project_id={project_id} env={environment} doc_tag={doc_tag}".format(
                project_id=scope.get("project_id"),
                environment=scope.get("environment"),
                doc_tag=scope.get("doc_tag"),
            )
        )
        lines.append("")
        lines.append("## Timings (ms)")
        lines.append(f"- bm25: {timing.get('bm25', 0.0)}")
        lines.append(f"- vector_search: {timing.get('vector_search', 0.0)}")
        lines.append(f"- fusion: {timing.get('fusion', 0.0)}")
        lines.append(f"- rerank: {timing.get('rerank', 0.0)}")
        lines.append(f"- graph_expansion: {timing.get('graph_expansion', 0.0)}")
        lines.append(f"- total: {timing.get('total', 0.0)}")
        lines.append("")
        lines.append("## Counts")
        lines.append(f"- candidates_initial: {counts.get('candidates_initial', 0)}")
        lines.append(
            f"- candidates_post_filter: {counts.get('candidates_post_filter', 0)}"
        )
        lines.append(
            f"- candidates_post_dedupe: {counts.get('candidates_post_dedupe', 0)}"
        )
        lines.append(f"- returned: {counts.get('returned', 0)}")
        dropped = counts.get("dropped", {}) or {}
        lines.append(
            "- dropped: dedupe={dedupe} veto={veto} scope_mismatch={scope_mismatch}".format(
                dedupe=dropped.get("dedupe", 0),
                veto=dropped.get("veto", 0),
                scope_mismatch=dropped.get("scope_mismatch", 0),
            )
        )
        lines.append("")
        lines.append("## Top Results")
        lines.append(
            "| rank | chunk_id | doc_tag | source | fused | rerank | graph | token_count |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")
        for item in (record.get("results") or [])[:10]:
            scores = item.get("scores", {}) or {}
            lines.append(
                "| {rank} | {chunk_id} | {doc_tag} | {source} | {fused} | {rerank} | {graph} | {token_count} |".format(
                    rank=item.get("rank"),
                    chunk_id=item.get("chunk_id"),
                    doc_tag=item.get("doc_tag"),
                    source=item.get("source"),
                    fused=scores.get("fused", 0.0),
                    rerank=scores.get("rerank", 0.0),
                    graph=scores.get("graph", 0.0),
                    token_count=item.get("token_count", 0),
                )
            )
        lines.append("")
        lines.append("## Budgets")
        lines.append(f"- response_bytes: {budgets.get('response_bytes', 0)}")
        lines.append(f"- tokens_estimate: {budgets.get('tokens_estimate', 0)}")
        lines.append(f"- partial: {budgets.get('partial', False)}")
        lines.append(f"- limit_reason: {budgets.get('limit_reason', 'none')}")
        lines.append("")
        return "\n".join(lines)

    async def emit(
        self,
        record: dict[str, Any],
        *,
        force: bool = False,
        error: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not self._should_emit(force=force, error=error):
            return None

        self._maybe_cleanup()

        now = datetime.now(timezone.utc)
        record.setdefault("schema_version", 1)
        record.setdefault("timestamp", now.isoformat())
        record.setdefault("diagnostic_id", str(uuid4()))

        self._redact_query(record)

        day = now.strftime("%Y-%m-%d")
        dir_path = self.base_dir / day
        jsonl_path = dir_path / "retrieval_diagnostics.jsonl"
        md_path = dir_path / f"{record['diagnostic_id']}.md"

        async with self._lock:
            dir_path.mkdir(parents=True, exist_ok=True)
            with open(jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            with open(md_path, "w", encoding="utf-8") as handle:
                handle.write(self._render_markdown(record))

        logger.info(
            "retrieval_diagnostics_emitted",
            diagnostic_id=record.get("diagnostic_id"),
            jsonl_path=str(jsonl_path),
            md_path=str(md_path),
        )

        return {
            "diagnostic_id": record.get("diagnostic_id"),
            "jsonl_path": str(jsonl_path),
            "md_path": str(md_path),
            "date": day,
        }
