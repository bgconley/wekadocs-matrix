# =============================================================================
# @status: ACTIVE
# @called-by: mcp_app.py (kb_retrieve_evidence, graph.expand, kb.read_excerpt)
# =============================================================================
"""
Retrieval trace system: full-pipeline observability for the evidence pack path.

Every kb.retrieve_evidence call produces a human-readable trace file showing
exactly what happened at each decision point. Follow-up tool calls (graph.expand,
kb.read_excerpt) append to the same trace via session correlation.

Trace access:
  - File: logs/retrieval_traces/<timestamp>_<trace_id>.txt
  - JSON sidecar: logs/retrieval_traces/<timestamp>_<trace_id>.json
  - MCP resource: wekadocs://traces/latest, wekadocs://traces/<trace_id>
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACE_DIR = os.getenv("MCP_RETRIEVAL_TRACE_DIR", "logs/retrieval_traces")
TRACE_RETENTION_HOURS = int(os.getenv("MCP_RETRIEVAL_TRACE_RETENTION_HOURS", "72"))
_DIVIDER = "=" * 65
_SECTION = "-" * 50


@dataclass
class TraceCandidate:
    """Compact representation of a retrieval candidate for trace output."""

    chunk_id: str
    score: float
    heading: str = ""
    parent_path: Optional[str] = None
    text_preview: str = ""  # first 150 chars


@dataclass
class TraceQuote:
    """Quote in the evidence pack output."""

    rank: int
    confidence: float
    doc_tag: Optional[str] = None
    parent_path: Optional[str] = None
    source: str = ""
    text: str = ""  # full quote text (max 500 chars)


@dataclass
class TraceFollowup:
    """A follow-up tool call correlated to this trace."""

    tool_name: str
    timestamp: str
    arguments_summary: str
    result_summary: str


class RetrievalTraceBuilder:
    """Accumulates data at each pipeline stage, then formats a trace."""

    def __init__(self, trace_id: str, session_id: str) -> None:
        self.trace_id = trace_id
        self.session_id = session_id
        self.timestamp = datetime.utcnow()
        self._start_time = time.time()

        # Section data (populated by record_* methods)
        self._query: Optional[Dict[str, Any]] = None
        self._candidates: Dict[str, List[TraceCandidate]] = {}
        self._signal_pool: Optional[Dict[str, Any]] = None
        self._reranker: Optional[Dict[str, Any]] = None
        self._related_to: Optional[Dict[str, Any]] = None
        self._graph_enrichment: Optional[Dict[str, Any]] = None
        self._evidence_pack: Optional[Dict[str, Any]] = None
        self._followups: List[TraceFollowup] = []
        self._appendix_chunks: List[Dict[str, Any]] = []
        self._stage_snapshots: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._colbert: Optional[Dict[str, Any]] = None

    # ── Record methods ────────────────────────────────────────────────

    def record_query(
        self,
        client_query: str,
        reformulated: str,
        method: str,
        latency_ms: float,
        dual_query_active: bool,
    ) -> None:
        self._query = {
            "client_query": client_query,
            "reformulated": reformulated,
            "method": method,
            "latency_ms": round(latency_ms, 1),
            "dual_query_active": dual_query_active,
            "lexical_query": client_query if dual_query_active else reformulated,
        }

    def record_candidates(
        self, signal_name: str, candidates: List[TraceCandidate]
    ) -> None:
        self._candidates[signal_name] = candidates

    def record_signal_pool(
        self,
        enabled: bool,
        pool_size: int,
        slot_fills: Dict[str, int],
        degraded: bool,
    ) -> None:
        self._signal_pool = {
            "enabled": enabled,
            "pool_size": pool_size,
            "slot_fills": slot_fills,
            "degraded": degraded,
        }

    def record_reranker(
        self,
        model: str,
        instruction: Optional[str],
        input_count: int,
        output_count: int,
        latency_ms: float,
        top_results: List[Dict[str, Any]],
    ) -> None:
        self._reranker = {
            "model": model,
            "instruction": (instruction or "")[:100]
            + ("..." if instruction and len(instruction) > 100 else ""),
            "input_count": input_count,
            "output_count": output_count,
            "latency_ms": round(latency_ms, 1),
            "top_results": top_results,
        }

    def record_related_to_expansion(
        self,
        seed_docs: int,
        related_docs_found: int,
        chunks_added: int,
        avg_edge_score: float,
        blended_count: int,
        blend_lambda: float,
    ) -> None:
        self._related_to = {
            "seed_docs": seed_docs,
            "related_docs_found": related_docs_found,
            "chunks_added": chunks_added,
            "avg_edge_score": round(avg_edge_score, 4),
            "blended_count": blended_count,
            "blend_lambda": blend_lambda,
        }

    def record_graph_enrichment(
        self,
        seeds: int,
        neighbors_added: int,
        neighbor_details: List[Dict[str, str]],
    ) -> None:
        self._graph_enrichment = {
            "seeds": seeds,
            "neighbors_added": neighbors_added,
            "neighbor_details": neighbor_details,
        }

    def record_evidence_pack(
        self,
        quotes: List[TraceQuote],
        coverage: Dict[str, Any],
    ) -> None:
        self._evidence_pack = {
            "quotes": quotes,
            "coverage": coverage,
        }

    def record_appendix_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store full text of top reranked candidates for the appendix."""
        self._appendix_chunks = chunks[:20]

    def record_followup_call(
        self,
        tool_name: str,
        arguments_summary: str,
        result_summary: str,
    ) -> None:
        self._followups.append(
            TraceFollowup(
                tool_name=tool_name,
                timestamp=datetime.utcnow().isoformat() + "Z",
                arguments_summary=arguments_summary,
                result_summary=result_summary,
            )
        )

    def record_stage_snapshots(
        self, snapshots: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Record per-stage candidate snapshots for pipeline observability."""
        self._stage_snapshots = snapshots

    def record_colbert(
        self,
        applied: bool,
        runtime_available: bool,
        query_embedding_ok: bool,
        rank_deltas: Optional[List[int]] = None,
        candidates: int = 0,
        hydrated: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        self._colbert = {
            "applied": applied,
            "runtime_available": runtime_available,
            "query_embedding_ok": query_embedding_ok,
            "rank_deltas_top10": rank_deltas or [],
            "candidates": candidates,
            "hydrated": hydrated,
            "latency_ms": round(latency_ms, 1),
        }

    # ── Format: Human-readable text ──────────────────────────────────

    def format(self) -> str:
        elapsed = (time.time() - self._start_time) * 1000
        lines: List[str] = []
        lines.append("")
        lines.append(_DIVIDER)
        lines.append(
            f"  RETRIEVAL TRACE  |  {self.timestamp.isoformat()}Z  |  {self.trace_id[:12]}"
        )
        lines.append(f"  Session: {self.session_id[:16]}  |  Total: {elapsed:.0f}ms")
        lines.append(_DIVIDER)

        # Section 1: Query
        lines.append("")
        lines.append(f"-- 1. QUERY {_SECTION}")
        if self._query:
            q = self._query
            lines.append(f"  Client query:      {q['client_query'][:200]}")
            lines.append(f"  Reformulated:      {q['reformulated'][:200]}")
            lines.append(f"  Method:            {q['method']} ({q['latency_ms']}ms)")
            lines.append(
                f"  Dual-query:        {'active' if q['dual_query_active'] else 'inactive'}"
            )
            if q["dual_query_active"]:
                lines.append(f"  Lexical query:     {q['lexical_query'][:200]}")
        else:
            lines.append("  (not recorded)")

        # Section 2: Candidates by signal
        lines.append("")
        lines.append(f"-- 2. CANDIDATES BY SIGNAL {_SECTION}")
        if self._candidates:
            for signal, candidates in self._candidates.items():
                total = len(candidates)
                shown = min(5, total)
                lines.append(f"  {signal} (top {shown} of {total}):")
                for c in candidates[:5]:
                    path = f"  {c.parent_path}" if c.parent_path else ""
                    lines.append(
                        f"    {c.score:7.3f}  {c.chunk_id[:16]}  {c.heading[:40]}{path}"
                    )
        else:
            lines.append("  (per-signal breakdown not available)")

        # Section 3: Signal pool
        lines.append("")
        lines.append(f"-- 3. SIGNAL POOL {_SECTION}")
        if self._signal_pool:
            sp = self._signal_pool
            lines.append(f"  Enabled: {sp['enabled']}    Pool size: {sp['pool_size']}")
            if sp["slot_fills"]:
                lines.append(f"  {'Slot':<22} {'Filled':>6}")
                lines.append(f"  {'-'*22} {'-'*6}")
                for slot, count in sp["slot_fills"].items():
                    lines.append(f"  {slot:<22} {count:>6}")
            lines.append(f"  Degraded: {sp['degraded']}")
        else:
            lines.append("  (signal pool not active)")

        # Section 4: Reranker
        lines.append("")
        lines.append(f"-- 4. RERANKER {_SECTION}")
        if self._reranker:
            r = self._reranker
            lines.append(f"  Model:       {r['model']}")
            if r["instruction"]:
                lines.append(f"  Instruction: {r['instruction']}")
            lines.append(
                f"  Candidates:  {r['input_count']} -> {r['output_count']} returned"
            )
            lines.append(f"  Latency:     {r['latency_ms']}ms")
            if r["top_results"]:
                lines.append("")
                lines.append("  Top results after rerank:")
                for item in r["top_results"][:10]:
                    orig = item.get("original_rank", "?")
                    new = item.get("rank", "?")
                    lines.append(
                        f"    {item.get('score', 0):7.3f}  [{orig}->{new}]  "
                        f"{item.get('chunk_id', '')[:16]}  {item.get('heading', '')[:50]}"
                    )
        else:
            lines.append("  (reranker not applied)")

        # Section 5: RELATED_TO expansion
        lines.append("")
        lines.append(f"-- 5. RELATED_TO EXPANSION {_SECTION}")
        if self._related_to:
            rt = self._related_to
            lines.append(
                f"  Seed docs: {rt['seed_docs']}    Related docs found: {rt['related_docs_found']}"
            )
            lines.append(
                f"  Chunks added: {rt['chunks_added']}    Avg edge score: {rt['avg_edge_score']}"
            )
            lines.append(
                f"  Blended: {rt['blended_count']} chunks    Lambda: {rt['blend_lambda']}"
            )
        else:
            lines.append("  (RELATED_TO expansion not applied)")

        # Section 6: Graph enrichment
        lines.append("")
        lines.append(f"-- 6. GRAPH ENRICHMENT {_SECTION}")
        if self._graph_enrichment:
            g = self._graph_enrichment
            lines.append(
                f"  Seeds: {g['seeds']}    Neighbors added: {g['neighbors_added']}"
            )
            for detail in g["neighbor_details"][:8]:
                lines.append(
                    f"    {detail.get('chunk_id', '')[:16]}  ({detail.get('rel', 'neighbor')})  {detail.get('heading', '')[:40]}"
                )
        else:
            lines.append("  (graph enrichment not applied)")

        # Section 7: Evidence pack
        lines.append("")
        lines.append(f"-- 7. EVIDENCE PACK {_SECTION}")
        if self._evidence_pack:
            ep = self._evidence_pack
            cov = ep["coverage"]
            quotes = ep["quotes"]
            lines.append(
                f"  Quotes: {len(quotes)}    Retrieval depth: {cov.get('retrieval_depth', '?')}"
            )
            lines.append("  Coverage:")
            lines.append(
                f"    docs_searched: {cov.get('documents_searched', '?')}   docs_with_evidence: {cov.get('documents_with_evidence', '?')}"
            )
            lines.append(
                f"    reranker: {'yes' if cov.get('reranker_applied') else 'no'}    signal_pool: {'yes' if cov.get('signal_pool_active') else 'no'}    graph: {'yes' if cov.get('graph_expansion_applied') else 'no'}"
            )
            lines.append("")
            for q in quotes:
                lines.append(
                    f"  Quote {q.rank}  confidence={q.confidence:.2f}  source={q.source}  rank={q.rank}"
                )
                if q.doc_tag:
                    lines.append(f"    doc:  {q.doc_tag}")
                if q.parent_path:
                    lines.append(f"    path: {q.parent_path}")
                lines.append(
                    f"    text: {q.text[:200]}{'...' if len(q.text) > 200 else ''}"
                )
                lines.append("")
        else:
            lines.append("  (not recorded)")

        # Section 8: Follow-up calls
        lines.append(f"-- 8. FOLLOW-UP CALLS {_SECTION}")
        if self._followups:
            for fu in self._followups:
                lines.append(f"  [{fu.timestamp}] {fu.tool_name}")
                lines.append(f"    args: {fu.arguments_summary[:150]}")
                lines.append(f"    result: {fu.result_summary[:150]}")
        else:
            lines.append("  (none)")

        # Section 9: Stage snapshots
        lines.append("")
        lines.append(f"-- 9. STAGE SNAPSHOTS {_SECTION}")
        if self._stage_snapshots:
            for stage_name, entries in self._stage_snapshots.items():
                shown = entries[:10]
                lines.append(f"  {stage_name} (top {len(shown)} of {len(entries)}):")
                for entry in shown:
                    rs = entry.get("rerank_score")
                    rs_str = f"  rerank={rs:.5f}" if rs is not None else ""
                    lines.append(
                        f"    {entry.get('fused_score', 0):9.5f}{rs_str}  "
                        f"{entry.get('chunk_id', '')[:16]}  {entry.get('doc_tag', '') or ''}"
                    )
        else:
            lines.append("  (no stage snapshots recorded)")

        # Section 10: ColBERT
        lines.append("")
        lines.append(f"-- 10. COLBERT {_SECTION}")
        if self._colbert:
            cb = self._colbert
            lines.append(f"  Applied:            {cb['applied']}")
            lines.append(f"  Runtime available:  {cb['runtime_available']}")
            lines.append(f"  Query embedding OK: {cb['query_embedding_ok']}")
            lines.append(f"  Candidates:         {cb['candidates']}")
            lines.append(f"  Hydrated:           {cb['hydrated']}")
            lines.append(f"  Latency:            {cb['latency_ms']}ms")
            if cb["rank_deltas_top10"]:
                lines.append(f"  Rank deltas (top10): {cb['rank_deltas_top10']}")
        else:
            lines.append("  (ColBERT not recorded)")

        # Section 11: Full text appendix
        if self._appendix_chunks:
            lines.append("")
            lines.append(
                f"-- 11. FULL TEXT APPENDIX (top {len(self._appendix_chunks)} reranked) {_SECTION}"
            )
            for idx, chunk in enumerate(self._appendix_chunks):
                lines.append("")
                lines.append(f"  [{idx+1}] {chunk.get('chunk_id', '')}")
                lines.append(
                    f"  score: {chunk.get('rerank_score', chunk.get('fused_score', '?'))}"
                )
                if chunk.get("doc_tag"):
                    lines.append(f"  doc:   {chunk['doc_tag']}")
                if chunk.get("parent_path_norm"):
                    lines.append(f"  path:  {chunk['parent_path_norm']}")
                if chunk.get("heading"):
                    lines.append(f"  head:  {chunk['heading']}")
                text = chunk.get("text", "")
                lines.append("  text:")
                # Indent full text for readability
                for text_line in text.split("\n"):
                    lines.append(f"    {text_line}")

        lines.append("")
        lines.append(_DIVIDER)
        lines.append("")
        return "\n".join(lines)

    # ── Format: Structured JSON ──────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "elapsed_ms": round((time.time() - self._start_time) * 1000, 1),
            "query": self._query,
            "candidates": {
                signal: [
                    {"chunk_id": c.chunk_id, "score": c.score, "heading": c.heading}
                    for c in candidates
                ]
                for signal, candidates in self._candidates.items()
            },
            "signal_pool": self._signal_pool,
            "reranker": self._reranker,
            "related_to": self._related_to,
            "graph_enrichment": self._graph_enrichment,
            "evidence_pack": {
                "quotes": (
                    [
                        {
                            "rank": q.rank,
                            "confidence": q.confidence,
                            "doc_tag": q.doc_tag,
                            "parent_path": q.parent_path,
                            "source": q.source,
                            "text": q.text,
                        }
                        for q in self._evidence_pack["quotes"]
                    ]
                    if self._evidence_pack
                    else []
                ),
                "coverage": (
                    self._evidence_pack["coverage"] if self._evidence_pack else {}
                ),
            },
            "followups": [
                {
                    "tool": f.tool_name,
                    "timestamp": f.timestamp,
                    "args": f.arguments_summary,
                    "result": f.result_summary,
                }
                for f in self._followups
            ],
            "appendix_chunk_count": len(self._appendix_chunks),
            "stage_snapshots": self._stage_snapshots or {},
            "colbert": self._colbert,
        }


# ── File writer ──────────────────────────────────────────────────────


def write_trace(trace: RetrievalTraceBuilder) -> str:
    """Write trace to file. Returns the file path."""
    os.makedirs(TRACE_DIR, exist_ok=True)

    # Cleanup old traces
    _cleanup_old_traces()

    filename = f"{trace.timestamp:%Y%m%d_%H%M%S}_{trace.trace_id[:8]}"
    txt_path = os.path.join(TRACE_DIR, f"{filename}.txt")
    json_path = os.path.join(TRACE_DIR, f"{filename}.json")

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(trace.format())
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)
        logger.info(
            f"retrieval_trace_written: path={txt_path}, trace_id={trace.trace_id[:12]}"
        )
    except Exception as exc:
        logger.warning(f"retrieval_trace_write_failed: {exc}")

    return txt_path


def _cleanup_old_traces() -> None:
    """Remove trace files older than TRACE_RETENTION_HOURS."""
    if not os.path.isdir(TRACE_DIR):
        return
    cutoff = time.time() - (TRACE_RETENTION_HOURS * 3600)
    try:
        for fname in os.listdir(TRACE_DIR):
            fpath = os.path.join(TRACE_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
    except Exception:
        pass  # best-effort cleanup


def get_latest_trace_path(session_id: Optional[str] = None) -> Optional[str]:
    """Return path to the most recent trace file."""
    if not os.path.isdir(TRACE_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(TRACE_DIR) if f.endswith(".txt")],
        reverse=True,
    )
    return os.path.join(TRACE_DIR, files[0]) if files else None


def read_trace_file(trace_id: str) -> Optional[str]:
    """Read a trace file by trace_id prefix."""
    if not os.path.isdir(TRACE_DIR):
        return None
    for fname in os.listdir(TRACE_DIR):
        if trace_id[:8] in fname and fname.endswith(".txt"):
            path = os.path.join(TRACE_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


# ── Session correlation ──────────────────────────────────────────────

# Active traces keyed by session_id for follow-up call correlation
_ACTIVE_TRACES: Dict[str, RetrievalTraceBuilder] = {}


def set_active_trace(session_id: str, trace: RetrievalTraceBuilder) -> None:
    _ACTIVE_TRACES[session_id] = trace


def get_active_trace(session_id: str) -> Optional[RetrievalTraceBuilder]:
    return _ACTIVE_TRACES.get(session_id)


def append_followup_and_write(
    session_id: str,
    tool_name: str,
    arguments_summary: str,
    result_summary: str,
) -> None:
    """Append a follow-up call to the active trace and re-write the file."""
    trace = _ACTIVE_TRACES.get(session_id)
    if trace:
        trace.record_followup_call(tool_name, arguments_summary, result_summary)
        write_trace(trace)
