"""
Delta cache used to avoid resending graph elements across paged MCP tool calls.

Docs reference: docs/cdx-outputs/retrieval_fix.json (session_state config).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Set


@dataclass
class SessionEntry:
    node_ids: Set[str] = field(default_factory=set)
    edge_keys: Set[str] = field(default_factory=set)
    updated_at: float = field(default_factory=lambda: time.time())

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionDeltaCache:
    """
    Simple in-memory cache keyed by session_id. Not persistent but sufficient for
    the MCP server process lifetime.
    """

    def __init__(self, ttl_seconds: int = 900):
        self._ttl = ttl_seconds
        self._sessions: Dict[str, SessionEntry] = {}

    def _purge(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, entry in self._sessions.items()
            if now - entry.updated_at > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def _entry(self, session_id: str) -> SessionEntry:
        self._purge()
        entry = self._sessions.get(session_id)
        if not entry:
            entry = SessionEntry()
            self._sessions[session_id] = entry
        entry.touch()
        return entry

    def filter_nodes(
        self, session_id: str, nodes: Iterable[dict]
    ) -> tuple[list[dict], int]:
        entry = self._entry(session_id)
        filtered = []
        suppressed = 0
        for node in nodes:
            node_id = node.get("id")
            if node_id and node_id in entry.node_ids:
                suppressed += 1
                continue
            if node_id:
                entry.node_ids.add(node_id)
            filtered.append(node)
        return filtered, suppressed

    def filter_edges(
        self, session_id: str, edges: Iterable[dict]
    ) -> tuple[list[dict], int]:
        entry = self._entry(session_id)
        filtered = []
        suppressed = 0
        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            etype = edge.get("type")
            key = f"{src}->{dst}:{etype}"
            if key in entry.edge_keys:
                suppressed += 1
                continue
            entry.edge_keys.add(key)
            filtered.append(edge)
        return filtered, suppressed
