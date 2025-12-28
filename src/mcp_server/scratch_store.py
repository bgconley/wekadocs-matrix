from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ScratchEntry:
    payload: Dict[str, Any]
    size_bytes: int
    created_at: float
    last_access: float


class ScratchStore:
    def __init__(
        self,
        *,
        max_bytes: int,
        ttl_seconds: int,
        logger=None,
    ) -> None:
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._logger = logger
        self._entries: "OrderedDict[Tuple[str, str], ScratchEntry]" = OrderedDict()
        self._bytes = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def build_uri(session_id: str, passage_id: str) -> str:
        return f"wekadocs://scratch/{session_id}/{passage_id}"

    async def put(
        self, session_id: str, passage_id: str, payload: Dict[str, Any]
    ) -> int:
        text_value = payload.get("text") or ""
        size_bytes = int(payload.get("size_bytes") or len(text_value.encode("utf-8")))
        now = time.time()

        async with self._lock:
            self._evict_expired_locked(now)

            if size_bytes > self._max_bytes:
                truncated = text_value.encode("utf-8")[: self._max_bytes].decode(
                    "utf-8", errors="ignore"
                )
                payload["text"] = truncated
                payload["truncated"] = True
                size_bytes = len(truncated.encode("utf-8"))

            payload["size_bytes"] = size_bytes
            entry = ScratchEntry(
                payload=payload,
                size_bytes=size_bytes,
                created_at=now,
                last_access=now,
            )
            self._insert_locked((session_id, passage_id), entry)
            evicted = self._evict_if_needed_locked()

        if evicted and self._logger:
            self._logger.warning(
                "scratch_store_evicted",
                evicted=evicted,
                current_bytes=self._bytes,
                max_bytes=self._max_bytes,
            )

        return size_bytes

    async def get(self, session_id: str, passage_id: str) -> Optional[Dict[str, Any]]:
        key = (session_id, passage_id)
        now = time.time()
        async with self._lock:
            self._evict_expired_locked(now)
            entry = self._entries.get(key)
            if not entry:
                return None
            entry.last_access = now
            self._entries.move_to_end(key)
            return dict(entry.payload)

    async def update(
        self, session_id: str, passage_id: str, updates: Dict[str, Any]
    ) -> bool:
        key = (session_id, passage_id)
        now = time.time()
        async with self._lock:
            self._evict_expired_locked(now)
            entry = self._entries.get(key)
            if not entry:
                return False
            entry.payload.update(updates)
            entry.last_access = now
            self._entries.move_to_end(key)
            return True

    def _insert_locked(self, key: Tuple[str, str], entry: ScratchEntry) -> None:
        if key in self._entries:
            prior = self._entries.pop(key)
            self._bytes -= prior.size_bytes
        self._entries[key] = entry
        self._bytes += entry.size_bytes

    def _evict_expired_locked(self, now: float) -> int:
        if self._ttl_seconds <= 0:
            return 0
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if (now - entry.last_access) > self._ttl_seconds
        ]
        for key in expired_keys:
            self._remove_locked(key)
        return len(expired_keys)

    def _evict_if_needed_locked(self) -> int:
        evicted = 0
        while self._bytes > self._max_bytes and self._entries:
            key, entry = self._entries.popitem(last=False)
            self._bytes -= entry.size_bytes
            evicted += 1
        return evicted

    def _remove_locked(self, key: Tuple[str, str]) -> None:
        entry = self._entries.pop(key, None)
        if entry:
            self._bytes -= entry.size_bytes
