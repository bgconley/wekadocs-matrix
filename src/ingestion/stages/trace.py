"""Lightweight trace events for atomic ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class IngestionTraceEvent:
    """One noteworthy ingestion-stage event."""

    stage: str
    kind: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class IngestionTrace:
    """Accumulates ingestion-stage events for result diagnostics."""

    def __init__(self) -> None:
        self.started_at = datetime.utcnow()
        self._events: List[IngestionTraceEvent] = []

    def add_event(
        self,
        *,
        stage: str,
        kind: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._events.append(
            IngestionTraceEvent(
                stage=stage,
                kind=kind,
                message=message,
                data=data or {},
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "events": [event.to_dict() for event in self._events],
        }
