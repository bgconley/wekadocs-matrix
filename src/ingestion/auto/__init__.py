"""
Phase 6: Auto-Ingestion Layer

Automated document ingestion with watchers, resumable jobs, progress tracking,
and verification reports.

Architecture:
- Watchers monitor FS/S3/HTTP for new documents
- Orchestrator runs resumable FSM through ingestion stages
- Progress events stream to Redis for CLI consumption
- Verification checks graph/vector alignment and sample queries
- Reports generated per job + phase artifacts

See: /docs/app-spec-phase6.md
See: /docs/implementation-plan-phase-6.md
"""

from .backpressure import BackPressureMonitor
from .orchestrator import JobState, Orchestrator
from .progress import JobStage, ProgressEvent, ProgressReader, ProgressTracker

__version__ = "0.1.0"

__all__ = [
    "BackPressureMonitor",
    "JobStage",
    "JobState",
    "Orchestrator",
    "ProgressEvent",
    "ProgressReader",
    "ProgressTracker",
]
