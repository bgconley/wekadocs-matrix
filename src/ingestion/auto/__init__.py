# =============================================================================
# @status: ACTIVE
# @reason: Package init for auto-ingestion. Eager imports of Orchestrator,
#          BackPressureMonitor, and ProgressTracker were removed (Phase B cleanup)
#          to eliminate phantom loading of 8 modules at worker startup.
#          Consumers needing those classes should import directly from submodules.
# =============================================================================
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

__version__ = "0.1.0"
