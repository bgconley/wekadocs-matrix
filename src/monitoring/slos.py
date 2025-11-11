"""
Phase 7E-4: SLO Monitoring
Service Level Objectives tracking and violation detection

Reference: Canonical Spec L4916-4976, L3513-3528, L5091
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"  # Informational, no action needed
    ALERT = "alert"  # Warning, investigate soon
    PAGE = "page"  # Critical, immediate action required


class SLOType(str, Enum):
    """Types of SLO checks."""

    TARGET = "target"  # Single threshold (lower is better)
    RANGE = "range"  # Min/max range
    ZERO_TOLERANCE = "zero_tolerance"  # Must be exactly zero


@dataclass
class SLODefinition:
    """
    Service Level Objective definition.

    Attributes:
        name: SLO identifier
        type: SLO check type
        target: Target value (for TARGET type)
        unit: Measurement unit
        alert_threshold: Threshold for ALERT level
        page_threshold: Threshold for PAGE level
        min_threshold: Minimum acceptable value (for RANGE type)
        max_threshold: Maximum acceptable value (for RANGE type)
        description: Human-readable description
    """

    name: str
    type: SLOType
    unit: str
    description: str
    target: Optional[float] = None
    alert_threshold: Optional[float] = None
    page_threshold: Optional[float] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None


@dataclass
class SLOViolation:
    """An SLO violation with severity and details."""

    slo_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.slo_name}: {self.message}"


# CRITICAL SLOs - Must be monitored in production
# Reference: Canonical Spec L4916-4941
SLO_DEFINITIONS = {
    "retrieval_p95_latency": SLODefinition(
        name="retrieval_p95_latency",
        type=SLOType.TARGET,
        target=500.0,  # ms - REQUIRED
        unit="ms",
        alert_threshold=600.0,  # Alert at 20% over target
        page_threshold=1000.0,  # Page at 2x target
        description="Retrieval p95 latency must be ≤500ms (Phase 7E requirement)",
    ),
    "ingestion_per_doc": SLODefinition(
        name="ingestion_per_doc",
        type=SLOType.TARGET,
        target=10.0,  # seconds
        unit="s",
        alert_threshold=15.0,  # Alert at 50% over target
        page_threshold=30.0,  # Page at 3x target
        description="Document ingestion should complete in ≤10s",
    ),
    "oversized_chunk_rate": SLODefinition(
        name="oversized_chunk_rate",
        type=SLOType.ZERO_TOLERANCE,
        target=0.0,  # ZERO tolerance
        unit="ratio",
        alert_threshold=0.0,  # Alert on ANY oversized chunk
        page_threshold=0.01,  # Page if >1% oversized
        description="ZERO oversized chunks allowed (>ABSOLUTE_MAX tokens)",
    ),
    "integrity_failure_rate": SLODefinition(
        name="integrity_failure_rate",
        type=SLOType.ZERO_TOLERANCE,
        target=0.0,  # ZERO tolerance
        unit="ratio",
        alert_threshold=0.0,  # Alert on ANY integrity failure
        page_threshold=0.001,  # Page if >0.1% failures
        description="ZERO integrity failures allowed (SHA256 verification)",
    ),
    "expansion_rate": SLODefinition(
        name="expansion_rate",
        type=SLOType.RANGE,
        target=0.25,  # 25% of queries
        unit="ratio",
        min_threshold=0.10,  # Guardrail: too low = not helping
        max_threshold=0.40,  # Guardrail: too high = performance issue
        description="Adjacency expansion rate should be 10-40% (bounded)",
    ),
}


class SLOMonitor:
    """
    Track and alert on SLO violations.

    Usage:
        monitor = SLOMonitor()
        violations = monitor.check_slos(metrics)
        if violations:
            for v in violations:
                logger.warning(str(v))
    """

    def __init__(self, slo_definitions: Optional[Dict[str, SLODefinition]] = None):
        """
        Initialize SLO monitor.

        Args:
            slo_definitions: Custom SLO definitions (defaults to SLO_DEFINITIONS)
        """
        self.slo_definitions = slo_definitions or SLO_DEFINITIONS

    def check_slos(self, metrics: Dict[str, float]) -> List[SLOViolation]:
        """
        Check metrics against SLO definitions.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            List of SLO violations (empty if all SLOs met)
        """
        violations = []

        for slo_name, slo_def in self.slo_definitions.items():
            if slo_name not in metrics:
                # Metric not provided, skip check
                continue

            value = metrics[slo_name]

            # Check based on SLO type
            if slo_def.type == SLOType.RANGE:
                violation = self._check_range_slo(slo_def, value)
            elif slo_def.type == SLOType.ZERO_TOLERANCE:
                violation = self._check_zero_tolerance_slo(slo_def, value)
            else:  # TARGET
                violation = self._check_target_slo(slo_def, value)

            if violation:
                violations.append(violation)

        return violations

    def _check_range_slo(
        self, slo_def: SLODefinition, value: float
    ) -> Optional[SLOViolation]:
        """Check range-based SLO (must be within min/max)."""
        if value < slo_def.min_threshold or value > slo_def.max_threshold:
            # Determine severity
            if value < slo_def.min_threshold:
                level = AlertLevel.ALERT  # Below minimum is warning
                message = (
                    f"{slo_def.name}={value:.3f} below minimum "
                    f"(expected {slo_def.min_threshold}-{slo_def.max_threshold})"
                )
                threshold = slo_def.min_threshold
            else:  # Above maximum
                level = AlertLevel.PAGE  # Above maximum is critical
                message = (
                    f"{slo_def.name}={value:.3f} above maximum "
                    f"(expected {slo_def.min_threshold}-{slo_def.max_threshold})"
                )
                threshold = slo_def.max_threshold

            return SLOViolation(
                slo_name=slo_def.name,
                level=level,
                message=message,
                value=value,
                threshold=threshold,
            )

        return None

    def _check_zero_tolerance_slo(
        self, slo_def: SLODefinition, value: float
    ) -> Optional[SLOViolation]:
        """Check zero-tolerance SLO (must be exactly 0)."""
        if value > 0:
            # Determine severity based on magnitude
            if value >= slo_def.page_threshold:
                level = AlertLevel.PAGE
            else:
                level = AlertLevel.ALERT

            return SLOViolation(
                slo_name=slo_def.name,
                level=level,
                message=f"{slo_def.name}={value:.6f} (ZERO TOLERANCE - must be 0)",
                value=value,
                threshold=0.0,
            )

        return None

    def _check_target_slo(
        self, slo_def: SLODefinition, value: float
    ) -> Optional[SLOViolation]:
        """Check target-based SLO (lower is better)."""
        if value > slo_def.alert_threshold:
            # Determine severity
            if value >= slo_def.page_threshold:
                level = AlertLevel.PAGE
            else:
                level = AlertLevel.ALERT

            return SLOViolation(
                slo_name=slo_def.name,
                level=level,
                message=(
                    f"{slo_def.name}={value:.2f}{slo_def.unit} "
                    f"(target={slo_def.target}{slo_def.unit}, "
                    f"threshold={slo_def.alert_threshold if level == AlertLevel.ALERT else slo_def.page_threshold}{slo_def.unit})"
                ),
                value=value,
                threshold=(
                    slo_def.alert_threshold
                    if level == AlertLevel.ALERT
                    else slo_def.page_threshold
                ),
            )

        return None

    def get_slo_status(self, metrics: Dict[str, float]) -> Dict[str, any]:
        """
        Get comprehensive SLO status report.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            Dictionary with SLO status, violations, and summary
        """
        violations = self.check_slos(metrics)

        # Count by severity
        alerts = sum(1 for v in violations if v.level == AlertLevel.ALERT)
        pages = sum(1 for v in violations if v.level == AlertLevel.PAGE)

        # Overall health
        if pages > 0:
            health = "unhealthy"
        elif alerts > 0:
            health = "degraded"
        else:
            health = "healthy"

        return {
            "health": health,
            "violations": [
                {
                    "slo": v.slo_name,
                    "level": v.level.value,
                    "message": v.message,
                    "value": v.value,
                    "threshold": v.threshold,
                    "timestamp": v.timestamp.isoformat(),
                }
                for v in violations
            ],
            "summary": {
                "total_slos": len(self.slo_definitions),
                "checked": len(
                    [k for k in self.slo_definitions.keys() if k in metrics]
                ),
                "violations": len(violations),
                "alerts": alerts,
                "pages": pages,
            },
        }


def check_slos_and_log(
    metrics: Dict[str, float], logger_instance: logging.Logger = None
) -> List[SLOViolation]:
    """
    Convenience function to check SLOs and log violations.

    Args:
        metrics: Dictionary of metric name -> value
        logger_instance: Logger to use (defaults to module logger)

    Returns:
        List of violations
    """
    log = logger_instance or logger
    monitor = SLOMonitor()
    violations = monitor.check_slos(metrics)

    if violations:
        log.warning(f"SLO violations detected: {len(violations)} issue(s)")
        for v in violations:
            if v.level == AlertLevel.PAGE:
                log.error(str(v))
            else:
                log.warning(str(v))
    else:
        log.info("✅ All SLOs met")

    return violations
