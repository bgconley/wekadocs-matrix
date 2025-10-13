# Implements Phase 1, Task 1.4 (Security layer)
# See: /docs/spec.md §6 (Security), §7 (Observability)
# See: /docs/implementation-plan.md → Task 1.4 DoD & Tests
# Audit logging with correlation IDs

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Request

from ..config import Config, get_config
from ..observability import get_correlation_id, get_logger

logger = get_logger(__name__)


class AuditLogger:
    """Audit logger for security-relevant events"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.enabled = self.config.audit.enabled
        self.log_params = self.config.audit.log_params
        self.log_results = self.config.audit.log_results

    def _hash_data(self, data: Any) -> str:
        """Hash sensitive data for audit log"""
        if data is None:
            return "null"
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def log_request(
        self,
        request: Request,
        endpoint: str,
        method: str,
        client_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an API request for audit purposes.

        Args:
            request: FastAPI request
            endpoint: Endpoint path
            method: HTTP method
            client_id: Client identifier (from JWT or IP)
            params: Request parameters
        """
        if not self.enabled:
            return

        correlation_id = get_correlation_id()

        audit_entry = {
            "event_type": "api_request",
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "endpoint": endpoint,
            "method": method,
            "client_id": client_id or "anonymous",
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        }

        if self.log_params and params:
            # Hash params instead of logging raw values to avoid leaking sensitive data
            audit_entry["params_hash"] = self._hash_data(params)

        logger.info("Audit: API request", **audit_entry)

    def log_response(
        self,
        request: Request,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log an API response for audit purposes.

        Args:
            request: FastAPI request
            endpoint: Endpoint path
            method: HTTP method
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            result: Response result (optional)
            error: Error message if failed (optional)
        """
        if not self.enabled:
            return

        correlation_id = get_correlation_id()

        audit_entry = {
            "event_type": "api_response",
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "success": status_code < 400,
        }

        if error:
            audit_entry["error"] = error

        if self.log_results and result is not None:
            # Hash result instead of logging raw values
            audit_entry["result_hash"] = self._hash_data(result)

        logger.info("Audit: API response", **audit_entry)

    def log_auth_event(
        self,
        event_type: str,
        client_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an authentication/authorization event.

        Args:
            event_type: Type of auth event (login, logout, token_refresh, etc.)
            client_id: Client identifier
            success: Whether the event was successful
            details: Additional event details
        """
        if not self.enabled:
            return

        correlation_id = get_correlation_id()

        audit_entry = {
            "event_type": f"auth_{event_type}",
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "client_id": client_id or "anonymous",
            "success": success,
        }

        if details:
            audit_entry.update(details)

        logger.info("Audit: Auth event", **audit_entry)

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a security-relevant event.

        Args:
            event_type: Type of security event
            severity: Event severity (low, medium, high, critical)
            description: Event description
            details: Additional event details
        """
        if not self.enabled:
            return

        correlation_id = get_correlation_id()

        audit_entry = {
            "event_type": f"security_{event_type}",
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "severity": severity,
            "description": description,
        }

        if details:
            audit_entry.update(details)

        logger.warning("Audit: Security event", **audit_entry)


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
