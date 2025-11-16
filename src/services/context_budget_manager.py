"""
ContextBudgetManager enforces the token and byte guardrails described in
docs/cdx-outputs/retrieval_fix.json.

It provides phase-aware accounting (seeds/neighbors/snippets/instructions) and
throws BudgetExceeded with a limit_reason so MCP handlers can return graceful
partials rather than blindly truncating responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from src.providers.tokenizer_service import TokenizerService
except Exception:  # pragma: no cover - tokenizer may not be available in tests
    TokenizerService = None  # type: ignore


DEFAULT_PHASE_ALLOCATION = {
    "seeds": 0.15,
    "neighbors": 0.35,
    "snippets": 0.30,
    "instructions": 0.20,
}


class BudgetExceeded(RuntimeError):
    """Raised when a budget (token or byte) would be exceeded."""

    def __init__(self, limit_reason: str, message: str, usage: Dict[str, int]):
        super().__init__(message)
        self.limit_reason = limit_reason
        self.usage = usage


@dataclass
class PhaseUsage:
    tokens: int = 0
    bytes: int = 0


@dataclass
class ContextBudgetManager:
    token_budget: int
    byte_budget: int
    phase_allocation: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_ALLOCATION)
    )
    tokenizer: Optional["TokenizerService"] = None

    def __post_init__(self) -> None:
        if self.token_budget <= 0 or self.byte_budget <= 0:
            raise ValueError("Budgets must be positive integers")

        if not self.tokenizer and TokenizerService is not None:
            try:
                self.tokenizer = TokenizerService()
            except Exception:
                # Fallback to heuristic token estimation
                self.tokenizer = None

        self._phase_usage: Dict[str, PhaseUsage] = {
            phase: PhaseUsage() for phase in self.phase_allocation
        }
        self._tokens_used = 0
        self._bytes_used = 0

    @property
    def usage(self) -> Dict[str, int]:
        return {
            "tokens": self._tokens_used,
            "bytes": self._bytes_used,
        }

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            estimator = getattr(self.tokenizer, "estimate_tokens", None)
            if callable(estimator):
                return estimator(text)
            counter = getattr(self.tokenizer, "count_tokens", None)
            if callable(counter):
                return counter(text)
        # Rough heuristic: 4 characters per token
        return max(1, len(text) // 4)

    def estimate_tokens(self, payload: str | bytes) -> int:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="ignore")
        return self._estimate_tokens(payload)

    def can_consume(self, tokens: int, bytes_: int, phase: str) -> bool:
        if phase not in self._phase_usage:
            raise ValueError(f"Unknown budget phase '{phase}'")

        projected_tokens = self._tokens_used + tokens
        projected_bytes = self._bytes_used + bytes_
        if projected_tokens > self.token_budget or projected_bytes > self.byte_budget:
            return False

        allocation = self.phase_allocation.get(phase, 0)
        phase_token_cap = int(self.token_budget * allocation)
        phase_byte_cap = int(self.byte_budget * allocation)
        phase_usage = self._phase_usage[phase]

        if phase_token_cap and phase_usage.tokens + tokens > phase_token_cap:
            return False
        if phase_byte_cap and phase_usage.bytes + bytes_ > phase_byte_cap:
            return False
        return True

    def consume(self, tokens: int, bytes_: int, phase: str) -> None:
        if not self.can_consume(tokens, bytes_, phase):
            limit_reason = "token_cap"
            if self._bytes_used + bytes_ > self.byte_budget:
                limit_reason = "byte_cap"
            raise BudgetExceeded(
                limit_reason=limit_reason,
                message=(
                    f"Budget exhausted (reason={limit_reason}, "
                    f"phase={phase}, tokens={tokens}, bytes={bytes_})"
                ),
                usage=self.usage,
            )

        self._tokens_used += tokens
        self._bytes_used += bytes_
        phase_usage = self._phase_usage[phase]
        phase_usage.tokens += tokens
        phase_usage.bytes += bytes_
