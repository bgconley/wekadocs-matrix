"""Async client pool for the OpenAI-compatible Qwen3.6-27B VLM endpoints.

Responsibilities:
  * Probe ``/v1/models`` to resolve the served model id at runtime (never hardcode).
  * Issue a single chat/completions request to a NAMED endpoint (routing and
    retries live in ``convert.py`` so a failed page can retry on the other host).
  * Disable thinking (`chat_template_kwargs={"enable_thinking": false}`) -- verified
    to suppress reasoning tokens on both SGLang and vLLM.
  * Parse ``choices[0].message.content`` (agnostic to the engines' differing
    ``reasoning`` vs ``reasoning_content`` fields) and surface token usage.
  * Accumulate per-endpoint stats for the end-of-run report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from .config import Config, EndpointConfig
from .log import get_logger

logger = get_logger("vlm")

# 1x1 transparent PNG for health probes.
_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class VLMError(RuntimeError):
    """Base for all VLM call failures (all are retryable unless noted)."""


class VLMHTTPError(VLMError):
    def __init__(
        self,
        status: int,
        body: str,
        *,
        retryable: bool,
        retry_after_s: Optional[float] = None,
    ):
        self.status = status
        self.body = body[:500]
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        super().__init__(f"HTTP {status}: {self.body}")


def _http_retryable(status: int) -> bool:
    if status == 429 or status in {408, 409, 425}:
        return True
    if status >= 500:
        return True
    return False


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())


class VLMTimeout(VLMError):
    pass


class VLMBadResponse(VLMError):
    pass


@dataclass
class ChatResult:
    content: str
    endpoint: str
    finish_reason: Optional[str]
    image_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_s: float

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class EndpointStats:
    requests: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class _Endpoint:
    cfg: EndpointConfig
    client: httpx.AsyncClient
    stats: EndpointStats = field(default_factory=EndpointStats)


class VLMPool:
    """Manages one httpx client per endpoint. Use as an async context manager."""

    def __init__(self, config: Config):
        self.config = config
        self._endpoints: dict[str, _Endpoint] = {}
        self.model_id: Optional[str] = config.model_id

    async def __aenter__(self) -> "VLMPool":
        for cfg in self.config.active_endpoints:
            limits = httpx.Limits(
                max_connections=cfg.inflight + 4,
                max_keepalive_connections=cfg.inflight + 4,
            )
            timeout = httpx.Timeout(self.config.convert.timeout_s, connect=15.0)
            client = httpx.AsyncClient(
                headers=cfg.headers(), limits=limits, timeout=timeout
            )
            self._endpoints[cfg.name] = _Endpoint(cfg=cfg, client=client)
        return self

    async def __aexit__(self, *exc) -> None:
        for ep in self._endpoints.values():
            await ep.client.aclose()

    @property
    def endpoint_names(self) -> list[str]:
        return list(self._endpoints)

    def stats(self) -> dict[str, EndpointStats]:
        return {name: ep.stats for name, ep in self._endpoints.items()}

    # ---- probing --------------------------------------------------------------
    async def probe_models(self) -> dict[str, str]:
        """GET /v1/models on each endpoint -> {endpoint_name: served_model_id}."""

        out: dict[str, str] = {}
        for name, ep in self._endpoints.items():
            try:
                resp = await ep.client.get(ep.cfg.models_url)
                resp.raise_for_status()
                data = resp.json()
                served = data["data"][0]["id"]
                if not isinstance(served, str) or not served.strip():
                    raise VLMBadResponse("missing served model id")
            except Exception as exc:
                logger.warning(
                    "model probe failed",
                    extra={
                        "fields": {
                            "endpoint": name,
                            "url": ep.cfg.models_url,
                            "error": str(exc),
                        }
                    },
                )
                continue
            out[name] = served
        return out

    async def resolve_model_id(self) -> str:
        """Resolve the model id from config, else from the endpoints (and warn on mismatch)."""

        if self.model_id:
            return self.model_id
        served = await self.probe_models()
        if not served:
            raise VLMError("no live endpoints responded to /v1/models")
        ids = set(served.values())
        if len(ids) > 1:
            logger.warning(
                "model id mismatch across endpoints",
                extra={"fields": {"served": served}},
            )
        self.model_id = next(iter(served.values()))
        logger.info(
            "resolved model",
            extra={"fields": {"model_id": self.model_id, "served": served}},
        )
        return self.model_id

    async def health_image(self, endpoint: str) -> ChatResult:
        """Tiny image round-trip against one endpoint (used by ``doctor``)."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly the word: ok"},
                    {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
                ],
            }
        ]
        return await self.chat(endpoint, messages, max_tokens=16)

    # ---- the request ----------------------------------------------------------
    async def chat(
        self,
        endpoint: str,
        messages: list[dict],
        *,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """Single chat/completions call to ``endpoint``. Raises ``VLMError`` on failure."""

        ep = self._endpoints[endpoint]
        model = self.model_id or self.config.model_id
        if not model:
            raise VLMError("model_id not resolved; call resolve_model_id() first")

        body: dict = {
            "model": model,
            "temperature": self.config.convert.temperature,
            "top_p": self.config.convert.top_p,
            "frequency_penalty": self.config.convert.frequency_penalty,
            "max_tokens": max_tokens or self.config.convert.max_tokens,
            "messages": messages,
        }
        if self.config.convert.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        t0 = time.monotonic()
        ep.stats.requests += 1
        try:
            resp = await ep.client.post(ep.cfg.chat_url, json=body)
        except httpx.TimeoutException as exc:
            ep.stats.failures += 1
            raise VLMTimeout(f"timeout after {self.config.convert.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            ep.stats.failures += 1
            raise VLMError(f"transport error: {exc}") from exc

        latency = time.monotonic() - t0
        ep.stats.latency_s += latency

        if resp.status_code != 200:
            ep.stats.failures += 1
            raise VLMHTTPError(
                resp.status_code,
                resp.text,
                retryable=_http_retryable(resp.status_code),
                retry_after_s=_retry_after_seconds(resp.headers.get("Retry-After")),
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
            usage = data.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            image_tokens = (
                details.get("image_tokens") if isinstance(details, dict) else None
            )
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            ep.stats.failures += 1
            raise VLMBadResponse(f"unexpected response shape: {exc}") from exc

        ep.stats.prompt_tokens += prompt_tokens or 0
        ep.stats.completion_tokens += completion_tokens or 0
        return ChatResult(
            content=content,
            endpoint=endpoint,
            finish_reason=finish,
            image_tokens=image_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
        )
