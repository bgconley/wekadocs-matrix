import json

import httpx
import pytest

from docpipe.config import default_config
from docpipe.vlm_client import VLMError, VLMHTTPError, VLMPool, _Endpoint


def _pool_with_transports(
    transports: dict[str, httpx.MockTransport],
) -> VLMPool:
    cfg = default_config()
    pool = VLMPool(cfg)
    by_name = {ep.name: ep for ep in cfg.endpoints}
    pool._endpoints = {
        name: _Endpoint(
            by_name[name],
            httpx.AsyncClient(transport=transport),
        )
        for name, transport in transports.items()
    }
    return pool


async def _close_pool(pool: VLMPool) -> None:
    for ep in pool._endpoints.values():
        await ep.client.aclose()


def _live_model(model_id: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": model_id}]})

    return httpx.MockTransport(handler)


def _dead_endpoint() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("endpoint down", request=request)

    return httpx.MockTransport(handler)


def _chat_status(status: int, *, retry_after: str | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        return httpx.Response(status, headers=headers, text="nope")

    return httpx.MockTransport(handler)


def _chat_ok(seen: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "# ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                },
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_resolve_model_id_uses_live_endpoint_when_one_endpoint_is_dead():
    pool = _pool_with_transports(
        {
            "blackbird": _dead_endpoint(),
            "oxcart": _live_model("qwen36-27b"),
        }
    )
    try:
        assert await pool.resolve_model_id() == "qwen36-27b"
    finally:
        await _close_pool(pool)


@pytest.mark.asyncio
async def test_resolve_model_id_raises_vlm_error_when_all_endpoints_are_dead():
    pool = _pool_with_transports(
        {
            "blackbird": _dead_endpoint(),
            "oxcart": _dead_endpoint(),
        }
    )
    try:
        with pytest.raises(VLMError, match="no live endpoints"):
            await pool.resolve_model_id()
    finally:
        await _close_pool(pool)


@pytest.mark.asyncio
async def test_chat_marks_auth_http_error_non_retryable():
    pool = _pool_with_transports({"oxcart": _chat_status(401)})
    pool.model_id = "qwen36-27b"
    try:
        with pytest.raises(VLMHTTPError) as exc_info:
            await pool.chat("oxcart", [{"role": "user", "content": "hi"}])
    finally:
        await _close_pool(pool)

    assert exc_info.value.status == 401
    assert getattr(exc_info.value, "retryable", None) is False
    assert getattr(exc_info.value, "retry_after_s", None) is None


@pytest.mark.asyncio
async def test_chat_marks_rate_limit_retryable_with_retry_after():
    pool = _pool_with_transports({"oxcart": _chat_status(429, retry_after="3")})
    pool.model_id = "qwen36-27b"
    try:
        with pytest.raises(VLMHTTPError) as exc_info:
            await pool.chat("oxcart", [{"role": "user", "content": "hi"}])
    finally:
        await _close_pool(pool)

    assert exc_info.value.status == 429
    assert getattr(exc_info.value, "retryable", None) is True
    assert getattr(exc_info.value, "retry_after_s", None) == 3.0


@pytest.mark.asyncio
async def test_chat_sends_sampler_hardening_parameters():
    seen: list[dict] = []
    pool = _pool_with_transports({"oxcart": _chat_ok(seen)})
    pool.model_id = "qwen36-27b"
    try:
        await pool.chat("oxcart", [{"role": "user", "content": "hi"}])
    finally:
        await _close_pool(pool)

    assert seen
    body = seen[0]
    assert body["frequency_penalty"] == pytest.approx(0.2)
    assert body["top_p"] == pytest.approx(0.9)
    assert body["temperature"] == pytest.approx(0.1)
