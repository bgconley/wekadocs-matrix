import httpx
import pytest

from docpipe.config import default_config
from docpipe.vlm_client import VLMError, VLMPool, _Endpoint


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
