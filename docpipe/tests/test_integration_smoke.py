"""Live smoke test: convert a real page against the Blackbird endpoint.

Skipped unless ``DOCPIPE_LIVE=1`` (needs network to blackbird.lan.conley.ai).
"""

import os
from pathlib import Path

import pytest

from docpipe.config import default_config
from docpipe.prompts import build_messages
from docpipe.rasterize import render_page_data_url
from docpipe.vlm_client import VLMPool

pytestmark = pytest.mark.live

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_PDF = _REPO_ROOT / "ETL-for-corpus" / "5c-book-of-ahv-administration.pdf"


@pytest.mark.skipif(
    os.environ.get("DOCPIPE_LIVE") != "1", reason="set DOCPIPE_LIVE=1 to run"
)
async def test_convert_one_real_page():
    if not _SAMPLE_PDF.exists():
        pytest.skip(f"sample PDF not present: {_SAMPLE_PDF}")

    cfg = default_config()
    # Keep only Blackbird for a deterministic single-endpoint smoke.
    for e in cfg.endpoints:
        e.enabled = e.name == "blackbird"

    data_url = render_page_data_url(
        _SAMPLE_PDF, 1, cfg.rasterize.dpi, cfg.rasterize.max_long_px
    )
    messages = build_messages(
        image_data_url=data_url,
        page_no=1,
        total_pages=3,
        prev_tail=None,
        figures=cfg.figures,
    )
    async with VLMPool(cfg) as pool:
        model_id = await pool.resolve_model_id()
        assert model_id
        result = await pool.chat("blackbird", messages)

    assert result.content.strip(), "model returned empty content for a real page"
    # A transcription of a real doc page should contain a heading or prose, not a code fence wrapper.
    assert not result.content.strip().startswith(
        "```"
    ), "output should not be wrapped in a fence"
