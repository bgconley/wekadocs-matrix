from pathlib import Path

import pytest

import docpipe.config as config_mod


def test_defaults_match_environment():
    cfg = config_mod.default_config()
    names = {e.name for e in cfg.endpoints}
    assert names == {"blackbird", "oxcart"}
    # Oxcart (pinned vLLM) is the default target; Blackbird is configured but off.
    assert {e.name for e in cfg.active_endpoints} == {"oxcart"}
    assert cfg.total_inflight == 6
    assert cfg.rasterize.dpi == 218
    assert cfg.rasterize.max_long_px == 2408
    assert cfg.rasterize.max_long_px % 28 == 0
    assert cfg.rasterize.escalate_dpi == 300
    assert getattr(cfg.rasterize, "escalate_max_long_px", None) == 3304
    assert getattr(cfg.rasterize, "min_pixels", None) == 28 * 28 * 256
    assert getattr(cfg.rasterize, "max_pixels", None) == 28 * 28 * 10976


def test_endpoint_auth_headers():
    cfg = config_mod.default_config()
    by = {e.name: e for e in cfg.endpoints}
    assert "Authorization" not in by["blackbird"].headers()
    assert by["oxcart"].headers()["Authorization"] == "Bearer EMPTY"
    assert by["blackbird"].chat_url.endswith("/v1/chat/completions")
    assert by["oxcart"].models_url.endswith("/v1/models")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCPIPE_DPI", "300")
    monkeypatch.setenv("DOCPIPE_OXCART_BEARER", "TOKEN123")
    monkeypatch.setenv("DOCPIPE_FIGURES", "minimal")
    cfg = config_mod.load()
    assert cfg.rasterize.dpi == 300
    assert cfg.figures == "minimal"
    by = {e.name: e for e in cfg.endpoints}
    assert by["oxcart"].bearer == "TOKEN123"


def test_disable_env(monkeypatch):
    monkeypatch.setenv("DOCPIPE_DISABLE", "oxcart")
    cfg = config_mod.load()
    by = {e.name: e for e in cfg.endpoints}
    assert by["oxcart"].enabled is False
    # oxcart disabled + blackbird default-off -> nothing active.
    assert cfg.active_endpoints == []


def test_env_dpi_must_be_positive_integer(monkeypatch):
    monkeypatch.setenv("DOCPIPE_DPI", "300dpi")
    with pytest.raises(ValueError, match="DOCPIPE_DPI.*positive integer"):
        config_mod.load()

    monkeypatch.setenv("DOCPIPE_DPI", "0")
    with pytest.raises(ValueError, match="DOCPIPE_DPI.*positive integer"):
        config_mod.load()


def test_shipped_toml_keeps_engine_sampler_hardened():
    cfg_path = Path(__file__).resolve().parents[1] / "docpipe.toml"

    cfg = config_mod.load(cfg_path)

    assert cfg.rasterize.dpi == 218
    assert cfg.rasterize.max_long_px == 2408
    assert cfg.rasterize.escalate_max_long_px == 3304
    assert cfg.convert.temperature == pytest.approx(0.1)
    assert cfg.convert.top_p == pytest.approx(0.9)
    assert cfg.convert.frequency_penalty == pytest.approx(0.2)


def test_partial_toml_endpoint_merges_with_curated_defaults(tmp_path):
    cfg_path = tmp_path / "docpipe.toml"
    cfg_path.write_text(
        """
[[endpoints]]
name = "oxcart"
base_url = "http://local-oxcart:18002"
""".strip(),
        encoding="utf-8",
    )

    cfg = config_mod.load(cfg_path)
    by = {e.name: e for e in cfg.endpoints}

    assert set(by) == {"oxcart", "blackbird"}
    assert by["oxcart"].base_url == "http://local-oxcart:18002"
    assert by["oxcart"].bearer == "EMPTY"
    assert by["oxcart"].inflight == 6
    assert by["blackbird"].enabled is False


def test_unknown_toml_keys_are_rejected(tmp_path):
    cfg_path = tmp_path / "docpipe.toml"
    cfg_path.write_text(
        """
[rasterize]
dip = 300
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dip"):
        config_mod.load(cfg_path)
