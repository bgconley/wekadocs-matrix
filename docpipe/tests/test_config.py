import docpipe.config as config_mod


def test_defaults_match_environment():
    cfg = config_mod.default_config()
    names = {e.name for e in cfg.endpoints}
    assert names == {"blackbird", "oxcart"}
    # Oxcart (pinned vLLM) is the default target; Blackbird is configured but off.
    assert {e.name for e in cfg.active_endpoints} == {"oxcart"}
    assert cfg.total_inflight == 6


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
