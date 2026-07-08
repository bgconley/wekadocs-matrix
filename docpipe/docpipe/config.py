"""Configuration model and loader for docpipe.

Precedence (low -> high): built-in defaults  <  docpipe.toml  <  DOCPIPE_* env vars
<  explicit CLI flags (applied by ``cli.py`` after ``Config.load``).

Defaults are pinned to the live lab environment characterized on 2026-07-07:
two OpenAI-compatible Qwen3.6-27B-FP8 endpoints, both ``max_running_requests=4``,
both FP8 KV. Blackbird (SGLang) is lower-latency and needs no auth; Oxcart (vLLM)
requires ``Authorization: Bearer EMPTY``. The model id is left ``None`` so it is
probed from ``/v1/models`` at runtime rather than hardcoded.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FiguresMode = Literal["enriched", "minimal", "skip"]


class EndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    bearer: Optional[str] = None  # Authorization: Bearer <bearer> when set
    inflight: int = 5  # concurrent in-flight requests dispatched to this host
    weight: float = 1.0  # relative preference on ties (higher = preferred)
    enabled: bool = True

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/models"

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.bearer is not None:
            h["Authorization"] = f"Bearer {self.bearer}"
        return h


class RasterizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dpi: int = 200  # base render DPI (US-Letter@200 -> ~1700x2200 px)
    max_long_px: int = 2000  # clamp the long side; keeps image tokens ~3-5K/page
    escalate_dpi: int = 300  # retry DPI for pages QA flags as garbled


class ConvertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.1
    top_p: float = 0.9
    frequency_penalty: float = 0.2
    max_tokens: int = 6000  # dense spec tables run long
    timeout_s: float = 240.0
    max_retries: int = 4
    backoff_base_s: float = 2.0  # exponential: base * 2**attempt (+ jitter)
    prev_tail_chars: int = 800  # ~200 tokens of previous-page context
    anchor_max_chars: int = 6000  # cap text-layer anchor to ~1-2k tokens
    text_overlap_min: float = 0.60
    disable_thinking: bool = True  # verified to suppress reasoning on both engines


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Scratch default per the standing decision: write outside the git-tracked
    # data/ingest/nutanix/ carve-out until placement is finalized.
    dir: str = "ETL-for-corpus/transformed-corpus"
    layout: Literal["per-doc", "flat"] = "per-doc"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: str = "ETL-for-corpus"
    work_dir: str = ".docpipe_work"  # manifest + per-page artifact cache
    model_id: Optional[str] = None  # None -> probe /v1/models
    figures: FiguresMode = "enriched"
    endpoints: list[EndpointConfig] = Field(default_factory=list)
    rasterize: RasterizeConfig = Field(default_factory=RasterizeConfig)
    convert: ConvertConfig = Field(default_factory=ConvertConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # ---- derived helpers -------------------------------------------------------
    @property
    def active_endpoints(self) -> list[EndpointConfig]:
        return [e for e in self.endpoints if e.enabled]

    @property
    def total_inflight(self) -> int:
        return sum(e.inflight for e in self.active_endpoints) or 1


def _default_endpoints() -> list[EndpointConfig]:
    # Oxcart (pinned-stable vLLM) is the DEFAULT target. Blackbird (dev-nightly
    # SGLang, lower latency but no version pinning) is configured but disabled by
    # default -- add it with `--endpoint all` (both) or `--endpoint blackbird`.
    return [
        EndpointConfig(
            name="oxcart",
            base_url="http://oxcart.lan.conley.ai:18002",
            bearer="EMPTY",
            inflight=6,  # a shade above max_running=4 to keep the server batch full
            weight=1.0,
            enabled=True,
        ),
        EndpointConfig(
            name="blackbird",
            base_url="http://blackbird.lan.conley.ai:18002",
            bearer=None,
            inflight=6,
            weight=1.2,  # lower image latency -> prefer on ties, when enabled
            enabled=False,
        ),
    ]


def default_config() -> Config:
    return Config(endpoints=_default_endpoints())


def _parse_positive_int(value: str, env_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return parsed


def _merge_endpoint_list(
    base_endpoints: list[dict[str, Any]], override_endpoints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged_by_name = {str(ep.get("name")): dict(ep) for ep in base_endpoints}
    order = [str(ep.get("name")) for ep in base_endpoints]
    for endpoint in override_endpoints:
        name = str(endpoint.get("name"))
        if name in merged_by_name:
            merged_by_name[name] = {**merged_by_name[name], **endpoint}
        else:
            merged_by_name[name] = dict(endpoint)
            order.append(name)
    return [merged_by_name[name] for name in order]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "endpoints" and isinstance(value, list):
            merged[key] = _merge_endpoint_list(list(merged.get(key, [])), value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _apply_env(cfg: Config) -> Config:
    """Overlay DOCPIPE_* environment variables onto a config."""

    env = os.environ
    if v := env.get("DOCPIPE_MODEL_ID"):
        cfg.model_id = v
    if v := env.get("DOCPIPE_IN"):
        cfg.input_dir = v
    if v := env.get("DOCPIPE_OUT"):
        cfg.output.dir = v
    if v := env.get("DOCPIPE_WORK"):
        cfg.work_dir = v
    if v := env.get("DOCPIPE_FIGURES"):
        if v in ("enriched", "minimal", "skip"):
            cfg.figures = v  # type: ignore[assignment]
    if v := env.get("DOCPIPE_DPI"):
        cfg.rasterize.dpi = _parse_positive_int(v, "DOCPIPE_DPI")

    by_name = {e.name: e for e in cfg.endpoints}
    if (v := env.get("DOCPIPE_BLACKBIRD_URL")) and "blackbird" in by_name:
        by_name["blackbird"].base_url = v
    if (v := env.get("DOCPIPE_OXCART_URL")) and "oxcart" in by_name:
        by_name["oxcart"].base_url = v
    if (v := env.get("DOCPIPE_OXCART_BEARER")) and "oxcart" in by_name:
        by_name["oxcart"].bearer = v
    # Allow disabling an endpoint entirely, e.g. DOCPIPE_DISABLE=oxcart
    if v := env.get("DOCPIPE_DISABLE"):
        for name in (n.strip() for n in v.split(",")):
            if name in by_name:
                by_name[name].enabled = False
    return cfg


def _merge_toml(base: dict[str, Any], toml_path: Path) -> dict[str, Any]:
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    return _deep_merge(base, data)


def load(config_path: str | os.PathLike[str] | None = None) -> Config:
    """Build a :class:`Config` from defaults + optional TOML + env vars.

    If ``config_path`` is given it must exist; otherwise a ``docpipe.toml`` in the
    current directory is used when present.
    """

    cfg = default_config()

    path: Optional[Path] = None
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
    else:
        candidate = Path("docpipe.toml")
        if candidate.is_file():
            path = candidate

    if path is not None:
        raw = _merge_toml(cfg.model_dump(), path)
        cfg = Config.model_validate(raw)

    return _apply_env(cfg)
