# =============================================================================
# @status: ACTIVE
# @called-by: hybrid_retrieval.py (HybridRetriever.__init__)
# =============================================================================
"""
Retrieval control plane: profile-to-plan resolution.

Replaces 18+ scattered boolean flags with a single ResolvedRetrievalPlan
dataclass resolved once at init time. Profiles are named presets that
set all booleans coherently; individual flag overrides remain possible
for backward compatibility during migration.

Design: Pure functions, no IO, fully testable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RetrievalProfile(str, Enum):
    """Named retrieval behavior presets."""

    VECTOR_ONLY = "vector_only"
    PRECISION_VECTOR = "precision_vector"
    GRAPH_ASSISTED = "graph_assisted"
    GRAPH_FULL = "graph_full"


@dataclass(frozen=True)
class ResolvedRetrievalPlan:
    """Immutable retrieval behavior contract.

    Resolved once at HybridRetriever.__init__ time. Every boolean
    that controls a conditional branch in retrieve() lives here.
    """

    profile: RetrievalProfile

    # ── Vector pipeline ──────────────────────────────────────────
    use_weighted_fusion: bool = False
    use_signal_pool: bool = False
    signal_pool_before_colbert: bool = False
    use_colbert: bool = True

    # ── RELATED_TO graph edges ──────────────────────────────────
    use_related_to_expansion: bool = False
    use_related_to_blending: bool = False

    # ── Entity graph channel ─────────────────────────────────────
    use_entity_graph_channel: bool = False
    use_graph_enrichment: bool = False

    # ── Expansion ────────────────────────────────────────────────
    use_structure_expansion: bool = False

    # ── Reranker refinements ─────────────────────────────────────
    use_focused_rerank_text: bool = False
    use_specificity_adjustment: bool = False

    # ── Graph quality (always-on when any graph feature is active) ──
    graph_garbage_filter_on: bool = False
    graph_score_normalized_on: bool = False

    # ── Experimental (opt-in only via profile_overrides) ────────
    use_graph_score_override: bool = False


# ── Profile definitions ──────────────────────────────────────────

_PROFILE_PLANS: Dict[RetrievalProfile, Dict[str, bool]] = {
    RetrievalProfile.VECTOR_ONLY: dict(
        use_weighted_fusion=False,
        use_signal_pool=False,
        signal_pool_before_colbert=False,
        use_colbert=True,
        use_related_to_expansion=False,
        use_related_to_blending=False,
        use_entity_graph_channel=False,
        use_graph_enrichment=False,
        use_structure_expansion=False,
        use_focused_rerank_text=False,
        use_specificity_adjustment=False,
        graph_garbage_filter_on=False,
        graph_score_normalized_on=False,
        use_graph_score_override=False,
    ),
    RetrievalProfile.PRECISION_VECTOR: dict(
        use_weighted_fusion=True,
        use_signal_pool=True,
        signal_pool_before_colbert=True,
        use_colbert=True,
        use_related_to_expansion=False,
        use_related_to_blending=False,
        use_entity_graph_channel=False,
        use_graph_enrichment=False,
        use_structure_expansion=True,
        use_focused_rerank_text=True,
        use_specificity_adjustment=False,
        graph_garbage_filter_on=False,
        graph_score_normalized_on=False,
        use_graph_score_override=False,
    ),
    RetrievalProfile.GRAPH_ASSISTED: dict(
        use_weighted_fusion=True,
        use_signal_pool=True,
        signal_pool_before_colbert=True,
        use_colbert=True,
        use_related_to_expansion=True,
        use_related_to_blending=True,
        use_entity_graph_channel=True,
        use_graph_enrichment=False,
        use_structure_expansion=True,
        use_focused_rerank_text=True,
        use_specificity_adjustment=False,
        graph_garbage_filter_on=True,
        graph_score_normalized_on=True,
        use_graph_score_override=False,
    ),
    RetrievalProfile.GRAPH_FULL: dict(
        use_weighted_fusion=True,
        use_signal_pool=True,
        signal_pool_before_colbert=True,
        use_colbert=True,
        use_related_to_expansion=True,
        use_related_to_blending=True,
        use_entity_graph_channel=True,
        use_graph_enrichment=True,
        use_structure_expansion=True,
        use_focused_rerank_text=True,
        use_specificity_adjustment=False,
        graph_garbage_filter_on=True,
        graph_score_normalized_on=True,
        use_graph_score_override=False,
    ),
}

# Keys allowed in profile_overrides (restricted allowlist)
_ALLOWED_OVERRIDES = frozenset(
    {"use_specificity_adjustment", "use_graph_score_override"}
)


def resolve_retrieval_plan(
    profile_name: Optional[str],
    hybrid_config: Any,
    feature_flags: Any,
) -> ResolvedRetrievalPlan:
    """Resolve a ResolvedRetrievalPlan from config.

    Priority:
      1. If profile_name is set, use the named profile preset
      2. Otherwise, infer the plan from legacy flags (backward compat)

    When profile is set, legacy flags are ignored and a deprecation
    warning is emitted if any legacy flags differ from the profile defaults.
    """
    neo4j_disabled = bool(getattr(hybrid_config, "neo4j_disabled", False))

    if profile_name is not None:
        profile = RetrievalProfile(profile_name)  # Pydantic already validated
        plan_kwargs = dict(_PROFILE_PLANS[profile])
        plan_kwargs["profile"] = profile

        # Apply profile_overrides (restricted allowlist)
        overrides = getattr(hybrid_config, "profile_overrides", None) or {}
        for key, value in overrides.items():
            if key in _ALLOWED_OVERRIDES:
                plan_kwargs[key] = bool(value)
            else:
                logger.warning(
                    "Ignoring unrecognized profile_override '%s'. " "Allowed: %s",
                    key,
                    sorted(_ALLOWED_OVERRIDES),
                )

        # Zero graph booleans when neo4j is disabled
        if neo4j_disabled:
            _zero_graph_fields(plan_kwargs)

        # Warn about legacy flag drift
        _warn_legacy_flag_drift(profile, hybrid_config, feature_flags)

        return ResolvedRetrievalPlan(**plan_kwargs)

    # ── Legacy flag inference (no profile set) ───────────────────
    return _infer_from_legacy_flags(hybrid_config, feature_flags, neo4j_disabled)


def _zero_graph_fields(plan_kwargs: Dict[str, Any]) -> None:
    """Zero all graph-related booleans. Called when neo4j_disabled=True."""
    for field in (
        "use_related_to_expansion",
        "use_related_to_blending",
        "use_entity_graph_channel",
        "use_graph_enrichment",
        "graph_garbage_filter_on",
        "graph_score_normalized_on",
        "use_graph_score_override",
    ):
        plan_kwargs[field] = False


def _infer_from_legacy_flags(
    hybrid_config: Any,
    feature_flags: Any,
    neo4j_disabled: bool,
) -> ResolvedRetrievalPlan:
    """Infer a plan from scattered legacy flags. Preserves exact current behavior."""
    ff = feature_flags

    # Signal pool requires both config section AND feature flag
    signal_pool_cfg = getattr(hybrid_config, "signal_pool", None)
    use_signal_pool = bool(
        signal_pool_cfg
        and getattr(signal_pool_cfg, "enabled", False)
        and getattr(ff, "signal_diverse_rerank_pool", False)
    )

    use_weighted_fusion = bool(getattr(ff, "query_api_weighted_fusion", False))
    signal_pool_before_colbert = bool(getattr(ff, "signal_pool_before_colbert", False))
    use_colbert = bool(getattr(hybrid_config, "colbert_rerank_enabled", True))

    graph_channel_enabled = bool(getattr(hybrid_config, "graph_channel_enabled", False))

    # RELATED_TO expansion is independent of graph channel (only needs neo4j)
    use_related_to_expansion = not neo4j_disabled

    # BUG preserved: blending is trapped inside graph_as_reranker path.
    # Only activates when graph channel is on. This is the bug that profiles fix.
    use_related_to_blending = graph_channel_enabled and not neo4j_disabled

    # Entity graph channel — legacy has enrichment coupling (preserved here)
    use_entity_graph_channel = graph_channel_enabled and not neo4j_disabled

    # Graph enrichment — legacy derivation
    graph_enrichment_enabled = bool(
        getattr(hybrid_config, "graph_enrichment_enabled", False)
    )
    use_graph_enrichment = graph_enrichment_enabled and not neo4j_disabled

    use_structure_expansion = bool(getattr(ff, "structure_aware_expansion", False))
    use_focused_rerank_text = bool(getattr(ff, "precision_focused_rerank_text", False))
    use_specificity_adjustment = bool(
        getattr(ff, "precision_specificity_adjustment", False)
    )

    graph_garbage_filter_on = bool(getattr(ff, "graph_garbage_filter", False))
    graph_score_normalized_on = bool(getattr(ff, "graph_score_normalized", False))
    use_graph_score_override = bool(getattr(ff, "graph_as_reranker", False))

    # Infer closest profile name for logging
    if use_graph_enrichment:
        profile = RetrievalProfile.GRAPH_FULL
    elif use_entity_graph_channel:
        profile = RetrievalProfile.GRAPH_ASSISTED
    elif use_signal_pool:
        profile = RetrievalProfile.PRECISION_VECTOR
    else:
        profile = RetrievalProfile.VECTOR_ONLY

    return ResolvedRetrievalPlan(
        profile=profile,
        use_weighted_fusion=use_weighted_fusion,
        use_signal_pool=use_signal_pool,
        signal_pool_before_colbert=signal_pool_before_colbert,
        use_colbert=use_colbert,
        use_related_to_expansion=use_related_to_expansion,
        use_related_to_blending=use_related_to_blending,
        use_entity_graph_channel=use_entity_graph_channel,
        use_graph_enrichment=use_graph_enrichment,
        use_structure_expansion=use_structure_expansion,
        use_focused_rerank_text=use_focused_rerank_text,
        use_specificity_adjustment=use_specificity_adjustment,
        graph_garbage_filter_on=graph_garbage_filter_on,
        graph_score_normalized_on=graph_score_normalized_on,
        use_graph_score_override=use_graph_score_override,
    )


# ── Legacy flag names mapped to plan fields ──────────────────────

_LEGACY_FLAG_MAP: Dict[str, Tuple[str, str, str]] = {
    # (source_obj, attr_name, plan_field)
    "graph_channel_enabled": (
        "hybrid",
        "graph_channel_enabled",
        "use_entity_graph_channel",
    ),
    "graph_enrichment_enabled": (
        "hybrid",
        "graph_enrichment_enabled",
        "use_graph_enrichment",
    ),
    "signal_pool_before_colbert": (
        "ff",
        "signal_pool_before_colbert",
        "signal_pool_before_colbert",
    ),
    "precision_focused_rerank_text": (
        "ff",
        "precision_focused_rerank_text",
        "use_focused_rerank_text",
    ),
    "precision_specificity_adjustment": (
        "ff",
        "precision_specificity_adjustment",
        "use_specificity_adjustment",
    ),
    "graph_garbage_filter": ("ff", "graph_garbage_filter", "graph_garbage_filter_on"),
    "graph_score_normalized": (
        "ff",
        "graph_score_normalized",
        "graph_score_normalized_on",
    ),
    "graph_as_reranker": ("ff", "graph_as_reranker", "use_graph_score_override"),
}


def _warn_legacy_flag_drift(
    profile: RetrievalProfile,
    hybrid_config: Any,
    feature_flags: Any,
) -> None:
    """Emit deprecation warnings when legacy flags contradict the active profile."""
    plan_values = _PROFILE_PLANS[profile]

    for flag_name, (source, attr, plan_field) in _LEGACY_FLAG_MAP.items():
        obj = hybrid_config if source == "hybrid" else feature_flags
        legacy_val = bool(getattr(obj, attr, False))
        profile_val = plan_values.get(plan_field, False)
        if legacy_val != profile_val:
            warnings.warn(
                f"Retrieval profile '{profile.value}' overrides legacy flag "
                f"'{flag_name}={legacy_val}' with '{plan_field}={profile_val}'. "
                f"Remove '{flag_name}' from config to silence this warning.",
                DeprecationWarning,
                stacklevel=4,
            )
