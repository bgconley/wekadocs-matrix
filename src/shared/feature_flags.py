"""
Feature Flags for Runtime Toggles
Phase 5 Task 5.4 - Production Deployment

Provides runtime feature toggles for gradual rollout and A/B testing.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FeatureFlag:
    """Represents a single feature flag."""

    name: str
    enabled: bool
    rollout_percentage: int = 100  # 0-100
    description: str = ""


class FeatureFlagManager:
    """Manages feature flags with runtime updates."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature flag manager.

        Args:
            config_path: Path to feature flags config file (JSON)
        """
        self.config_path = config_path or os.getenv(
            "FEATURE_FLAGS_PATH", "/app/config/feature_flags.json"
        )
        self._flags: Dict[str, FeatureFlag] = {}
        self._load_flags()

    def _load_flags(self):
        """Load feature flags from config file or environment."""
        # Default flags
        self._flags = {
            "hybrid_search_v2": FeatureFlag(
                name="hybrid_search_v2",
                enabled=False,
                rollout_percentage=0,
                description="New hybrid search algorithm",
            ),
            "advanced_caching": FeatureFlag(
                name="advanced_caching",
                enabled=True,
                rollout_percentage=100,
                description="Advanced L1+L2 caching",
            ),
            "dual_vector_write": FeatureFlag(
                name="dual_vector_write",
                enabled=False,
                rollout_percentage=0,
                description="Write to both Qdrant and Neo4j vectors",
            ),
            "rate_limiting": FeatureFlag(
                name="rate_limiting",
                enabled=True,
                rollout_percentage=100,
                description="JWT-based rate limiting",
            ),
        }

        # Override from config file if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = json.load(f)
                for name, data in config.get("flags", {}).items():
                    self._flags[name] = FeatureFlag(
                        name=name,
                        enabled=data.get("enabled", False),
                        rollout_percentage=data.get("rollout_percentage", 100),
                        description=data.get("description", ""),
                    )

        # Override from environment variables
        for name in self._flags:
            env_key = f"FF_{name.upper()}"
            if env_key in os.environ:
                self._flags[name].enabled = os.environ[env_key].lower() == "true"

    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_name: Name of the feature flag
            user_id: Optional user ID for percentage-based rollout

        Returns:
            True if flag is enabled for this user
        """
        if flag_name not in self._flags:
            return False

        flag = self._flags[flag_name]

        if not flag.enabled:
            return False

        # If full rollout, enable for everyone
        if flag.rollout_percentage >= 100:
            return True

        # If no user ID, use random rollout
        if user_id is None:
            import random

            return random.randint(0, 100) < flag.rollout_percentage

        # Deterministic percentage rollout based on user ID
        user_hash = hash(user_id) % 100
        return user_hash < flag.rollout_percentage

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag details."""
        return self._flags.get(flag_name)

    def list_flags(self) -> Dict[str, Dict[str, Any]]:
        """List all feature flags."""
        return {
            name: {
                "enabled": flag.enabled,
                "rollout_percentage": flag.rollout_percentage,
                "description": flag.description,
            }
            for name, flag in self._flags.items()
        }

    def reload(self):
        """Reload feature flags from config."""
        self._load_flags()


# Global feature flag manager instance
_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _manager
    if _manager is None:
        _manager = FeatureFlagManager()
    return _manager


def is_enabled(flag_name: str, user_id: Optional[str] = None) -> bool:
    """Convenience function to check if a feature flag is enabled."""
    return get_feature_flag_manager().is_enabled(flag_name, user_id)
