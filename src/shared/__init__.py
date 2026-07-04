# =============================================================================
# @status: ACTIVE
# @reason: Package init for shared utilities. AuditLogger import was removed
#          (Phase B cleanup) — never called by active code. Config and
#          connections re-exports are actively used by main.py, worker.py.
# =============================================================================
# Shared utilities package
from .config import Config, Settings, get_config, get_settings, init_config
from .connections import (
    ConnectionManager,
    close_connections,
    get_connection_manager,
    initialize_connections,
)

__all__ = [
    "Config",
    "Settings",
    "get_config",
    "get_settings",
    "init_config",
    "ConnectionManager",
    "get_connection_manager",
    "initialize_connections",
    "close_connections",
]
