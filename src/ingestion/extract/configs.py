# Implements Phase 3, Task 3.2 (Entity extraction - Configurations)
# See: /docs/spec.md §3.1 (Domain entities)
# See: /docs/implementation-plan.md → Task 3.2
# See: /docs/pseudocode-reference.md → Task 3.2

import hashlib
import re
from typing import Dict, List, Tuple

from src.shared.observability import get_logger

logger = get_logger(__name__)


def extract_configurations(section: Dict[str, any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract Configuration entities from a section.

    Returns:
        Tuple of (configurations, mentions)
    """
    configurations = []
    mentions = []

    text = section["text"]
    section_id = section["id"]

    # Pattern 1: Configuration file patterns
    file_configs, file_mentions = _extract_config_files(text, section_id)
    configurations.extend(file_configs)
    mentions.extend(file_mentions)

    # Pattern 2: Configuration parameters (key=value, key: value)
    param_configs, param_mentions = _extract_config_parameters(text, section_id)
    configurations.extend(param_configs)
    mentions.extend(param_mentions)

    # Pattern 3: Environment variables
    env_configs, env_mentions = _extract_env_variables(text, section_id)
    configurations.extend(env_configs)
    mentions.extend(env_mentions)

    # Pattern 4: YAML/JSON config keys in code blocks
    for code_block in section.get("code_blocks", []):
        code_configs, code_mentions = _extract_from_config_code(
            code_block, section_id, text
        )
        configurations.extend(code_configs)
        mentions.extend(code_mentions)

    # Deduplicate
    configs_dict = {cfg["name"]: cfg for cfg in configurations}
    configurations = list(configs_dict.values())

    logger.debug(
        "Extracted configurations",
        section_id=section_id,
        configs_count=len(configurations),
        mentions_count=len(mentions),
    )

    return configurations, mentions


def _extract_config_files(text: str, section_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract configuration file references."""
    configurations = []
    mentions = []

    # Common config file patterns
    config_file_patterns = [
        r"(/etc/[a-z0-9/\-_\.]+\.conf)",
        r"(/etc/[a-z0-9/\-_\.]+\.cfg)",
        r"([a-z0-9\-_]+\.yaml)",
        r"([a-z0-9\-_]+\.yml)",
        r"([a-z0-9\-_]+\.json)",
        r"([a-z0-9\-_]+\.toml)",
        r"([a-z0-9\-_]+\.ini)",
        r"(\.env(?:\.[a-z]+)?)",
    ]

    for pattern in config_file_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            config_name = match.group(1)

            if _is_valid_config_name(config_name):
                config_entity = _create_config_entity(
                    config_name,
                    f"Configuration file: {config_name}",
                    "file",
                )
                configurations.append(config_entity)

                span = (match.start(1), match.end(1))
                mention = _create_mention(section_id, config_entity["id"], span, 0.85)
                mentions.append(mention)

    return configurations, mentions


def _extract_config_parameters(
    text: str, section_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """Extract configuration parameters."""
    configurations = []
    mentions = []

    # Patterns for config parameters
    patterns = [
        r"([A-Z_][A-Z0-9_]{2,})\s*=",  # UPPER_CASE_VAR = value
        r"`([a-z][a-z0-9\-_\.]+)`\s*:\s*",  # `config.key`: value
        r"--([a-z][a-z0-9\-]+)(?:\s|=)",  # --flag-name
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            param_name = match.group(1)

            if _is_valid_config_name(param_name):
                config_entity = _create_config_entity(
                    param_name,
                    f"Configuration parameter: {param_name}",
                    "parameter",
                )
                configurations.append(config_entity)

                span = (match.start(1), match.end(1))
                mention = _create_mention(section_id, config_entity["id"], span, 0.75)
                mentions.append(mention)

    return configurations, mentions


def _extract_env_variables(text: str, section_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract environment variables."""
    configurations = []
    mentions = []

    # Pattern: $VAR_NAME or ${VAR_NAME}
    patterns = [
        r"\$\{([A-Z][A-Z0-9_]+)\}",
        r"\$([A-Z][A-Z0-9_]+)(?:\s|$|/)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            env_var = match.group(1)

            if len(env_var) > 2:
                config_entity = _create_config_entity(
                    env_var,
                    f"Environment variable: {env_var}",
                    "environment",
                )
                configurations.append(config_entity)

                span = (match.start(1), match.end(1))
                mention = _create_mention(section_id, config_entity["id"], span, 0.9)
                mentions.append(mention)

    return configurations, mentions


def _extract_from_config_code(
    code_block: str, section_id: str, full_text: str
) -> Tuple[List[Dict], List[Dict]]:
    """Extract configuration keys from YAML/JSON code blocks."""
    configurations = []
    mentions = []

    # YAML/JSON key patterns
    lines = code_block.split("\n")
    for line in lines:
        # YAML: key: value
        yaml_match = re.match(r"^\s*([a-z][a-z0-9_\-]+):\s*", line, re.IGNORECASE)
        if yaml_match:
            key = yaml_match.group(1)
            if _is_valid_config_name(key):
                config_entity = _create_config_entity(
                    key,
                    f"Configuration key: {key}",
                    "yaml",
                )
                configurations.append(config_entity)

                span = _find_span(full_text, key)
                if span:
                    mention = _create_mention(
                        section_id, config_entity["id"], span, 0.8
                    )
                    mentions.append(mention)

        # JSON: "key": value
        json_match = re.match(r'^\s*"([a-z][a-z0-9_\-]+)":\s*', line, re.IGNORECASE)
        if json_match:
            key = json_match.group(1)
            if _is_valid_config_name(key):
                config_entity = _create_config_entity(
                    key,
                    f"Configuration key: {key}",
                    "json",
                )
                configurations.append(config_entity)

                span = _find_span(full_text, key)
                if span:
                    mention = _create_mention(
                        section_id, config_entity["id"], span, 0.8
                    )
                    mentions.append(mention)

    return configurations, mentions


def _is_valid_config_name(name: str) -> bool:
    """Check if name looks like a valid configuration name."""
    if not name or len(name) < 2:
        return False

    # Exclude common words
    excluded = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "name",
        "type",
        "value",
        "data",
        "info",
        "test",
    }
    return name.lower() not in excluded


def _create_config_entity(name: str, description: str, category: str) -> Dict:
    """Create a Configuration entity."""
    entity_id = hashlib.sha256(f"configuration:{name}".encode("utf-8")).hexdigest()

    return {
        "id": entity_id,
        "label": "Configuration",
        "name": name,
        "description": description,
        "category": category,
        "introduced_in": None,
        "deprecated_in": None,
        "updated_at": None,
        "vector_embedding": None,
        "embedding_version": None,
    }


def _create_mention(
    section_id: str, entity_id: str, span: Tuple[int, int], confidence: float
) -> Dict:
    """Create a MENTIONS relationship."""
    return {
        "section_id": section_id,
        "entity_id": entity_id,
        "confidence": confidence,
        "start": span[0],
        "end": span[1],
        "source_section_id": section_id,
    }


def _find_span(text: str, substring: str) -> Tuple[int, int]:
    """Find span of substring in text."""
    try:
        start = text.index(substring)
        return (start, start + len(substring))
    except ValueError:
        return None
