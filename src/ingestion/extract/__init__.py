# Implements Phase 3, Task 3.2 (Entity extraction - main module)
# See: /docs/spec.md §3.1 (Domain entities)
# See: /docs/implementation-plan.md → Task 3.2
# See: /docs/pseudocode-reference.md → Task 3.2

from typing import Dict, List, Tuple

from src.shared.observability import get_logger

from .commands import extract_commands
from .configs import extract_configurations
from .procedures import extract_procedures

logger = get_logger(__name__)


def extract_entities(sections: List[Dict[str, any]]) -> Tuple[Dict, List[Dict]]:
    """
    Extract all entities from sections.

    Args:
        sections: List of parsed sections

    Returns:
        Tuple of (entities_dict, mentions_list)
        - entities_dict: Dict keyed by entity_id
        - mentions_list: List of MENTIONS relationships
    """
    all_entities = {}
    all_mentions = []

    for section in sections:
        logger.debug("Extracting entities from section", section_id=section["id"])

        # Extract commands
        commands, cmd_mentions = extract_commands(section)
        for cmd in commands:
            all_entities[cmd["id"]] = cmd
        all_mentions.extend(cmd_mentions)

        # Extract configurations
        configs, cfg_mentions = extract_configurations(section)
        for cfg in configs:
            all_entities[cfg["id"]] = cfg
        all_mentions.extend(cfg_mentions)

        # Extract procedures and steps
        procedures, steps, proc_mentions = extract_procedures(section)
        for proc in procedures:
            all_entities[proc["id"]] = proc
        for step in steps:
            all_entities[step["id"]] = step
        all_mentions.extend(proc_mentions)

    logger.info(
        "Entity extraction complete",
        total_entities=len(all_entities),
        total_mentions=len(all_mentions),
        commands=len([e for e in all_entities.values() if e["label"] == "Command"]),
        configs=len(
            [e for e in all_entities.values() if e["label"] == "Configuration"]
        ),
        procedures=len([e for e in all_entities.values() if e["label"] == "Procedure"]),
        steps=len([e for e in all_entities.values() if e["label"] == "Step"]),
    )

    return all_entities, all_mentions
