# Implements Phase 3, Task 3.2 (Entity extraction - Procedures and Steps)
# See: /docs/spec.md §3 (Data model - Procedure, Step entities)
# See: /docs/implementation-plan.md → Task 3.2
# See: /docs/pseudocode-reference.md → Task 3.2

import hashlib
import re
from typing import Dict, List, Tuple

from src.shared.observability import get_logger

logger = get_logger(__name__)

# Pattern to match numbered steps in various formats
_STEP_RX = re.compile(
    r"(?m)^\s*(\d+)[\.\)]\s+(?P<body>.+?)(?=(?:\n\s*\d+[\.\)]\s+)|\Z)", flags=re.DOTALL
)


def _hash16(s: str) -> str:
    """Generate 16-char hash for entity IDs."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def extract_procedures(section: Dict) -> Tuple[List, List, List]:
    """
    Extract procedures and steps from a section.

    Returns:
        Tuple of (procedures, steps, mentions)
    """
    title = section.get("title", "")
    content = section.get("text", "")

    procedures = []
    steps = []
    mentions = []

    if not content:
        return procedures, steps, mentions

    # Heuristic: if title or content suggests a procedure/steps section
    is_proc = (
        any(
            k in title.lower()
            for k in [
                "procedure",
                "steps",
                "how to",
                "setup",
                "installation",
                "configuration",
                "getting started",
            ]
        )
        or "### steps" in content.lower()
        or "## steps" in content.lower()
        or "follow these steps" in content.lower()
        or "to install" in title.lower()
        or "to configure" in title.lower()
    )

    # Extract numbered steps
    found_steps = []
    for m in _STEP_RX.finditer(content):
        order = int(m.group(1))
        body = m.group("body").strip()
        # span positions in original text
        start = m.start("body")
        end = m.end("body")
        step_id = _hash16(f"step|{section['id']}|{order}|{body.lower()[:50]}")
        found_steps.append(
            {
                "id": step_id,
                "order": order,
                "instruction": body,
                "start": start,
                "end": end,
            }
        )

    # Sort steps by order
    found_steps.sort(key=lambda x: x["order"])

    # Create procedure if we found steps or if section looks procedural
    procedure_id = None
    if is_proc or found_steps:
        proc_id = _hash16(f"proc|{section['id']}|{title.lower()}")
        procedure = {
            "id": proc_id,
            "label": "Procedure",
            "name": title or "Procedure",
            "title": title or "Procedure",
            "description": content[:400] if content else "",
            "type": "operational",
        }
        procedures.append(procedure)
        procedure_id = proc_id

        # Add MENTIONS relationship from section to procedure
        mentions.append(
            {
                "section_id": section["id"],
                "entity_id": proc_id,
                "entity_label": "Procedure",
                "confidence": 0.9,
                "start": 0,
                "end": min(100, len(content)),
                "source_section_id": section["id"],
            }
        )

    # Create step entities
    for step_info in found_steps:
        step = {
            "id": step_info["id"],
            "label": "Step",
            "order": step_info["order"],
            "instruction": step_info["instruction"],
            "procedure_id": procedure_id,  # Link to procedure if exists
        }
        steps.append(step)

        # Add MENTIONS relationship from section to step
        mentions.append(
            {
                "section_id": section["id"],
                "entity_id": step_info["id"],
                "entity_label": "Step",
                "confidence": 0.95,
                "start": step_info["start"],
                "end": step_info["end"],
                "source_section_id": section["id"],
            }
        )

        # Add CONTAINS_STEP relationship from procedure to step
        if procedure_id:
            mentions.append(
                {
                    "from_id": procedure_id,
                    "from_label": "Procedure",
                    "to_id": step_info["id"],
                    "to_label": "Step",
                    "relationship": "CONTAINS_STEP",
                    "order": step_info["order"],
                    "confidence": 0.95,
                    "source_section_id": section["id"],
                }
            )

    if procedures or steps:
        logger.info(
            "Extracted procedure from section",
            section_id=section["id"],
            procedures_found=len(procedures),
            steps_count=len(steps),
        )

    return procedures, steps, mentions
