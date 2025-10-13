# Implements Phase 3, Task 3.2 (Entity extraction - Commands)
# See: /docs/spec.md §3.1 (Domain entities)
# See: /docs/implementation-plan.md → Task 3.2
# See: /docs/pseudocode-reference.md → Task 3.2

import hashlib
import re
from typing import Dict, List, Tuple

from src.shared.observability import get_logger

logger = get_logger(__name__)


def extract_commands(section: Dict[str, any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract Command entities from a section.

    Returns:
        Tuple of (commands, mentions)
        - commands: List of Command entities
        - mentions: List of MENTIONS relationships with spans
    """
    commands = []
    mentions = []

    text = section["text"]
    section_id = section["id"]

    # Pattern 1: CLI commands in code blocks
    for code_block in section.get("code_blocks", []):
        commands_from_code, mentions_from_code = _extract_from_code_block(
            code_block, section_id, text
        )
        commands.extend(commands_from_code)
        mentions.extend(mentions_from_code)

    # Pattern 2: Inline code with command-like patterns
    inline_commands, inline_mentions = _extract_inline_commands(text, section_id)
    commands.extend(inline_commands)
    mentions.extend(inline_mentions)

    # Pattern 3: Command documentation patterns (e.g., "The `weka` command...")
    doc_commands, doc_mentions = _extract_documented_commands(text, section_id)
    commands.extend(doc_commands)
    mentions.extend(doc_mentions)

    # Deduplicate commands by canonical name
    commands_dict = {cmd["name"]: cmd for cmd in commands}
    commands = list(commands_dict.values())

    logger.debug(
        "Extracted commands",
        section_id=section_id,
        commands_count=len(commands),
        mentions_count=len(mentions),
    )

    return commands, mentions


def _extract_from_code_block(
    code_block: str, section_id: str, full_text: str
) -> Tuple[List[Dict], List[Dict]]:
    """Extract commands from code blocks."""
    commands = []
    mentions = []

    lines = code_block.split("\n")
    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Common command patterns
        if _looks_like_command(line):
            cmd = _parse_command_line(line)
            if cmd:
                command_entity = _create_command_entity(
                    cmd["name"],
                    cmd["full_command"],
                    section_id,
                )
                commands.append(command_entity)

                # Find span in full text
                span = _find_span(full_text, cmd["full_command"])
                if span:
                    mention = _create_mention(
                        section_id,
                        command_entity["id"],
                        span,
                        confidence=0.9,  # High confidence for code blocks
                    )
                    mentions.append(mention)

    return commands, mentions


def _extract_inline_commands(
    text: str, section_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """Extract commands from inline code (backticks)."""
    commands = []
    mentions = []

    # Pattern: `command args...`
    pattern = r"`([^`]+)`"
    for match in re.finditer(pattern, text):
        code = match.group(1).strip()

        if _looks_like_command(code):
            cmd = _parse_command_line(code)
            if cmd:
                command_entity = _create_command_entity(
                    cmd["name"],
                    cmd["full_command"],
                    section_id,
                )
                commands.append(command_entity)

                span = (match.start(), match.end())
                mention = _create_mention(
                    section_id,
                    command_entity["id"],
                    span,
                    confidence=0.85,
                )
                mentions.append(mention)

    return commands, mentions


def _extract_documented_commands(
    text: str, section_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """Extract commands from documentation patterns."""
    commands = []
    mentions = []

    # Pattern: "The `command` command does..."
    # Pattern: "`command` - description"
    patterns = [
        r"The\s+`([a-z][a-z0-9\-_]+)`\s+command",
        r"`([a-z][a-z0-9\-_]+)`\s*-\s+",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            command_name = match.group(1).strip()

            if len(command_name) > 1 and _is_valid_command_name(command_name):
                command_entity = _create_command_entity(
                    command_name,
                    command_name,
                    section_id,
                )
                commands.append(command_entity)

                span = (match.start(1), match.end(1))
                mention = _create_mention(
                    section_id,
                    command_entity["id"],
                    span,
                    confidence=0.8,
                )
                mentions.append(mention)

    return commands, mentions


def _looks_like_command(line: str) -> bool:
    """Check if a line looks like a command."""
    line = line.strip()

    # Must not be too long
    if len(line) > 500:
        return False

    # Common command patterns
    command_starters = [
        "weka",
        "kubectl",
        "docker",
        "git",
        "npm",
        "pip",
        "python",
        "curl",
        "ssh",
        "scp",
        "rsync",
        "mount",
        "umount",
        "systemctl",
        "service",
    ]

    first_word = line.split()[0] if line.split() else ""
    return any(first_word.startswith(starter) for starter in command_starters)


def _parse_command_line(line: str) -> Dict:
    """Parse a command line into components."""
    parts = line.split()
    if not parts:
        return None

    # Remove shell prefixes
    if parts[0] in ["$", "#", ">"]:
        parts = parts[1:]

    if not parts:
        return None

    command_name = parts[0]

    # Extract subcommand if present
    if len(parts) > 1 and not parts[1].startswith("-"):
        command_name = f"{parts[0]} {parts[1]}"

    return {
        "name": command_name,
        "full_command": line,
    }


def _is_valid_command_name(name: str) -> bool:
    """Check if a name looks like a valid command name."""
    # Allow letters, numbers, hyphens, underscores
    if not re.match(r"^[a-z][a-z0-9\-_]+$", name, re.IGNORECASE):
        return False

    # Exclude common non-commands
    excluded = {"and", "the", "for", "with", "from", "this", "that", "are", "can"}
    return name.lower() not in excluded


def _create_command_entity(
    name: str, full_command: str, source_section_id: str
) -> Dict:
    """Create a Command entity."""
    # Deterministic ID based on canonical name
    entity_id = hashlib.sha256(f"command:{name}".encode("utf-8")).hexdigest()

    return {
        "id": entity_id,
        "label": "Command",
        "name": name,
        "description": full_command,
        "category": "cli",
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
