"""
NL→Cypher Query Planner (Task 2.1)
Implements templates-first approach with intent classification and entity linking.
See: /docs/spec.md §4 (Retrieval & query planning)
See: /docs/pseudocode-reference.md Phase 2, Task 2.1
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.shared.config import get_config


@dataclass
class QueryPlan:
    """Represents a planned query with intent, Cypher, and parameters."""

    intent: str
    cypher: str
    params: Dict[str, Any]
    template_name: Optional[str] = None
    confidence: float = 1.0


class TemplateLibrary:
    """Manages pre-approved Cypher templates for known intents."""

    def __init__(self):
        self.templates: Dict[str, Dict[str, str]] = {}
        self._load_templates()

    def _load_templates(self):
        """Load all .cypher template files from templates directory."""
        templates_dir = Path(__file__).parent / "templates"

        for template_file in templates_dir.glob("*.cypher"):
            intent = template_file.stem  # e.g., "search", "traverse", "troubleshoot"
            content = template_file.read_text()

            # Parse multiple versions from the file
            versions = self._parse_template_versions(content)
            self.templates[intent] = versions

    def _parse_template_versions(self, content: str) -> Dict[str, str]:
        """Parse multiple template versions from a single file."""
        versions = {}
        current_version = None
        current_lines = []

        for line in content.split("\n"):
            # Match version headers like "-- Version 1: ..."
            version_match = re.match(r"--\s*Version\s+(\d+):", line, re.IGNORECASE)
            if version_match:
                # Save previous version
                if current_version and current_lines:
                    cypher = "\n".join(current_lines).strip()
                    versions[f"v{current_version}"] = cypher

                current_version = version_match.group(1)
                current_lines = []
            elif current_version and not line.strip().startswith("--"):
                # Non-comment line belongs to current version
                current_lines.append(line)

        # Save last version
        if current_version and current_lines:
            cypher = "\n".join(current_lines).strip()
            versions[f"v{current_version}"] = cypher

        return versions

    def has(self, intent: str) -> bool:
        """Check if template exists for intent."""
        return intent in self.templates

    def get(self, intent: str, version: str = "v1") -> Optional[str]:
        """Get template for intent and version."""
        if intent not in self.templates:
            return None
        return self.templates[intent].get(version)

    def render(
        self, intent: str, entities: Dict[str, Any], version: str = "v1"
    ) -> Tuple[str, Dict[str, Any]]:
        """Render template with entities as parameters."""
        template = self.get(intent, version)
        if not template:
            raise ValueError(
                f"No template found for intent '{intent}' version '{version}'"
            )

        # Parameters are passed as-is; Cypher uses $param syntax
        return template, entities


class IntentClassifier:
    """Classifies natural language queries into intents."""

    # Intent patterns with keywords and regex
    INTENT_PATTERNS = {
        "search": [
            r"\b(search|find|show|list|get|retrieve)\b",
            r"\b(documentation|docs|sections?|about)\b",
        ],
        "traverse": [
            r"\b(traverse|explore|follow|navigate|relationships?)\b",
            r"\b(from|starting|connected to|related to)\b",
        ],
        "troubleshoot": [
            r"\b(troubleshoot|fix|resolve|solve|error|issue|problem)\b",
            r"\b(failed|failing|broken|not working)\b",
        ],
        "compare": [
            r"\b(compare|difference|versus|vs|between)\b",
            r"\b(similar|different|contrast)\b",
        ],
        "explain": [
            r"\b(explain|what is|what does|how does|describe|architecture)\b",
            r"\b(concept|component|system|work|node|backend|frontend)\b",
        ],
    }

    def classify(self, nl_query: str) -> str:
        """Classify query into an intent."""
        nl_lower = nl_query.lower()
        scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, nl_lower))
            scores[intent] = score

        # Return intent with highest score, default to 'search'
        if max(scores.values()) == 0:
            return "search"

        return max(scores, key=scores.get)


class EntityLinker:
    """Links entities mentioned in queries to graph nodes."""

    # Common entity patterns
    ENTITY_PATTERNS = {
        "error_code": r"\b[EW]\d{3,5}\b",  # E123, W4567
        "command": r"\b(?:weka|fs|snap|nfs|smb)\s+[a-z_-]+\b",
        "config": r"\b[A-Z_]{3,}\b",  # CONFIG_NAME
        "component": r"\b(?:cluster|node|client|backend|frontend)\b",
    }

    def link(self, nl_query: str) -> Dict[str, Any]:
        """Extract entities from natural language query."""
        entities = {}

        # Error codes
        error_codes = re.findall(self.ENTITY_PATTERNS["error_code"], nl_query)
        if error_codes:
            entities["error_code"] = error_codes[0]
            entities["error_name"] = error_codes[0]  # Fallback

        # Commands
        commands = re.findall(self.ENTITY_PATTERNS["command"], nl_query.lower())
        if commands:
            entities["command_name"] = commands[0]

        # Components
        components = re.findall(self.ENTITY_PATTERNS["component"], nl_query.lower())
        if components:
            entities["component_name"] = components[0]

        return entities


class QueryPlanner:
    """Main query planner - templates first, with LLM fallback."""

    def __init__(self):
        self.config = get_config()
        self.templates = TemplateLibrary()
        self.classifier = IntentClassifier()
        self.linker = EntityLinker()

    def plan(self, nl_query: str, filters: Optional[Dict] = None) -> QueryPlan:
        """
        Plan a query from natural language.
        Templates-first; LLM proposal only as fallback (not implemented yet).
        """
        # Step 1: Classify intent
        intent = self.classifier.classify(nl_query)

        # Step 2: Link entities
        entities = self.linker.link(nl_query)

        # Step 3: Add standard parameters
        params = self._build_params(entities, filters)

        # Step 4: Render template if available
        if self.templates.has(intent):
            try:
                cypher, params = self.templates.render(intent, params)
                cypher = self._inject_limits_and_constraints(cypher, params)
                return QueryPlan(
                    intent=intent,
                    cypher=cypher,
                    params=params,
                    template_name=f"{intent}_v1",
                    confidence=1.0,
                )
            except Exception:
                # Template rendering failed, fall through to LLM
                pass

        # Step 5: LLM fallback (not implemented - would call LLM to generate Cypher)
        # For now, return a basic search template
        return self._fallback_plan(nl_query, intent, params)

    def _build_params(
        self, entities: Dict[str, Any], filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build parameter dictionary with standard constraints."""
        params = entities.copy()

        # Add standard limits from config
        params.setdefault("limit", 100)  # Default limit
        params.setdefault("max_depth", self.config.validator.max_depth)
        params.setdefault("max_hops", self.config.validator.max_depth)

        # Add filters if provided
        if filters:
            params.update(filters)

        # For search intent, add empty section_ids if not provided
        if "section_ids" not in params:
            params["section_ids"] = []

        # For traverse intent, ensure rel_types is a list
        if "rel_types" not in params:
            params["rel_types"] = [
                "MENTIONS",
                "CONTAINS_STEP",
                "HAS_PARAMETER",
                "REQUIRES",
                "AFFECTS",
            ]

        return params

    def _inject_limits_and_constraints(
        self, cypher: str, params: Dict[str, Any]
    ) -> str:
        """Ensure LIMIT and depth constraints are present in query."""
        # If no LIMIT clause exists, add one
        if not re.search(r"\bLIMIT\b", cypher, re.IGNORECASE):
            cypher = cypher.rstrip(";") + "\nLIMIT $limit;"

        # Ensure variable-length patterns use parameter
        cypher = re.sub(r"\*(\d+)\.\.(\d+)", r"*1..$max_hops", cypher)

        return cypher

    def _fallback_plan(
        self, nl_query: str, intent: str, params: Dict[str, Any]
    ) -> QueryPlan:
        """Generate a safe fallback query when no template matches."""
        # Use basic search template as fallback
        fallback_cypher = """
        MATCH (s:Section)
        WHERE s.id IN $section_ids
        RETURN s
        ORDER BY s.document_id, s.order
        LIMIT $limit
        """

        return QueryPlan(
            intent=intent,
            cypher=fallback_cypher.strip(),
            params=params,
            template_name="fallback_search",
            confidence=0.5,
        )

    def normalize_and_parameterize(self, raw_cypher: str) -> Tuple[str, Dict[str, Any]]:
        """
        Normalize a raw Cypher query and extract literals as parameters.
        Used for LLM-generated queries (future).
        """
        params = {}
        cypher = raw_cypher

        # Extract string literals
        def replace_string(match):
            value = match.group(1) or match.group(2)
            param_name = f"param_{hashlib.md5(value.encode()).hexdigest()[:8]}"
            params[param_name] = value
            return f"${param_name}"

        cypher = re.sub(r"'([^']*)'|\"([^\"]*)\"", replace_string, cypher)

        # Extract numeric literals (excluding in patterns like *1..2)
        def replace_number(match):
            if match.group(0).startswith("*"):
                return match.group(0)
            value = match.group(1)
            param_name = f"num_{value}"
            params[param_name] = int(value) if "." not in value else float(value)
            return f"${param_name}"

        cypher = re.sub(r"\b(\d+(?:\.\d+)?)\b", replace_number, cypher)

        return cypher, params
