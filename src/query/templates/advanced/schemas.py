"""
Template schemas and guardrails for advanced query patterns.
Phase 4, Task 4.1
"""

from typing import Dict

from . import TEMPLATE_DIR, TemplateGuardrails, TemplateSchema

DEPENDENCY_CHAIN = TemplateSchema(
    name="dependency_chain",
    description="Trace dependency chains for components",
    input_schema={
        "type": "object",
        "properties": {
            "component_name": {"type": "string"},
            "max_depth": {"type": "integer", "default": 5},
        },
        "required": ["component_name"],
    },
    output_schema={"type": "array"},
    guardrails=TemplateGuardrails(
        max_depth=5,
        max_results=100,
        timeout_ms=30000,
        allowed_rel_types=["DEPENDS_ON", "CRITICAL_FOR"],
        estimated_row_limit=1000,
    ),
    file_path=str(TEMPLATE_DIR / "dependency_chain.cypher"),
)

IMPACT_ASSESSMENT = TemplateSchema(
    name="impact_assessment",
    description="Analyze impact of configuration changes",
    input_schema={
        "type": "object",
        "properties": {
            "config_name": {"type": "string"},
            "max_hops": {"type": "integer", "default": 3},
        },
        "required": ["config_name"],
    },
    output_schema={"type": "array"},
    guardrails=TemplateGuardrails(
        max_depth=3,
        max_results=100,
        timeout_ms=30000,
        allowed_rel_types=["AFFECTS", "CRITICAL_FOR"],
        estimated_row_limit=2000,
    ),
    file_path=str(TEMPLATE_DIR / "impact_assessment.cypher"),
)

COMPARISON = TemplateSchema(
    name="comparison",
    description="Compare configurations or procedures",
    input_schema={
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "entity_name_a": {"type": "string"},
            "entity_name_b": {"type": "string"},
        },
        "required": ["entity_type", "entity_name_a", "entity_name_b"],
    },
    output_schema={"type": "array"},
    guardrails=TemplateGuardrails(
        max_depth=2, max_results=100, timeout_ms=20000, estimated_row_limit=500
    ),
    file_path=str(TEMPLATE_DIR / "comparison.cypher"),
)

TEMPORAL = TemplateSchema(
    name="temporal",
    description="Query documentation state as of version",
    input_schema={
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "entity_type": {"type": "string"},
            "entity_name": {"type": "string"},
        },
        "required": ["version"],
    },
    output_schema={"type": "array"},
    guardrails=TemplateGuardrails(
        max_depth=2,
        max_results=100,
        timeout_ms=25000,
        estimated_row_limit=1000,
        requires_indexes=["introduced_in", "deprecated_in"],
    ),
    file_path=str(TEMPLATE_DIR / "temporal.cypher"),
)

TROUBLESHOOTING_PATH = TemplateSchema(
    name="troubleshooting_path",
    description="Advanced error troubleshooting with full resolution paths",
    input_schema={
        "type": "object",
        "properties": {
            "error_code": {"type": "string"},
            "error_name": {"type": "string"},
            "include_related": {"type": "boolean", "default": True},
        },
        "required": ["error_code"],
    },
    output_schema={"type": "array"},
    guardrails=TemplateGuardrails(
        max_depth=3,
        max_results=10,
        timeout_ms=30000,
        allowed_rel_types=[
            "RESOLVES",
            "CONTAINS_STEP",
            "EXECUTES",
            "RELATED_TO",
            "HAS_PARAMETER",
        ],
        estimated_row_limit=500,
    ),
    file_path=str(TEMPLATE_DIR / "troubleshooting_path.cypher"),
)

# Template registry
ADVANCED_TEMPLATES: Dict[str, TemplateSchema] = {
    "dependency_chain": DEPENDENCY_CHAIN,
    "impact_assessment": IMPACT_ASSESSMENT,
    "comparison": COMPARISON,
    "temporal": TEMPORAL,
    "troubleshooting_path": TROUBLESHOOTING_PATH,
}


def get_template(name: str) -> TemplateSchema:
    """Get template schema by name."""
    if name not in ADVANCED_TEMPLATES:
        raise ValueError(f"Unknown template: {name}")
    return ADVANCED_TEMPLATES[name]


def list_templates() -> Dict[str, TemplateSchema]:
    """List all available advanced templates."""
    return ADVANCED_TEMPLATES.copy()
