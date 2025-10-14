"""
Advanced query templates with schemas and guardrails.
Phase 4, Task 4.1
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

TEMPLATE_DIR = Path(__file__).parent


class TemplateGuardrails(BaseModel):
    """Guardrails for template execution."""

    max_depth: int = Field(default=3, ge=1, le=10)
    max_results: int = Field(default=100, ge=1, le=1000)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    allowed_rel_types: Optional[List[str]] = None
    requires_indexes: Optional[List[str]] = None
    estimated_row_limit: int = Field(default=10000, ge=100, le=100000)


class TemplateSchema(BaseModel):
    """Schema definition for a query template."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    guardrails: TemplateGuardrails
    file_path: str
    versions: List[str] = Field(default_factory=lambda: ["v1", "v2", "v3"])
