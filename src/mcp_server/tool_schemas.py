from __future__ import annotations

from src.mcp_server.mcp_utils import (
    KB_EVIDENCE_INTERNAL_FETCH_K,
    KB_EVIDENCE_MAX_QUOTES,
    KB_SEARCH_DEFAULT_TOP_K,
)

__all__ = [
    "_error_schema",
    "_with_error",
    "BASE_META_SCHEMA",
    "SCOPE_SCHEMA",
    "FILTERS_SCHEMA",
    "KB_SEARCH_OPTIONS_SCHEMA",
    "KB_SEARCH_INPUT_SCHEMA",
    "KB_EXCERPT_INPUT_SCHEMA",
    "KB_EXPAND_INPUT_SCHEMA",
    "KB_EXTRACT_INPUT_SCHEMA",
    "KB_RETRIEVE_INPUT_SCHEMA",
    "KB_SEARCH_OUTPUT_SCHEMA",
    "KB_EXCERPT_OUTPUT_SCHEMA",
    "KB_EVIDENCE_OUTPUT_SCHEMA",
    "SEARCH_SECTIONS_INPUT_SCHEMA",
    "GENERIC_GRAPH_INPUT_SCHEMA",
    "GENERIC_GRAPH_OUTPUT_SCHEMA",
    "GRAPH_NODE_SCHEMA",
    "GRAPH_EDGE_SCHEMA",
    "GRAPH_PATH_SCHEMA",
    "GRAPH_DESCRIBE_OUTPUT_SCHEMA",
    "GRAPH_EXPAND_OUTPUT_SCHEMA",
    "GRAPH_PATHS_OUTPUT_SCHEMA",
    "GRAPH_CHILDREN_OUTPUT_SCHEMA",
    "GRAPH_PARENTS_OUTPUT_SCHEMA",
    "GRAPH_ENTITIES_OUTPUT_SCHEMA",
    "GRAPH_SECTIONS_OUTPUT_SCHEMA",
]


def _error_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "error": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": "object"},
                },
                "required": ["code", "message"],
                "additionalProperties": True,
            }
        },
        "required": ["error"],
        "additionalProperties": True,
    }


def _with_error(normal_schema: dict) -> dict:
    return {"oneOf": [normal_schema, _error_schema()]}


BASE_META_SCHEMA = {
    "type": "object",
    "properties": {
        "usage": {
            "type": "object",
            "properties": {
                "tokens_estimate": {"type": "number"},
                "bytes_returned": {"type": "number"},
                "duplicates_suppressed": {"type": "number"},
            },
            "additionalProperties": True,
        }
    },
    "additionalProperties": True,
}

SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "project_id": {"type": "string"},
        "environment": {"type": "string"},
        "doc_tags": {
            "oneOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "string"},
            ]
        },
        "repositories": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

FILTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_tag": {"type": "array", "items": {"type": "string"}},
        "path_prefix": {"type": ["string", "null"]},
        "updated_after": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

KB_SEARCH_OPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "max_snippet_chars": {"type": "integer"},
        "max_per_doc": {"type": "integer"},
        "include_scores": {"type": "boolean"},
        "include_debug": {"type": "boolean"},
        "mode": {"type": "string"},
    },
    "additionalProperties": True,
}

KB_SEARCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": KB_SEARCH_DEFAULT_TOP_K},
        "cursor": {"type": ["string", "null"]},
        "page_size": {"type": "integer", "default": KB_SEARCH_DEFAULT_TOP_K},
        "scope": SCOPE_SCHEMA,
        "filters": FILTERS_SCHEMA,
        "options": KB_SEARCH_OPTIONS_SCHEMA,
        "session_id": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": True,
}

KB_EXCERPT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "passage_id": {"type": "string"},
        "max_tokens": {"type": "integer", "default": 300},
        "start_char": {"type": "integer", "default": 0},
        "scope": SCOPE_SCHEMA,
        "options": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "include_citation": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "session_id": {"type": "string"},
    },
    "required": ["passage_id"],
    "additionalProperties": True,
}

KB_EXPAND_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "passage_id": {"type": "string"},
        "before_tokens": {"type": "integer", "default": 150},
        "after_tokens": {"type": "integer", "default": 150},
        "scope": SCOPE_SCHEMA,
        "options": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "include_citation": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "session_id": {"type": "string"},
    },
    "required": ["passage_id"],
    "additionalProperties": True,
}

KB_EXTRACT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "passage_ids": {"type": "array", "items": {"type": "string"}},
        "max_quotes": {"type": "integer", "default": 6},
        "max_quote_tokens": {"type": "integer", "default": 80},
        "include_context_tokens": {"type": "integer", "default": 20},
        "scope": SCOPE_SCHEMA,
        "session_id": {"type": "string"},
    },
    "required": ["question", "passage_ids"],
    "additionalProperties": True,
}

KB_RETRIEVE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to find evidence for",
        },
        "max_quotes": {
            "type": "integer",
            "default": 6,
            "description": f"Max evidence quotes to return (1-{KB_EVIDENCE_MAX_QUOTES})",
        },
        "max_quote_tokens": {
            "type": "integer",
            "default": 80,
            "description": "Max tokens per quote",
        },
        "include_context_tokens": {
            "type": "integer",
            "default": 20,
            "description": "Context tokens around each quote",
        },
        "retrieval_depth": {
            "type": "integer",
            "default": KB_EVIDENCE_INTERNAL_FETCH_K,
            "description": "Internal search depth (default 60, max 150). Searches this many candidates to find the best quotes.",
        },
        "graph_enrichment": {
            "type": "boolean",
            "description": "Enable evidence-only structural graph expansion (default false for precision mode).",
        },
        "response_mode": {
            "type": "string",
            "enum": ["evidence_only", "evidence_plus_draft"],
            "default": "evidence_only",
            "description": (
                "Return evidence only (default), or evidence plus a "
                "citation-preserving draft answer. Draft mode is experimental "
                "and active only when the server sets KB_EVIDENCE_DRAFT_ENABLED; "
                "otherwise the server returns evidence only."
            ),
        },
        "top_k": {
            "type": "integer",
            "default": KB_SEARCH_DEFAULT_TOP_K,
            "description": "Backward compat alias for max_quotes",
        },
        "scope": SCOPE_SCHEMA,
        "filters": FILTERS_SCHEMA,
        "options": KB_SEARCH_OPTIONS_SCHEMA,
        "session_id": {"type": "string"},
    },
    "required": ["question"],
    "additionalProperties": True,
}

KB_SEARCH_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": {"type": "object"}},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
            "diagnostic_id": {"type": "string"},
            "diagnostic_hint": {"type": "string"},
            "diagnostic_uri": {"type": "string"},
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

KB_EXCERPT_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "passage_id": {"type": "string"},
            "excerpt": {"type": "string"},
            "truncated": {"type": "boolean"},
            "next_start_char": {"type": "integer"},
            "citation": {"type": ["object", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": [
            "passage_id",
            "excerpt",
            "truncated",
            "next_start_char",
            "partial",
            "limit_reason",
            "session_id",
            "meta",
        ],
        "additionalProperties": True,
    }
)

KB_EVIDENCE_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "quotes": {"type": "array", "items": {"type": "object"}},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
            "diagnostic_id": {"type": "string"},
            "diagnostic_hint": {"type": "string"},
            "diagnostic_uri": {"type": "string"},
        },
        "required": ["quotes", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

SEARCH_SECTIONS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 20},
        "filters": {"type": "object"},
        "cursor": {"type": ["string", "null"]},
        "page_size": {"type": "integer"},
        "session_id": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": True,
}

GENERIC_GRAPH_INPUT_SCHEMA = {"type": "object", "additionalProperties": True}

GENERIC_GRAPH_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "label": {"type": "string"},
        "title": {"type": "string"},
        "level": {"type": ["integer", "null"]},
        "tokens": {"type": ["integer", "null"]},
        "doc_tag": {"type": ["string", "null"]},
        "anchor": {"type": ["string", "null"]},
        "snippet": {"type": "string"},
    },
    "required": ["id"],
    "additionalProperties": True,
}

GRAPH_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "src": {"type": "string"},
        "dst": {"type": "string"},
        "type": {"type": "string"},
    },
    "required": ["src", "dst", "type"],
    "additionalProperties": True,
}

GRAPH_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {"type": "array", "items": {"type": "string"}},
        "types": {"type": "array", "items": {"type": "string"}},
        "length": {"type": "integer"},
    },
    "required": ["nodes", "types", "length"],
    "additionalProperties": True,
}

GRAPH_DESCRIBE_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": GRAPH_NODE_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_EXPAND_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "items": GRAPH_NODE_SCHEMA},
            "edges": {"type": "array", "items": GRAPH_EDGE_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "dedupe_applied": {"type": "boolean"},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["nodes", "edges", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_PATHS_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": GRAPH_PATH_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["paths", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_CHILDREN_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "children": {"type": "array", "items": GRAPH_NODE_SCHEMA},
            "cursor": {"type": ["string", "null"]},
            "next_cursor": {"type": ["string", "null"]},
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["children", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_PARENTS_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "parent_id": {"type": "string"},
                        "parent_title": {"type": "string"},
                    },
                    "required": ["section_id", "parent_id", "parent_title"],
                    "additionalProperties": True,
                },
            },
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_ENTITIES_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                                "required": ["id", "label", "name"],
                                "additionalProperties": True,
                            },
                        },
                    },
                    "required": ["section_id", "entities"],
                    "additionalProperties": True,
                },
            },
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)

GRAPH_SECTIONS_OUTPUT_SCHEMA = _with_error(
    {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "section_id": {"type": "string"},
                                    "title": {"type": "string"},
                                },
                                "required": ["section_id", "title"],
                                "additionalProperties": True,
                            },
                        },
                    },
                    "required": ["entity_id", "sections"],
                    "additionalProperties": True,
                },
            },
            "partial": {"type": "boolean"},
            "limit_reason": {"type": "string"},
            "session_id": {"type": "string"},
            "meta": BASE_META_SCHEMA,
        },
        "required": ["results", "partial", "limit_reason", "session_id", "meta"],
        "additionalProperties": True,
    }
)
