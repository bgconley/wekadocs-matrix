"""
Shared Neo4j schema metadata.

Centralizes the canonical node/relationship allow-lists so safety guards
stay aligned with ingestion output. Update these sets whenever the schema
expands to keep ExplainGuard and other validators in sync.
"""

# Enumerates relationship types materialized by ingestion and referenced by
# query templates / traversal utilities. Keep sorted for readability.
RELATIONSHIP_TYPES = {
    "AFFECTS",
    "ANSWERED_AS",
    "CHILD_OF",
    "CONTAINS_STEP",
    "CRITICAL_FOR",
    "DEFINES",  # C.1.1: heading concept → chunk
    "DEPENDS_ON",
    "EXECUTES",
    "FOCUSED_ON",
    "HAS_CITATION",
    "HAS_PARAMETER",
    "HAS_QUERY",
    "HAS_SECTION",
    "IN_CHUNK",
    "IN_SECTION",  # Structure expansion: chunk → section
    "MENTIONED_IN",  # Entity → chunk (inverse of MENTIONS)
    "MENTIONS",
    "NEXT",
    "NEXT_CHUNK",
    "PARENT_OF",
    "PREV",
    "RELATED_TO",
    "REQUIRES",
    "RESOLVES",
    "RETRIEVED",
    "SAME_HEADING",
    "SUPPORTED_BY",
}
