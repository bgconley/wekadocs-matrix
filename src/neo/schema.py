"""
Shared Neo4j schema metadata.

Centralizes the canonical node/relationship allow-lists so safety guards
stay aligned with ingestion output. Update these sets whenever the schema
expands to keep ExplainGuard and other validators in sync.
"""

# Enumerates relationship types materialized by ingestion and referenced by
# query templates / traversal utilities. Keep sorted for readability.
# Phase 2 Cleanup: Removed dead types that were never materialized by ingestion
# or queried at runtime: AFFECTS, CRITICAL_FOR, DEPENDS_ON, RELATED_TO, REQUIRES
# Also removed redundant edges: PREV (use <-[:NEXT]-), SAME_HEADING (O(n²) fanout)
RELATIONSHIP_TYPES = {
    "ANSWERED_AS",
    "CHILD_OF",
    "CONTAINS_STEP",
    "DEFINES",  # C.1.1: heading concept → chunk
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
    "REFERENCES",  # Phase 3: Cross-document connectivity (Chunk → Document)
    "RESOLVES",
    "RETRIEVED",
    "SUPPORTED_BY",
}
