#!/usr/bin/env python3
"""
Test script for Phase 5 (Pre-Phase 7) implementation.
Tests E1-E3 (response safeguards) and F1-F3 (schema v2.1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_byte_truncation():
    """Test E1: 32KB byte cap."""
    print("\n=== E1: Byte Truncation ===")
    try:
        from src.query.response_builder import ResponseBuilder

        b = ResponseBuilder(max_text_bytes=32768)

        assert b._truncate_to_bytes("short", 100) == "short"
        assert b._truncate_to_bytes("x" * 200, 100).endswith("...[truncated]")
        assert len(b._truncate_to_bytes("x" * 50000).encode("utf-8")) <= 32800
        print("✓ Byte truncation working")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_graph_mode_caps():
    """Test E2: GRAPH mode safety caps."""
    print("\n=== E2: GRAPH Mode Safety Caps ===")
    try:
        from src.query.response_builder import ResponseBuilder

        b = ResponseBuilder()

        # Check max_entities parameter exists
        import inspect

        sig = inspect.signature(b._get_related_entities)
        assert "max_entities" in sig.parameters
        assert sig.parameters["max_entities"].default == 20
        assert "max_depth" in sig.parameters
        print("✓ Safety caps present (max_entities=20, max_depth=1)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_citation_helper():
    """Test E3: Citation formatting."""
    print("\n=== E3: Citation Helper ===")
    try:
        from src.query.ranking import RankedResult, RankingFeatures, SearchResult
        from src.query.response_builder import ResponseBuilder

        b = ResponseBuilder()

        # Mock results
        r1 = RankedResult(
            rank=1,
            result=SearchResult(
                node_id="s1",
                node_label="Section",
                score=0.9,
                distance=0,
                metadata={
                    "title": "Install Guide",
                    "document_uri": "install.md",
                    "anchor": "prereqs",
                },
            ),
            features=RankingFeatures(0.9, 0.8, 0.7, 0.6, 0.5, 0.85),
        )

        citations = b.format_citations([r1], max_citations=5)
        assert "[1]" in citations
        assert "Install Guide" in citations
        assert "install.md#prereqs" in citations
        print("✓ Citation helper working")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_schema_v2_1_file_exists():
    """Test F1: Schema DDL exists."""
    print("\n=== F1: Schema v2.1 DDL ===")
    try:
        ddl_path = (
            Path(__file__).parent.parent / "scripts/neo4j/create_schema_v2_1.cypher"
        )
        assert ddl_path.exists(), f"DDL not found at {ddl_path}"

        content = ddl_path.read_text()
        assert "SET s:Chunk" in content
        assert "session_id_unique" in content
        assert "query_id_unique" in content
        assert "answer_id_unique" in content
        assert "SchemaVersion" in content
        print("✓ Schema v2.1 DDL exists and contains required statements")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_schema_v2_1_runner():
    """Test F2: Runner wired in schema.py."""
    print("\n=== F2: Schema v2.1 Runner ===")
    try:
        import inspect

        from src.shared import schema

        # Check apply_schema_v2_1 function exists
        assert hasattr(schema, "apply_schema_v2_1")

        # Check it's called in create_schema
        source = inspect.getsource(schema.create_schema)
        assert "apply_schema_v2_1" in source
        print("✓ Schema v2.1 runner wired correctly")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_response_builder_imports():
    """Verify response builder has all enhancements."""
    print("\n=== Response Builder Verification ===")
    try:
        from src.query.response_builder import ResponseBuilder

        b = ResponseBuilder()
        assert hasattr(b, "_truncate_to_bytes")
        assert hasattr(b, "format_citations")
        assert hasattr(b, "max_text_bytes")
        print("✓ Response builder has E1-E3 enhancements")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Run Phase 5 tests."""
    print("=" * 60)
    print("Phase 5: Response Safeguards & Schema v2.1")
    print("=" * 60)

    results = [
        ("E1: Byte truncation", test_byte_truncation()),
        ("E2: GRAPH mode caps", test_graph_mode_caps()),
        ("E3: Citation helper", test_citation_helper()),
        ("F1: Schema DDL exists", test_schema_v2_1_file_exists()),
        ("F2: Schema runner wired", test_schema_v2_1_runner()),
        ("Response builder complete", test_response_builder_imports()),
    ]

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"{'✓ PASS' if result else '✗ FAIL'}: {name}")

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 Phase 5 complete! Ready for Phase 6.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
