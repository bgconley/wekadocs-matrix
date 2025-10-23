#!/usr/bin/env python3
"""
Phase 4: Ranking Stability & Coverage Tests
Tests for Pre-Phase 7 workstream D (Ranking stability & signals)
"""

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def test_ranking_normalization():
    """Test D1: Score normalization for similarity vs distance"""
    print("Testing ranking score normalization...")

    try:
        # Properly set up the import path
        from src.query.ranking import Ranker
    except ImportError:
        # If direct import fails, manually load with proper mocking
        spec = importlib.util.spec_from_file_location(
            "ranking", os.path.join(project_root, "src/query/ranking.py")
        )
        ranking = importlib.util.module_from_spec(spec)

        # Create complete mock structure
        import types

        # Mock the config module
        mock_config = types.ModuleType("config")

        class MockSettings:
            class search:
                class hybrid:
                    vector_weight = 0.6
                    graph_weight = 0.4

        mock_config.get_config = lambda: MockSettings()
        sys.modules["src.shared.config"] = mock_config

        # Mock the observability module
        mock_obs = types.ModuleType("observability")

        class MockLogger:
            def debug(self, *args, **kwargs):
                pass

            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        mock_obs.get_logger = lambda name: MockLogger()
        sys.modules["src.shared.observability"] = mock_obs

        # Mock SearchResult
        mock_search = types.ModuleType("hybrid_search")

        class MockSearchResult:
            def __init__(self, node_id, node_label, score, distance, metadata):
                self.node_id = node_id
                self.node_label = node_label
                self.score = score
                self.distance = distance
                self.metadata = metadata

        mock_search.SearchResult = MockSearchResult
        sys.modules["src.query.hybrid_search"] = mock_search

        # Now load the module
        spec.loader.exec_module(ranking)
        Ranker = ranking.Ranker

    # Create ranker instance and test actual functionality
    ranker = Ranker()

    # Test similarity normalization
    similarity_score = ranker._normalize_score(0.8, "similarity")
    assert (
        0.7 <= similarity_score <= 0.9
    ), f"Similarity score {similarity_score} out of range"

    # Test distance normalization (distance -> similarity)
    distance_score = ranker._normalize_score(
        0.5, "distance"
    )  # 0.5 distance -> 0.75 similarity
    assert (
        0.7 <= distance_score <= 0.8
    ), f"Distance conversion {distance_score} incorrect"

    # Test negative similarity handling
    neg_similarity = ranker._normalize_score(-0.5, "similarity")
    assert (
        0.2 <= neg_similarity <= 0.3
    ), f"Negative similarity {neg_similarity} not handled correctly"

    print("✓ PASS: Score normalization working correctly")
    return True


def test_utc_recency_scoring():
    """Test D1: UTC handling in recency scoring"""
    print("Testing UTC handling in recency scoring...")

    try:
        from src.query.ranking import Ranker
    except ImportError:
        # Reuse the mocking setup
        import types

        # Load module with mocks
        spec = importlib.util.spec_from_file_location(
            "ranking", os.path.join(project_root, "src/query/ranking.py")
        )
        ranking = importlib.util.module_from_spec(spec)

        # Mock dependencies
        mock_config = types.ModuleType("config")

        class MockSettings:
            class search:
                class hybrid:
                    vector_weight = 0.6
                    graph_weight = 0.4

        mock_config.get_config = lambda: MockSettings()
        sys.modules["src.shared.config"] = mock_config

        mock_obs = types.ModuleType("observability")

        class MockLogger:
            def debug(self, *args, **kwargs):
                pass

            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        mock_obs.get_logger = lambda name: MockLogger()
        sys.modules["src.shared.observability"] = mock_obs

        mock_search = types.ModuleType("hybrid_search")

        class MockSearchResult:
            pass

        mock_search.SearchResult = MockSearchResult
        sys.modules["src.query.hybrid_search"] = mock_search

        spec.loader.exec_module(ranking)
        Ranker = ranking.Ranker

    ranker = Ranker()

    # Test with UTC timestamp
    now = datetime.now(timezone.utc)
    recent_doc = {
        "updated_at": (now - timedelta(days=7)).isoformat()  # Already has timezone
    }
    recent_score = ranker._recency_score(recent_doc)
    assert 0.9 <= recent_score <= 1.0, f"Recent doc score {recent_score} too low"

    # Test with old document
    old_doc = {
        "created_at": (
            now - timedelta(days=365 * 2)
        ).isoformat()  # Already has timezone
    }
    old_score = ranker._recency_score(old_doc)
    assert old_score < 0.4, f"Old doc score {old_score} too high"

    # Test naive timestamp conversion
    naive_doc = {"last_edited": datetime.now().isoformat()}  # No timezone
    naive_score = ranker._recency_score(naive_doc)
    assert 0.9 <= naive_score <= 1.0, "Naive timestamp not handled correctly"

    print("✓ PASS: UTC recency scoring working correctly")
    return True


def test_tie_breaking_stability():
    """Test D1: Tie-breaking with label priority and node_id"""
    print("Testing tie-breaking stability...")

    try:
        from src.query.hybrid_search import SearchResult
        from src.query.ranking import RankedResult, Ranker, RankingFeatures
    except ImportError:
        import types

        # Load module with complete mocking
        spec = importlib.util.spec_from_file_location(
            "ranking", os.path.join(project_root, "src/query/ranking.py")
        )
        ranking = importlib.util.module_from_spec(spec)

        # Mock config
        mock_config = types.ModuleType("config")

        class MockSettings:
            class search:
                class hybrid:
                    vector_weight = 0.6
                    graph_weight = 0.4

        mock_config.get_config = lambda: MockSettings()
        sys.modules["src.shared.config"] = mock_config

        # Mock observability
        mock_obs = types.ModuleType("observability")

        class MockLogger:
            def debug(self, *args, **kwargs):
                pass

            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        mock_obs.get_logger = lambda name: MockLogger()
        sys.modules["src.shared.observability"] = mock_obs

        # Mock SearchResult
        mock_search = types.ModuleType("hybrid_search")

        class MockSearchResult:
            def __init__(self, node_id, node_label, score=0.5):
                self.node_id = node_id
                self.node_label = node_label
                self.score = score
                self.distance = 0
                self.metadata = {}

        mock_search.SearchResult = MockSearchResult
        sys.modules["src.query.hybrid_search"] = mock_search

        spec.loader.exec_module(ranking)
        Ranker = ranking.Ranker
        RankedResult = ranking.RankedResult
        RankingFeatures = ranking.RankingFeatures
        SearchResult = MockSearchResult

    ranker = Ranker()

    # Create results with same scores but different labels
    results = [
        SearchResult("node3", "Section", 0.5, 0, {}),  # Priority 1.0
        SearchResult("node1", "Procedure", 0.5, 0, {}),  # Priority 1.0 (tie)
        SearchResult("node2", "Command", 0.5, 0, {}),  # Priority 0.9
    ]

    # Manually create ranked results with same final score
    ranked = []
    for r in results:
        rr = RankedResult(result=r, features=RankingFeatures())
        rr.features.final_score = 0.5  # Same score for all
        ranked.append(rr)

    # Apply tie-breaking
    ranked = ranker._break_ties(ranked)

    # Check ordering: Should be by priority (Section/Procedure > Command), then node_id
    assert (
        ranked[0].result.node_id == "node1"
    ), "Procedure should be first (alphabetically)"
    assert ranked[1].result.node_id == "node3", "Section should be second"
    assert (
        ranked[2].result.node_id == "node2"
    ), "Command should be last (lower priority)"

    print("✓ PASS: Tie-breaking works correctly")
    return True


def test_coverage_enrichment():
    """Test D2: Coverage enrichment in hybrid_search"""
    print("Testing coverage enrichment...")

    # Check that hybrid_search has _enrich_with_coverage method
    hybrid_path = os.path.join(project_root, "src/query/hybrid_search.py")
    with open(hybrid_path, "r") as f:
        content = f.read()

    # Check for coverage enrichment method
    assert "_enrich_with_coverage" in content, "Coverage enrichment method not found"
    assert "connection_count" in content, "Connection count not tracked"
    assert "mention_count" in content, "Mention count not tracked"

    # Check for batched Cypher query
    assert "UNWIND $ids AS sid" in content, "Batched query not found"
    assert "count(DISTINCT r) AS conn_count" in content, "Connection counting missing"
    assert "count(DISTINCT e) AS mention_count" in content, "Mention counting missing"

    print("✓ PASS: Coverage enrichment implemented")
    return True


def test_score_kind_tagging():
    """Test D3: Score kind tagging in hybrid_search"""
    print("Testing score kind tagging...")

    hybrid_path = os.path.join(project_root, "src/query/hybrid_search.py")
    with open(hybrid_path, "r") as f:
        content = f.read()

    # Check that score_kind is tagged
    assert (
        '"score_kind"' in content or "'score_kind'" in content
    ), "Score kind not tagged"
    assert (
        'metadata["score_kind"] = "similarity"' in content
    ), "Similarity tagging missing"

    print("✓ PASS: Score kind tagging implemented")
    return True


def test_entity_focus_hook():
    """Test D1: Entity focus hook exists (no-op for now)"""
    print("Testing entity focus hook...")

    ranking_path = os.path.join(project_root, "src/query/ranking.py")
    with open(ranking_path, "r") as f:
        content = f.read()

    # Check for entity focus hook
    assert "_entity_focus_bonus" in content, "Entity focus hook not found"
    assert "return 0.0" in content, "Hook should return 0.0 in Pre-Phase 7"
    assert "Phase 7" in content, "Should mention Phase 7 implementation"

    print("✓ PASS: Entity focus hook present (no-op)")
    return True


def test_chunk_label_support():
    """Test D1: Chunk label treated as Section"""
    print("Testing Chunk label support...")

    ranking_path = os.path.join(project_root, "src/query/ranking.py")
    with open(ranking_path, "r") as f:
        content = f.read()

    # Check that Chunk is in LABEL_PRIORITIES
    assert '"Chunk": 1.0' in content, "Chunk label not added to priorities"
    assert (
        "Treat Chunk as Section" in content or "Chunk" in content
    ), "Chunk support not documented"

    print("✓ PASS: Chunk label properly supported")
    return True


def test_import_logger():
    """Test that logger is imported in modified files"""
    print("Testing logger imports...")

    # Check ranking.py
    ranking_path = os.path.join(project_root, "src/query/ranking.py")
    with open(ranking_path, "r") as f:
        ranking_content = f.read()

    assert (
        "from src.shared.observability import get_logger" in ranking_content
    ), "Logger not imported in ranking.py"
    assert (
        "logger = get_logger(__name__)" in ranking_content
    ), "Logger not initialized in ranking.py"

    print("✓ PASS: Logger properly imported")
    return True


def main():
    """Run all Phase 4 tests"""
    print("=" * 60)
    print("Phase 4 Ranking & Coverage Tests")
    print("=" * 60)

    tests = [
        ("D1: Score normalization", test_ranking_normalization),
        ("D1: UTC recency scoring", test_utc_recency_scoring),
        ("D1: Tie-breaking stability", test_tie_breaking_stability),
        ("D1: Entity focus hook", test_entity_focus_hook),
        ("D1: Chunk label support", test_chunk_label_support),
        ("D2: Coverage enrichment", test_coverage_enrichment),
        ("D3: Score kind tagging", test_score_kind_tagging),
        ("Logger imports", test_import_logger),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR in {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("🎉 All Phase 4 tests passed! Ready to proceed to Phase 5.")
        return 0
    else:
        print(f"❌ {failed} tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
