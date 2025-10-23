#!/usr/bin/env python3
"""
Phase 3: Qdrant Safety & Incremental Tests
Tests for Pre-Phase 7 workstream C (Vector dimensionality invariants & Qdrant safety)
"""

import importlib.util
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def test_incremental_placeholder_vectors():
    """Test C1: Incremental.py uses config-driven dimensions"""
    print("Testing incremental.py placeholder vectors...")

    # Read the file and check for hardcoded dimensions
    incremental_path = os.path.join(project_root, "src/ingestion/incremental.py")
    with open(incremental_path, "r") as f:
        content = f.read()

    # Check that hardcoded 384 is gone from placeholder vectors
    assert (
        "[0.0] * 384" not in content
    ), "Found hardcoded 384 dimension in incremental.py"

    # Check that config.embedding.dims is used
    assert (
        "config.embedding.dims" in content or "embedding_dims" in content
    ), "incremental.py doesn't use config-driven dimensions"

    print("✓ PASS: Incremental uses config-driven dimensions")
    return True


def test_upsert_validated_exists():
    """Test C2: CompatQdrantClient has upsert_validated method"""
    print("Testing upsert_validated method exists...")

    # Import connections module
    try:
        from src.shared.connections import CompatQdrantClient
    except ImportError:
        # Fallback to direct module loading
        spec = importlib.util.spec_from_file_location(
            "connections", os.path.join(project_root, "src/shared/connections.py")
        )
        connections = importlib.util.module_from_spec(spec)
        # Temporarily add required modules to sys.modules
        sys.modules["src"] = type(sys)("src")
        sys.modules["src.shared"] = type(sys)("src.shared")
        sys.modules["src.shared.config"] = type(sys)("src.shared.config")
        sys.modules["src.shared.observability"] = type(sys)("src.shared.observability")
        spec.loader.exec_module(connections)
        CompatQdrantClient = connections.CompatQdrantClient

    # Check that CompatQdrantClient has upsert_validated
    assert hasattr(
        CompatQdrantClient, "upsert_validated"
    ), "CompatQdrantClient missing upsert_validated method"

    # Check method signature
    import inspect

    sig = inspect.signature(CompatQdrantClient.upsert_validated)
    params = list(sig.parameters.keys())

    assert "expected_dim" in params, "upsert_validated missing expected_dim parameter"
    assert "points" in params, "upsert_validated missing points parameter"

    print("✓ PASS: upsert_validated method exists with correct signature")
    return True


def test_blue_green_helper_exists():
    """Test C4: Blue/green collection helper exists"""
    print("Testing blue/green collection helper...")

    # Import connections module
    try:
        from src.shared.connections import CompatQdrantClient
    except ImportError:
        # Fallback to direct module loading
        spec = importlib.util.spec_from_file_location(
            "connections", os.path.join(project_root, "src/shared/connections.py")
        )
        connections = importlib.util.module_from_spec(spec)
        # Temporarily add required modules to sys.modules
        sys.modules["src"] = type(sys)("src")
        sys.modules["src.shared"] = type(sys)("src.shared")
        sys.modules["src.shared.config"] = type(sys)("src.shared.config")
        sys.modules["src.shared.observability"] = type(sys)("src.shared.observability")
        spec.loader.exec_module(connections)
        CompatQdrantClient = connections.CompatQdrantClient

    # Check that CompatQdrantClient has create_collection_with_dims
    assert hasattr(
        CompatQdrantClient, "create_collection_with_dims"
    ), "CompatQdrantClient missing create_collection_with_dims helper"

    # Check method signature
    import inspect

    sig = inspect.signature(CompatQdrantClient.create_collection_with_dims)
    params = list(sig.parameters.keys())

    assert "size" in params, "create_collection_with_dims missing size parameter"
    assert (
        "distance" in params
    ), "create_collection_with_dims missing distance parameter"

    print("✓ PASS: Blue/green collection helper exists")
    return True


def test_all_upserts_use_validated():
    """Test C3: All upsert calls use upsert_validated"""
    print("Testing all upsert calls use validated version...")

    files_to_check = [
        "src/ingestion/incremental.py",
        "src/ingestion/reconcile.py",
        "src/ingestion/build_graph.py",
        "src/ingestion/auto/orchestrator.py",
    ]

    for file_path in files_to_check:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            print(f"  Skipping {file_path} (not found)")
            continue

        with open(full_path, "r") as f:
            content = f.read()

        # Count upsert calls
        # Look for .upsert( but NOT .upsert_validated(
        import re

        # Find all upsert calls
        upsert_pattern = r"\.upsert\s*\("
        upsert_validated_pattern = r"\.upsert_validated\s*\("

        all_upserts = re.findall(upsert_pattern, content)
        validated_upserts = re.findall(upsert_validated_pattern, content)

        # The only plain upsert should be in connections.py itself
        # All others should use upsert_validated
        plain_upserts = len(all_upserts) - len(validated_upserts)

        if "connections.py" not in file_path:
            # For non-connections files, all upserts should be validated
            # (allowing for the one in upsert_validated implementation itself)
            if plain_upserts > 0:
                # Check if these are comments or inside upsert_validated definition
                lines = content.split("\n")
                actual_plain = 0
                for i, line in enumerate(lines):
                    if ".upsert(" in line and ".upsert_validated(" not in line:
                        # Check if it's a comment or inside the validated method
                        if not line.strip().startswith("#"):
                            # Check context - might be inside upsert_validated
                            context_start = max(0, i - 5)
                            context = "\n".join(lines[context_start : i + 1])
                            if "def upsert_validated" not in context:
                                actual_plain += 1

                assert (
                    actual_plain == 0
                ), f"{file_path} has {actual_plain} non-validated upsert calls"

        print(f"  ✓ {file_path}: All upserts are validated")

    print("✓ PASS: All upsert calls use upsert_validated")
    return True


def test_dimension_validation_logic():
    """Test that dimension validation would catch mismatches"""
    print("Testing dimension validation logic...")

    # This is a logical test - we verify the validation code exists
    connections_path = os.path.join(project_root, "src/shared/connections.py")
    with open(connections_path, "r") as f:
        content = f.read()

    # Check for dimension validation logic in upsert_validated
    assert "expected_dim" in content, "No expected_dim handling"
    assert "ValueError" in content, "No ValueError for dimension mismatch"
    assert (
        "Dimension mismatch" in content or "dimension mismatch" in content.lower()
    ), "No clear dimension mismatch error message"

    print("✓ PASS: Dimension validation logic present")
    return True


def test_documentation_exists():
    """Test C4: Blue/green migration documentation exists"""
    print("Testing blue/green migration documentation...")

    doc_path = os.path.join(project_root, "docs/blue-green-migration.md")
    assert os.path.exists(doc_path), "Blue/green migration documentation not found"

    with open(doc_path, "r") as f:
        content = f.read()

    # Check for key sections
    assert "Blue/Green Collection Migration" in content, "Missing migration title"
    assert "Pre-Phase 7" in content, "Missing Pre-Phase 7 reference"
    assert "create_collection_with_dims" in content, "Missing helper method reference"
    assert "Rollback" in content, "Missing rollback procedure"

    print("✓ PASS: Blue/green migration documentation complete")
    return True


def main():
    """Run all Phase 3 tests"""
    print("=" * 60)
    print("Phase 3 Qdrant Safety Tests")
    print("=" * 60)

    tests = [
        ("C1: Incremental placeholder vectors", test_incremental_placeholder_vectors),
        ("C2: upsert_validated exists", test_upsert_validated_exists),
        ("C3: All upserts validated", test_all_upserts_use_validated),
        ("C4: Blue/green helper", test_blue_green_helper_exists),
        ("Dimension validation", test_dimension_validation_logic),
        ("Documentation", test_documentation_exists),
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
        print("🎉 All Phase 3 tests passed! Ready to proceed to Phase 4.")
        return 0
    else:
        print(f"❌ {failed} tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
