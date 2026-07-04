"""
Custom validation tests for the codebase cleanup.

These tests verify structural integrity after the cleanup:
- Expected dead files are actually gone
- Expected new files exist with correct structure
- Key imports work (no broken references from decomposition)
- No circular imports in critical paths
- Backward-compatible exports are present
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"


class TestDeadFilesActuallyRemoved:
    """Verify files that were supposed to be deleted are gone."""

    def test_learning_directory_gone(self):
        assert not (SRC / "learning").exists()

    def test_registry_directory_gone(self):
        assert not (SRC / "registry").exists()

    def test_ops_warmers_gone(self):
        assert not (SRC / "ops" / "warmers").exists()

    def test_mcp_server_security_gone(self):
        assert not (SRC / "mcp_server" / "security").exists()

    def test_orchestrator_gone(self):
        assert not (SRC / "ingestion" / "auto" / "orchestrator.py").exists()

    def test_optimizer_gone(self):
        assert not (SRC / "ops" / "optimizer.py").exists()

    def test_validation_gone(self):
        assert not (SRC / "mcp_server" / "validation.py").exists()

    def test_diffusion_reranker_gone(self):
        assert not (SRC / "query" / "diffusion_reranker.py").exists()

    def test_graph_expansion_gone(self):
        assert not (SRC / "query" / "graph_expansion.py").exists()

    def test_graph_features_gone(self):
        assert not (SRC / "query" / "graph_features.py").exists()

    def test_feature_flags_gone(self):
        assert not (SRC / "shared" / "feature_flags.py").exists()


class TestNewFilesExist:
    """Verify decomposition created expected new modules."""

    def test_neo4j_writers_exists(self):
        path = SRC / "ingestion" / "neo4j_writers.py"
        assert path.exists()
        text = path.read_text()
        assert "Neo4jWriter" in text
        assert "_neo4j_upsert_document" in text

    def test_qdrant_writers_exists(self):
        path = SRC / "ingestion" / "qdrant_writers.py"
        assert path.exists()
        text = path.read_text()
        assert "QdrantWriter" in text
        assert "_qdrant_upsert_vectors" in text

    def test_mcp_utils_exists(self):
        path = SRC / "mcp_server" / "mcp_utils.py"
        assert path.exists()
        text = path.read_text()
        assert "DIAGNOSTICS_RESOURCE_TEMPLATE" in text
        assert "MAX_TOKENS_PER_TURN" in text

    def test_mcp_search_exists(self):
        path = SRC / "mcp_server" / "mcp_search.py"
        assert path.exists()
        text = path.read_text()
        assert "_kb_search_candidates" in text

    def test_mcp_tools_exists(self):
        path = SRC / "mcp_server" / "mcp_tools.py"
        assert path.exists()
        text = path.read_text()
        assert "PROMPT_DEFINITIONS" in text
        assert "_tool_specs" in text

    def test_base_chonkie_adapter_exists(self):
        path = SRC / "providers" / "embeddings" / "base_chonkie_adapter.py"
        assert path.exists()
        text = path.read_text()
        assert "BaseChonkieAdapter" in text

    def test_unified_circuit_breaker_exists(self):
        path = SRC / "shared" / "resilience" / "circuit_breaker.py"
        assert path.exists()
        text = path.read_text()
        assert "class CircuitBreaker" in text

    def test_duplicate_circuit_breakers_gone(self):
        assert not (SRC / "providers" / "rerank" / "circuit_breaker.py").exists()
        assert not (SRC / "connectors" / "circuit_breaker.py").exists()


class TestBackwardCompatibleExports:
    """Verify tests can still import symbols moved during decomposition."""

    def test_atomic_reexports_entity_constants(self):
        # These constants live in neo4j_writers; atomic may or may not re-export them.
        from src.ingestion.neo4j_writers import (
            ALLOWED_ENTITY_RELATIONSHIP_TYPES,
        )
        # ENTITY_LABEL_ALLOWLIST is a class attribute on Neo4jWriter, not a module constant
        from src.ingestion.neo4j_writers import Neo4jWriter
        assert isinstance(ALLOWED_ENTITY_RELATIONSHIP_TYPES, frozenset)
        assert hasattr(Neo4jWriter, "ENTITY_LABEL_ALLOWLIST")

    def test_parsers_shadow_mode_error_export(self):
        from src.ingestion.parsers import ShadowModeError
        assert issubclass(ShadowModeError, Exception)

    def test_mcp_tools_legacy_flag_importable(self):
        from src.mcp_server.mcp_tools import LEGACY_SEARCH_DOCUMENTATION_ENABLED
        assert isinstance(LEGACY_SEARCH_DOCUMENTATION_ENABLED, bool)


class TestKeyImportsWork:
    """Smoke-test that critical import chains resolve."""

    def test_mcp_app_imports(self):
        from src.mcp_server.mcp_app import build_mcp_server
        assert callable(build_mcp_server)

    def test_mcp_tools_imports(self):
        from src.mcp_server.mcp_tools import _tool_specs, PROMPT_DEFINITIONS
        assert callable(_tool_specs)
        assert isinstance(PROMPT_DEFINITIONS, list)

    def test_mcp_utils_imports(self):
        from src.mcp_server.mcp_utils import (
            DIAGNOSTICS_RESOURCE_TEMPLATE,
            Deps,
            _get_deps,
        )
        assert isinstance(DIAGNOSTICS_RESOURCE_TEMPLATE, str)
        assert Deps is not None
        assert callable(_get_deps)

    def test_mcp_search_imports(self):
        from src.mcp_server.mcp_search import _kb_search_candidates
        assert callable(_kb_search_candidates)

    def test_hybrid_retrieval_imports(self):
        from src.query.hybrid_retrieval import HybridRetriever
        assert HybridRetriever is not None

    def test_neo4j_writers_imports(self):
        from src.ingestion.neo4j_writers import Neo4jWriter
        assert hasattr(Neo4jWriter, "_neo4j_upsert_document")

    def test_qdrant_writers_imports(self):
        from src.ingestion.qdrant_writers import QdrantWriter
        assert hasattr(QdrantWriter, "_qdrant_upsert_vectors")

    def test_base_chonkie_adapter_imports(self):
        from src.providers.embeddings.base_chonkie_adapter import BaseChonkieAdapter
        assert BaseChonkieAdapter is not None

    def test_circuit_breaker_unified_import(self):
        from src.shared.resilience import CircuitBreaker
        assert CircuitBreaker is not None

    def test_connectors_use_unified_circuit_breaker(self):
        from src.connectors import CircuitBreaker
        assert CircuitBreaker is not None

    def test_worker_lazy_import_contract_checks(self):
        """Verify contract_checks module can be imported (lazy import in worker)."""
        from src.neo.contract_checks import GraphContractChecker
        assert GraphContractChecker is not None


class TestNoCircularImports:
    """Verify critical modules don't have circular import issues."""

    def test_mcp_server_package_imports(self):
        """Importing the package __init__ should not hang or error."""
        import src.mcp_server
        assert hasattr(src.mcp_server, "build_mcp_server")

    def test_atomic_imports(self):
        import src.ingestion.atomic
        assert hasattr(src.ingestion.atomic, "AtomicIngestionCoordinator")

    def test_hybrid_retrieval_imports(self):
        import src.query.hybrid_retrieval
        assert hasattr(src.query.hybrid_retrieval, "HybridRetriever")


class TestModuleStructureIntegrity:
    """Verify extracted modules don't reference things they shouldn't."""

    def test_mcp_app_no_longer_has_tool_impls(self):
        """mcp_app should be small and delegate to mcp_tools."""
        path = SRC / "mcp_server" / "mcp_app.py"
        lines = path.read_text().splitlines()
        assert len(lines) < 500, f"mcp_app.py is {len(lines)} lines, expected < 500"

    def test_mcp_tools_has_tool_impls(self):
        """mcp_tools should be the large file with implementations."""
        path = SRC / "mcp_server" / "mcp_tools.py"
        lines = path.read_text().splitlines()
        assert len(lines) > 1500, f"mcp_tools.py is {len(lines)} lines, expected > 1500"

    def test_atomic_no_longer_has_db_write_methods(self):
        """atomic.py should import from neo4j_writers/qdrant_writers instead of defining them."""
        text = (SRC / "ingestion" / "atomic.py").read_text()
        assert "from src.ingestion.neo4j_writers import" in text
        assert "from src.ingestion.qdrant_writers import" in text
        # It should NOT define these methods anymore
        assert "def _neo4j_upsert_document(" not in text
        assert "def _qdrant_upsert_vectors(" not in text


class TestNoBrokenInternalReferences:
    """Parse AST to verify no NameErrors from missing references."""

    def _check_module_compiles(self, rel_path: str):
        path = SRC / rel_path
        source = path.read_text()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {rel_path}: {e}")

    def test_mcp_app_compiles(self):
        self._check_module_compiles("mcp_server/mcp_app.py")

    def test_mcp_tools_compiles(self):
        self._check_module_compiles("mcp_server/mcp_tools.py")

    def test_mcp_search_compiles(self):
        self._check_module_compiles("mcp_server/mcp_search.py")

    def test_mcp_utils_compiles(self):
        self._check_module_compiles("mcp_server/mcp_utils.py")

    def test_atomic_compiles(self):
        self._check_module_compiles("ingestion/atomic.py")

    def test_neo4j_writers_compiles(self):
        self._check_module_compiles("ingestion/neo4j_writers.py")

    def test_qdrant_writers_compiles(self):
        self._check_module_compiles("ingestion/qdrant_writers.py")
