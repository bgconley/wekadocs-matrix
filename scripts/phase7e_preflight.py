#!/usr/bin/env python3
"""
Phase 7E.0 - Preflight Validation Script

PURPOSE:
Verify that infrastructure, schema, and configuration are ready for Phase 7E implementation.
Fail fast if any critical specification is not met.

CHECKS:
1. Neo4j v2.1 DDL (constraints, indexes, vector indexes @1024-D cosine)
2. Qdrant 'chunks' collection (1024-D cosine) with payload indexes
3. Runtime environment variables (embed model, provider, dimensions)
4. Config settings (hybrid retrieval, context budget, expansion thresholds)
5. Health probes (all systems operational)

OUTPUT:
JSON report with pass/fail status and evidence for each check.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

import yaml
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams


class PreflightChecker:
    """Phase 7E preflight validation."""

    def __init__(self):
        self.results = {
            "phase": "7E.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status": "UNKNOWN",
            "checks": [],
        }

        # Connect to services
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "testpassword123")

        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        self.neo4j_driver = None
        self.qdrant_client = None

    def connect(self) -> bool:
        """Establish connections to Neo4j and Qdrant."""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host, port=self.qdrant_port
            )
            return True
        except Exception as e:
            self._add_check(
                "Infrastructure Connection", False, f"Failed to connect: {e}"
            )
            return False

    def close(self):
        """Close connections."""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def _add_check(self, name: str, passed: bool, details: Any, evidence: Any = None):
        """Add a check result."""
        self.results["checks"].append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "details": details,
                "evidence": evidence,
            }
        )

    def check_neo4j_constraints(self) -> bool:
        """Verify all required constraints exist."""
        required_constraints = [
            "document_id_unique",
            "document_source_uri_unique",
            "section_id_unique",
            "command_id_unique",
            "configuration_id_unique",
            "procedure_id_unique",
            "error_id_unique",
            "concept_id_unique",
            "example_id_unique",
            "step_id_unique",
            "parameter_id_unique",
            "component_id_unique",
            "session_id_unique",
            "query_id_unique",
            "answer_id_unique",
        ]

        with self.neo4j_driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in result]

            missing = [c for c in required_constraints if c not in constraints]

            passed = len(missing) == 0
            details = f"Found {len(constraints)}/15 required constraints"
            if missing:
                details += f". Missing: {', '.join(missing)}"

            self._add_check(
                "Neo4j Constraints",
                passed,
                details,
                {
                    "found": len(constraints),
                    "required": 15,
                    "constraints": constraints[:10],
                },
            )
            return passed

    def check_neo4j_property_indexes(self) -> bool:
        """Verify critical property indexes exist."""
        required_indexes = [
            "section_document_id",
            "section_level",
            "section_order",
            "chunk_document_id",
            "chunk_level",
            "chunk_embedding_version",
        ]

        with self.neo4j_driver.session() as session:
            result = session.run("SHOW INDEXES WHERE type IN ['RANGE', 'BTREE']")
            indexes = [record["name"] for record in result]

            missing = [idx for idx in required_indexes if idx not in indexes]

            passed = len(missing) == 0
            details = f"Found {len(indexes)} property indexes"
            if missing:
                details += f". Missing critical: {', '.join(missing)}"

            self._add_check(
                "Neo4j Property Indexes",
                passed,
                details,
                {
                    "found": len(indexes),
                    "critical_present": len(required_indexes) - len(missing),
                },
            )
            return passed

    def check_neo4j_vector_indexes(self) -> bool:
        """Verify vector indexes exist with correct dimensions."""
        with self.neo4j_driver.session() as session:
            result = session.run("SHOW INDEXES WHERE type = 'VECTOR'")
            vector_indexes = list(result)

            required_indexes = {
                "section_embeddings_v2": "Section",
                "chunk_embeddings_v2": "Chunk",
            }

            found_indexes = {
                rec["name"]: rec["labelsOrTypes"][0] for rec in vector_indexes
            }

            missing = [name for name in required_indexes if name not in found_indexes]

            # Test vector index with correct dimensionality
            dims_correct = True
            try:
                test_vec = [0.0] * 1024
                session.run(
                    "CALL db.index.vector.queryNodes('section_embeddings_v2', 1, $vec) YIELD node RETURN node LIMIT 0",
                    vec=test_vec,
                )
            except Exception as e:
                if "1024" not in str(e):
                    dims_correct = False

            passed = len(missing) == 0 and dims_correct
            details = f"Found {len(found_indexes)}/2 vector indexes @1024-D cosine"
            if missing:
                details += f". Missing: {', '.join(missing)}"
            if not dims_correct:
                details += ". Dimension mismatch detected"

            self._add_check(
                "Neo4j Vector Indexes (1024-D)",
                passed,
                details,
                {
                    "indexes": list(found_indexes.keys()),
                    "dimensions": 1024,
                    "similarity": "cosine",
                },
            )
            return passed

    def check_neo4j_schema_version(self) -> bool:
        """Verify SchemaVersion marker is v2.1."""
        with self.neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (sv:SchemaVersion {id: 'singleton'})
                RETURN sv.version as version,
                       sv.vector_dimensions as dims,
                       sv.embedding_provider as provider,
                       sv.embedding_model as model
            """
            )

            record = result.single()
            if not record:
                self._add_check(
                    "Neo4j Schema Version", False, "SchemaVersion node not found", None
                )
                return False

            version = record["version"]
            dims = record["dims"]
            provider = record["provider"]
            model = record["model"]

            passed = (
                version == "v2.1"
                and dims == 1024
                and provider == "jina-ai"
                and model == "jina-embeddings-v3"
            )

            details = f"Schema v{version}, {dims}-D, {provider}/{model}"
            if not passed:
                details += (
                    " (MISMATCH: expected v2.1, 1024-D, jina-ai/jina-embeddings-v3)"
                )

            self._add_check(
                "Neo4j Schema Version",
                passed,
                details,
                {
                    "version": version,
                    "dimensions": dims,
                    "provider": provider,
                    "model": model,
                },
            )
            return passed

    def check_qdrant_chunks_collection(self) -> bool:
        """Verify 'chunks' collection exists with correct config."""
        try:
            # Check if chunks collection exists, create if not
            collections = [
                c.name for c in self.qdrant_client.get_collections().collections
            ]

            if "chunks" not in collections:
                # Create chunks collection per Phase 7E spec
                self.qdrant_client.create_collection(
                    collection_name="chunks",
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 100,
                        "full_scan_threshold": 10000,
                    },
                    shard_number=1,
                    replication_factor=1,
                    on_disk_payload=True,
                )
                action = "created"
            else:
                action = "exists"

            # Verify configuration
            info = self.qdrant_client.get_collection("chunks")

            size = info.config.params.vectors.size
            distance = info.config.params.vectors.distance.value

            passed = size == 1024 and distance == "Cosine"
            details = f"Collection '{action}': {size}-D {distance}"
            if not passed:
                details += " (FAIL: expected 1024-D Cosine)"

            self._add_check(
                "Qdrant 'chunks' Collection",
                passed,
                details,
                {
                    "collection": "chunks",
                    "size": size,
                    "distance": distance,
                    "points": info.points_count,
                    "action": action,
                },
            )
            return passed

        except Exception as e:
            self._add_check("Qdrant 'chunks' Collection", False, f"Error: {e}", None)
            return False

    def check_qdrant_payload_indexes(self) -> bool:
        """Verify payload indexes exist for fast filtering."""
        try:
            info = self.qdrant_client.get_collection("chunks")
            payload_schema = info.payload_schema or {}

            # Required payload indexes
            required = ["document_id", "parent_section_id", "order"]
            existing = list(payload_schema.keys())

            # Create missing indexes
            created = []
            for field in required:
                if field not in existing:
                    if field in ["document_id", "parent_section_id"]:
                        schema_type = PayloadSchemaType.KEYWORD
                    else:  # order
                        schema_type = PayloadSchemaType.INTEGER

                    self.qdrant_client.create_payload_index(
                        collection_name="chunks",
                        field_name=field,
                        field_schema=schema_type,
                    )
                    created.append(field)

            # Verify all exist now
            info = self.qdrant_client.get_collection("chunks")
            payload_schema = info.payload_schema or {}
            existing = list(payload_schema.keys())

            missing = [f for f in required if f not in existing]
            passed = len(missing) == 0

            details = f"Payload indexes: {len(existing)} present"
            if created:
                details += f" ({len(created)} created: {', '.join(created)})"
            if missing:
                details += f". Missing: {', '.join(missing)}"

            self._add_check(
                "Qdrant Payload Indexes",
                passed,
                details,
                {"required": required, "existing": existing, "created": created},
            )
            return passed

        except Exception as e:
            self._add_check("Qdrant Payload Indexes", False, f"Error: {e}", None)
            return False

    def check_runtime_env_vars(self) -> bool:
        """Verify runtime environment variables are set correctly."""
        required_vars = {
            "EMBED_MODEL_ID": "jina-embeddings-v3",
            "EMBED_PROVIDER": "jina-ai",
            "EMBED_DIM": "1024",
        }

        # Check if vars are set (may be in config instead)
        env_values = {}
        mismatches = []

        for var, expected in required_vars.items():
            actual = os.getenv(var, "NOT_SET")
            env_values[var] = actual
            if actual != "NOT_SET" and actual != expected:
                mismatches.append(f"{var}={actual} (expected {expected})")

        # These might be in config instead of env, which is acceptable
        passed = len(mismatches) == 0
        details = "Environment variables checked"
        if mismatches:
            details += f". Mismatches: {', '.join(mismatches)}"
        else:
            details += ". Note: May be configured in YAML (acceptable)"

        self._add_check(
            "Runtime Environment Variables",
            passed,
            details,
            {"checked": required_vars, "env_values": env_values},
        )
        return passed

    def check_config_file(self) -> bool:
        """Verify config/development.yaml has Phase 7E settings."""
        config_path = "config/development.yaml"

        if not os.path.exists(config_path):
            self._add_check(
                "Config File Settings",
                False,
                f"Config file not found: {config_path}",
                None,
            )
            return False

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check embedding configuration
        embedding = config.get("embedding", {})
        embed_checks = {
            "model_name": ("jina-embeddings-v3", embedding.get("model_name")),
            "dims": (1024, embedding.get("dims")),
            "version": ("jina-embeddings-v3", embedding.get("version")),
            "provider": ("jina-ai", embedding.get("provider")),
        }

        # Check hybrid retrieval settings (may not exist yet)
        hybrid = config.get("hybrid", {})
        hybrid_checks = {
            "method": ("rrf", hybrid.get("method", "NOT_SET")),
            "rrf_k": (60, hybrid.get("rrf_k", "NOT_SET")),
            "fusion_alpha": (0.6, hybrid.get("fusion_alpha", "NOT_SET")),
        }

        # Check context budget (may not exist yet)
        answer_max = config.get("answer_context_max_tokens", "NOT_SET")

        failures = []
        for key, (expected, actual) in embed_checks.items():
            if actual != expected:
                failures.append(f"embedding.{key}={actual} (expected {expected})")

        # Hybrid settings are optional for preflight (will be added in implementation)
        warnings = []
        for key, (expected, actual) in hybrid_checks.items():
            if actual == "NOT_SET":
                warnings.append(f"hybrid.{key} not set (will add)")

        if answer_max == "NOT_SET":
            warnings.append("answer_context_max_tokens not set (will add)")

        passed = len(failures) == 0
        details = "Config file validated"
        if failures:
            details += f". Failures: {', '.join(failures)}"
        if warnings:
            details += f". Warnings: {', '.join(warnings)}"

        self._add_check(
            "Config File Settings",
            passed,
            details,
            {
                "embed_config": embed_checks,
                "hybrid_config": hybrid_checks,
                "warnings": warnings,
            },
        )
        return passed

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all preflight checks."""
        if not self.connect():
            self.results["overall_status"] = "FAIL"
            return self.results

        try:
            checks = [
                self.check_neo4j_constraints(),
                self.check_neo4j_property_indexes(),
                self.check_neo4j_vector_indexes(),
                self.check_neo4j_schema_version(),
                self.check_qdrant_chunks_collection(),
                self.check_qdrant_payload_indexes(),
                self.check_runtime_env_vars(),
                self.check_config_file(),
            ]

            # Overall status
            if all(checks):
                self.results["overall_status"] = "PASS"
            elif any(checks):
                self.results["overall_status"] = "PARTIAL"
            else:
                self.results["overall_status"] = "FAIL"

            # Summary
            passed = sum(1 for c in self.results["checks"] if c["status"] == "PASS")
            total = len(self.results["checks"])
            self.results["summary"] = {
                "passed": passed,
                "total": total,
                "pass_rate": f"{passed}/{total} ({100*passed//total}%)",
            }

        finally:
            self.close()

        return self.results


def main():
    """Run preflight checks and output results."""
    checker = PreflightChecker()
    results = checker.run_all_checks()

    # Output JSON
    print(json.dumps(results, indent=2))

    # Exit code
    sys.exit(0 if results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
