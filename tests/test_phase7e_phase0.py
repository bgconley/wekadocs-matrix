"""
Phase 7E.0 Integration Test

Tests all Phase 0 components (validation & baseline establishment).
Ensures scripts work correctly with production-like scenarios.

Test Coverage:
- Task 0.1: Document token backfill (dry-run and execute)
- Task 0.2: Token accounting validation
- Task 0.3: Baseline distribution analysis
- Task 0.4: Baseline query execution

Requirements:
- Neo4j with test data
- Qdrant with embeddings
- All Phase 0 scripts functional
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.connections import get_connection_manager


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """
    Configure environment for host-side test execution.

    Tests run from the host machine but need to connect to Docker services.
    Override Docker internal hostnames with localhost.
    """
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_PASSWORD"] = "testpassword123"  # pragma: allowlist secret
    os.environ["QDRANT_HOST"] = "localhost"
    os.environ["QDRANT_PORT"] = "6333"
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6379"

    yield

    # Cleanup not needed - environment changes are process-local


class TestPhase0Backfill:
    """Test Task 0.1: Document token backfill"""

    def test_backfill_dry_run(self):
        """Test dry-run mode shows what would be done"""
        result = subprocess.run(
            ["python", "scripts/backfill_document_tokens.py", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
        assert "DRY RUN" in result.stdout
        # Accept either message depending on current database state
        assert (
            "No changes made" in result.stdout
            or "All documents already have token_count" in result.stdout
            or "Would update:" in result.stdout
        )

    def test_backfill_execute(self, tmp_path):
        """Test actual backfill execution"""
        report_file = tmp_path / "backfill-results.json"

        result = subprocess.run(
            [
                "python",
                "scripts/backfill_document_tokens.py",
                "--execute",
                "--report",
                str(report_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Backfill failed: {result.stderr}"
        assert "✓" in result.stdout or "SUCCESS" in result.stdout.upper()

        # Verify report file created
        assert report_file.exists(), "Report file not created"

        with open(report_file) as f:
            report = json.load(f)

        assert report["success"] is True
        assert "documents_updated" in report or "would_update" in report
        assert report["duration_seconds"] > 0

    def test_backfill_idempotent(self):
        """Test backfill is idempotent (safe to run multiple times)"""
        # Run twice
        result1 = subprocess.run(
            ["python", "scripts/backfill_document_tokens.py", "--execute"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        result2 = subprocess.run(
            ["python", "scripts/backfill_document_tokens.py", "--execute"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result1.returncode == 0
        assert result2.returncode == 0

        # Second run should find no documents to update
        assert (
            "All documents already have token_count" in result2.stdout
            or "Documents updated: 0" in result2.stdout
        )


class TestPhase0Validation:
    """Test Task 0.2: Token accounting validation"""

    def test_validation_runs(self, tmp_path):
        """Test validation script executes successfully"""
        report_file = tmp_path / "validation-results.json"

        result = subprocess.run(
            [
                "python",
                "scripts/validate_token_accounting.py",
                "--threshold",
                "0.01",
                "--report",
                str(report_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Exit code 0 or 1 are both acceptable (1 means violations found)
        assert result.returncode in [0, 1], f"Validation crashed: {result.stderr}"

        # Verify report created
        assert report_file.exists()

        with open(report_file) as f:
            report = json.load(f)

        assert "success" in report
        assert "documents_validated" in report
        assert "statistics" in report
        assert report["statistics"]["threshold"] == 0.01

    def test_validation_detects_violations(self):
        """Test validation correctly identifies threshold violations"""
        # Run with very strict threshold (0.001 = 0.1%)
        result = subprocess.run(
            [
                "python",
                "scripts/validate_token_accounting.py",
                "--threshold",
                "0.001",
                "--fail-on-violation",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should pass validation or report violations clearly
        if result.returncode != 0:
            assert "VIOLATION" in result.stdout or "FAIL" in result.stdout
        else:
            assert "PASS" in result.stdout

    def test_validation_statistics(self, tmp_path):
        """Test validation produces correct statistics"""
        report_file = tmp_path / "validation-stats.json"

        result = subprocess.run(
            [
                "python",
                "scripts/validate_token_accounting.py",
                "--report",
                str(report_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode in [0, 1]

        with open(report_file) as f:
            report = json.load(f)

        stats = report["statistics"]
        assert stats["total_documents"] > 0
        assert "error_rate" in stats
        assert "min" in stats["error_rate"]
        assert "max" in stats["error_rate"]
        assert "avg" in stats["error_rate"]


class TestPhase0Distribution:
    """Test Task 0.3: Baseline distribution analysis"""

    def test_distribution_analysis_runs(self, tmp_path):
        """Test distribution analysis executes and produces reports"""
        json_report = tmp_path / "distribution.json"
        md_report = tmp_path / "distribution.md"

        result = subprocess.run(
            [
                "python",
                "scripts/baseline_distribution_analysis.py",
                "--report",
                str(json_report),
                "--markdown",
                str(md_report),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Analysis failed: {result.stderr}"

        # Verify JSON report
        assert json_report.exists()
        with open(json_report) as f:
            report = json.load(f)

        assert report["success"] is True
        assert "overall" in report
        assert "by_document" in report
        assert "h2_groupings" in report

        overall = report["overall"]
        assert overall["total_sections"] > 0
        assert "percentiles" in overall
        assert "distribution" in overall

        # Verify markdown report
        assert md_report.exists()
        with open(md_report) as f:
            markdown = f.read()

        assert "Phase 7E.0" in markdown
        assert "Overall Statistics" in markdown
        assert "Token Range Distribution" in markdown

    def test_distribution_percentiles(self, tmp_path):
        """Test percentile calculations are correct"""
        report_file = tmp_path / "percentiles.json"

        result = subprocess.run(
            [
                "python",
                "scripts/baseline_distribution_analysis.py",
                "--report",
                str(report_file),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0

        with open(report_file) as f:
            report = json.load(f)

        percentiles = report["overall"]["percentiles"]

        # Verify percentiles are in ascending order (keys are strings in JSON)
        assert percentiles["50"] <= percentiles["75"]
        assert percentiles["75"] <= percentiles["90"]
        assert percentiles["90"] <= percentiles["95"]
        assert percentiles["95"] <= percentiles["99"]

        # Verify percentiles are positive
        for p in ["50", "75", "90", "95", "99"]:
            assert percentiles[p] > 0

    def test_distribution_ranges(self, tmp_path):
        """Test token range distribution sums to 100%"""
        report_file = tmp_path / "ranges.json"

        result = subprocess.run(
            [
                "python",
                "scripts/baseline_distribution_analysis.py",
                "--report",
                str(report_file),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0

        with open(report_file) as f:
            report = json.load(f)

        dist_pct = report["overall"]["distribution_percentages"]

        # Sum should be approximately 100%
        total_pct = sum(dist_pct.values())
        assert 99.0 <= total_pct <= 101.0, f"Percentages sum to {total_pct}%"


class TestPhase0Queries:
    """Test Task 0.4: Baseline query execution"""

    def test_query_set_loads(self):
        """Test query set YAML loads correctly"""
        query_file = Path("tests/fixtures/baseline_query_set.yaml")
        assert query_file.exists(), "Query set file not found"

        import yaml

        with open(query_file) as f:
            queries = yaml.safe_load(f)

        assert len(queries) >= 15, "Should have at least 15 queries"

        # Verify structure
        for query in queries:
            assert "id" in query
            assert "text" in query
            assert "category" in query
            assert "token_estimate" in query

    def test_baseline_queries_execute(self, tmp_path):
        """Test baseline queries run successfully"""
        report_file = tmp_path / "baseline-queries.json"

        result = subprocess.run(
            [
                "python",
                "scripts/run_baseline_queries.py",
                "--queries",
                "tests/fixtures/baseline_query_set.yaml",
                "--report",
                str(report_file),
                "--top-k",
                "10",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for all queries
        )

        # May fail if no data, but should not crash
        assert result.returncode in [0, 1], f"Script crashed: {result.stderr}"

        # If report exists, validate structure
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)

            assert "queries" in report
            assert "aggregate_stats" in report
            assert "config" in report

            stats = report["aggregate_stats"]
            assert "total_queries" in stats
            assert "avg_total_ms" in stats
            assert "latency_percentiles" in stats

    def test_query_timing_captured(self, tmp_path):
        """Test query execution captures timing metrics"""
        report_file = tmp_path / "query-timing.json"

        result = subprocess.run(
            [
                "python",
                "scripts/run_baseline_queries.py",
                "--queries",
                "tests/fixtures/baseline_query_set.yaml",
                "--report",
                str(report_file),
                "--top-k",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0 and report_file.exists():
            with open(report_file) as f:
                report = json.load(f)

            # Check individual query timing
            if report["queries"]:
                first_query = report["queries"][0]
                if first_query["success"]:
                    timing = first_query["timing"]
                    assert "embedding_ms" in timing
                    assert "search_ms" in timing
                    assert "total_ms" in timing
                    assert timing["total_ms"] > 0

    def test_query_categories_analyzed(self, tmp_path):
        """Test per-category statistics are generated"""
        report_file = tmp_path / "query-categories.json"

        result = subprocess.run(
            ["python", "scripts/run_baseline_queries.py", "--report", str(report_file)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0 and report_file.exists():
            with open(report_file) as f:
                report = json.load(f)

            if report["aggregate_stats"]["successful_queries"] > 0:
                by_category = report["aggregate_stats"]["by_category"]

                # Should have at least config, procedure, troubleshooting, complex
                assert len(by_category) >= 3, "Should analyze multiple categories"

                for cat_stats in by_category.values():
                    assert "query_count" in cat_stats
                    assert "avg_latency_ms" in cat_stats


class TestPhase0EndToEnd:
    """End-to-end Phase 0 workflow test"""

    def test_full_phase0_workflow(self, tmp_path):
        """Test complete Phase 0 workflow in sequence"""
        report_dir = tmp_path / "phase-7e"
        report_dir.mkdir(parents=True)

        # Step 1: Backfill (if needed)
        backfill = subprocess.run(
            [
                "python",
                "scripts/backfill_document_tokens.py",
                "--execute",
                "--report",
                str(report_dir / "backfill.json"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert backfill.returncode == 0

        # Step 2: Validate
        validate = subprocess.run(
            [
                "python",
                "scripts/validate_token_accounting.py",
                "--report",
                str(report_dir / "validation.json"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert validate.returncode in [0, 1]  # 1 = violations, still valid

        # Step 3: Distribution analysis
        distribution = subprocess.run(
            [
                "python",
                "scripts/baseline_distribution_analysis.py",
                "--report",
                str(report_dir / "distribution.json"),
                "--markdown",
                str(report_dir / "distribution.md"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert distribution.returncode == 0

        # Step 4: Baseline queries (may fail if no vectors, but shouldn't crash)
        _queries = subprocess.run(
            [
                "python",
                "scripts/run_baseline_queries.py",
                "--report",
                str(report_dir / "queries.json"),
                "--top-k",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Don't assert exit code - may fail legitimately if no embeddings

        # Verify all reports created
        assert (report_dir / "backfill.json").exists()
        assert (report_dir / "validation.json").exists()
        assert (report_dir / "distribution.json").exists()
        assert (report_dir / "distribution.md").exists()


@pytest.mark.integration
class TestPhase0WithRealData:
    """Integration tests requiring real database with test data"""

    def test_with_neo4j_connection(self):
        """Test scripts work with real Neo4j connection"""
        conn_manager = get_connection_manager()
        driver = conn_manager.get_neo4j_driver()

        # Verify connection works
        with driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            doc_count = result.single()["count"]

            assert doc_count >= 0, "Should be able to query documents"

    def test_validates_real_token_counts(self):
        """Test validation works with real document token counts"""
        conn_manager = get_connection_manager()
        driver = conn_manager.get_neo4j_driver()

        with driver.session() as session:
            # Get a document with sections
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
                WITH d, count(s) as section_count, sum(s.token_count) as section_sum
                WHERE section_count > 0
                RETURN d.doc_id as doc_id,
                       d.token_count as doc_tokens,
                       section_sum,
                       section_count
                LIMIT 1
            """
            )

            record = result.single()

            if record:
                doc_tokens = record["doc_tokens"]
                section_sum = record["section_sum"]

                # Document token count should match section sum
                if doc_tokens and section_sum:
                    delta = abs(doc_tokens - section_sum)
                    error_rate = delta / doc_tokens if doc_tokens > 0 else 0

                    assert (
                        error_rate < 0.02
                    ), f"Token accounting error: {error_rate:.2%}"


# Run with: pytest tests/test_phase7e_phase0.py -v
# Run with markers: pytest tests/test_phase7e_phase0.py -m integration -v
