"""
Phase 6, Task 6.4: Ingestion Report Generation

Generates JSON and Markdown reports for completed ingestion jobs.

See: /docs/pseudocode-phase6.md → Task 6.4
See: /docs/implementation-plan-phase-6.md → Task 6.4
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from neo4j import Driver

from src.shared.config import Config
from src.shared.observability import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates ingestion reports with stats, verification results, and readiness verdict.
    """

    def __init__(self, driver: Driver, config: Config, qdrant_client=None):
        self.driver = driver
        self.config = config
        self.qdrant_client = qdrant_client
        from src.shared.config import get_embedding_settings

        self.embedding_version = get_embedding_settings(config).version
        self.vector_primary = config.search.vector.primary

    def generate_report(
        self,
        job_id: str,
        tag: str,
        parsed: Dict,
        verdict: Dict,
        timings: Dict[str, int],
        errors: list = None,
    ) -> Dict:
        """
        Generate comprehensive ingestion report.

        Args:
            job_id: Job identifier
            tag: Classification tag
            parsed: Parsed document structure
            verdict: Verification verdict
            timings: Stage timings in milliseconds
            errors: List of errors encountered

        Returns:
            Report dict
        """
        logger.info("Generating ingestion report", job_id=job_id)

        # Build report structure
        report = {
            "job_id": job_id,
            "tag": tag,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "doc": self._get_doc_stats(parsed),
            "graph": self._get_graph_stats(),
            "vector": self._get_vector_stats(),
            "drift_pct": verdict["drift"]["pct"],
            "sample_queries": verdict["answers"],
            "ready_for_queries": verdict["ready"],
            "timings_ms": timings,
            "errors": errors or [],
        }

        logger.info(
            "Report generated", job_id=job_id, ready=report["ready_for_queries"]
        )
        return report

    def write_report(self, report: Dict, output_dir: Optional[str] = None):
        """
        Write report to JSON and Markdown files.

        Args:
            report: Report dict
            output_dir: Output directory (default: reports/ingest/{timestamp})
        """
        job_id = report["job_id"]

        if not output_dir:
            # Create timestamped directory
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = f"reports/ingest/{timestamp}_{job_id[:8]}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write JSON
        json_path = output_path / "ingest_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("JSON report written", path=str(json_path))

        # Write Markdown
        md_path = output_path / "ingest_report.md"
        md_content = self._render_markdown(report)
        with open(md_path, "w") as f:
            f.write(md_content)

        logger.info("Markdown report written", path=str(md_path))

        return {"json": str(json_path), "markdown": str(md_path)}

    def _get_doc_stats(self, parsed: Dict) -> Dict:
        """Extract document stats from parsed structure."""
        document = parsed.get("Document", {})
        sections = parsed.get("Sections", [])

        return {
            "source_uri": document.get("source_uri", "N/A"),
            "checksum": document.get("checksum", "N/A"),
            "sections": len(sections),
            "title": document.get("title", "Untitled"),
        }

    def _get_graph_stats(self) -> Dict:
        """Get current graph stats from Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)
                    RETURN
                        count(n) AS total_nodes,
                        count{(n)-[]->()} AS total_rels,
                        count{(n:Section)} AS sections,
                        count{(n:Document)} AS documents
                    """
                )
                record = result.single()

                if record:
                    return {
                        "nodes_total": record["total_nodes"],
                        "rels_total": record["total_rels"],
                        "sections_total": record["sections"],
                        "documents_total": record["documents"],
                    }
        except Exception as e:
            logger.error("Failed to get graph stats", error=str(e))

        return {
            "nodes_total": 0,
            "rels_total": 0,
            "sections_total": 0,
            "documents_total": 0,
        }

    def _get_vector_stats(self) -> Dict:
        """Get vector store stats."""
        try:
            if self.vector_primary == "qdrant" and self.qdrant_client:
                collection_name = self.config.search.vector.qdrant.collection_name
                coll_info = self.qdrant_client.get_collection(collection_name)

                return {
                    "sot": "qdrant",
                    "sections_indexed": coll_info.points_count,
                    "embedding_version": self.embedding_version,
                }

            elif self.vector_primary == "neo4j":
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (s:Section)
                        WHERE s.vector_embedding IS NOT NULL
                          AND s.embedding_version = $version
                        RETURN count(s) AS count
                        """,
                        version=self.embedding_version,
                    )
                    record = result.single()
                    count = record["count"] if record else 0

                return {
                    "sot": "neo4j",
                    "sections_indexed": count,
                    "embedding_version": self.embedding_version,
                }

        except Exception as e:
            logger.error("Failed to get vector stats", error=str(e))

        return {
            "sot": self.vector_primary,
            "sections_indexed": 0,
            "embedding_version": self.embedding_version,
        }

    def _render_markdown(self, report: Dict) -> str:
        """Render report as Markdown."""
        lines = [
            "# Ingestion Report",
            "",
            f"**Job ID:** `{report['job_id']}`  ",
            f"**Tag:** `{report['tag']}`  ",
            f"**Timestamp:** {report['timestamp_utc']}  ",
            f"**Ready for Queries:** {'✅ YES' if report['ready_for_queries'] else '❌ NO'}  ",
            "",
            "## Document",
            "",
            f"- **Title:** {report['doc']['title']}",
            f"- **Source:** `{report['doc']['source_uri']}`",
            f"- **Sections:** {report['doc']['sections']}",
            f"- **Checksum:** `{report['doc']['checksum'][:16]}...`",
            "",
            "## Graph Stats",
            "",
            f"- **Total Nodes:** {report['graph']['nodes_total']}",
            f"- **Total Relationships:** {report['graph']['rels_total']}",
            f"- **Sections:** {report['graph']['sections_total']}",
            f"- **Documents:** {report['graph']['documents_total']}",
            "",
            "## Vector Store",
            "",
            f"- **Primary:** {report['vector']['sot']}",
            f"- **Sections Indexed:** {report['vector']['sections_indexed']}",
            f"- **Embedding Version:** `{report['vector']['embedding_version']}`",
            "",
            "## Drift Analysis",
            "",
            f"- **Drift Percentage:** {report['drift_pct']}%",
            f"- **Status:** {'✅ OK (<0.5%)' if report['drift_pct'] <= 0.5 else '⚠️ HIGH (>0.5%)'}",
            "",
            "## Sample Queries",
            "",
        ]

        if report["sample_queries"]:
            for i, sq in enumerate(report["sample_queries"], 1):
                lines.append(f"### Query {i}")
                lines.append("")
                lines.append(f"**Question:** {sq['q']}")
                lines.append(f"**Confidence:** {sq['confidence']}")
                lines.append(f"**Evidence:** {sq['evidence_count']} items")
                lines.append(f"**Status:** {'✅' if sq['has_evidence'] else '❌'}")
                if "error" in sq:
                    lines.append(f"**Error:** `{sq['error']}`")
                lines.append("")
        else:
            lines.append("*No sample queries configured*")
            lines.append("")

        lines.extend(
            [
                "## Timings",
                "",
            ]
        )

        for stage, ms in report["timings_ms"].items():
            lines.append(f"- **{stage.title()}:** {ms}ms")

        lines.append("")

        if report["errors"]:
            lines.extend(
                [
                    "## Errors",
                    "",
                ]
            )
            for err in report["errors"]:
                lines.append(f"- {err}")
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                "*Generated by WekaDocs GraphRAG MCP - Phase 6 Auto-Ingestion*",
            ]
        )

        return "\n".join(lines)
