#!/usr/bin/env python3
"""
Phase 7E.0 - Task 0.3: Baseline Distribution Analysis

Analyzes current section token distribution before chunking implementation.
Captures baseline metrics for comparison after Phase 2 (combiner) completion.

Usage:
    python scripts/baseline_distribution_analysis.py
    python scripts/baseline_distribution_analysis.py --report reports/phase-7e/baseline-distribution.json
    python scripts/baseline_distribution_analysis.py --markdown reports/phase-7e/baseline-distribution.md

Features:
- Percentile analysis (p50, p75, p90, p95, p99)
- Token range histograms
- Per-document statistics
- H2 grouping analysis
- Export to JSON and Markdown
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from neo4j import Driver

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.connections import get_connection_manager
from src.shared.observability.logging import setup_logging

logger = logging.getLogger(__name__)


class BaselineDistributionAnalyzer:
    """Analyzes token distribution of current micro-sections"""

    def __init__(self, driver: Driver):
        self.driver = driver

    def get_all_sections(self) -> List[Dict]:
        """Fetch all sections with metadata"""
        query = """
        MATCH (d:Document)-[:HAS_CHUNK]->(s:Chunk)
        RETURN d.id as doc_id,
               s.id as section_id,
               s.title as heading,
               s.level as level,
               s.tokens as token_count,
               s.order as position
        ORDER BY d.id, s.order
        """

        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def calculate_percentiles(
        self, values: List[float], percentiles: List[int]
    ) -> Dict[int, float]:
        """Calculate percentiles from sorted values"""
        if not values:
            return {p: 0.0 for p in percentiles}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        results = {}
        for p in percentiles:
            idx = int(n * p / 100.0)
            if idx >= n:
                idx = n - 1
            results[p] = sorted_vals[idx]

        return results

    def analyze_token_distribution(self, sections: List[Dict]) -> Dict:
        """Analyze overall token distribution"""
        token_counts = [s["token_count"] for s in sections if s["token_count"]]

        if not token_counts:
            return {}

        # Calculate statistics
        percentiles = self.calculate_percentiles(token_counts, [50, 75, 90, 95, 99])

        # Token range buckets (matching Phase 7E targets)
        ranges = {
            "under_200": 0,
            "range_200_800": 0,
            "range_800_1500": 0,
            "range_1500_7900": 0,
            "over_7900": 0,
        }

        for tc in token_counts:
            if tc < 200:
                ranges["under_200"] += 1
            elif tc < 800:
                ranges["range_200_800"] += 1
            elif tc < 1500:
                ranges["range_800_1500"] += 1
            elif tc < 7900:
                ranges["range_1500_7900"] += 1
            else:
                ranges["over_7900"] += 1

        return {
            "total_sections": len(token_counts),
            "total_tokens": sum(token_counts),
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": sum(token_counts) / len(token_counts),
            "median": percentiles[50],
            "percentiles": percentiles,
            "distribution": ranges,
            "distribution_percentages": {
                k: (v / len(token_counts) * 100) for k, v in ranges.items()
            },
        }

    def analyze_by_document(self, sections: List[Dict]) -> List[Dict]:
        """Analyze token distribution per document"""
        by_doc = defaultdict(list)

        for section in sections:
            by_doc[section["doc_id"]].append(section)

        doc_stats = []
        for doc_id, doc_sections in by_doc.items():
            token_counts = [s["token_count"] for s in doc_sections if s["token_count"]]

            if not token_counts:
                continue

            percentiles = self.calculate_percentiles(token_counts, [50, 75, 90, 95])

            doc_stats.append(
                {
                    "doc_id": doc_id,
                    "section_count": len(doc_sections),
                    "total_tokens": sum(token_counts),
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "avg": sum(token_counts) / len(token_counts),
                    "p50": percentiles[50],
                    "p75": percentiles[75],
                    "p90": percentiles[90],
                    "p95": percentiles[95],
                }
            )

        return sorted(doc_stats, key=lambda x: x["section_count"], reverse=True)

    def analyze_h2_groupings(self, sections: List[Dict]) -> Dict:
        """Analyze sections grouped by H2 headings (future chunking boundaries)"""
        by_doc = defaultdict(list)

        for section in sections:
            by_doc[section["doc_id"]].append(section)

        h2_groups = []

        for doc_id, doc_sections in by_doc.items():
            # Sort by position
            sorted_sections = sorted(doc_sections, key=lambda s: s.get("position", 0))

            current_h2 = None
            current_group = []

            for section in sorted_sections:
                level = section.get("level", 3)

                # H2 starts a new group
                if level == 2:
                    # Save previous group
                    if current_group:
                        tokens = sum(
                            s["token_count"] for s in current_group if s["token_count"]
                        )
                        h2_groups.append(
                            {
                                "doc_id": doc_id,
                                "h2_heading": current_h2,
                                "section_count": len(current_group),
                                "total_tokens": tokens,
                            }
                        )

                    # Start new group
                    current_h2 = section.get("heading", "Unknown")
                    current_group = [section]
                else:
                    # Add to current group
                    current_group.append(section)

            # Save final group
            if current_group:
                tokens = sum(
                    s["token_count"] for s in current_group if s["token_count"]
                )
                h2_groups.append(
                    {
                        "doc_id": doc_id,
                        "h2_heading": current_h2,
                        "section_count": len(current_group),
                        "total_tokens": tokens,
                    }
                )

        # Calculate H2 group statistics
        if h2_groups:
            group_tokens = [g["total_tokens"] for g in h2_groups]
            group_sections = [g["section_count"] for g in h2_groups]

            return {
                "total_h2_groups": len(h2_groups),
                "avg_tokens_per_group": sum(group_tokens) / len(group_tokens),
                "avg_sections_per_group": sum(group_sections) / len(group_sections),
                "min_tokens": min(group_tokens),
                "max_tokens": max(group_tokens),
                "groups": h2_groups[:20],  # First 20 for inspection
            }

        return {}

    def generate_markdown_report(self, analysis: Dict) -> str:
        """Generate human-readable markdown report"""
        lines = []

        lines.append("# Phase 7E.0 - Baseline Distribution Analysis")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(
            "\n**Purpose:** Establish baseline metrics before chunking implementation"
        )
        lines.append("\n---\n")

        # Overall statistics
        overall = analysis.get("overall", {})
        lines.append("## Overall Statistics")
        lines.append(f"\n- **Total Sections:** {overall.get('total_sections', 0):,}")
        lines.append(f"- **Total Tokens:** {overall.get('total_tokens', 0):,}")
        lines.append(f"- **Average Tokens/Section:** {overall.get('avg', 0):.1f}")
        lines.append(f"- **Median Tokens/Section:** {overall.get('median', 0):.1f}")
        lines.append(f"- **Min:** {overall.get('min', 0)}")
        lines.append(f"- **Max:** {overall.get('max', 0)}")

        # Percentiles
        lines.append("\n### Percentile Distribution")
        percentiles = overall.get("percentiles", {})
        lines.append("\n| Percentile | Token Count |")
        lines.append("|------------|-------------|")
        for p in [50, 75, 90, 95, 99]:
            lines.append(f"| p{p} | {percentiles.get(p, 0):.0f} |")

        # Token range distribution
        lines.append("\n### Token Range Distribution")
        dist = overall.get("distribution", {})
        dist_pct = overall.get("distribution_percentages", {})
        lines.append("\n| Range | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        lines.append(
            f"| < 200 | {dist.get('under_200', 0):,} | {dist_pct.get('under_200', 0):.1f}% |"
        )
        lines.append(
            f"| 200-800 | {dist.get('range_200_800', 0):,} | {dist_pct.get('range_200_800', 0):.1f}% |"
        )
        lines.append(
            f"| 800-1,500 | {dist.get('range_800_1500', 0):,} | {dist_pct.get('range_800_1500', 0):.1f}% |"
        )
        lines.append(
            f"| 1,500-7,900 | {dist.get('range_1500_7900', 0):,} | {dist_pct.get('range_1500_7900', 0):.1f}% |"
        )
        lines.append(
            f"| > 7,900 | {dist.get('over_7900', 0):,} | {dist_pct.get('over_7900', 0):.1f}% |"
        )

        # Problem identification
        lines.append("\n### Analysis")
        under_200_pct = dist_pct.get("under_200", 0)
        under_800_pct = under_200_pct + dist_pct.get("range_200_800", 0)

        if under_200_pct > 50:
            lines.append(
                f"\n⚠️ **CRITICAL:** {under_200_pct:.1f}% of sections are under 200 tokens (severe fragmentation)"
            )
        if under_800_pct > 80:
            lines.append(
                f"\n⚠️ **HIGH:** {under_800_pct:.1f}% of sections are under 800 tokens (needs combining)"
            )

        optimal_pct = dist_pct.get("range_800_1500", 0)
        if optimal_pct < 10:
            lines.append(
                f"\n⚠️ **TARGET:** Only {optimal_pct:.1f}% in optimal range (800-1,500 tokens)"
            )

        # Per-document stats
        by_doc = analysis.get("by_document", [])
        if by_doc:
            lines.append("\n## Per-Document Statistics")
            lines.append(
                "\n| Document | Sections | Total Tokens | Avg | p50 | p90 | p95 |"
            )
            lines.append(
                "|----------|----------|--------------|-----|-----|-----|-----|"
            )
            for doc in by_doc[:10]:  # Top 10
                lines.append(
                    f"| {doc['doc_id'][:30]} | {doc['section_count']} | "
                    f"{doc['total_tokens']:,} | {doc['avg']:.0f} | "
                    f"{doc['p50']:.0f} | {doc['p90']:.0f} | {doc['p95']:.0f} |"
                )

        # H2 grouping analysis
        h2_analysis = analysis.get("h2_groupings", {})
        if h2_analysis:
            lines.append("\n## H2 Grouping Analysis")
            lines.append(
                f"\n- **Total H2 Groups:** {h2_analysis.get('total_h2_groups', 0)}"
            )
            lines.append(
                f"- **Avg Tokens/Group:** {h2_analysis.get('avg_tokens_per_group', 0):.1f}"
            )
            lines.append(
                f"- **Avg Sections/Group:** {h2_analysis.get('avg_sections_per_group', 0):.1f}"
            )
            lines.append(f"- **Min Tokens:** {h2_analysis.get('min_tokens', 0)}")
            lines.append(f"- **Max Tokens:** {h2_analysis.get('max_tokens', 0):,}")

        lines.append("\n---")
        lines.append(
            "\n*This baseline will be compared with post-chunking metrics in Phase 2*"
        )

        return "\n".join(lines)

    def run_analysis(self) -> Dict:
        """Run complete baseline analysis"""
        start_time = datetime.now()

        logger.info("Fetching all sections...")
        sections = self.get_all_sections()

        if not sections:
            logger.warning("No sections found in database")
            return {
                "success": False,
                "error": "No sections found",
                "timestamp": datetime.now().isoformat(),
            }

        logger.info(f"Analyzing {len(sections)} sections...")

        # Run analyses
        overall = self.analyze_token_distribution(sections)
        by_document = self.analyze_by_document(sections)
        h2_groupings = self.analyze_h2_groupings(sections)

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Analysis completed in {duration:.2f}s")

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "overall": overall,
            "by_document": by_document,
            "h2_groupings": h2_groupings,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline token distribution before chunking"
    )
    parser.add_argument("--report", type=str, help="Path to save JSON report")
    parser.add_argument("--markdown", type=str, help="Path to save Markdown report")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("=" * 80)
    logger.info("Phase 7E.0 - Task 0.3: Baseline Distribution Analysis")
    logger.info("=" * 80)
    logger.info(f"JSON report: {args.report if args.report else 'None'}")
    logger.info(f"Markdown report: {args.markdown if args.markdown else 'None'}")
    logger.info("")

    try:
        # Get Neo4j connection
        conn_manager = get_connection_manager()
        driver = conn_manager.get_neo4j_driver()

        # Run analysis
        analyzer = BaselineDistributionAnalyzer(driver)
        results = analyzer.run_analysis()

        if not results["success"]:
            logger.error(f"Analysis failed: {results.get('error')}")
            sys.exit(1)

        # Print summary
        overall = results.get("overall", {})
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total sections: {overall.get('total_sections', 0):,}")
        logger.info(f"Average tokens: {overall.get('avg', 0):.1f}")
        logger.info(f"Median tokens: {overall.get('median', 0):.1f}")
        logger.info(f"p95 tokens: {overall.get('percentiles', {}).get(95, 0):.0f}")

        dist_pct = overall.get("distribution_percentages", {})
        logger.info("")
        logger.info("Token range distribution:")
        logger.info(f"  < 200: {dist_pct.get('under_200', 0):.1f}%")
        logger.info(f"  200-800: {dist_pct.get('range_200_800', 0):.1f}%")
        logger.info(f"  800-1,500: {dist_pct.get('range_800_1500', 0):.1f}% ← TARGET")
        logger.info(f"  1,500-7,900: {dist_pct.get('range_1500_7900', 0):.1f}%")
        logger.info(f"  > 7,900: {dist_pct.get('over_7900', 0):.1f}%")

        # Save JSON report
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"\nJSON report saved: {report_path}")

        # Save Markdown report
        if args.markdown:
            md_path = Path(args.markdown)
            md_path.parent.mkdir(parents=True, exist_ok=True)

            markdown = analyzer.generate_markdown_report(results)
            with open(md_path, "w") as f:
                f.write(markdown)

            logger.info(f"Markdown report saved: {md_path}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
