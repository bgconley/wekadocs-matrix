#!/usr/bin/env python3
"""
Parser benchmark script for Phase 4 migration validation.

Compares performance and output equivalence between legacy and markdown-it-py parsers.

Usage:
    # Benchmark single file
    python scripts/benchmark_parsers.py --file data/ingest/test.md

    # Benchmark directory
    python scripts/benchmark_parsers.py --dir data/ingest/

    # Generate JSON report
    python scripts/benchmark_parsers.py --dir data/ingest/ --report benchmark_report.json

    # Verbose output
    python scripts/benchmark_parsers.py --file test.md -v
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.parsers.markdown import parse_markdown as parse_legacy
from src.ingestion.parsers.markdown_it_parser import parse_markdown as parse_mit
from src.ingestion.parsers.shadow_comparison import compare_parser_results


@dataclass
class ParserTiming:
    """Timing results for a single parser run."""

    parser: str
    duration_ms: float
    section_count: int
    document_title: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single document."""

    file_path: str
    file_size_bytes: int
    legacy_timing: ParserTiming
    mit_timing: ParserTiming
    speedup_factor: float  # > 1 means mit is faster
    has_differences: bool
    difference_summary: str
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""

    total_files: int
    successful_files: int
    failed_files: int
    total_bytes: int

    # Timing statistics (ms)
    legacy_mean_ms: float
    legacy_median_ms: float
    legacy_p95_ms: float
    mit_mean_ms: float
    mit_median_ms: float
    mit_p95_ms: float

    # Speedup
    mean_speedup: float
    median_speedup: float

    # Equivalence
    files_with_differences: int
    difference_rate_percent: float

    # Individual results
    results: List[BenchmarkResult] = field(default_factory=list)


def benchmark_file(file_path: Path, iterations: int = 3) -> BenchmarkResult:
    """
    Benchmark a single file with both parsers.

    Args:
        file_path: Path to markdown file
        iterations: Number of iterations for timing (takes median)

    Returns:
        BenchmarkResult with timing and comparison data
    """
    try:
        raw_text = file_path.read_text(encoding="utf-8")
        file_size = file_path.stat().st_size
        source_uri = f"file://{file_path}"

        # Benchmark legacy parser
        legacy_times = []
        legacy_result = None
        for _ in range(iterations):
            start = time.perf_counter()
            legacy_result = parse_legacy(source_uri, raw_text)
            elapsed = (time.perf_counter() - start) * 1000
            legacy_times.append(elapsed)

        legacy_timing = ParserTiming(
            parser="legacy",
            duration_ms=statistics.median(legacy_times),
            section_count=len(legacy_result.get("Sections", [])),
            document_title=legacy_result.get("Document", {}).get("title"),
        )

        # Benchmark markdown-it-py parser
        mit_times = []
        mit_result = None
        for _ in range(iterations):
            start = time.perf_counter()
            mit_result = parse_mit(source_uri, raw_text)
            elapsed = (time.perf_counter() - start) * 1000
            mit_times.append(elapsed)

        mit_timing = ParserTiming(
            parser="markdown-it-py",
            duration_ms=statistics.median(mit_times),
            section_count=len(mit_result.get("Sections", [])),
            document_title=mit_result.get("Document", {}).get("title"),
        )

        # Calculate speedup (> 1 means mit is faster)
        speedup = (
            legacy_timing.duration_ms / mit_timing.duration_ms
            if mit_timing.duration_ms > 0
            else 0
        )

        # Compare results
        comparison = compare_parser_results(source_uri, legacy_result, mit_result)

        return BenchmarkResult(
            file_path=str(file_path),
            file_size_bytes=file_size,
            legacy_timing=legacy_timing,
            mit_timing=mit_timing,
            speedup_factor=speedup,
            has_differences=comparison.has_differences,
            difference_summary=comparison.summary(),
        )

    except Exception as e:
        return BenchmarkResult(
            file_path=str(file_path),
            file_size_bytes=0,
            legacy_timing=ParserTiming(parser="legacy", duration_ms=0, section_count=0),
            mit_timing=ParserTiming(
                parser="markdown-it-py", duration_ms=0, section_count=0
            ),
            speedup_factor=0,
            has_differences=False,
            difference_summary="",
            error=str(e),
        )


def generate_report(results: List[BenchmarkResult]) -> BenchmarkReport:
    """Generate aggregate report from individual results."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    if not successful:
        return BenchmarkReport(
            total_files=len(results),
            successful_files=0,
            failed_files=len(failed),
            total_bytes=0,
            legacy_mean_ms=0,
            legacy_median_ms=0,
            legacy_p95_ms=0,
            mit_mean_ms=0,
            mit_median_ms=0,
            mit_p95_ms=0,
            mean_speedup=0,
            median_speedup=0,
            files_with_differences=0,
            difference_rate_percent=0,
            results=results,
        )

    legacy_times = [r.legacy_timing.duration_ms for r in successful]
    mit_times = [r.mit_timing.duration_ms for r in successful]
    speedups = [r.speedup_factor for r in successful]
    files_with_diffs = sum(1 for r in successful if r.has_differences)

    def percentile(data: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    return BenchmarkReport(
        total_files=len(results),
        successful_files=len(successful),
        failed_files=len(failed),
        total_bytes=sum(r.file_size_bytes for r in successful),
        legacy_mean_ms=statistics.mean(legacy_times),
        legacy_median_ms=statistics.median(legacy_times),
        legacy_p95_ms=percentile(legacy_times, 95),
        mit_mean_ms=statistics.mean(mit_times),
        mit_median_ms=statistics.median(mit_times),
        mit_p95_ms=percentile(mit_times, 95),
        mean_speedup=statistics.mean(speedups),
        median_speedup=statistics.median(speedups),
        files_with_differences=files_with_diffs,
        difference_rate_percent=(files_with_diffs / len(successful)) * 100,
        results=results,
    )


def print_report(report: BenchmarkReport, verbose: bool = False) -> None:
    """Print report to console."""
    print("\n" + "=" * 70)
    print("PARSER BENCHMARK REPORT")
    print("=" * 70)

    print(
        f"\nFiles: {report.successful_files} successful, {report.failed_files} failed"
    )
    print(f"Total size: {report.total_bytes / 1024:.1f} KB")

    print("\n--- Timing (ms) ---")
    print(f"{'Parser':<15} {'Mean':>10} {'Median':>10} {'P95':>10}")
    print("-" * 45)
    print(
        f"{'Legacy':<15} {report.legacy_mean_ms:>10.2f} {report.legacy_median_ms:>10.2f} {report.legacy_p95_ms:>10.2f}"
    )
    print(
        f"{'markdown-it-py':<15} {report.mit_mean_ms:>10.2f} {report.mit_median_ms:>10.2f} {report.mit_p95_ms:>10.2f}"
    )

    print("\n--- Speedup ---")
    print(f"Mean speedup:   {report.mean_speedup:.2f}x", end="")
    if report.mean_speedup > 1:
        print(" (markdown-it-py is faster)")
    elif report.mean_speedup < 1:
        print(" (legacy is faster)")
    else:
        print(" (same speed)")
    print(f"Median speedup: {report.median_speedup:.2f}x")

    print("\n--- Equivalence ---")
    print(f"Files with differences: {report.files_with_differences}")
    print(f"Difference rate: {report.difference_rate_percent:.1f}%")

    if verbose and report.results:
        print("\n--- Individual Results ---")
        for r in report.results:
            status = "ERROR" if r.error else ("DIFF" if r.has_differences else "OK")
            print(
                f"  [{status}] {Path(r.file_path).name}: "
                f"legacy={r.legacy_timing.duration_ms:.1f}ms, "
                f"mit={r.mit_timing.duration_ms:.1f}ms, "
                f"speedup={r.speedup_factor:.2f}x"
            )
            if r.has_differences:
                print(f"        ^ {r.difference_summary}")
            if r.error:
                print(f"        ^ Error: {r.error}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark legacy vs markdown-it-py parsers"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Single markdown file to benchmark",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory of markdown files to benchmark",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output JSON report file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per file (default: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Must specify --file or --dir")

    # Collect files to benchmark
    files: List[Path] = []
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        files.append(args.file)
    if args.dir:
        if not args.dir.is_dir():
            print(f"Error: Not a directory: {args.dir}")
            sys.exit(1)
        files.extend(args.dir.glob("**/*.md"))

    if not files:
        print("No markdown files found")
        sys.exit(1)

    print(f"Benchmarking {len(files)} files...")

    # Run benchmarks
    results = []
    for i, file_path in enumerate(files, 1):
        if args.verbose:
            print(f"  [{i}/{len(files)}] {file_path.name}...", end="", flush=True)
        result = benchmark_file(file_path, iterations=args.iterations)
        results.append(result)
        if args.verbose:
            status = "ERROR" if result.error else "OK"
            print(f" {status} ({result.mit_timing.duration_ms:.1f}ms)")

    # Generate and print report
    report = generate_report(results)
    print_report(report, verbose=args.verbose)

    # Write JSON report if requested
    if args.report:
        report_dict = asdict(report)
        with open(args.report, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\nJSON report written to: {args.report}")

    # Exit with error if any files had differences and we're in strict mode
    if report.files_with_differences > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
