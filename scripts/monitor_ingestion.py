#!/usr/bin/env python3
"""
Real-time ingestion monitor for WekaDocs production document ingestion.
Tracks file drops, processing status, and database updates.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: str, shell: bool = True) -> Optional[str]:
    """Run command and return output, or None on error."""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def count_files_in_directory(path: str) -> Dict[str, int]:
    """Count files in ingestion directory by extension."""
    counts = {"total": 0, "md": 0, "html": 0, "other": 0}
    try:
        p = Path(path)
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    counts["total"] += 1
                    ext = f.suffix.lower()
                    if ext in [".md", ".markdown"]:
                        counts["md"] += 1
                    elif ext in [".html", ".htm"]:
                        counts["html"] += 1
                    else:
                        counts["other"] += 1
    except Exception:
        pass
    return counts


def get_neo4j_counts() -> Dict[str, int]:
    """Get current node counts from Neo4j."""
    counts = {}
    cmd = '''docker exec weka-neo4j cypher-shell -u neo4j -p testpassword123 "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"'''
    output = run_command(cmd)
    if output:
        for line in output.split("\n")[1:]:  # Skip header
            if line.strip():
                parts = line.split(",")
                if len(parts) == 2:
                    label = parts[0].strip('"')
                    count = int(parts[1].strip())
                    counts[label] = count
    return counts


def get_qdrant_counts() -> Dict[str, int]:
    """Get vector counts from Qdrant collections."""
    counts = {}
    # Get list of collections
    collections_cmd = "curl -s http://localhost:6333/collections"
    output = run_command(collections_cmd)
    if output:
        try:
            data = json.loads(output)
            for coll in data.get("result", {}).get("collections", []):
                name = coll.get("name", "")
                # Get count for each collection
                count_cmd = f'curl -s "http://localhost:6333/collections/{name}"'
                coll_output = run_command(count_cmd)
                if coll_output:
                    coll_data = json.loads(coll_output)
                    points_count = coll_data.get("result", {}).get("points_count", 0)
                    if points_count > 0:
                        counts[name] = points_count
        except Exception:
            pass
    return counts


def get_redis_queue_status() -> Dict[str, int]:
    """Get Redis queue lengths."""
    stats = {"pending": 0, "processing": 0, "failed": 0}

    # Check pending queue
    cmd = 'docker exec weka-redis redis-cli -a testredis123 LLEN "ingestion:queue:pending" 2>/dev/null'
    output = run_command(cmd)
    if output and output.isdigit():
        stats["pending"] = int(output)

    # Check processing set
    cmd = 'docker exec weka-redis redis-cli -a testredis123 SCARD "ingestion:queue:processing" 2>/dev/null'
    output = run_command(cmd)
    if output and output.isdigit():
        stats["processing"] = int(output)

    # Check failed queue
    cmd = 'docker exec weka-redis redis-cli -a testredis123 LLEN "ingestion:queue:failed" 2>/dev/null'
    output = run_command(cmd)
    if output and output.isdigit():
        stats["failed"] = int(output)

    return stats


def get_worker_logs(lines: int = 5) -> List[str]:
    """Get recent worker logs."""
    cmd = f"docker logs weka-ingestion-worker --tail {lines} 2>&1"
    output = run_command(cmd)
    if output:
        return output.split("\n")
    return []


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def format_timestamp() -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def display_status(
    start_time: datetime, initial_neo4j: Dict[str, int], initial_qdrant: Dict[str, int]
):
    """Display current ingestion status."""
    clear_screen()

    # Header
    print("=" * 80)
    print(f"🔍 WEKADOCS INGESTION MONITOR - {format_timestamp()}")
    print(f"   Runtime: {str(datetime.now() - start_time).split('.')[0]}")
    print("=" * 80)

    # Ingest directory status
    print("\n📁 INGEST DIRECTORY (/data/ingest):")
    file_counts = count_files_in_directory(
        "/Users/brennanconley/vibecode/wekadocs-matrix/data/ingest"
    )
    print(f"   Total Files: {file_counts['total']}")
    print(
        f"   Markdown: {file_counts['md']} | HTML: {file_counts['html']} | Other: {file_counts['other']}"
    )

    # Redis queue status
    print("\n📋 REDIS QUEUE STATUS:")
    queue_status = get_redis_queue_status()
    print(
        f"   Pending: {queue_status['pending']} | Processing: {queue_status['processing']} | Failed: {queue_status['failed']}"
    )

    # Neo4j status
    print("\n🗂️  NEO4J DATABASE:")
    neo4j_counts = get_neo4j_counts()
    if neo4j_counts:
        for label, count in list(neo4j_counts.items())[:5]:  # Top 5 labels
            initial = initial_neo4j.get(label, 0)
            delta = count - initial
            delta_str = f"(+{delta})" if delta > 0 else ""
            print(f"   {label}: {count} {delta_str}")

        # Total nodes
        total = sum(neo4j_counts.values())
        initial_total = sum(initial_neo4j.values())
        total_delta = total - initial_total
        print(f"   TOTAL NODES: {total} (+{total_delta} new)")
    else:
        print("   No data available")

    # Qdrant status
    print("\n🔍 QDRANT VECTORS:")
    qdrant_counts = get_qdrant_counts()
    if qdrant_counts:
        for coll, count in list(qdrant_counts.items())[:3]:  # Top 3 collections
            initial = initial_qdrant.get(coll, 0)
            delta = count - initial
            delta_str = f"(+{delta})" if delta > 0 else ""
            print(f"   {coll}: {count} {delta_str}")
    else:
        print("   No collections with data")

    # Recent worker activity
    print("\n📝 RECENT WORKER ACTIVITY:")
    logs = get_worker_logs(3)
    for log in logs:
        if log.strip():
            # Truncate long lines
            if len(log) > 77:
                log = log[:74] + "..."
            print(f"   {log}")

    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Updates every 2 seconds")


def main():
    """Main monitoring loop."""
    print("Starting WekaDocs Ingestion Monitor...")
    print("Collecting initial baseline...")

    # Get initial counts
    start_time = datetime.now()
    initial_neo4j = get_neo4j_counts()
    initial_qdrant = get_qdrant_counts()

    print(
        f"Baseline: {sum(initial_neo4j.values())} Neo4j nodes, {sum(initial_qdrant.values())} vectors"
    )
    print("\nMonitoring started. Waiting for file drops in /data/ingest...")
    time.sleep(2)

    try:
        while True:
            display_status(start_time, initial_neo4j, initial_qdrant)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
        print(f"Final stats after {str(datetime.now() - start_time).split('.')[0]}:")

        # Final summary
        final_neo4j = get_neo4j_counts()
        final_qdrant = get_qdrant_counts()

        neo4j_delta = sum(final_neo4j.values()) - sum(initial_neo4j.values())
        qdrant_delta = sum(final_qdrant.values()) - sum(initial_qdrant.values())

        print(f"  - Neo4j: {neo4j_delta} new nodes added")
        print(f"  - Qdrant: {qdrant_delta} new vectors added")

        queue_status = get_redis_queue_status()
        if queue_status["failed"] > 0:
            print(f"  ⚠️  {queue_status['failed']} failed jobs in queue")

        sys.exit(0)


if __name__ == "__main__":
    main()
