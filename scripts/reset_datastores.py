#!/usr/bin/env python3
"""
Reset Qdrant + Neo4j + Redis contents while preserving schema objects.
Also validates Qdrant/Neo4j schema snapshots after clearing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_QDRANT_VALIDATE_COLLECTION = "chunks_multi_voyage_context_3"
ENV_FILES = [Path(".env.local"), Path(".env"), Path(".env.docker")]
DEFAULT_SCHEMA_VERSION = "v2.2"

# Neo4j metadata labels that must be preserved.
PRESERVED_LABELS = {
    "SchemaVersion",  # Required for health checks (v2.2)
    "SystemMetadata",
    "MigrationHistory",
    "RelationshipTypesMarker",
    "_metadata",
    "_system",
}

# Neo4j data labels to delete.
DATA_LABELS = {
    "Document",
    "Section",
    "Chunk",
    "Entity",
    "CitationUnit",
    "Example",
    "Parameter",
    "Component",
    "Procedure",
    "Command",
    "Configuration",
    "Step",
    "Concept",
    "Topic",
    "Person",
    "Organization",
    "Location",
    "Date",
    "Software",
    "Hardware",
    "Metric",
    "Session",
    "Query",
    "Answer",
    "PendingReference",
    "GhostDocument",
    "Error",
    "QueryFeedback",
}


def _read_env_var(key: str) -> Tuple[Optional[str], Optional[str]]:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() != key:
                continue
            value = v.strip().strip('"').strip("'")
            if value:
                return value, str(path)
    return None, None


def _require_env_value(key: str, args_value: Optional[str]) -> str:
    if args_value:
        return args_value
    env_value = os.environ.get(key)
    if env_value:
        return env_value
    file_value, _source = _read_env_var(key)
    if file_value:
        return file_value
    raise RuntimeError(f"{key} is required (env or env file).")


def _maybe_env_value(key: str, args_value: Optional[str]) -> Optional[str]:
    if args_value:
        return args_value
    env_value = os.environ.get(key)
    if env_value:
        return env_value
    file_value, _source = _read_env_var(key)
    return file_value


def _print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def clear_qdrant(host: str, port: int, *, dry_run: bool = False) -> List[str]:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(host=host, port=port)
    collections = [c.name for c in client.get_collections().collections]
    if not collections:
        print("No Qdrant collections found.")
        return []

    _print_header("Qdrant: clearing collections" + (" [DRY RUN]" if dry_run else ""))
    for name in collections:
        info = client.get_collection(name)
        points_count = info.points_count
        vectors_count = getattr(info, "vectors_count", None)

        if dry_run:
            # Just report what would be deleted
            if vectors_count is not None:
                print(
                    f"{name}: WOULD DELETE {points_count} points, "
                    f"{vectors_count} vectors"
                )
            else:
                print(f"{name}: WOULD DELETE {points_count} points")
            continue

        # Actually delete
        deleted = 0
        while True:
            points, _ = client.scroll(
                collection_name=name,
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            ids = [p.id for p in points]
            client.delete(
                collection_name=name,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )
            deleted += len(ids)
        info = client.get_collection(name)
        vectors_count = getattr(info, "vectors_count", None)
        if vectors_count is not None:
            print(
                f"{name}: deleted={deleted} points={info.points_count} "
                f"vectors={vectors_count} status={info.status}"
            )
        else:
            print(
                f"{name}: deleted={deleted} points={info.points_count} "
                f"status={info.status}"
            )
    return collections


def _ensure_schema_version(session, schema_version: str) -> None:
    result = session.run(
        "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv.version AS version"
    ).single()
    if result and result.get("version"):
        return
    session.run(
        """
        MERGE (sv:SchemaVersion {id: 'singleton'})
        SET sv.version = $version,
            sv.updated_at = datetime()
        """,
        version=schema_version,
    )


def clear_neo4j(
    uri: str,
    user: str,
    password: str,
    database: Optional[str],
    schema_version: str,
    *,
    dry_run: bool = False,
) -> None:
    from neo4j import GraphDatabase

    _print_header(
        "Neo4j: clearing nodes and relationships" + (" [DRY RUN]" if dry_run else "")
    )
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        if database:
            session = driver.session(database=database)
        else:
            session = driver.session()
        with session:
            label_counts = session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(n) AS count
                ORDER BY label
                """
            ).data()

            preserved_labels = {}
            deletable_labels = {}
            for row in label_counts:
                label = row["label"]
                count = row["count"]
                if (
                    label in PRESERVED_LABELS
                    or label.endswith("_metadata")
                    or label.endswith("_system")
                ):
                    preserved_labels[label] = count
                elif label in DATA_LABELS:
                    deletable_labels[label] = count
                else:
                    preserved_labels[label] = count

            if dry_run:
                # Just report what would be deleted
                total_deletable = sum(deletable_labels.values())
                print(
                    f"WOULD DELETE {total_deletable} nodes across {len(deletable_labels)} labels:"
                )
                for label, count in sorted(deletable_labels.items()):
                    print(f"  - {label}: {count} nodes")
                print(
                    f"WOULD PRESERVE {sum(preserved_labels.values())} nodes across {len(preserved_labels)} labels:"
                )
                for label, count in sorted(preserved_labels.items()):
                    print(f"  - {label}: {count} nodes")
                return

            deleted_count = 0
            for label in deletable_labels:
                while True:
                    result = session.run(
                        f"""
                        MATCH (n:{label})
                        WITH n LIMIT 10000
                        DETACH DELETE n
                        RETURN count(n) AS deleted
                        """
                    )
                    batch_deleted = result.single()["deleted"]
                    deleted_count += batch_deleted
                    if batch_deleted == 0:
                        break

            _ensure_schema_version(session, schema_version)

            counts = session.run("MATCH (n) RETURN count(n) AS nodes").single()
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS rels").single()
        print(f"nodes={counts['nodes']} rels={rels['rels']}")
    finally:
        driver.close()


def report_neo4j_schema(
    uri: str, user: str, password: str, database: Optional[str]
) -> None:
    from neo4j import GraphDatabase

    _print_header("Neo4j: schema status")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        if database:
            session = driver.session(database=database)
        else:
            session = driver.session()
        with session:
            try:
                indexes = session.run(
                    """
                    SHOW INDEXES
                    YIELD name, type, entityType, labelsOrTypes, properties, state, owningConstraint
                    WHERE type <> 'LOOKUP'
                    RETURN name, type, entityType, labelsOrTypes, properties, state, owningConstraint
                    ORDER BY name
                    """
                ).data()
            except Exception as exc:
                if "state" not in str(exc):
                    raise
                indexes = session.run(
                    """
                    SHOW INDEXES
                    YIELD name, type, entityType, labelsOrTypes, properties, owningConstraint
                    WHERE type <> 'LOOKUP'
                    RETURN name, type, entityType, labelsOrTypes, properties, owningConstraint
                    ORDER BY name
                    """
                ).data()

            try:
                constraints = session.run(
                    """
                    SHOW CONSTRAINTS
                    YIELD name, type, entityType, labelsOrTypes, properties, state
                    RETURN name, type, entityType, labelsOrTypes, properties, state
                    ORDER BY name
                    """
                ).data()
            except Exception as exc:
                if "state" not in str(exc):
                    raise
                constraints = session.run(
                    """
                    SHOW CONSTRAINTS
                    YIELD name, type, entityType, labelsOrTypes, properties
                    RETURN name, type, entityType, labelsOrTypes, properties
                    ORDER BY name
                    """
                ).data()
    finally:
        driver.close()

    print("Indexes:")
    if not indexes:
        print("  (none)")
    for entry in indexes:
        labels = ",".join(entry.get("labelsOrTypes") or [])
        props = ",".join(entry.get("properties") or [])
        state = entry.get("state")
        owning = entry.get("owningConstraint")
        owning_str = f" owningConstraint={owning}" if owning else ""
        state_str = f" state={state}" if state else ""
        print(
            f"  - {entry['name']} type={entry['type']} entity={entry['entityType']} "
            f"labels=[{labels}] props=[{props}]{state_str}{owning_str}"
        )

    print("Constraints:")
    if not constraints:
        print("  (none)")
    for entry in constraints:
        labels = ",".join(entry.get("labelsOrTypes") or [])
        props = ",".join(entry.get("properties") or [])
        state = entry.get("state")
        state_str = f" state={state}" if state else ""
        print(
            f"  - {entry['name']} type={entry['type']} entity={entry['entityType']} "
            f"labels=[{labels}] props=[{props}]{state_str}"
        )


def flush_redis(
    redis_uri: Optional[str],
    host: str,
    port: int,
    password: Optional[str],
    *,
    dry_run: bool = False,
) -> None:
    _print_header("Redis: flushing DB" + (" [DRY RUN]" if dry_run else ""))
    try:
        import redis  # type: ignore
    except Exception:
        redis = None

    if redis:
        if redis_uri:
            client = redis.Redis.from_url(redis_uri)
        else:
            client = redis.Redis(host=host, port=port, password=password, db=0)

        if dry_run:
            size = client.dbsize()
            info = client.info("keyspace") or {}
            print(f"WOULD FLUSH {size} keys")
            print(f"Current keyspace: {info.get('db0', {})}")
            return

        client.flushdb()
        size = client.dbsize()
        info = client.info("keyspace") or {}
        print(f"dbsize={size} keyspace={info.get('db0', {})}")
        return

    redis_cli = shutil.which("redis-cli")
    if not redis_cli:
        raise RuntimeError("redis library and redis-cli not available.")

    cmd = [redis_cli]
    if redis_uri:
        cmd += ["-u", redis_uri]
    else:
        cmd += ["-h", host, "-p", str(port)]
        if password:
            cmd += ["-a", password]

    if dry_run:
        print("WOULD RUN: FLUSHDB")
        _run(cmd + ["DBSIZE"])
        return

    _run(cmd + ["FLUSHDB"])
    _run(cmd + ["DBSIZE"])


def validate_snapshots(
    qdrant_collections: Iterable[str],
    qdrant_host: str,
    qdrant_port: int,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: Optional[str],
) -> None:
    _print_header("Schema validation")
    qdrant_script = Path(
        "scripts/qdrant_snapshots_20251206_canonical/qdrant_schema_snapshot.py"
    )
    if qdrant_script.exists():
        for name in qdrant_collections:
            _run(
                [
                    sys.executable,
                    str(qdrant_script),
                    "validate",
                    "--collection",
                    name,
                    "--host",
                    qdrant_host,
                    "--port",
                    str(qdrant_port),
                ]
            )
    else:
        print(f"Qdrant snapshot script not found: {qdrant_script}")

    neo4j_script = Path("scripts/neo4j/neo4j_schema_snapshot.py")
    if neo4j_script.exists():
        env = os.environ.copy()
        env["NEO4J_PASSWORD"] = neo4j_password
        env["NEO4J_URI"] = neo4j_uri
        env["NEO4J_USER"] = neo4j_user
        if neo4j_database:
            env["NEO4J_DATABASE"] = neo4j_database
        _run(
            [sys.executable, str(neo4j_script), "validate", "--snapshot-name", "neo4j"],
            env=env,
        )
    else:
        print(f"Neo4j snapshot script not found: {neo4j_script}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear Qdrant/Neo4j/Redis contents, keep schema, and validate snapshots."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without executing",
    )
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--skip-redis", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")

    parser.add_argument(
        "--qdrant-host", default=os.environ.get("QDRANT_HOST", "localhost")
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=int(os.environ.get("QDRANT_PORT", "6333"))
    )
    parser.add_argument(
        "--qdrant-validate-collection",
        action="append",
        dest="qdrant_validate_collections",
    )

    parser.add_argument(
        "--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password")
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE"))

    parser.add_argument("--redis-uri")
    parser.add_argument(
        "--redis-host", default=os.environ.get("REDIS_HOST", "localhost")
    )
    parser.add_argument(
        "--redis-port", type=int, default=int(os.environ.get("REDIS_PORT", "6379"))
    )
    parser.add_argument("--redis-password")

    args = parser.parse_args()
    dry_run = args.dry_run

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No data will be deleted")
        print("=" * 60)

    qdrant_collections = []
    if not args.skip_qdrant:
        qdrant_collections = clear_qdrant(
            args.qdrant_host, args.qdrant_port, dry_run=dry_run
        )

    if not args.skip_neo4j:
        neo4j_password = _require_env_value("NEO4J_PASSWORD", args.neo4j_password)
        schema_version = os.environ.get("SCHEMA_VERSION", DEFAULT_SCHEMA_VERSION)
        clear_neo4j(
            args.neo4j_uri,
            args.neo4j_user,
            neo4j_password,
            args.neo4j_database,
            schema_version,
            dry_run=dry_run,
        )
        report_neo4j_schema(
            args.neo4j_uri, args.neo4j_user, neo4j_password, args.neo4j_database
        )
    else:
        neo4j_password = _maybe_env_value("NEO4J_PASSWORD", args.neo4j_password)

    if not args.skip_redis:
        redis_uri = _maybe_env_value("REDIS_URI", args.redis_uri)
        redis_password = _maybe_env_value("REDIS_PASSWORD", args.redis_password)
        flush_redis(
            redis_uri, args.redis_host, args.redis_port, redis_password, dry_run=dry_run
        )

    if not args.skip_validate:
        validate_collections = args.qdrant_validate_collections or [
            DEFAULT_QDRANT_VALIDATE_COLLECTION
        ]
        if qdrant_collections:
            validate_collections = [
                name for name in validate_collections if name in qdrant_collections
            ]
        if not validate_collections:
            print("No Qdrant collections selected for validation.")
        if not neo4j_password:
            raise RuntimeError(
                "NEO4J_PASSWORD is required for Neo4j schema validation."
            )
        validate_snapshots(
            validate_collections,
            args.qdrant_host,
            args.qdrant_port,
            args.neo4j_uri,
            args.neo4j_user,
            neo4j_password,
            args.neo4j_database,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
