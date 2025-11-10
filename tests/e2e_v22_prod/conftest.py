"""E2E v2.2 production validation fixtures (spec-only; tests not executed yet).

These fixtures prepare a production-like run by:
- Copying hand-selected Markdown docs into the ingestion watch directory
  so the auto-ingest pipeline processes them as in production.
- Creating an artifacts directory to capture verbose logs/metrics when tests run.
- Exposing connection clients for Neo4j and Qdrant, plus tokenizer and a snapshot manifest.

Environment requirements:
- E2E_PROD_DOCS_DIR: Path to a folder with Markdown files to ingest.
- E2E_PROD_DOC_TAG: Doc tag to stamp and to filter during retrieval.
- E2E_PROD_SNAPSHOT_ID (optional): Identifier for this test run; defaults to timestamp.
- E2E_INGEST_SPOOL_PATTERN (optional): set to 'ready' to use .part→.ready rename pattern.

NOTE: This suite does not run now; tests will use these fixtures once approved to execute.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from src.providers.tokenizer_service import TokenizerService
from src.shared.config import init_config
from src.shared.connections import get_connection_manager


def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


@pytest.fixture(scope="session")
def prod_env() -> Dict[str, Any]:
    # Allow default to local test_docs path when E2E_PROD_DOCS_DIR not provided
    default_docs = Path.cwd() / "tests" / "e2e_v22_prod" / "test_docs"
    docs_dir = os.environ.get("E2E_PROD_DOCS_DIR") or str(default_docs)
    snapshot_scope = os.environ.get("E2E_PROD_DOC_TAG")
    if not snapshot_scope:
        pytest.skip("E2E_PROD_DOC_TAG must be set for e2e suite")

    # Validate docs_dir exists and has at least one .md
    dd_path = Path(docs_dir)
    if not dd_path.exists() or not any(dd_path.rglob("*.md")):
        pytest.skip(f"E2E_PROD_DOCS_DIR '{docs_dir}' missing or contains no .md files")

    snapshot_id = os.environ.get("E2E_PROD_SNAPSHOT_ID") or _now_id()
    artifacts_root = Path(__file__).parent / "artifacts" / snapshot_id
    logs_dir = artifacts_root / "logs"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize config and clients (no network actions yet)
    config, settings = init_config()
    manager = get_connection_manager()

    env: Dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "snapshot_scope": snapshot_scope,
        "docs_dir": Path(docs_dir),
        "artifacts_root": artifacts_root,
        "logs_dir": logs_dir,
        "config": config,
        "settings": settings,
        "neo4j": manager.get_neo4j_driver(),
        "qdrant": manager.get_qdrant_client(),
        "tokenizer": TokenizerService(),
    }

    # Persist a manifest placeholder (actual doc_ids filled by tests after ingest)
    (artifacts_root / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "snapshot_scope": snapshot_scope,
                "docs_dir": str(docs_dir),
                "created_at": datetime.utcnow().isoformat(),
                "doc_ids": [],
            },
            indent=2,
        )
    )
    return env


@pytest.fixture(scope="session", autouse=True)
def prepare_ingest(prod_env: Dict[str, Any]):
    """Copy selected Markdown files into the production watch directory and wait for ingest."""
    ingest_dir = os.environ.get("E2E_INGEST_DIR") or str(Path.cwd() / "data" / "ingest")
    watch_dir = Path(ingest_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    src = prod_env["docs_dir"]
    snapshot_scope = prod_env["snapshot_scope"]
    mode = os.environ.get("E2E_INGEST_SPOOL_PATTERN", "direct")

    # Files must reside directly under the watch directory (no subfolders) for the
    # current ingestion service, so copy them to the root while prefixing doc_tag in
    # the filename to retain tagging information.
    dest_dir = watch_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    for path in src.rglob("*.md"):
        rel = path.relative_to(src)
        safe_name = f"{snapshot_scope}__{str(rel).replace('/', '__')}"
        target = dest_dir / safe_name
        if mode == "ready":
            part = target.with_suffix(target.suffix + ".part")
            ready = target.with_suffix(target.suffix + ".ready")
            shutil.copy2(path, part)
            os.replace(part, ready)
        else:
            shutil.copy2(path, target)

    # Wait for ingestion pipeline to pick up files
    driver = prod_env["neo4j"]
    qdrant = prod_env["qdrant"]
    timeout = int(os.environ.get("E2E_INGEST_TIMEOUT", "90"))
    deadline = time.time() + timeout

    def _docs_ready() -> bool:
        with driver.session() as sess:
            row = sess.run(
                "MATCH (d:Document {snapshot_scope:$tag}) RETURN count(d) as c",
                {"tag": snapshot_scope},
            ).single()
            return bool(row and row["c"] >= 1)

    def _vectors_ready() -> bool:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            must=[
                FieldCondition(
                    key="snapshot_scope", match=MatchValue(value=snapshot_scope)
                )
            ]
        )
        res = qdrant.scroll(
            collection_name="chunks_multi",
            limit=1,
            with_payload=False,
            with_vectors=False,
            scroll_filter=flt,
        )
        points = res[0] if isinstance(res, tuple) else res
        return bool(points)

    while time.time() < deadline:
        if _docs_ready() and _vectors_ready():
            break
        time.sleep(2)
    else:
        pytest.fail(
            "Ingestion pipeline did not ingest snapshot_scope within timeout. "
            "Verify watcher/service/worker containers are running."
        )

    return {"watch_dir": str(watch_dir), "dest_dir": str(dest_dir)}


def _capture_container_logs(names: list[str], out_dir: Path) -> None:
    """Best-effort capture of docker logs for requested containers."""
    for name in names:
        try:
            # Lazy import to avoid adding a dependency during collection
            import subprocess  # noqa: WPS433

            result = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=10
            )
            (out_dir / f"{name}.log").write_text(result.stdout + "\n" + result.stderr)
        except Exception as exc:  # noqa: BLE001
            (out_dir / f"{name}.log").write_text(f"log capture failed: {exc}\n")


@pytest.fixture(scope="function")
def capture_logs(prod_env: Dict[str, Any]):
    """Per-test log capture hook (no-op until tests call it).

    Usage in tests:
        capture_logs(before=True)
        ... run action ...
        capture_logs(after=True)
    """
    logs_dir: Path = prod_env["logs_dir"]

    # Build container list from env or use defaults aligning to production stack
    default_containers = [
        "weka-neo4j",
        "weka-qdrant",
        "weka-redis",
        "weka-jaeger",
        "weka-mcp-server",
        "weka-ingestion-service",
        "weka-ingestion-worker",
    ]
    env_val = os.environ.get("E2E_LOG_CONTAINERS")
    container_names = (
        [c.strip() for c in env_val.split(",") if c.strip()]
        if env_val
        else default_containers
    )

    def _do_capture(label: str):
        run_dir = logs_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        # Capture the main service containers if present
        _capture_container_logs(container_names, run_dir)

    return _do_capture
