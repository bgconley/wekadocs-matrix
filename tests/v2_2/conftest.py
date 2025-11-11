"""Shared fixtures for v2.2 integration tests."""

import hashlib
import os
import socket
from pathlib import Path
from typing import List

import pytest

from src.providers.tokenizer_service import TokenizerService
from src.shared.config import init_config
from src.shared.connections import get_connection_manager

DEFAULT_ENV = {
    "ENV": "development",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "testpassword123",  # pragma: allowlist secret
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_PASSWORD": "testredis123",  # pragma: allowlist secret
    "JWT_SECRET": "test-secret-key-change-in-production-min-32-chars",  # pragma: allowlist secret
    "TRANSFORMERS_OFFLINE": "true",
    "HF_TOKENIZER_ID": "jinaai/jina-embeddings-v3",
    "HF_CACHE": str(Path.home() / ".cache" / "huggingface"),
}
for key, value in DEFAULT_ENV.items():
    os.environ.setdefault(key, value)


class DeterministicEmbeddingProvider:
    """Offline-friendly embedding stub that mimics 1024-D Jina vectors."""

    provider_name = "deterministic-test"
    model_id = "deterministic-v1"
    task = "retrieval.passage"

    def __init__(self, dims: int = 1024):
        self.dims = dims

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest() or b"seed"
        values: List[float] = []
        while len(values) < self.dims:
            for idx in range(0, len(digest), 4):
                chunk = digest[idx : idx + 4]
                if not chunk:
                    break
                val = int.from_bytes(chunk, "big") % 1000
                values.append(val / 1000.0)
                if len(values) == self.dims:
                    break
            digest = hashlib.sha256(digest).digest()
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def _ensure_service_ready(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


@pytest.fixture(scope="session")
def integration_env():
    neo4j_host = os.environ["NEO4J_URI"].split("://")[-1].split(":")[0]
    required = [
        (neo4j_host, 7687),
        (os.environ["QDRANT_HOST"], int(os.environ["QDRANT_PORT"])),
        (os.environ["REDIS_HOST"], int(os.environ["REDIS_PORT"])),
    ]
    missing = [host for host, port in required if not _ensure_service_ready(host, port)]
    if missing:
        pytest.skip(
            "Integration services unavailable: " + ", ".join(sorted(set(missing)))
        )

    config, settings = init_config()
    manager = get_connection_manager()
    driver = manager.get_neo4j_driver()
    qdrant = manager.get_qdrant_client()
    tokenizer = TokenizerService()
    embedder = DeterministicEmbeddingProvider(dims=config.embedding.dims)

    return {
        "config": config,
        "settings": settings,
        "driver": driver,
        "qdrant": qdrant,
        "tokenizer": tokenizer,
        "embedder": embedder,
    }
