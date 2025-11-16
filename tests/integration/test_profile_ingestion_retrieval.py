import os
import uuid
from contextlib import contextmanager
from datetime import datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.providers.factory import ProviderFactory
from src.query.hybrid_retrieval import QdrantMultiVectorRetriever
from src.shared import config as config_module
from src.shared.config import get_embedding_settings
from src.shared.embedding_fields import canonicalize_embedding_metadata

PROFILE_REQUIREMENTS = {
    "jina_v3": ["JINA_API_KEY"],
    "bge_m3": ["BGE_M3_API_URL", "BGE_M3_CLIENT_PATH"],
    "st_minilm": ["RUN_ST_MINILM_PROFILE"],
}


def _should_run() -> bool:
    return os.getenv("RUN_PROFILE_MATRIX_INTEGRATION") == "1"


def _missing_env(vars_):
    return [var for var in vars_ if not os.getenv(var)]


def _ensure_requirements(profile: str):
    missing = _missing_env(PROFILE_REQUIREMENTS.get(profile, []))
    if missing:
        pytest.skip(f"Profile {profile} missing required environment: {missing}")


def _get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


@contextmanager
def _profile_env(profile: str):
    patch_env = {"EMBEDDINGS_PROFILE": profile}  # pragma: allowlist secret
    if "NEO4J_PASSWORD" not in os.environ:
        patch_env["NEO4J_PASSWORD"] = "placeholder"  # pragma: allowlist secret
    original = {k: os.environ.get(k) for k in patch_env}
    try:
        os.environ.update(patch_env)
        config_module._config = None
        config_module._settings = None
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        config_module._config = None
        config_module._settings = None


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.parametrize("profile", ["jina_v3", "bge_m3", "st_minilm"])
def test_profile_ingestion_and_retrieval(profile):
    if not _should_run():
        pytest.skip("RUN_PROFILE_MATRIX_INTEGRATION=1 required for real-provider tests")
    _ensure_requirements(profile)

    with _profile_env(profile):
        settings = get_embedding_settings()
        try:
            embedder = ProviderFactory.create_embedding_provider(settings=settings)
        except Exception as exc:
            pytest.skip(f"Unable to create provider for {profile}: {exc}")

        client = _get_qdrant_client()
        collection_name = f"profile_matrix_{profile}_{uuid.uuid4().hex}"
        try:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "content": VectorParams(
                        size=settings.dims, distance=Distance.COSINE
                    )
                },
            )
        except Exception as exc:
            pytest.skip(f"Qdrant not reachable: {exc}")

        try:
            payload = canonicalize_embedding_metadata(
                embedding_model=settings.version,
                dimensions=settings.dims,
                provider=settings.provider,
                task=settings.task,
                profile=settings.profile,
                timestamp=datetime.utcnow(),
            )
            payload.update(
                {
                    "node_id": "profile-test",
                    "node_label": "Section",
                    "document_id": "doc-profile",
                }
            )

            text = "Profile modularization integration test"
            vector = embedder.embed_documents([text])[0]
            point_id = str(uuid.uuid4())
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"content": vector},
                        payload=payload,
                    )
                ],
            )

            search = client.search(
                collection_name=collection_name,
                query_vector={"name": "content", "vector": vector},
                limit=1,
                with_payload=True,
            )
            assert search, "Qdrant search returned no results"
            assert (
                search[0].payload.get("embedding_version") == settings.version
            ), "Payload missing embedding_version"

            retriever = QdrantMultiVectorRetriever(
                client,
                embedder,
                collection_name=collection_name,
                field_weights={"content": 1.0},
                embedding_settings=settings,
            )
            results = retriever.search(
                "integration test profile matrix", top_k=1, filters=None
            )
            assert results, "QdrantMultiVectorRetriever returned no results"
            assert results[0].embedding_version == settings.version
        finally:
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception:
                pass
