# Baseline Counts for Embedding Field Migration

**Generated:** 2025-10-28T23:52:13.888258
**Note:** Section IDs below are SHA-256 hashes, not secrets

## Neo4j Baseline

| Metric | Count |
|--------|-------|
| Sections with `embedding_model` | 0 |
| Sections with `embedding_version` | 371 |
| Chunks with `embedding_model` | 0 |
| Chunks with `embedding_version` | 371 |
| Total Sections | 371 |
| Total Chunks | 371 |

### Sample Section Nodes
```json
[
  {
    "id": "2b161dfb0d1023344bbc3bb46029d05d51225f3d0b130fc12f15cd15595a4680",
    "model": null,
    "version": "jina-embeddings-v3",
    "dims": 1024,
    "provider": "jina-ai"
  },
  {
    "id": "5a69421f8aa97dcf65edc7fcb2fb7a9a9f183c12c0565258486289b952c9d227",
    "model": null,
    "version": "jina-embeddings-v3",
    "dims": 1024,
    "provider": "jina-ai"
  },
  {
    "id": "c613b196f124ee49e3c3cdc3137f46ff9792d9b29848cb4dbdc08d6a83c51175",
    "model": null,
    "version": "jina-embeddings-v3",
    "dims": 1024,
    "provider": "jina-ai"
  },
  {
    "id": "0ef2746006c043e4c996007bd843404be767c49258ed9b4a3c7d2eb2947ef965",
    "model": null,
    "version": "jina-embeddings-v3",
    "dims": 1024,
    "provider": "jina-ai"
  },
  {
    "id": "ed4a5052d1a851ccc4f73629066e2f7515672b42e3ac45d7dcb3f3a9226410eb",
    "model": null,
    "version": "jina-embeddings-v3",
    "dims": 1024,
    "provider": "jina-ai"
  }
]
```

## Qdrant Baseline

| Metric | Value |
|--------|-------|
| Collection exists | True |
| Points count | 371 |
| Vector size | 1024 |
| Distance metric | Cosine |
| Sample with `embedding_model` | 0/100 |
| Sample with `embedding_version` | 100/100 |

### Sample Payloads
```json
[
  {
    "id": "005e5358-3603-5d2f-ac45-38ba5e841fc3",
    "has_embedding_model": false,
    "has_embedding_version": true,
    "embedding_model": null,
    "embedding_version": "jina-embeddings-v3",
    "embedding_dimensions": 1024,
    "embedding_provider": "jina-ai"
  },
  {
    "id": "00783b37-bda2-50d1-939e-33768ef658c3",
    "has_embedding_model": false,
    "has_embedding_version": true,
    "embedding_model": null,
    "embedding_version": "jina-embeddings-v3",
    "embedding_dimensions": 1024,
    "embedding_provider": "jina-ai"
  },
  {
    "id": "00a623c0-217a-55b5-b4e4-f0dbd5b576f6",
    "has_embedding_model": false,
    "has_embedding_version": true,
    "embedding_model": null,
    "embedding_version": "jina-embeddings-v3",
    "embedding_dimensions": 1024,
    "embedding_provider": "jina-ai"
  }
]
```

## Summary

- **Neo4j**: 0 nodes need field migration
- **Qdrant**: 0% of sampled points have legacy field
- **Target**: Migrate all `embedding_model` → `embedding_version`
