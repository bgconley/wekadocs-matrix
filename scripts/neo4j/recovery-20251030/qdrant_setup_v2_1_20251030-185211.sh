#!/usr/bin/env bash
# === Phase 7E-4 / GraphRAG v2.1: Qdrant collections (idempotent) ===
QDRANT="${QDRANT:-http://localhost:6333}"
set -euo pipefail
echo "Ensuring Qdrant collection: -"
curl -sS -X PUT "$QDRANT/collections/-" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: CONFIGURATION"
curl -sS -X PUT "$QDRANT/collections/CONFIGURATION" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: Canonical"
curl -sS -X PUT "$QDRANT/collections/Canonical" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: Schema"
curl -sS -X PUT "$QDRANT/collections/Schema" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: Section"
curl -sS -X PUT "$QDRANT/collections/Section" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: See"
curl -sS -X PUT "$QDRANT/collections/See" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: Setup"
curl -sS -X PUT "$QDRANT/collections/Setup" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: already"
curl -sS -X PUT "$QDRANT/collections/already" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: chunks"
curl -sS -X PUT "$QDRANT/collections/chunks" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: collection"
curl -sS -X PUT "$QDRANT/collections/collection" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: configuration"
curl -sS -X PUT "$QDRANT/collections/configuration" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: curl"
curl -sS -X PUT "$QDRANT/collections/curl" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: indexes"
curl -sS -X PUT "$QDRANT/collections/indexes" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: logs"
curl -sS -X PUT "$QDRANT/collections/logs" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: points"
curl -sS -X PUT "$QDRANT/collections/points" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: stats"
curl -sS -X PUT "$QDRANT/collections/stats" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: string"
curl -sS -X PUT "$QDRANT/collections/string" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: upsert"
curl -sS -X PUT "$QDRANT/collections/upsert" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true

echo "Ensuring Qdrant collection: with"
curl -sS -X PUT "$QDRANT/collections/with" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true



echo "Ensuring Qdrant collection: sections"
curl -sS -X PUT "$QDRANT/collections/sections" \
  -H 'Content-Type: application/json' \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": false
    },
    "always_ram": false
  },
  "optimizers_config": {
    "default_segment_number": 2
  },
  "replication_factor": 1,
  "write_consistency_factor": 1,
  "shard_number": 1,
  "on_disk_payload": true
}' || true
