#!/bin/bash
# ============================================================================
# Qdrant Collections Setup - Phase 7E-4 Production Ready
# Generated: 2025-10-30 15:27:31
# Qdrant Version: 1.7.4
# ============================================================================
#
# This script creates/verifies Qdrant collections for Phase 7E-4
# Collections use 1024-D vectors with cosine distance for Jina embeddings
#
# Usage:
#   bash qdrant_collections_setup.sh
#   QDRANT=http://custom-host:6333 bash qdrant_collections_setup.sh
#
# ============================================================================

set -euo pipefail

# Configuration (can be overridden via environment)
QDRANT="${QDRANT:-http://localhost:6333}"

echo "=============================================="
echo "Qdrant Collections Setup - Phase 7E-4"
echo "Target: $QDRANT"
echo "=============================================="

# Function to create or update a collection
create_collection() {
    local COLLECTION_NAME=$1
    echo ""
    echo "Setting up collection: $COLLECTION_NAME"
    
    # First check if collection exists
    RESPONSE=$(curl -sS "$QDRANT/collections/$COLLECTION_NAME" 2>/dev/null || echo '{"status":"error"}')
    
    if echo "$RESPONSE" | grep -q '"status":"ok"'; then
        echo "  Collection exists, verifying configuration..."
        
        # Extract current config
        CURRENT_DIM=$(echo "$RESPONSE" | grep -o '"size":[0-9]*' | cut -d: -f2 || echo "0")
        CURRENT_DISTANCE=$(echo "$RESPONSE" | grep -o '"distance":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        
        if [ "$CURRENT_DIM" = "1024" ] && [ "${CURRENT_DISTANCE^^}" = "COSINE" ]; then
            echo "  ✅ Collection correctly configured (1024-D, Cosine)"
        else
            echo "  ⚠️  Collection exists with different config: ${CURRENT_DIM}-D, $CURRENT_DISTANCE"
            echo "  ℹ️  Manual intervention may be required to migrate data"
        fi
    else
        echo "  Creating new collection..."
        
        RESPONSE=$(curl -sS -X PUT "$QDRANT/collections/$COLLECTION_NAME" \
          -H 'Content-Type: application/json' \
          -d '{
            "vectors": {
              "size": 1024,
              "distance": "Cosine"
            },
            "optimizers_config": {
              "default_segment_number": 2,
              "indexing_threshold": 20000
            },
            "hnsw_config": {
              "m": 16,
              "ef_construct": 200,
              "full_scan_threshold": 10000
            },
            "quantization_config": {
              "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": false
              }
            },
            "replication_factor": 1,
            "write_consistency_factor": 1,
            "on_disk_payload": true
          }')
        
        if echo "$RESPONSE" | grep -q '"status":"ok"'; then
            echo "  ✅ Collection created successfully"
        else
            echo "  ❌ Failed to create collection: $RESPONSE"
            exit 1
        fi
    fi
}

# Create both required collections
create_collection "chunks"
create_collection "sections"

# Additional collections that might be needed (uncomment if required)
# create_collection "documents"
# create_collection "entities"

echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="

# List all collections and their configurations
echo ""
echo "All collections in Qdrant:"
curl -sS "$QDRANT/collections" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | while read -r collection; do
    if [ ! -z "$collection" ]; then
        CONFIG=$(curl -sS "$QDRANT/collections/$collection" 2>/dev/null)
        DIM=$(echo "$CONFIG" | grep -o '"size":[0-9]*' | cut -d: -f2 || echo "?")
        DISTANCE=$(echo "$CONFIG" | grep -o '"distance":"[^"]*"' | cut -d'"' -f4 || echo "?")
        POINTS=$(echo "$CONFIG" | grep -o '"points_count":[0-9]*' | cut -d: -f2 || echo "0")
        echo "  - $collection: ${DIM}-D, $DISTANCE distance, $POINTS points"
    fi
done

echo ""
echo "✅ Qdrant collections setup complete!"
echo ""
echo "To verify manually:"
echo "  curl -sS '$QDRANT/collections' | jq '.result.collections'"
echo "  curl -sS '$QDRANT/collections/chunks' | jq '.result.config.params.vectors'"
echo "  curl -sS '$QDRANT/collections/sections' | jq '.result.config.params.vectors'"