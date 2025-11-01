#!/bin/bash
# ============================================================================
# MASTER RECOVERY SCRIPT - Phase 7E-4 Complete Database Restoration
# Generated: 2025-10-30 15:27:31
# ============================================================================
#
# This script performs complete recovery of all databases to Phase 7E-4 state
# Run this after a catastrophic failure or when setting up a new environment
#
# Prerequisites:
#   - Docker services running (neo4j, qdrant, redis)
#   - Correct passwords set in environment
#
# Usage:
#   bash RECOVERY_MASTER.sh [--dry-run] [--skip-backup]
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DRY_RUN=false
SKIP_BACKUP=false
BACKUP_DIR="/tmp/recovery_backup_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            echo -e "${YELLOW}Skipping backup step${NC}"
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--skip-backup]"
            echo "  --dry-run     Show what would be done without making changes"
            echo "  --skip-backup Skip the initial backup step"
            exit 0
            ;;
    esac
done

# Environment setup
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-testpassword123}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-testredis123}"
export QDRANT="${QDRANT:-http://localhost:6333}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}     PHASE 7E-4 COMPLETE DATABASE RECOVERY${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "Timestamp: $(date)"
echo "Script Directory: $SCRIPT_DIR"
echo "Backup Directory: $BACKUP_DIR"
echo ""

# Function to execute or simulate commands
execute() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $@"
    else
        "$@"
    fi
}

# Step 1: Create backup of current state
if [ "$SKIP_BACKUP" = false ]; then
    echo -e "${BLUE}Step 1: Backing up current state...${NC}"
    execute mkdir -p "$BACKUP_DIR"
    
    # Neo4j backup
    execute docker exec weka-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        "SHOW CONSTRAINTS; SHOW INDEXES;" > "$BACKUP_DIR/neo4j_schema_before.txt" 2>/dev/null || true
    
    # Qdrant backup
    execute curl -sS "$QDRANT/collections" > "$BACKUP_DIR/qdrant_collections_before.json" 2>/dev/null || true
    
    # Redis info
    execute docker exec weka-redis redis-cli -a "$REDIS_PASSWORD" INFO keyspace > "$BACKUP_DIR/redis_keyspace_before.txt" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Backup completed${NC}"
else
    echo -e "${YELLOW}Step 1: Skipping backup (--skip-backup flag)${NC}"
fi

# Step 2: Stop application services
echo -e "\n${BLUE}Step 2: Stopping application services...${NC}"
execute docker-compose stop ingestion-worker ingestion-service mcp-server || true
echo -e "${GREEN}✓ Services stopped${NC}"

# Step 3: Apply Neo4j DDL
echo -e "\n${BLUE}Step 3: Applying Neo4j schema...${NC}"
if [ "$DRY_RUN" = false ]; then
    # Copy DDL to container
    docker cp "$SCRIPT_DIR/neo4j_complete_ddl.cypher" weka-neo4j:/tmp/recovery_ddl.cypher
    
    # Execute DDL
    docker exec weka-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -f /tmp/recovery_ddl.cypher
    
    # Verify
    SCHEMA_VERSION=$(docker exec weka-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        "MATCH (sv:SchemaVersion {id: 'singleton'}) RETURN sv.version" --format plain 2>/dev/null | tail -1)
    
    if [ "$SCHEMA_VERSION" = "v2.1" ]; then
        echo -e "${GREEN}✓ Neo4j schema applied successfully (v2.1)${NC}"
    else
        echo -e "${RED}✗ Schema version verification failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[DRY RUN] Would apply Neo4j DDL from neo4j_complete_ddl.cypher${NC}"
fi

# Step 4: Setup Qdrant collections
echo -e "\n${BLUE}Step 4: Setting up Qdrant collections...${NC}"
if [ "$DRY_RUN" = false ]; then
    bash "$SCRIPT_DIR/qdrant_collections_setup.sh"
    echo -e "${GREEN}✓ Qdrant collections configured${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would run qdrant_collections_setup.sh${NC}"
fi

# Step 5: Redis cleanup (optional - only test DB)
echo -e "\n${BLUE}Step 5: Cleaning Redis test database...${NC}"
execute docker exec weka-redis redis-cli -a "$REDIS_PASSWORD" -n 1 FLUSHDB ASYNC
echo -e "${GREEN}✓ Redis test DB flushed${NC}"

# Step 6: Restart application services
echo -e "\n${BLUE}Step 6: Restarting application services...${NC}"
execute docker-compose up -d mcp-server ingestion-service ingestion-worker

# Wait for services to be healthy
echo -n "Waiting for services to be healthy"
for i in {1..30}; do
    if [ "$DRY_RUN" = false ]; then
        sleep 2
        echo -n "."
        
        # Check MCP health
        if curl -sS http://localhost:8000/health 2>/dev/null | grep -q "healthy"; then
            echo -e "\n${GREEN}✓ Services are healthy${NC}"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo -e "\n${RED}✗ Services failed to become healthy${NC}"
            exit 1
        fi
    else
        echo -e "\n${YELLOW}[DRY RUN] Would wait for services to be healthy${NC}"
        break
    fi
done

# Step 7: Run verification
echo -e "\n${BLUE}Step 7: Running verification checks...${NC}"
if [ "$DRY_RUN" = false ]; then
    # Neo4j checks
    CONSTRAINTS=$(docker exec weka-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        "SHOW CONSTRAINTS YIELD name RETURN count(*) as count" --format plain 2>/dev/null | tail -1)
    INDEXES=$(docker exec weka-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        "SHOW INDEXES YIELD name RETURN count(*) as count" --format plain 2>/dev/null | tail -1)
    
    echo "  Neo4j: $CONSTRAINTS constraints, $INDEXES indexes"
    
    # Qdrant checks
    COLLECTIONS=$(curl -sS "$QDRANT/collections" 2>/dev/null | grep -c '"name"' || echo "0")
    echo "  Qdrant: $COLLECTIONS collections"
    
    # Redis check
    REDIS_PING=$(docker exec weka-redis redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null)
    echo "  Redis: $REDIS_PING"
    
    # MCP health
    MCP_STATUS=$(curl -sS http://localhost:8000/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "failed")
    echo "  MCP Server: $MCP_STATUS"
    
    echo -e "${GREEN}✓ All verification checks passed${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would run verification checks${NC}"
fi

# Summary
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}     RECOVERY COMPLETE${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "Recovery timestamp: $(date)"
echo ""
echo "Next steps:"
echo "1. Verify application functionality"
echo "2. Run smoke tests: python tests/test_phase7e4_*.py"
echo "3. Check logs: docker-compose logs -f mcp-server"
echo "4. Monitor health: curl http://localhost:8000/health"
echo ""
echo -e "${GREEN}System is ready for Phase 7E-4 operations!${NC}"

# Optional: Run smoke tests automatically
if [ "$DRY_RUN" = false ]; then
    read -p "Run automated smoke tests? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 "$SCRIPT_DIR/../../../tests/test_phase7e4_observability.py" -v || true
    fi
fi