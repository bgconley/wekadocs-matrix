# WekaDocs Matrix Configuration Guide

**Pre-Phase 7 Edition**
**Version:** 2.0 (Post Pre-Phase7 Foundation)
**Status:** Production Ready

## Overview

The WekaDocs Matrix application uses a hierarchical configuration system that combines:
1. YAML configuration files for application settings (single source of truth)
2. Environment variables for sensitive values and runtime overrides
3. Startup validation for critical configuration parameters

**Pre-Phase 7 Enhancements:**
- ✅ Eliminated phantom defaults (all values from config)
- ✅ Added embedding provider abstraction
- ✅ Dimension validation on startup
- ✅ Provenance tracking via embedding.version
- ✅ Feature flags for gradual rollout

This guide explains how to configure the application for different environments.

## Configuration Loading

### Environment Selection

The application determines which configuration file to load based on the `ENV` environment variable:

```bash
# Development (default)
ENV=development  # Loads config/development.yaml

# Staging
ENV=staging      # Loads config/staging.yaml

# Production
ENV=production   # Loads config/production.yaml
```

### Custom Configuration Path

You can override the default configuration file location using `CONFIG_PATH`:

```bash
# Use a custom configuration file
CONFIG_PATH=/path/to/custom/config.yaml python src/main.py

# Use relative path
CONFIG_PATH=./my-config.yaml python src/main.py
```

## Environment Variables

### Required Variables

These environment variables must be set for the application to start:

```bash
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password  # Optional if no auth

# JWT Authentication (for MCP server)
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
```

### Optional Variables

```bash
# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# OpenTelemetry (if using)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=weka-mcp-server
```

## Configuration Structure

### Key Configuration Sections

#### Embedding Configuration (Critical - Pre-Phase 7)
Controls vector embedding generation. **This is now the single source of truth** for all embedding parameters:

```yaml
embedding:
  # Model configuration
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dims: 384  # MUST match model output dimensions EXACTLY
  similarity: "cosine"  # Options: cosine, dot, euclidean
  version: "miniLM-L6-v2-2024-01-01"  # REQUIRED for provenance tracking

  # Provider configuration (Pre-Phase 7)
  provider: "sentence-transformers"  # Future: "jina" in Phase 7
  task: "retrieval.passage"  # Task-specific embeddings (for Jina)

  # Performance tuning
  batch_size: 32  # Batch size for embedding generation
  max_sequence_length: 512  # Max tokens per input
```

**Critical Fields:**
- `dims`: Must match the model's actual output. Validation runs on startup.
- `version`: Required for tracking which embeddings need re-generation during model upgrades.
- `similarity`: Must match Qdrant collection distance metric.

**Pre-Phase 7 Safety:**
- Application validates `dims > 0` on startup
- Application validates `version` is not empty
- Provider validates actual model output matches configured `dims`
- Dimension mismatches cause immediate failure (fail-fast)

#### Search Configuration
Controls retrieval behavior:

```yaml
search:
  hybrid:
    top_k: 20
    vector_weight: 0.7
    text_weight: 0.3
```

#### Feature Flags
Enable/disable features for gradual rollout:

```yaml
feature_flags:
  use_embedding_provider: true
  validate_dimensions: true
  coverage_enrichment: true
  byte_cap_responses: true
  schema_v2_1_dev: true  # Dev only
```

## Development Setup

### Quick Start

1. Create a `.env` file in the project root:

```bash
# .env
ENV=development
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword123
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET=dev-secret-key
LOG_LEVEL=DEBUG
```

2. Load environment variables:

```bash
# Using python-dotenv
python -m dotenv run python src/main.py

# Or source manually
source .env && python src/main.py

# Or export individually
export ENV=development
export NEO4J_PASSWORD=testpassword123
# ... etc
```

3. The application will:
   - Load `config/development.yaml`
   - Override with any environment variables
   - Validate configuration at startup
   - Log the loaded configuration

## Production Setup

### Best Practices

1. **Never commit secrets** - Use environment variables for all sensitive values
2. **Use separate configs** - Maintain separate YAML files per environment
3. **Version your configs** - Track embedding versions for migration safety
4. **Validate at startup** - The application validates critical settings on start

### Example Production Setup

```bash
# Set environment
export ENV=production
export CONFIG_PATH=/etc/wekadocs/config.yaml

# Database connections (from secret manager)
export NEO4J_URI=$(vault kv get -field=uri secret/neo4j)
export NEO4J_PASSWORD=$(vault kv get -field=password secret/neo4j)

# Start application
python src/main.py
```

## Configuration Validation (Pre-Phase 7 Enhanced)

The application performs comprehensive validation at startup:

### Embedding Configuration Validation

1. **Embedding dimensions** - Must be positive integer (`dims > 0`)
2. **Embedding version** - Must be non-empty string (required for provenance)
3. **Similarity metric** - Must be one of: `cosine`, `dot`, `euclidean`
4. **Provider validation** - Provider generates test embedding and verifies dimensions match

**Example startup log:**
```
INFO     Embedding configuration loaded: model=sentence-transformers/all-MiniLM-L6-v2, dims=384, version=miniLM-L6-v2-2024-01-01, provider=sentence-transformers
INFO     Configuration validation successful
```

### Environment Variables Validation

5. **Required env vars** - All database passwords must be set
6. **Connection strings** - NEO4J_URI, QDRANT_HOST, REDIS_HOST must be valid

### Fail-Fast Behavior

If validation fails, the application will:
1. Log the specific error with context
2. Refuse to start (no partial initialization)
3. Exit with non-zero status
4. Provide actionable error messages

**Example validation failure:**
```
ERROR    embedding.dims must be positive, got 0
ERROR    Configuration validation failed
```

### Startup Checklist

When application starts successfully, you should see:
- ✅ Configuration file loaded from path
- ✅ Embedding configuration logged
- ✅ Feature flags status logged
- ✅ Configuration validation successful
- ✅ Database connections established

## Pre-Phase 7 Foundation (New in v2.0)

### What Changed in Pre-Phase 7?

Pre-Phase 7 foundation work eliminated hardcoded values and added safety validations. Key changes:

**1. Configuration is Now Single Source of Truth**
- Before: Dimensions hardcoded as `384` in multiple files
- After: All values read from `config.embedding.dims`
- Impact: Easy model switching, no phantom defaults

**2. Embedding Provider Abstraction**
- Before: Direct `SentenceTransformer` usage throughout codebase
- After: Provider interface with validation
- Impact: Ready for Jina integration in Phase 7

**3. Dimension Safety**
- Before: Dimension mismatches could corrupt vectors silently
- After: Pre-upsert validation, fail-fast on mismatch
- Impact: Prevents vector corruption

**4. Provenance Tracking**
- Before: No tracking of which model generated embeddings
- After: `embedding.version` stored with every vector
- Impact: Safe model migrations, can filter by version

### Feature Flags (Pre-Phase 7)

New feature flags control pre-phase7 enhancements:

```yaml
feature_flags:
  use_embedding_provider: true   # Use provider abstraction (vs direct model)
  validate_dimensions: true       # Pre-upsert dimension validation
  coverage_enrichment: true       # Add connection/mention counts to ranking
  byte_cap_responses: true        # 32KB cap in FULL mode
  schema_v2_1_dev: true          # Schema v2.1 (dev only, optional)
```

**Recommendation:** Keep all flags `true` in development and production (these are battle-tested improvements).

### Migration from Hardcoded Configuration

If upgrading from a version with hardcoded values:

**Step 1: Verify Configuration**
```bash
# Check your config file has all required embedding fields
grep -A5 "embedding:" config/development.yaml

# Should show:
#   model_name: "sentence-transformers/all-MiniLM-L6-v2"
#   dims: 384
#   version: "miniLM-L6-v2-2024-01-01"
```

**Step 2: Test Startup Validation**
```bash
# Start application and verify startup logs
python -m src.mcp_server.main 2>&1 | grep -E "(Embedding configuration|validation)"

# Should show:
#   INFO Embedding configuration loaded: model=..., dims=384, ...
#   INFO Configuration validation successful
```

**Step 3: Verify No Hardcoded Values Remain**
```bash
# Search for hardcoded dimensions (should find none in src/)
grep -rn "384" src/ --include="*.py" | grep -v "# OK:" | grep -v "test"

# Search for hardcoded model names
grep -rn "all-MiniLM-L6-v2" src/ --include="*.py" | grep -v "test"
```

**Step 4: Test Dimension Validation**
```python
# Optional: Test that dimension validation works
# Temporarily change dims in config to wrong value
# Application should refuse to start with clear error
```

## Troubleshooting

### Common Issues

#### Configuration File Not Found
```
FileNotFoundError: Configuration file not found: config/development.yaml
```
**Solution:** Ensure the config file exists or set `CONFIG_PATH` to the correct location.

#### Invalid Embedding Dimensions
```
ValueError: embedding.dims must be positive, got 0
```
**Solution:** Check your YAML configuration for valid dimension value (typically 384 or 768).

#### Missing Environment Variables
```
pydantic.error_wrappers.ValidationError: NEO4J_PASSWORD
  field required
```
**Solution:** Set all required environment variables before starting.

### Debug Configuration Loading

Enable debug logging to see configuration details:

```bash
export LOG_LEVEL=DEBUG
python src/main.py
```

This will log:
- Configuration file path being loaded
- Embedding configuration details
- Feature flags status
- Validation results

## Migration Notes

### From Hardcoded Values

If migrating from hardcoded configuration:

1. Search for hardcoded values:
```bash
grep -r "384" src/  # Find hardcoded dimensions
grep -r "all-MiniLM" src/  # Find hardcoded models
```

2. Replace with configuration references:
```python
# Before
dims = 384

# After
from src.shared.config import get_config
config = get_config()
dims = config.embedding.dims
```

3. Test thoroughly with different configurations

## Configuration Schema Reference

See `config/development.yaml` for the complete configuration schema with all available options and their descriptions.
