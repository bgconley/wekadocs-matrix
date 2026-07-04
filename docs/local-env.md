# Local vs Docker Environment Variables

This repo uses separate env files to avoid mixing host-only and container-only values.

## Docker / containers
- Docker Compose loads `.env.docker` via `env_file`.
- Use container-safe endpoints (`host.docker.internal`) and `/opt/hf-cache` paths.
- `.env` remains container-safe for Compose interpolation.

## Local test runs
1) Copy `.env.docker` to `.env.local`.
2) Update host endpoints, e.g.:
   - `NEO4J_URI=bolt://localhost:7687`
   - `QDRANT_HOST=localhost`
   - `REDIS_HOST=localhost`
   - `BGE_M3_API_URL=http://127.0.0.1:9000`
   - `GLINER_SERVICE_URL=http://127.0.0.1:9002`
   - `RERANKER_BASE_URL=http://127.0.0.1:9005`
3) Use the helper to load `.env.local`:
   - `scripts/run_local.sh <command>`

## GLiNER override
- Local runs can set `GLINER_SERVICE_URL` to avoid changing YAML config.
- Containers should use `GLINER_SERVICE_URL=http://host.docker.internal:9002`.
