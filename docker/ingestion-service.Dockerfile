# Phase 6, Task 6.1: Auto-Ingestion Service Dockerfile
# See: /docs/implementation-plan-phase-6.md → Task 6.1

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Prefetch tokenizers during build (Phase 7C hotfix)
# This eliminates runtime downloads and enables offline operation
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache && \
    python - <<'PY' && \
    echo "Tokenizers prefetched successfully"
from transformers import AutoTokenizer

models = [
    "jinaai/jina-embeddings-v3",
    "voyageai/voyage-context-3",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Reranker-0.6B",
]
for model_id in models:
    AutoTokenizer.from_pretrained(model_id, cache_dir="/opt/hf-cache")
PY

# Copy application code
COPY src/ src/
COPY config/ config/

# Create watch and reports directories
RUN mkdir -p /app/ingest/watch /app/reports/ingest

# Expose metrics port
EXPOSE 9108

# Health check on /health endpoint
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:9108/health || exit 1

# Run service
CMD ["python", "-m", "src.ingestion.auto.service"]
