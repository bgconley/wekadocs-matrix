# Implements Phase 1, Task 1.1 (Docker environment setup)
# Ingestion Worker container

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Prefetch tokenizers during build (Phase 7C hotfix)
# This eliminates runtime downloads and enables offline operation
ARG HF_HUB_OFFLINE=1
ARG TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache/hub \
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} \
    HF_HUB_DISABLE_TELEMETRY=1
RUN mkdir -p /opt/hf-cache && \
    python - <<'PY' && \
    echo "Tokenizers prefetched successfully"
from transformers import AutoTokenizer
import os

offline = os.getenv("HF_HUB_OFFLINE", "1").lower() in {"1", "true", "yes"}
models = [
    "voyageai/voyage-context-3",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Reranker-0.6B",
]
for model_id in models:
    try:
        AutoTokenizer.from_pretrained(
            model_id,
            cache_dir="/opt/hf-cache",
            local_files_only=offline,
        )
    except Exception as exc:
        print(f"Tokenizer prefetch skipped for {model_id}: {exc}")
PY

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run the ingestion worker
CMD ["python", "-m", "src.ingestion.worker"]
