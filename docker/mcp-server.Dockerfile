# Implements Phase 1, Task 1.1 (Docker environment setup)
# MCP Server container

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
ARG HF_HUB_OFFLINE=0
ARG TRANSFORMERS_OFFLINE=0
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache/hub \
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} \
    HF_HUB_DISABLE_TELEMETRY=1
RUN mkdir -p /opt/hf-cache && \
    python - <<'PY' && \
    echo "Tokenizers prefetched successfully"
from transformers import AutoTokenizer
import sys

# Always download during build (not local_files_only).
# Models that are gated or missing fall back to bert-base-uncased.
required = [
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "BAAI/bge-m3",
]
optional = [
    ("naver/splade-v3", "bert-base-uncased"),       # Gated repo
    ("colbert-ai/colbertv2.0", "bert-base-uncased"), # No HF tokenizer
    ("voyageai/voyage-context-3", None),
]

for model_id in required:
    try:
        t = AutoTokenizer.from_pretrained(model_id, cache_dir="/opt/hf-cache")
        print(f"  OK: {model_id} (vocab={t.vocab_size})")
    except Exception as exc:
        print(f"  REQUIRED FAIL: {model_id}: {exc}")
        sys.exit(1)

for model_id, fallback in optional:
    try:
        t = AutoTokenizer.from_pretrained(model_id, cache_dir="/opt/hf-cache")
        print(f"  OK: {model_id} (vocab={t.vocab_size})")
    except Exception:
        if fallback:
            try:
                t = AutoTokenizer.from_pretrained(fallback, cache_dir="/opt/hf-cache")
                print(f"  FALLBACK: {model_id} -> {fallback} (vocab={t.vocab_size})")
            except Exception as exc2:
                print(f"  SKIP: {model_id} and fallback {fallback}: {exc2}")
        else:
            print(f"  SKIP: {model_id}")
PY

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the MCP server
CMD ["python", "-m", "uvicorn", "src.mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
