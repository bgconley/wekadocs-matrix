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

# Prefetch jina-embeddings-v3 tokenizer during build (Phase 7C hotfix)
# This eliminates runtime downloads and enables offline operation
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache && \
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', cache_dir='/opt/hf-cache')" && \
    echo "Tokenizer prefetched successfully"

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run the ingestion worker
CMD ["python", "-m", "src.ingestion.worker"]
