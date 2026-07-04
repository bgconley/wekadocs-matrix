# ── mxbai-rerank service ──────────────────────────────────────────────
# Cross-encoder reranking via mixedbread-ai/mxbai-rerank-large-v2
#
# Build:  docker build -f docker/mxbai-reranker.Dockerfile -t mxbai-reranker .
# Run:    docker run --gpus all -p 9006:9006 -v ./hf-cache:/opt/hf-cache mxbai-reranker

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY services/mxbai-reranker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code
COPY services/mxbai-reranker/server.py .
COPY services/mxbai-reranker/run.sh .
RUN chmod +x run.sh

# HF cache support
ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache

# Service defaults
ENV MXBAI_MODEL_ID=mixedbread-ai/mxbai-rerank-large-v2
ENV MXBAI_DEVICE=cuda
ENV MXBAI_DTYPE=float16
ENV MXBAI_MAX_LENGTH=8192
ENV MXBAI_BATCH_SIZE=8
ENV MXBAI_MAX_CONCURRENCY=1
ENV MXBAI_PORT=9006

EXPOSE 9006

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9006/health || exit 1

CMD ["python3", "server.py"]
