# mxbai-rerank Service

Cross-encoder reranking service using mixedbread-ai models.

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Run (auto-detects GPU)
./run.sh

# Or directly
MXBAI_MODEL_ID=mixedbread-ai/mxbai-rerank-large-v2 python3 server.py
```

## API

### GET /health

Returns service status, model info, device, and warmup state.

### POST /v1/rerank

```json
{
  "query": "search query",
  "documents": ["doc1", "doc2", "doc3"],
  "instruction": "optional reranking instruction",
  "top_k": 3
}
```

Response:
```json
{
  "results": [{"index": 0, "score": 0.97}, {"index": 2, "score": 0.85}],
  "model": "mixedbread-ai/mxbai-rerank-large-v2",
  "latency_ms": 123.4
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| MXBAI_MODEL_ID | mixedbread-ai/mxbai-rerank-large-v2 | HuggingFace model ID |
| MXBAI_DEVICE | auto | cuda, mps, cpu, or auto |
| MXBAI_DTYPE | float16 | float16 or float32 |
| MXBAI_MAX_LENGTH | 8192 | Max input token length |
| MXBAI_MAX_CONCURRENCY | 1 | Max concurrent requests |
| MXBAI_PORT | 9006 | Listen port |

## Models

- **large-v2** (1.5B, ~3 GB FP16): Best quality, BEIR 57.49
- **base-v2** (0.5B, ~1 GB FP16): Faster, BEIR 55.57

Switch by changing MXBAI_MODEL_ID only.

## Serial GPU Deployment

Qwen3-Reranker-4B and mxbai-rerank-large-v2 cannot coexist on a 24 GB GPU.

1. Download tokenizer to host cache:
   ```bash
   python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mixedbread-ai/mxbai-rerank-large-v2', cache_dir='./hf-cache')"
   ```
2. Stop Qwen reranker to free VRAM
3. Start mxbai service: `docker compose up -d mxbai-reranker`
4. Verify: `curl http://localhost:9006/health`
5. Smoke test: `curl -X POST http://localhost:9006/v1/rerank -H 'Content-Type: application/json' -d '{"query":"test","documents":["doc1","doc2"]}'`
6. Repoint MCP: set `RERANKER_BASE_URL=http://host:9006`
7. Rollback: stop mxbai, restart Qwen, repoint MCP back
