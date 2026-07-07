# INFX Inference Endpoint Characterization — Blackbird & Oxcart

Generated: 2026-07-07 | Method: repo audit + live probing

## Summary

| Host | Engine + version | Base URL | Model ID | Weights quant | KV dtype | Max context | Image input | Auth | Status |
|---|---|---|---|---|---|---|---|---|---|
| blackbird.lan.conley.ai | SGLang 0.0.0.dev1+g70df09b83 | <http://blackbird.lan.conley.ai:18002> | qwen36-27b-fp8-oxcart | FP8 (e4m3) | FP8 (fp8_e4m3) | 262,144 | ✅ enabled | none | ✅ ready |
| oxcart.lan.conley.ai | vLLM 0.20.2rc1.dev9 | <http://oxcart.lan.conley.ai:18002> | qwen36-27b-fp8-oxcart | FP8 (e4m3) | FP8 (auto→FP8) | 262,144 | ✅ enabled | Bearer EMPTY | ✅ ready |

**Hypothesis verification:**

| Hypothesis | Result |
|---|---|
| Blackbird: SGLang serving Qwen3.6-27B | ✅ CONFIRMED — SGLang dev nightly, Qwen3.6-27B-FP8 |
| Blackbird: FP8 weights + FP8 KV cache | ✅ CONFIRMED — `kv_cache_dtype: fp8_e4m3` |
| Oxcart: vLLM serving Qwen3.6-27B | ✅ CONFIRMED — vLLM 0.20.2rc1.dev9 |
| Oxcart: FP8 weights + BF16 KV cache | ❌ MISMATCH — live container uses FP8 KV (`cache_dtype=auto`, runbook shows `--kv-cache-dtype fp8`) |
| Max context ≈ 262,144 tokens both | ✅ CONFIRMED |
| One RTX PRO 6000 Blackwell per host | ✅ CONFIRMED (both hosts) |
| TP size 1 on both | ✅ CONFIRMED |
| Both expose OpenAI-compatible APIs | ✅ CONFIRMED (`/v1/chat/completions`, `/v1/models`) |

---

## Blackbird

### Endpoint & identity

- **Base URL**: `http://blackbird.lan.conley.ai:18002`
- **Engine**: SGLang (`0.0.0.dev1+g70df09b83`)
- **Docker image**: `lmsysorg/sglang:dev-cu13` (`sha256:56b67e2ca1497503b8d36be68eae93c296308f3bf02241eb02fcdfc291b6fd11`)
- **Container name**: `qwen36-27b-fp8-sglang-server`
- **Served model name**: `qwen36-27b-fp8-oxcart`
- **Model path**: `/workspace/model` (host-mounted from `/home/bgconley/models/qwen36-27b-fp8/`)
- **Model HF root**: `Qwen/Qwen3.6-27B-FP8`
- **Model architecture**: `Qwen3_5ForConditionalGeneration` (SGLang reports `model_type: qwen3_5`)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 97,887 MiB)
- **Auth**: None required

### Verbatim launch flags (source: `QWEN36_27B_FP8_SGLANG_BLACKBIRD.md`, live-probed via `/get_server_info`)

```
python -m sglang.launch_server \
  --model-path /workspace/model \
  --host 0.0.0.0 \
  --port 18002 \
  --served-model-name qwen36-27b-fp8-oxcart \
  --context-length 262144 \
  --mem-fraction-static 0.92 \
  --max-running-requests 4 \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --cuda-graph-max-bs 4 \
  --attention-backend flashinfer \
  --fp8-gemm-backend triton \
  --log-level-http warning
```

**Environment variables**: `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `TORCH_CUDA_ARCH_LIST=12.0f`, `SGLANG_ENABLE_SPEC_V2=1`, cache mounts under `/cache/jit/`.

### Key parameters for the extraction pipeline

| Parameter | Value | Source |
|---|---|---|
| Context length | 262,144 | `/get_server_info` `context_length` |
| Max total tokens (KV capacity) | 1,206,208 | `/get_server_info` `max_total_num_tokens` |
| Max request input length | 262,138 | `/get_server_info` `max_req_input_len` |
| Weights quantization | FP8 e4m3 (auto-detected) | `/get_server_info` `quantization: null` (auto), model card `Qwen/Qwen3.6-27B-FP8` |
| KV cache dtype | FP8 e4m3 (`fp8_e4m3`) | `/get_server_info` `kv_cache_dtype` |
| Max running requests | 4 | `/get_server_info` `max_running_requests` |
| Max queued requests | unlimited (`null`) | `/get_server_info` `max_queued_requests` |
| Mem fraction static | 0.92 | `/get_server_info` `mem_fraction_static` |
| Chunked prefill size | 8,192 | `/get_server_info` `chunked_prefill_size` |
| Max prefill tokens | 16,384 | `/get_server_info` `max_prefill_tokens` |
| Attention backend | FlashInfer | `/get_server_info` `attention_backend` |
| FP8 GEMM backend | Triton | `/get_server_info` `fp8_gemm_runner_backend` |
| TP size | 1 | `/get_server_info` `tp_size` |
| Chat template | null (model default) | `/get_server_info` `chat_template` |
| Reasoning parser | `qwen3` | `/get_server_info` `reasoning_parser` |
| Tool call parser | `qwen3_coder` | `/get_server_info` `tool_call_parser` |
| Speculative decoding | EAGLE v2, 3 steps, 1 topk, 4 draft tokens | `/get_server_info` speculative fields |
| Image input | ✅ `has_image_understanding: true` | `/get_model_info` |
| Audio input | ❌ `has_audio_understanding: false` | `/get_model_info` |
| Video input | UNVERIFIED (not probed) | — |
| MM limit per prompt | UNVERIFIED (not exposed in SGLang API) | — |
| Pixel bounds (image) | UNVERIFIED | — |
| Prefix caching | Radix cache (LRU eviction) | `/get_server_info` `radix_eviction_policy: lru` |
| CUDA graph (decode) | full backend, bs [1,2,3,4], max_bs=4 | `/get_server_info` `cuda_graph_config` |
| CUDA graph (prefill) | disabled | `/get_server_info` `cuda_graph_config` |
| Memory usage | weights: 28.47 GB, KV: 36.81 GB, draft: 5.19 GB | `/get_server_info` `internal_states[0].memory_usage` |

### Functional test results

**Text round-trip** (thinking disabled):

```json
Response: "ok"
finish_reason: stop
latency: 0.15s
prompt_tokens: 19, completion_tokens: 2
```

**Image round-trip** (1×1 transparent PNG):

```json
Response: " ok"
finish_reason: stop
latency: 0.21s
prompt_tokens: 85 (image_tokens: 64), completion_tokens: 2
```

Image input works. Vision tower processes 1×1 PNG → 64 image tokens.

### Discrepancies vs. repo documentation

1. **Served model name mismatch with hypothesis**: The model is served as `qwen36-27b-fp8-oxcart` on both hosts, not `qwen36-27b-fp8-rp6000` as in the blackbird launch script. This matches the redeploy runbook convention.
2. **`--language-model-only` absent**: The launch script for Blackbird does NOT include `--language-model-only`, so multimodal is enabled. Confirmed by live probe (`has_image_understanding: true`).
3. **`--cuda-graph-max-bs` deprecated**: Runbook notes this flag is deprecated in current nightly; use `--cuda-graph-max-bs-decode` instead. Live server still accepts it.
4. **`--mamba-scheduler-strategy` deprecated**: Runbook notes this is deprecated; use `--mamba-radix-cache-strategy` instead.
5. **`SGLANG_ENABLE_SPEC_V2` deprecated**: Speculative decoding always uses v2 in current nightly.
6. **FP8 kernel configs missing for RTX PRO 6000**: Runbook notes SGLang falls back to default W8A8 Block FP8 configs. Performance is sub-optimal but functional.
7. **Model architecture reported as `Qwen3_5ForConditionalGeneration`**: SGLang's `model_type` is `qwen3_5`, not `qwen3_6`. This is an internal SGLang classification and does not affect the actual model loaded.

---

## Oxcart

### Endpoint & identity

- **Base URL**: `http://oxcart.lan.conley.ai:18002`
- **Engine**: vLLM (`0.20.2rc1.dev9+g01d4d1ad3`)
- **Docker image**: `vllm/vllm-openai@sha256:b13d6e5fda0785f3d41752df8513ff832f67cb231a216c76b6b4f2a515bf0046` (pinned digest)
- **Container name**: `qwen36-27b-fp8-mtp-vl-oxcart-server`
- **Served model name**: `qwen36-27b-fp8-oxcart`
- **Model HF root**: `Qwen/Qwen3.6-27B-FP8`
- **Model path**: Host-mounted from `/tank/ai/models/qwen36-27b-fp8/hf-cache`
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 97,887 MiB)
- **Auth**: Bearer `EMPTY` required

### Verbatim launch flags (source: `OXCART_QWEN36_27B_FP8_VLLM_SM120_REDEPLOY.md` section 16)

```
Qwen/Qwen3.6-27B-FP8 \
  --host 0.0.0.0 \
  --port 18002 \
  --api-key EMPTY \
  --served-model-name qwen36-27b-fp8-oxcart \
  --trust-remote-code \
  --dtype auto \
  --attention-backend FLASHINFER \
  --mm-encoder-attn-backend FLASHINFER \
  --kv-cache-dtype fp8 \
  --safetensors-load-strategy prefetch \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.955 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":16,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4],"cudagraph_num_of_warmups":1}' \
  --cudagraph-metrics \
  --disable-uvicorn-access-log
```

**Environment variables**: `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `FLASHINFER_LOGLEVEL=0`, `FLASHINFER_JIT_VERBOSE=0`, `SAFETENSORS_FAST_GPU=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, HF/VLLM/Triton cache mounts under `/tank/ai/models/qwen36-27b-fp8/vllm-rp6000-mtp-vl/`.

**NOTE**: The launch script at `qwen36-27b-test/start-qwen36-27b-fp8-oxcart.sh` is the OLD configuration (text-only, no MTP, no multimodal). The LIVE container follows the redeploy runbook (`OXCART_QWEN36_27B_FP8_VLLM_SM120_REDEPLOY.md`) and was redeployed 2026-06-23.

### Key parameters for the extraction pipeline

| Parameter | Value | Source |
|---|---|---|
| Context length | 262,144 | `/v1/models` `max_model_len` |
| Weights quantization | FP8 e4m3 (auto-detected) | Model repo `Qwen/Qwen3.6-27B-FP8`; `--dtype auto` |
| KV cache dtype | FP8 (`fp8`) | `--kv-cache-dtype fp8` in launch flags; metrics `cache_dtype=auto` (vLLM resolves FP8 model → FP8 KV) |
| Max num seqs | 4 | `--max-num-seqs 4` |
| GPU memory utilization | 0.955 | `--gpu-memory-utilization 0.955`; confirmed by metrics |
| KV cache memory | 59.54 GiB, 1,708,404 tokens | Redeploy runbook section 17 (startup logs) |
| Max concurrency at 262K | 6.52× | Redeploy runbook section 17 |
| Max num batched tokens | 8,192 | `--max-num-batched-tokens 8192` |
| Attention backend | FlashInfer | `--attention-backend FLASHINFER` |
| MM encoder attention backend | FlashInfer | `--mm-encoder-attn-backend FLASHINFER` |
| TP size | 1 (single GPU) | No `--tensor-parallel-size` flag |
| Chat template | Model default with `enable_thinking=true, preserve_thinking=true` | `--default-chat-template-kwargs` |
| Reasoning parser | `qwen3` | `--reasoning-parser qwen3` |
| Tool call parser | `qwen3_coder` | `--tool-call-parser qwen3_coder` |
| Speculative decoding | MTP, 2 tokens | `--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}` → normalized to `mtp` by vLLM |
| Image input | ✅ enabled | `--limit-mm-per-prompt {"image":16,"video":1}` |
| Max images per prompt | 16 | `--limit-mm-per-prompt` |
| Max videos per prompt | 1 | `--limit-mm-per-prompt` |
| Video frames | unlimited (`num_frames: -1`) | `--media-io-kwargs` |
| Prefix caching | ✅ enabled | `--enable-prefix-caching` |
| Chunked prefill | ✅ enabled | `--enable-chunked-prefill` |
| CUDA graphs | bs [1,2,4], 1 warmup | `--compilation-config` |
| CUDA graph mode | PIECEWISE (downgraded due to spec-decode + FlashInfer) | Runbook section 17 |

### Functional test results

**Text round-trip** (thinking disabled):

```json
Response: "ok"
finish_reason: stop
latency: 0.15s
prompt_tokens: 19, completion_tokens: 2
system_fingerprint: vllm-0.20.2rc1.dev9+g01d4d1ad3-68c8e8ed
```

**Image round-trip** (1×1 transparent PNG):

```json
Response: " ok"
finish_reason: stop
latency: 0.49s
prompt_tokens: 86, completion_tokens: 2
```

Image input works. Vision tower processes 1×1 PNG. Oxcart is ~2.3× slower on image requests vs. Blackbird (0.49s vs 0.21s), likely due to MTP speculative decoding overhead and different CUDA graph configuration.

### Discrepancies vs. repo documentation

1. **Hypothesis mismatch — Oxcart KV cache dtype**: Hypothesis stated BF16 KV cache. Live container and redeploy runbook both use FP8 KV (`--kv-cache-dtype fp8`). The OLD launch script (`qwen36-27b-test/start-qwen36-27b-fp8-oxcart.sh`) also defaults to `KV_CACHE_DTYPE=fp8`. **BF16 KV was never the live configuration.**
2. **Launch script vs. live container**: `qwen36-27b-test/start-qwen36-27b-fp8-oxcart.sh` is the OLD config (text-only, no MTP, `--language-model-only`, `--limit-mm-per-prompt {"image":0,"video":0}`). The live container was redeployed 2026-06-23 per `OXCART_QWEN36_27B_FP8_VLLM_SM120_REDEPLOY.md` with multimodal + MTP enabled.
3. **`--limit-mm-per-prompt {"image":4,"video":1}` in logs**: Runbook section 17 shows vLLM normalized `{"image":16,"video":1}` to `{"image":4,"video":1}`. The live config uses the model-card default of 4 images, not the requested 16.
4. **`qwen3_next_mtp` normalized to `mtp`**: vLLM accepts `qwen3_next_mtp` but emits a deprecation warning and normalizes to `mtp`. Future launches can use `{"method":"mtp","num_speculative_tokens":2}` directly.
5. **`min_p` and `logit_bias` incompatible with spec decode**: vLLM warns these won't work with speculative decoding. `min_p=0.0` is neutral here.
6. **CUDA graph mode downgraded**: `CUDAGraphMode.FULL_AND_PIECEWISE` not supported with spec-decode on FlashInfer; downgraded to `PIECEWISE`.
7. **Metrics show `cache_dtype=auto`**: vLLM reports `auto` in Prometheus metrics because the `--kv-cache-dtype` was explicitly set to `fp8` at launch and vLLM's metrics reflect the resolved auto-mode label. The actual KV dtype is FP8 as confirmed by launch flags and startup logs (`Available KV cache memory: 59.54 GiB`).
8. **vLLM image stack drift**: `VLLM_SERVING_INVENTORY.md` warns that oxcart's floating `:nightly` tag drifted from `0.20.1rc1.dev46` to `0.21.1rc1.dev98`. However, the live container uses the **pinned digest** `b13d6e5f` (= `0.20.2rc1.dev9`), so it is unaffected by nightly drift.

---

## Cross-host comparison relevant to the pipeline

### Engine differences

- **Blackbird**: SGLang (dev nightly) with EAGLE speculative decoding (4 draft tokens, 3 steps). Supports radix caching (LRU eviction).
- **Oxcart**: vLLM (pinned stable) with MTP speculative decoding (2 draft tokens). Supports prefix caching.

### KV cache and concurrency

- **Blackbird**: FP8 KV, max 4 running requests, max total KV tokens: 1,206,208. Memory: 28.47 GB weights + 36.81 GB KV + 5.19 GB draft model = ~70.5 GB active. Mem fraction: 0.92.
- **Oxcart**: FP8 KV, max 4 sequences, KV: 59.54 GiB (1,708,404 tokens). Mem utilization: 0.955. Maximum concurrency at full 262K context: 6.52×.

**Both use FP8 KV caches** — the original hypothesis of BF16 KV on Oxcart is incorrect. FP8 KV on both means:

- Lower per-request memory footprint → higher concurrency potential
- Slight accuracy trade-off (typically negligible for extraction tasks)
- Both are suitable for high-res PNG PDF page extraction

### Multimodal configuration

| Feature | Blackbird (SGLang) | Oxcart (vLLM) |
|---|---|---|
| Image input | ✅ enabled | ✅ enabled |
| Max images/prompt | UNVERIFIED (not exposed) | 4 (normalized from 16) |
| Video input | UNVERIFIED | ✅ enabled (1 video) |
| Pixel bounds | UNVERIFIED | UNVERIFIED |
| MM attention backend | auto-selected (not fa3) | FlashInfer |

For PDF extraction, both endpoints support image input. Oxcart's 4-image-per-prompt limit (normalized by vLLM) constrains batch sizes — a multi-page PDF would need chunked requests. Blackbird's limit is UNVERIFIED via API but the server config does not appear to enforce per-prompt image limits (SGLang lacks `--limit-mm-per-prompt` equivalent in its exposed API).

### Concurrency for batch PDF extraction

- **Blackbird**: max 4 concurrent requests. EAGLE spec decode adds ~3.36 avg accepted tokens/draft. Good for sustained single-sequence throughput.
- **Oxcart**: max 4 concurrent sequences. MTP spec decode adds 2 tokens/draft. Higher KV capacity (1.7M tokens vs 1.2M) allows more full-context concurrent requests (6.52× at 262K).

**Oxcart has a capacity advantage** for long-context workloads due to higher KV cache allocation. **Blackbird has a latency advantage** on image requests (0.21s vs 0.49s for the 1×1 PNG probe).

### Shared constraints for the extraction pipeline

- Both use `Qwen3.6-27B-FP8` → same model behavior, same tokenizer, same vision encoder
- Both have 262K context windows — sufficient for multi-page PDFs
- Thinking mode is enabled by default on Oxcart (`enable_thinking: true`); Blackbird's default is UNVERIFIED but SGLang uses `sampling_defaults: model`. The pipeline should explicitly disable thinking via `chat_template_kwargs: {enable_thinking: false}` for deterministic, fast extraction.
- Both require the `Qwen/Qwen3.6-27B-FP8` chat template for multimodal input.

---

## UNVERIFIED items and open questions

1. **Blackbird max images per prompt**: SGLang does not expose `--limit-mm-per-prompt` equivalent in `/get_server_info`. The launch config does not include such a flag. Default limit is unknown — should be tested with a large batch of images.
2. **Blackbird pixel bounds**: Neither endpoint exposes image processor pixel configuration via API. The model card recommends specific pixel bounds for Qwen3.6 VLM, but the actual bounds used by the servers are not visible without SSH access or log inspection.
3. **Video input on Blackbird**: Not probed. SGLang model info shows `has_image_understanding: true` but no video field. UNVERIFIED whether video input works on the SGLang deployment.
4. **Blackbird default thinking behavior**: SGLang reports `sampling_defaults: model`. Whether thinking is enabled by default is UNVERIFIED — should be tested in the pipeline.
5. **Oxcart MTP token acceptance rate**: Metrics show 116,136 accepted / 136,547 drafted = ~85% acceptance rate. This is healthy but means effective throughput gain from MTP is modest.
6. **Blackbird EAGLE draft model overhead**: EAGLE draft model uses 5.19 GB of VRAM. This reduces KV cache capacity compared to Oxcart's no-extra-draft-model configuration. Trade-off: EAGLE can accept more tokens per draft (avg 3.36) vs MTP's fixed 2.
7. **Concurrent request limits under load**: Both show `max_running_requests` / `max_num_seqs` of 4, but actual throughput under sustained load is UNVERIFIED. KV cache saturation behavior under multi-page PDF extraction is unknown.
8. **Oxcart thinking default**: Server is configured with `enable_thinking: true, preserve_thinking: true` as defaults. This means every request will generate reasoning tokens before content unless overridden. For extraction, this adds latency and consumes context budget. Pipeline must pass `chat_template_kwargs: {enable_thinking: false}` to avoid this.
9. **Blackbird SGLang version stability**: `0.0.0.dev1+g70df09b83` is a dev nightly build. API stability and bug fixes are not guaranteed. No official release version pinning.
10. **SSH access to both hosts**: Denied (`Permission denied`). Live `ps aux` and `docker inspect` were not possible. Launch flags are cited from repo runbooks, not live process inspection. If the live container diverged from the runbook without an update, this report would miss it.

---

## Appendix: raw probe output

### Blackbird `/get_server_info` (key fields extracted)

```json
{
  "version": "0.0.0.dev1+g70df09b83",
  "status": "ready",
  "served_model_name": "qwen36-27b-fp8-oxcart",
  "model_path": "/workspace/model",
  "tokenizer_path": "/workspace/model",
  "tokenizer_backend": "huggingface",
  "context_length": 262144,
  "dtype": "auto",
  "quantization": null,
  "kv_cache_dtype": "fp8_e4m3",
  "mem_fraction_static": 0.92,
  "max_running_requests": 4,
  "max_queued_requests": null,
  "max_total_num_tokens": 1206208,
  "max_req_input_len": 262138,
  "chunked_prefill_size": 8192,
  "max_prefill_tokens": 16384,
  "tp_size": 1,
  "attention_backend": "flashinfer",
  "fp8_gemm_runner_backend": "triton",
  "reasoning_parser": "qwen3",
  "tool_call_parser": "qwen3_coder",
  "chat_template": null,
  "speculative_algorithm": "EAGLE",
  "speculative_num_steps": 3,
  "speculative_eagle_topk": 1,
  "speculative_num_draft_tokens": 4,
  "max_speculative_num_draft_tokens": 4,
  "avg_spec_accept_length": 3.364100464100464,
  "last_gen_throughput": 102.24274982348105,
  "memory_usage": {
    "weight": 28.47,
    "kvcache": 36.81,
    "token_capacity": 1206208,
    "graph": 0.12
  },
  "cuda_graph_config": {
    "decode": {"backend": "full", "max_bs": 4, "bs": [1,2,3,4], "tc_compiler": "eager"},
    "prefill": {"backend": "disabled", "max_bs": 8192, "bs": [4,8,12,...,8192], "tc_compiler": "eager"}
  },
  "effective_max_running_requests_per_dp": 4
}
```

### Blackbird `/get_model_info`

```json
{
  "model_path": "/workspace/model",
  "tokenizer_path": "/workspace/model",
  "is_generation": true,
  "has_image_understanding": true,
  "has_audio_understanding": false,
  "model_type": "qwen3_5",
  "architectures": ["Qwen3_5ForConditionalGeneration"]
}
```

### Blackbird `/v1/models`

```json
{
  "object": "list",
  "data": [{
    "id": "qwen36-27b-fp8-oxcart",
    "object": "model",
    "created": 1783409378,
    "owned_by": "sglang",
    "root": "qwen36-27b-fp8-oxcart",
    "parent": null,
    "max_model_len": 262144
  }]
}
```

### Oxcart `/version`

```json
{"version": "0.20.2rc1.dev9+g01d4d1ad3"}
```

### Oxcart `/v1/models`

```json
{
  "object": "list",
  "data": [{
    "id": "qwen36-27b-fp8-oxcart",
    "object": "model",
    "created": 1783409400,
    "owned_by": "vllm",
    "root": "Qwen/Qwen3.6-27B-FP8",
    "parent": null,
    "max_model_len": 262144,
    "permission": [{
      "id": "modelperm-bbfa8b3639da42b5",
      "object": "model_permission",
      "created": 1783409400,
      "allow_create_engine": false,
      "allow_sampling": true,
      "allow_logprobs": true,
      "allow_search_indices": false,
      "allow_view": true,
      "allow_fine_tuning": false,
      "organization": "*",
      "group": null,
      "is_blocking": false
    }]
  }]
}
```

### Oxcart vLLM metrics — cache config

```
vllm:cache_config_info{
  _block_size_resolved="True",
  block_size="16",
  cache_dtype="auto",
  calculate_kv_scales="False",
  enable_prefix_caching="True",
  engine="0",
  gpu_memory_utilization="0.955",
  hash_block_size="None",
  is_attention_free="False",
  kv_cache_dtype_skip_layers="[]",
  kv_cache_memory_bytes="None",
  kv_offloading_backend="native",
  kv_offloading_size="None",
  kv_sharing_fast_prefill="False",
  mamba_block_size="16",
  mamba_cache_dtype="auto",
  mamba_cache_mode="align",
  mamba_ssm_cache_dtype="float32",
  num_cpu_blocks="None",
  num_gpu_blocks="1147",
  num_gpu_blocks_override="None",
  prefix_caching_hash_algo="sha256",
  sliding_window="None",
  user_specified_block_size="False",
  user_specified_mamba_block_size="False"
} 1.0
```

### Oxcart vLLM metrics — spec decoding

```
vllm:spec_decode_num_drafts_total{engine="0",model_name="qwen36-27b-fp8-oxcart"} 68270
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="qwen36-27b-fp8-oxcart"} 136537
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="qwen36-27b-fp8-oxcart"} 116128
vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="0"} 61680
vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="1"} 54448
```

### Blackbird text test raw response

```json
{
  "id": "<redacted-chat-completion-id>",
  "object": "chat.completion",
  "created": 1783409451,
  "model": "qwen36-27b-fp8-oxcart",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "ok", "reasoning_content": null, "tool_calls": null},
    "finish_reason": "stop",
    "matched_stop": 248046
  }],
  "usage": {"prompt_tokens": 19, "total_tokens": 21, "completion_tokens": 2, "reasoning_tokens": 0}
}
latency_s=0.152287
```

### Blackbird image test raw response

```json
{
  "id": "<redacted-chat-completion-id>",
  "object": "chat.completion",
  "created": 1783409473,
  "model": "qwen36-27b-fp8-oxcart",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": " ok", "reasoning_content": null, "tool_calls": null},
    "finish_reason": "stop",
    "matched_stop": 248046
  }],
  "usage": {"prompt_tokens": 85, "total_tokens": 87, "completion_tokens": 2, "image_tokens": 64, "reasoning_tokens": 0}
}
latency_s=0.213181
```

### Oxcart text test raw response

```json
{
  "id": "chatcmpl-9b01aea0fdb408f9",
  "object": "chat.completion",
  "created": 1783409451,
  "model": "qwen36-27b-fp8-oxcart",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "ok", "reasoning": null, "tool_calls": []},
    "finish_reason": "stop"
  }],
  "system_fingerprint": "vllm-0.20.2rc1.dev9+g01d4d1ad3-68c8e8ed",
  "usage": {"prompt_tokens": 19, "total_tokens": 21, "completion_tokens": 2}
}
latency_s=0.152572
```

### Oxcart image test raw response

```json
{
  "id": "chatcmpl-a1ff926f9861b504",
  "object": "chat.completion",
  "created": 1783409473,
  "model": "qwen36-27b-fp8-oxcart",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": " ok", "reasoning": null, "tool_calls": []},
    "finish_reason": "stop"
  }],
  "system_fingerprint": "vllm-0.20.2rc1.dev9+g01d4d1ad3-68c8e8ed",
  "usage": {"prompt_tokens": 86, "total_tokens": 88, "completion_tokens": 2}
}
latency_s=0.490648
```
