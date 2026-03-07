#!/usr/bin/env bash
set -euo pipefail

echo "=== mxbai-rerank Service ==="
echo "  Model:     ${MXBAI_MODEL_ID:-mixedbread-ai/mxbai-rerank-large-v2}"
echo "  Device:    ${MXBAI_DEVICE:-auto}"
echo "  Dtype:     ${MXBAI_DTYPE:-float16}"
echo "  MaxLen:    ${MXBAI_MAX_LENGTH:-8192}"
echo "  Port:      ${MXBAI_PORT:-9006}"

# Device detection
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA:    {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  Device:  Apple MPS')
else:
    print(f'  Device:  CPU only')
"

echo "Starting mxbai-rerank service..."
exec python3 server.py "$@"
