#!/bin/bash
# GLiNER NER Service - Startup Script
#
# Usage:
#   ./run.sh              # Run with defaults (MPS auto-detect, port 9002)
#   ./run.sh --device cpu # Force CPU mode
#   ./run.sh --port 9003  # Custom port
#
# For development with auto-reload:
#   RELOAD=1 ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}GLiNER NER Service${NC}"
echo "=================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if running in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment${NC}"
    echo "Consider: python3 -m venv venv && source venv/bin/activate"
fi

# Check for MPS availability
echo ""
echo "Checking device availability..."
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
mps_avail = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
cuda_avail = torch.cuda.is_available()
print(f'  MPS available: {mps_avail}')
print(f'  CUDA available: {cuda_avail}')
if mps_avail:
    print('  -> Will use MPS (Apple Silicon)')
elif cuda_avail:
    print('  -> Will use CUDA')
else:
    print('  -> Will use CPU')
"

echo ""

# Run with or without reload
if [[ "$RELOAD" == "1" ]]; then
    echo "Starting with auto-reload (development mode)..."
    exec uvicorn server:app --host 0.0.0.0 --port 9002 --reload "$@"
else
    echo "Starting server..."
    exec python3 server.py "$@"
fi
