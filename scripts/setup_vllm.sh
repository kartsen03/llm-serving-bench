#!/usr/bin/env bash
# Creates the WSL venv that holds vLLM and the benchmark client's dependencies.
set -euo pipefail
VENV="$HOME/llmbench/venv"
mkdir -p "$HOME/llmbench"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip uv
"$VENV/bin/uv" pip install --python "$VENV/bin/python" vllm aiohttp matplotlib
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())"
