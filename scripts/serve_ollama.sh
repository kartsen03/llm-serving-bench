#!/usr/bin/env bash
# Ollama server used by bench/study.py (user-space install from scripts/setup_ollama.sh).
#   OLLAMA_NUM_PARALLEL=16      16 request slots, so it can batch up to the highest concurrency
#   OLLAMA_CONTEXT_LENGTH=2048  per-request context, same as vLLM --max-model-len
#   port 11435                  keeps clear of the Windows Ollama app on 11434
#   LLAMA_ARG_CACHE_RAM=0       turns off llama-server's host-RAM prompt cache. Its default 8 GiB limit
#                               is more than this WSL instance's 7.6 GB of RAM, and the workload gives it
#                               nothing to reuse (see results/aborted/attempt2_1c2d0a3/NOTE.md)
# Everything else (flash attention, KV cache type f16) stays at Ollama's defaults.
set -euo pipefail
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$HOME/llmbench/ollama-models"
export OLLAMA_NUM_PARALLEL=16
export OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NOHISTORY=1
export LLAMA_ARG_CACHE_RAM=0
exec "$HOME/llmbench/ollama/bin/ollama" serve
