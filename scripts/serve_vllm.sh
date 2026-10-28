#!/usr/bin/env bash
# vLLM OpenAI-compatible server used by bench/study.py. Each result file stores this script.
#   --dtype float16            the checkpoint is BF16; FP16 matches Ollama's F16 GGUF
#   --max-model-len 2048       same per-request context as Ollama's OLLAMA_CONTEXT_LENGTH
#   --max-num-seqs 16          at most 16 sequences in a batch, same as OLLAMA_NUM_PARALLEL
#   --generation-config vllm   ignore the checkpoint's sampling defaults (repetition_penalty 1.1)
#   --no-enable-prefix-caching  caching off on both servers (Ollama: LLAMA_ARG_CACHE_RAM=0); the
#                               client's per-request tag would defeat it anyway
set -euo pipefail
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
exec "$HOME/llmbench/venv/bin/vllm" serve Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --served-model-name qwen2.5-1.5b-instruct \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80 \
  --generation-config vllm \
  --no-enable-prefix-caching \
  --seed 0 \
  --disable-uvicorn-access-log \
  --host 127.0.0.1 --port 8000
