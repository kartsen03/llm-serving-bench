#!/usr/bin/env bash
# Pulls the FP16 Qwen2.5-1.5B-Instruct GGUF and creates the benchmark alias from ollama/Modelfile.
set -euo pipefail
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="$HOME/llmbench/ollama-models"
O="$HOME/llmbench/ollama/bin/ollama"
"$O" serve > "$HOME/llmbench/ollama-setup.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for i in $(seq 60); do curl -sf http://127.0.0.1:11435/api/version >/dev/null && break; sleep 1; done
"$O" pull qwen2.5:1.5b-instruct-fp16 2>&1 | tail -2
echo "== base model parameters"; "$O" show --parameters qwen2.5:1.5b-instruct-fp16
echo "== base model info"; "$O" show qwen2.5:1.5b-instruct-fp16
"$O" create qwen2.5-1.5b-bench -f ollama/Modelfile 2>&1 | tail -1
echo "== bench model parameters"; "$O" show --parameters qwen2.5-1.5b-bench
"$O" list
