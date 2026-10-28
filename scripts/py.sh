#!/usr/bin/env bash
# Runs a bench module inside WSL with the study venv:  bash scripts/py.sh -m bench.study --smoke
cd "$(dirname "$0")/.."
exec "$HOME/llmbench/venv/bin/python" "$@"
