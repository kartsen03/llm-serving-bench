#!/usr/bin/env bash
# Installs a user-space Ollama (no sudo) inside WSL so both servers run in the same environment.
# The release ships only as .tar.zst and this distro has no zstd binary, so a tiny venv
# with the `zstandard` module does the decompression.
set -euo pipefail
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.40.12}"
DEST="$HOME/llmbench/ollama"
TOOLS="$HOME/llmbench/tools-venv"
mkdir -p "$DEST" "$HOME/llmbench/dl"
[ -x "$TOOLS/bin/python" ] || python3 -m venv "$TOOLS"
"$TOOLS/bin/pip" install --quiet zstandard
ARCHIVE="$HOME/llmbench/dl/ollama-linux-amd64-$OLLAMA_VERSION.tar.zst"
if [ ! -s "$ARCHIVE" ]; then
  curl -fL --retry 3 -o "$ARCHIVE.part" \
    "https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
  mv "$ARCHIVE.part" "$ARCHIVE"
fi
"$TOOLS/bin/python" -c "
import sys, zstandard
with open(sys.argv[1], 'rb') as f:
    zstandard.ZstdDecompressor().copy_stream(f, sys.stdout.buffer)
" "$ARCHIVE" | tar -xf - -C "$DEST"
ls "$DEST" "$DEST/lib/ollama" | head -20
"$DEST/bin/ollama" --version || true
