#!/usr/bin/env bash
# Starts bench.study detached from the calling wsl.exe, so that the client going away
# (a closed terminal, a tool timeout) does not kill the study. Progress goes to
# results/logs/study_console.log, ending with a "study exit <code>" line; the study
# resumes from saved runs if started again.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
setsid nohup bash -c '"$HOME/llmbench/venv/bin/python" -m bench.study "$@"; echo "study exit $? at $(date -Is)"' \
  bench "$@" >> results/logs/study_console.log 2>&1 < /dev/null &
echo $! > results/logs/study.pid
echo "study started detached, pid $(cat results/logs/study.pid)"
