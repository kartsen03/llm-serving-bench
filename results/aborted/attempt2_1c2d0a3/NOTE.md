# Aborted attempt 2 (commit 1c2d0a3, 2026-10-27 22:16-22:21)

Kept for the record, not used in any table or plot. Stopped on purpose after vLLM repeat 1,
once attempt 1's Ollama log showed that llama-server's host-RAM prompt cache (default limit
8192 MiB) had grown to 8.1 GiB, more than this WSL instance's 7.6 GB of RAM. The workload makes
every request unique, so that cache can never be hit, and the swapping it causes is an artifact
of this WSL setup rather than Ollama's serving speed. The final study turns prompt/prefix
caching off on both servers. `vllm_c16_r1.json` was cut short by the stop (12 failed requests).
