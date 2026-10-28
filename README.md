# llm-serving-bench: vLLM vs Ollama on a 6 GB laptop GPU

A small, controlled benchmark of two LLM servers, [vLLM](https://github.com/vllm-project/vllm)
and [Ollama](https://github.com/ollama/ollama), serving the same model at the same precision
through their OpenAI-compatible APIs on one RTX 3060 Laptop GPU under WSL2.

> Status: methodology and tooling are in place; results sections are filled in from
> `results/runs/*.json` by `bench/report.py` once the study has run.

## What is measured

| Metric | Definition (client-side, `time.perf_counter`) |
|---|---|
| Time to first token (TTFT) | request sent -> first streamed chunk with non-empty content |
| Inter-token latency (ITL) | per request: (last content chunk - first content chunk) / (output tokens - 1) |
| Token gap p99 | 99th percentile of the gaps between consecutive content chunks, over all requests in a run |
| Output tokens / s | all completion tokens in a run / run wall time |
| Requests / s | completed requests / run wall time |
| GPU memory | device-wide `memory.used`, sampled every 100 ms by `nvidia-smi` during each run |

Token counts come from each server's own `usage` field (`stream_options.include_usage`).

## How

**Same model, same precision.** Qwen2.5-1.5B-Instruct in FP16 on both servers, FP16 KV cache on both.

- vLLM loads the Hugging Face checkpoint (`Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`). The checkpoint is stored in BF16; vLLM casts it
  to FP16 (`--dtype float16`).
- Ollama loads `qwen2.5:1.5b-instruct-fp16` (GGUF, quantization F16), through an alias
  (`ollama/Modelfile`) that only changes sampling defaults.
- Not byte-identical: the GGUF was converted from the same release by Ollama, and GGUF keeps
  1-D tensors such as norm weights in F32. Both runs do FP16 matrix multiplies on the same weights.

**Same work.** 64 seeded synthetic prompts (`data/prompts.jsonl`, built by `bench/prompts.py`),
each a list of 3-30 "field notes" plus an instruction to write a long report. Greedy decoding,
every penalty off, `max_tokens=256`. Ollama's OpenAI API has no `ignore_eos`, so the prompts ask
for 800+ words, and the results report how often each server actually reached 256 tokens.

**Same limits.** Both servers allow at most 16 sequences at once (vLLM `--max-num-seqs 16`,
Ollama `OLLAMA_NUM_PARALLEL=16`) and a 2048-token context per request.

**No cache tricks.** Every request in the study starts with a unique tag (`Session r2c08i017. ...`),
identical across servers, so neither server can reuse a cached prompt prefix from an earlier request.

**Load pattern.** Closed loop: at concurrency *c* the client keeps exactly *c* requests in flight
(levels 1, 2, 4, 8, 16), sending max(16, 8c) requests per run. Each server launch starts with a
30 s warmup at concurrency 16, and each level gets one unmeasured wave of *c* requests first.

**Repeats and ordering.** Only one server fits in 6 GB, so the study alternates whole server
launches: vLLM r1 | Ollama r1 r2 | vLLM r2 r3 | Ollama r3. Each server gets an early, a middle
and a late repeat, so the GPU heating up over the evening does not favour either one. Within a
repeat the concurrency levels run in a seeded shuffled order, the same for both servers. Tables
report the median of the 3 repeats, with min-max ranges.

**Provenance.** Every run file embeds the git commit, GPU, driver, CUDA, package and server
versions, power plan, the launch script and the prompt-file hash. `bench/study.py` refuses to
record with uncommitted changes to tracked files, or when Windows is not on the Turbo power plan.

## Reproduce

Inside WSL2 (Ubuntu 24.04):

```bash
bash scripts/setup_vllm.sh          # venv at ~/llmbench/venv with vLLM
bash scripts/setup_ollama.sh        # user-space Ollama at ~/llmbench/ollama
bash scripts/setup_ollama_model.sh  # pulls the FP16 GGUF, creates the benchmark alias
bash scripts/py.sh -m bench.prompts # regenerates data/prompts.jsonl (seeded)
bash scripts/run_study_detached.sh  # full study, about an hour; progress in results/logs/study_console.log
bash scripts/py.sh -m bench.report  # table, plots and resume numbers from results/runs
```
