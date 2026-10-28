Assumed ITL floor at concurrency 1: 3.09 GB of weights / 336 GB/s spec bandwidth = 9.2 ms per token.

| | vLLM | Ollama |
|---|---|---|
| Prefill at c=1, fitted (prompt tok/s) | 7,226 | 5,989 |
| Fixed TTFT overhead at c=1, fitted (ms) | 26 | 11 |
| Fit quality r² (TTFT vs prompt tokens) | 0.98 | 0.95 |
| ITL at c=1, all requests pooled (ms) | 11.2 | 13.4 |
| Memory clock while busy, all runs (MHz, min-max) | 7,100–7,100 | 7,100–7,100 |
| Weight bandwidth implied at c=1 (GB/s) | 276 | 231 |
| Share of spec bandwidth at c=1 | 82% | 69% |
| Aggregate decode at c=2 (tok/s, efficiency vs ideal) | 157 (88%) | 145 (97%) |
| Aggregate decode at c=4 (tok/s, efficiency vs ideal) | 308 (86%) | 240 (80%) |
| Aggregate decode at c=8 (tok/s, efficiency vs ideal) | 544 (76%) | 287 (48%) |
| Aggregate decode at c=16 (tok/s, efficiency vs ideal) | 870 (61%) | 476 (40%) |
| Token gap at c=1: p99 / max (ms) | 13 / 16 | 18 / 28 |
| Token gap at c=2: p99 / max (ms) | 15 / 169 | 35 / 329 |
| Token gap at c=4: p99 / max (ms) | 14 / 292 | 86 / 135 |
| Token gap at c=8: p99 / max (ms) | 18 / 313 | 113 / 153 |
| Token gap at c=16: p99 / max (ms) | 200 / 323 | 137 / 165 |
| GPU utilization at c=1 (nvidia-smi, run mean) | 99% | 87% |
| GPU utilization at c=2 (nvidia-smi, run mean) | 99% | 84% |
| GPU utilization at c=4 (nvidia-smi, run mean) | 99% | 81% |
| GPU utilization at c=8 (nvidia-smi, run mean) | 99% | 74% |
| GPU utilization at c=16 (nvidia-smi, run mean) | 99% | 64% |
| c=16: requests still in prefill when a request arrives (mean) | 5.9 (4,197 prompt tokens) | 1.4 (1,054 prompt tokens) |
| c=16: TTFT p50 when 0 requests are ahead in prefill (ms, n) | 98 (n=21) | 270 (n=60) |
| c=16: TTFT p50 when 1 requests are ahead in prefill (ms, n) | 277 (n=21) | 361 (n=136) |
| c=16: TTFT p50 when 2-3 requests are ahead in prefill (ms, n) | 493 (n=42) | 497 (n=131) |
| c=16: TTFT p50 when 4+ requests are ahead in prefill (ms, n) | 719 (n=252) | 880 (n=9) |

Per-request ITL modes: share of requests whose ITL is over 1.4x the fastest quartile at that level (median ITL of the fast / slow requests in brackets).

| Server | Concurrency | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---|---|---|---|
| vLLM | 1 | 0% (11.1 / - ms) | 0% (11.3 / - ms) | 0% (11.1 / - ms) |
| vLLM | 2 | 0% (12.4 / - ms) | 0% (13.2 / - ms) | 0% (12.7 / - ms) |
| vLLM | 4 | 0% (12.8 / - ms) | 0% (12.9 / - ms) | 0% (13.1 / - ms) |
| vLLM | 8 | 0% (14.2 / - ms) | 0% (15.2 / - ms) | 0% (14.7 / - ms) |
| vLLM | 16 | 0% (18.1 / - ms) | 0% (18.6 / - ms) | 0% (18.2 / - ms) |
| Ollama | 1 | 0% (13.6 / - ms) | 0% (13.4 / - ms) | 0% (13.2 / - ms) |
| Ollama | 2 | 0% (13.9 / - ms) | 12% (13.7 / 29.6 ms) | 0% (13.8 / - ms) |
| Ollama | 4 | 0% (16.2 / - ms) | 25% (16.8 / 32.6 ms) | 50% (16.4 / 32.4 ms) |
| Ollama | 8 | 0% (21.6 / - ms) | 50% (21.3 / 37.7 ms) | 100% (- / 37.0 ms) |
| Ollama | 16 | 0% (34.0 / - ms) | 0% (33.4 / - ms) | 0% (33.4 / - ms) |
