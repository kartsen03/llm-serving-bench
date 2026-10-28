"""Mechanism checks behind the headline numbers, computed from the per-request records.

- Prefill speed: at concurrency 1 there is no batching, so TTFT ~= overhead + prompt_tokens /
  prefill_rate. A least-squares line over all c=1 requests gives both terms per server.
- Decode efficiency: at concurrency 1 each step reads every weight once, so the spec memory
  bandwidth sets a floor on ITL. Measured ITL against that floor says how close each server gets.
- Batching efficiency: aggregate decode rate c / ITL(c), relative to c / ITL(1), the ideal where a
  bigger batch costs nothing extra per step.
- Prefill interference: the token-gap tail (p99, max) at each concurrency; gaps much longer than the
  ITL are steps where a new prompt was being prefilled alongside the running decodes.
"""

import statistics
from collections import defaultdict

# Bytes that every decode step has to read: the safetensors file vLLM loads (3,087,467,144 B, as
# downloaded for revision 989aa79). The F16 GGUF is 3,093,682,443 B (0.2% larger: F32 norms,
# metadata and the embedded tokenizer).
WEIGHT_BYTES = 3_087_467_144
# RTX 3060 Laptop: 192-bit GDDR6, nvidia-smi max memory clock 7001 MHz = 14 Gbps per pin.
SPEC_BW_BYTES_S = 192 / 8 * 14e9


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns (a, b, r^2)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot if ss_tot else float("nan")


def analyze(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[(r["server"], r["concurrency"])].append(r)
    servers = sorted({r["server"] for r in runs})
    levels = sorted({r["concurrency"] for r in runs})
    out: dict = {"assumptions": {"weight_bytes": WEIGHT_BYTES,
                                 "spec_bandwidth_GB_s": SPEC_BW_BYTES_S / 1e9,
                                 "itl_floor_ms": WEIGHT_BYTES / SPEC_BW_BYTES_S * 1000}}
    prompt_tokens = [q["prompt_tokens"] for r in runs for q in r["requests"]
                     if q["ok"] and q["prompt_tokens"]]
    out["prompt_tokens_range"] = [min(prompt_tokens, default=None), max(prompt_tokens, default=None)]

    for s in servers:
        res: dict = {}
        c1 = [q for r in by[(s, 1)] for q in r["requests"] if q["ok"] and q["prompt_tokens"]]
        if len(c1) >= 3:
            a, b, r2 = _fit([q["prompt_tokens"] for q in c1], [q["ttft"] for q in c1])
            res["prefill_c1"] = {"n": len(c1), "overhead_ms": a * 1000,
                                 "ms_per_1k_prompt_tokens": b * 1e6,
                                 "prefill_tok_per_s": 1 / b if b > 0 else None, "r2": r2}
        itl = {c: statistics.median(q["itl"] for r in by[(s, c)] for q in r["requests"]
                                    if q["ok"] and q["itl"])
               for c in levels if by[(s, c)]}
        if 1 in itl:
            res["decode_c1"] = {"itl_ms": itl[1] * 1000,
                                "effective_weight_bw_GB_s": WEIGHT_BYTES / itl[1] / 1e9,
                                "share_of_spec_bw": WEIGHT_BYTES / itl[1] / SPEC_BW_BYTES_S}
            res["batching"] = {c: {"itl_ms": itl[c] * 1000,
                                   "aggregate_decode_tok_s": c / itl[c],
                                   "speedup_vs_c1": (c / itl[c]) / (1 / itl[1]),
                                   "efficiency_vs_ideal": (c / itl[c]) / (c / itl[1])}
                               for c in sorted(itl)}
        gaps = {}
        for c in levels:
            g = [b - a for r in by[(s, c)] for q in r["requests"] if q["ok"]
                 for a, b in zip(q["chunk_offsets"], q["chunk_offsets"][1:])]
            if g:
                g.sort()
                med = g[len(g) // 2]
                gaps[c] = {"p50_ms": med * 1000, "p99_ms": g[int(0.99 * (len(g) - 1))] * 1000,
                           "max_ms": g[-1] * 1000,
                           "share_over_3x_median": sum(x > 3 * med for x in g) / len(g)}
        res["token_gaps"] = gaps
        out[s] = res
    return out


def to_markdown(a: dict) -> str:
    lines = [f"Assumed ITL floor at concurrency 1: {a['assumptions']['weight_bytes'] / 1e9:.2f} GB of "
             f"weights / {a['assumptions']['spec_bandwidth_GB_s']:.0f} GB/s spec bandwidth = "
             f"{a['assumptions']['itl_floor_ms']:.1f} ms per token.", ""]
    servers = [s for s in ("vllm", "ollama") if s in a]
    lines += ["| | " + " | ".join({"vllm": "vLLM", "ollama": "Ollama"}[s] for s in servers) + " |",
              "|---|" + "---|" * len(servers)]

    def row(label, f):
        cells = []
        for s in servers:
            try:
                cells.append(f(a[s]))
            except (KeyError, TypeError):
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Prefill at c=1, fitted (prompt tok/s)", lambda r: f"{r['prefill_c1']['prefill_tok_per_s']:,.0f}")
    row("Fixed TTFT overhead at c=1, fitted (ms)", lambda r: f"{r['prefill_c1']['overhead_ms']:.0f}")
    row("Fit quality r² (TTFT vs prompt tokens)", lambda r: f"{r['prefill_c1']['r2']:.2f}")
    row("ITL at c=1 (ms)", lambda r: f"{r['decode_c1']['itl_ms']:.1f}")
    row("Weight bandwidth implied at c=1 (GB/s)", lambda r: f"{r['decode_c1']['effective_weight_bw_GB_s']:.0f}")
    row("Share of spec bandwidth at c=1", lambda r: f"{r['decode_c1']['share_of_spec_bw']:.0%}")
    for c in sorted({c for s in servers for c in a[s].get("batching", {})}):
        if c == 1:
            continue
        row(f"Aggregate decode at c={c} (tok/s, efficiency vs ideal)",
            lambda r, c=c: f"{r['batching'][c]['aggregate_decode_tok_s']:,.0f} "
                           f"({r['batching'][c]['efficiency_vs_ideal']:.0%})")
    for c in sorted({c for s in servers for c in a[s].get("token_gaps", {})}):
        row(f"Token gap at c={c}: p99 / max (ms)",
            lambda r, c=c: f"{r['token_gaps'][c]['p99_ms']:.0f} / {r['token_gaps'][c]['max_ms']:.0f}")
    return "\n".join(lines)
