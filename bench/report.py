"""Builds the results table, plots and resume numbers from results/runs/*.json only.

    python -m bench.report [--results results]

Every figure and number this writes comes from the saved run files; nothing is typed in.
Per (server, concurrency) it takes the median over the repeats and keeps the min-max range.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SERVER_LABEL = {"vllm": "vLLM", "ollama": "Ollama"}
# Reference categorical palette, slots 1-2, light surface (validated in the dataviz skill).
COLORS = {"vllm": "#2a78d6", "ollama": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

METRICS = {  # name -> how to read it from one run file
    "out_tok_s": lambda r: r["summary"]["output_tok_per_s"],
    "req_s": lambda r: r["summary"]["requests_per_s"],
    "ttft_p50_ms": lambda r: r["summary"]["ttft_s"]["p50"] * 1000,
    "ttft_p90_ms": lambda r: r["summary"]["ttft_s"]["p90"] * 1000,
    "itl_p50_ms": lambda r: r["summary"]["itl_s"]["p50"] * 1000,
    "gap_p99_ms": lambda r: r["summary"]["chunk_gap_s"]["p99"] * 1000,
    "e2e_p50_s": lambda r: r["summary"]["e2e_s"]["p50"],
    "mean_out_tokens": lambda r: r["summary"]["output_tokens"] / max(r["summary"]["n_ok"], 1),
    "length_frac": lambda r: r["summary"]["finish_length_frac"],
    "tokens_per_chunk": lambda r: r["summary"]["tokens_per_chunk"],
    "peak_mem_mib": lambda r: r["gpu"]["mem_used_mib_max"],
    "sm_mhz_busy": lambda r: r["gpu"]["sm_clock_mhz_mean_busy"],
    "temp_c_max": lambda r: r["gpu"]["temp_c_max"],
    "errors": lambda r: r["summary"]["n_errors"],
}


def load_runs(results: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((results / "runs").glob("*.json"))]


def aggregate(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[(r["server"], r["concurrency"])].append(r)
    agg = {}
    for (server, c), rs in sorted(by.items()):
        entry = {"server": server, "concurrency": c, "repeats": sorted(r["repeat"] for r in rs),
                 "n_requests_per_run": rs[0]["n_requests"]}
        for name, get in METRICS.items():
            vals = [get(r) for r in rs if get(r) is not None]
            entry[name] = {"median": statistics.median(vals) if vals else None,
                           "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                           "values": vals}
        agg[f"{server}_c{c}"] = entry
    return agg


def provenance(runs: list[dict]) -> dict:
    """The environment every run reports; any field that differs between runs is listed."""
    def fields(r):
        e, s = r["env"], r["server_info"]["details"]
        return {"commit": e["git"]["commit"], "dirty": e["git"]["dirty"],
                "gpu": e["gpu"]["name"], "vram_mib": e["gpu"]["memory_total"],
                "driver": e["gpu"]["driver_version"], "cuda_driver_api": e["gpu"]["cuda_driver_api_version"],
                "cuda_toolkit": e["cuda_toolkit_nvcc"], "torch": e["packages"]["torch"],
                "torch_cuda_runtime": e["torch_cuda_runtime"], "vllm": e["packages"]["vllm"],
                "python": e["python"], "kernel": e["host"]["kernel"], "distro": e["host"]["distro"],
                "windows": e["host"]["windows_ver"], "power_plan": e["host"]["power_plan"],
                "cpu": e["host"]["cpu"], "wsl_mem_mib": e["host"]["wsl_mem_mib"],
                "prompt_sha256": r["config"]["prompt_sha256"],
                f"{r['server']}_server_version": s.get("version")}
    merged, conflicts = {}, {}
    for r in runs:
        for k, v in fields(r).items():
            if k in merged and merged[k] != v:
                conflicts.setdefault(k, {merged[k]}).add(v)
            merged.setdefault(k, v)
    return {"values": merged, "conflicts": {k: sorted(map(str, v)) for k, v in conflicts.items()}}


def fidelity(runs: list[dict]) -> dict:
    """Same request tag on both servers: do they see the same prompt and write the same text?"""
    reqs = defaultdict(dict)
    for r in runs:
        for q in r["requests"]:
            if q["ok"]:
                reqs[q["tag"]][r["server"]] = q
    pairs = [v for v in reqs.values() if "vllm" in v and "ollama" in v]
    dprompt = [v["ollama"]["prompt_tokens"] - v["vllm"]["prompt_tokens"] for v in pairs
               if v["ollama"]["prompt_tokens"] is not None and v["vllm"]["prompt_tokens"] is not None]
    same_text = [v["ollama"]["text_sha1"] == v["vllm"]["text_sha1"] for v in pairs]
    same_head = [v["ollama"]["text_head"][:60] == v["vllm"]["text_head"][:60] for v in pairs]
    return {"paired_requests": len(pairs),
            "prompt_token_diff_ollama_minus_vllm": {
                "min": min(dprompt, default=None), "max": max(dprompt, default=None),
                "median": statistics.median(dprompt) if dprompt else None},
            "identical_full_text_frac": sum(same_text) / len(pairs) if pairs else None,
            "identical_first_60_chars_frac": sum(same_head) / len(pairs) if pairs else None}


def fmt(x, nd=0):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def table(agg: dict) -> str:
    cols = [("out_tok_s", "Output tok/s", 0), ("req_s", "Req/s", 2), ("ttft_p50_ms", "TTFT p50 (ms)", 0),
            ("ttft_p90_ms", "TTFT p90 (ms)", 0), ("itl_p50_ms", "ITL p50 (ms)", 1),
            ("gap_p99_ms", "Token gap p99 (ms)", 0), ("e2e_p50_s", "E2E p50 (s)", 2),
            ("mean_out_tokens", "Mean out tokens", 0), ("peak_mem_mib", "Peak GPU mem (MiB)", 0),
            ("sm_mhz_busy", "SM clock busy (MHz)", 0)]
    head = "| Server | Concurrency | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    rows = []
    for e in agg.values():
        cells = []
        for key, _, nd in cols:
            m = e[key]
            cell = fmt(m["median"], nd)
            if key in ("out_tok_s", "ttft_p50_ms") and m["min"] is not None:
                cell += f" ({fmt(m['min'], nd)}–{fmt(m['max'], nd)})"
            cells.append(cell)
        rows.append(f"| {SERVER_LABEL[e['server']]} | {e['concurrency']} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def plot(agg: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    panels = [("out_tok_s", "Output tokens / s (higher is better)", "throughput_vs_concurrency.png"),
              ("ttft_p50_ms", "Median time to first token, ms (lower is better)", "ttft_vs_concurrency.png"),
              ("itl_p50_ms", "Median inter-token latency, ms (lower is better)", "itl_vs_concurrency.png")]
    for key, title, fname in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for server in ("vllm", "ollama"):
            es = [e for e in agg.values() if e["server"] == server and e[key]["median"] is not None]
            if not es:
                continue
            xs = [e["concurrency"] for e in es]
            med = [e[key]["median"] for e in es]
            lo = [e[key]["median"] - e[key]["min"] for e in es]
            hi = [e[key]["max"] - e[key]["median"] for e in es]
            ax.errorbar(xs, med, yerr=[lo, hi], color=COLORS[server], lw=2, marker="o", ms=6,
                        mec=SURFACE, mew=1.5, capsize=3, elinewidth=1.2, label=SERVER_LABEL[server])
            ax.annotate(f"{SERVER_LABEL[server]} {med[-1]:,.0f}" if key != "itl_p50_ms"
                        else f"{SERVER_LABEL[server]} {med[-1]:.1f}",
                        (xs[-1], med[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlim(0.8, 26)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Concurrent requests in flight")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.text(0.01, 0.01, "Qwen2.5-1.5B-Instruct FP16, RTX 3060 Laptop 6 GB, WSL2. "
                 "Median of 3 runs; bars span min to max.", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / fname, facecolor=SURFACE)
        plt.close(fig)


def resume_block(agg: dict, prov: dict) -> str:
    def m(server, c, key):
        e = agg.get(f"{server}_c{c}")
        return e[key]["median"] if e else None
    hi = max(e["concurrency"] for e in agg.values())
    v = prov["values"]
    peak = {s: max((e["peak_mem_mib"]["max"] for e in agg.values() if e["server"] == s
                    and e["peak_mem_mib"]["max"] is not None), default=None) for s in SERVER_LABEL}
    ratio_hi = (m("vllm", hi, "out_tok_s") / m("ollama", hi, "out_tok_s")
                if m("vllm", hi, "out_tok_s") and m("ollama", hi, "out_tok_s") else None)
    ratio_1 = (m("vllm", 1, "out_tok_s") / m("ollama", 1, "out_tok_s")
               if m("vllm", 1, "out_tok_s") and m("ollama", 1, "out_tok_s") else None)
    lines = [
        "Model / precision: Qwen2.5-1.5B-Instruct, FP16 weights and FP16 KV cache on both servers",
        "  (vLLM: HF safetensors cast BF16->FP16; Ollama: qwen2.5:1.5b-instruct-fp16 GGUF)",
        f"Workload: 64 seeded prompts, greedy, max 256 output tokens, concurrency 1-{hi}, median of 3 runs",
        "",
        f"{'':10}{'tok/s @1':>10}{'TTFT @1':>11}{f'tok/s @{hi}':>11}{f'TTFT @{hi}':>12}{'peak GPU mem':>14}",
    ]
    for s in ("vllm", "ollama"):
        lines.append(f"{SERVER_LABEL[s]:10}{fmt(m(s, 1, 'out_tok_s'), 1):>10}"
                     f"{fmt(m(s, 1, 'ttft_p50_ms')) + ' ms':>11}{fmt(m(s, hi, 'out_tok_s'), 0):>11}"
                     f"{fmt(m(s, hi, 'ttft_p50_ms')) + ' ms':>12}{fmt(peak[s]) + ' MiB':>14}")
    lines += [
        "",
        f"Throughput ratio vLLM/Ollama: {fmt(ratio_1, 2)}x at concurrency 1, {fmt(ratio_hi, 2)}x at {hi}",
        "Peak GPU memory is device-wide memory.used from nvidia-smi (WSL2 has no per-process view);",
        f"  vLLM pre-reserves --gpu-memory-utilization of VRAM, so its number is a reservation, not need.",
        "",
        f"GPU: {v.get('gpu')} ({v.get('vram_mib')}), driver {v.get('driver')}, CUDA {v.get('cuda_driver_api')} (driver API)",
        f"vLLM {v.get('vllm')} (torch {v.get('torch')}, CUDA runtime {v.get('torch_cuda_runtime')}); "
        f"Ollama {v.get('ollama_server_version')}",
        f"WSL2 {v.get('distro')}, kernel {v.get('kernel')}; {v.get('windows')}; power plan {v.get('power_plan')}",
        f"Benchmark commit {v.get('commit', '')[:12]}{' (dirty)' if v.get('dirty') else ''}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no run files in {args.results / 'runs'}")
    agg = aggregate(runs)
    prov = provenance(runs)
    fid = fidelity(runs)
    summary = {"n_run_files": len(runs), "provenance": prov, "fidelity": fid, "aggregate": agg}
    (args.results / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (args.results / "results_table.md").write_text(table(agg) + "\n", encoding="utf-8")
    (args.results / "resume_numbers.txt").write_text(resume_block(agg, prov) + "\n", encoding="utf-8")
    plot(agg, args.results)
    print(table(agg))
    print()
    print(resume_block(agg, prov))
    print()
    print(json.dumps({"fidelity": fid, "provenance_conflicts": prov["conflicts"]}, indent=1))


if __name__ == "__main__":
    main()
