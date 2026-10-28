"""Runs the whole study and writes one JSON file per (server, concurrency, repeat).

Only one server fits in 6 GB at a time, so servers cannot be interleaved request by request.
Instead the schedule is A1 | B1 B2 | A2 A3 | B3 (A = vLLM, B = Ollama): each server gets an
early, a middle and a late repeat, so slow thermal drift over the evening does not favour
either one. Inside a repeat the concurrency levels run in a seeded shuffled order, the same
order for both servers.

    python -m bench.study                 # full study -> results/runs/*.json
    python -m bench.study --smoke         # tiny end-to-end check -> results/smoke/
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from bench import client, envinfo, gpu, prompts

REPO = Path(__file__).resolve().parents[1]
LEVELS = [1, 2, 4, 8, 16]
SCHEDULE = [("vllm", [1]), ("ollama", [1, 2]), ("vllm", [2, 3]), ("ollama", [3])]
MAX_TOKENS = 256
SEED = 20261027
WARMUP_SECONDS = 30
BASELINE_LIMIT_MIB = 800  # more than this before a server starts means something else is on the GPU

SERVERS = {
    "vllm": {"script": "scripts/serve_vllm.sh", "base_url": "http://127.0.0.1:8000",
             "model": "qwen2.5-1.5b-instruct", "ready": "/v1/models"},
    "ollama": {"script": "scripts/serve_ollama.sh", "base_url": "http://127.0.0.1:11435",
               "model": "qwen2.5-1.5b-bench", "ready": "/api/version"},
}
# Lines worth keeping from each server's startup log (KV cache size, backends, batch slots).
LOG_PATTERNS = {
    "vllm": r"KV cache|Maximum concurrency|weights took|Model loading took|[Gg]raph captur|"
            r"[Aa]ttention backend|torch.compile|chunked prefill|max_num_batched_tokens|"
            r"pin_memory|WSL|Using .*backend|non-default args",
    "ollama": r"load_backend|offloaded|flash|kv_cache|KvSize|KvCacheType|n_ctx|n_seq|"
              r"parallel|NumParallel|model weights|kv cache|compute graph|total memory|"
              r"inference compute|library=|runner|engine|OLLAMA_",
}


def n_requests(c: int) -> int:
    return max(16, 8 * c)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def http(url: str, body: dict | None = None, timeout: float = 10) -> dict | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception:
        return None


def start_server(name: str, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(["bash", str(REPO / SERVERS[name]["script"])], cwd=REPO,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(name: str, proc: subprocess.Popen, log_path: Path, timeout: float = 900) -> float:
    cfg, t0 = SERVERS[name], time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-40:]
            raise RuntimeError(f"{name} exited with {proc.returncode}:\n" + "\n".join(tail))
        if http(cfg["base_url"] + cfg["ready"], timeout=2) is not None:
            return time.perf_counter() - t0
        time.sleep(2)
    raise TimeoutError(f"{name} not ready after {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass


def wait_gpu_released(baseline_mib: float, timeout: float = 60) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        used = gpu.summarize(gpu.sample_for(1.0))["mem_used_mib_max"] or 0
        if used <= baseline_mib + 150:
            return used
        time.sleep(2)
    return used


def server_details(name: str) -> dict:
    cfg = SERVERS[name]
    base = cfg["base_url"]
    if name == "vllm":
        return {"version": (http(base + "/version") or {}).get("version"),
                "models": http(base + "/v1/models")}
    return {"version": (http(base + "/api/version") or {}).get("version"),
            "show": _trim_show(http(base + "/api/show", {"model": cfg["model"]})),
            "ps": http(base + "/api/ps"),
            "tags": [m for m in (http(base + "/api/tags") or {}).get("models", [])
                     if m.get("name", "").startswith(("qwen2.5", cfg["model"]))]}


def _trim_show(show: dict | None) -> dict | None:
    if not show:
        return show
    keep = {k: show.get(k) for k in ("details", "parameters", "modified_at", "capabilities")}
    info = show.get("model_info") or {}
    keep["model_info"] = {k: v for k, v in info.items() if not isinstance(v, list)}
    return keep


def log_highlights(name: str, log_path: Path) -> list[str]:
    pat = re.compile(LOG_PATTERNS[name])
    lines = log_path.read_text(errors="replace").splitlines()
    return [ln.strip()[:400] for ln in lines if pat.search(ln)][:120]


def warmup(name: str, prompt_set: list[dict], seconds: float, block: int) -> dict:
    """Keeps 16 requests in flight until `seconds` have passed (GPU boost, lazy init, caches)."""
    cfg, t0, sent, waves = SERVERS[name], time.perf_counter(), 0, 0
    while time.perf_counter() - t0 < seconds:
        jobs = [(prompt_set[(sent + i) % len(prompt_set)], f"warm-b{block}-w{waves}-i{i}")
                for i in range(16)]
        res, _ = asyncio.run(client.closed_loop(cfg["base_url"], cfg["model"], jobs, 16,
                                                MAX_TOKENS))
        bad = [r.error for r in res if not r.ok]
        if bad:
            raise RuntimeError(f"warmup errors on {name}: {bad[:3]}")
        sent, waves = sent + len(jobs), waves + 1
    return {"seconds": time.perf_counter() - t0, "requests": sent}


def run_level(name: str, prompt_set: list[dict], c: int, rep: int, n: int,
              max_tokens: int) -> dict:
    cfg = SERVERS[name]
    warm = [(prompt_set[i % len(prompt_set)], f"warm-r{rep}c{c:02d}i{i:03d}") for i in range(c)]
    asyncio.run(client.closed_loop(cfg["base_url"], cfg["model"], warm, c, max_tokens))
    jobs = [(prompt_set[i % len(prompt_set)], f"r{rep}c{c:02d}i{i:03d}") for i in range(n)]
    sampler = gpu.GpuSampler(100).start()
    started = now_iso()
    results, wall = asyncio.run(client.closed_loop(cfg["base_url"], cfg["model"], jobs, c,
                                                   max_tokens))
    samples = sampler.stop()
    return {"started_at": started, "summary": client.summarize(results, wall),
            "gpu": gpu.summarize(samples), "gpu_samples": samples,
            "requests": [r.to_json() for r in results]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--only", choices=list(SERVERS), help="smoke-test one server")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-power-plan", action="store_true",
                    help="record even if Windows is not on the Turbo plan")
    args = ap.parse_args()

    prompt_path = REPO / "data" / "prompts.jsonl"
    prompt_set = prompts.load(prompt_path)
    env = envinfo.collect(REPO)
    if env["git"]["dirty"] and not (args.allow_dirty or args.smoke):
        sys.exit(f"tracked files have uncommitted changes: {env['git']['dirty_files']}")
    if env["host"]["power_plan"] != "Turbo" and not (args.allow_power_plan or args.smoke):
        sys.exit(f"Windows power plan is {env['host']['power_plan']!r}; switch to Turbo first")

    if args.smoke:
        levels, schedule, max_tokens, warm_s = [1, 4], [(args.only or "vllm", [1])], 32, 5
        if not args.only:
            schedule.append(("ollama", [1]))
        out = REPO / "results" / "smoke"
    else:
        levels, schedule, max_tokens, warm_s = LEVELS, SCHEDULE, MAX_TOKENS, WARMUP_SECONDS
        out = REPO / "results"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    config = {"levels": levels, "schedule": schedule, "max_tokens": max_tokens, "seed": SEED,
              "requests_per_run": {c: (4 if args.smoke else n_requests(c)) for c in levels},
              "warmup_seconds_per_launch": warm_s, "per_level_warmup": "one wave of c requests",
              "sampling": client.SAMPLING, "prompt_file": str(prompt_path.relative_to(REPO)),
              "prompt_sha256": prompts.file_sha256(prompt_path), "n_prompts": len(prompt_set),
              "servers": SERVERS}
    study = {"started_at": now_iso(), "env": env, "config": config, "blocks": []}
    position = 0
    for block, (name, reps) in enumerate(schedule):
        # The Windows plan has been seen to change on its own, so check it at every launch.
        plan = envinfo.windows_info()["power_plan"]
        if plan != "Turbo" and not (args.allow_power_plan or args.smoke):
            sys.exit(f"power plan changed to {plan!r} before block {block}; switch back to Turbo")
        base = gpu.summarize(gpu.sample_for(3.0))
        if (base["mem_used_mib_max"] or 0) > BASELINE_LIMIT_MIB:
            sys.exit(f"GPU already has {base['mem_used_mib_max']} MiB in use before {name}")
        log_path = out / "logs" / f"block{block}_{name}.log"
        proc = start_server(name, log_path)
        info = {"block": block, "server": name, "repeats": reps, "baseline_gpu": base,
                "power_plan": plan,
                "launch_script": (REPO / SERVERS[name]["script"]).read_text()}
        try:
            info["ready_s"] = wait_ready(name, proc, log_path)
            if name == "ollama":  # first generate loads the weights; time it separately
                t0 = time.perf_counter()
                http(SERVERS[name]["base_url"] + "/api/generate",
                     {"model": SERVERS[name]["model"], "prompt": "", "keep_alive": -1}, 600)
                info["model_load_s"] = time.perf_counter() - t0
            info["idle_loaded_gpu"] = gpu.summarize(gpu.sample_for(3.0))
            info["details"] = server_details(name)
            info["warmup"] = warmup(name, prompt_set, warm_s, block)
            print(f"[{now_iso()}] block {block} {name} ready in {info['ready_s']:.0f}s, "
                  f"idle {info['idle_loaded_gpu']['mem_used_mib_max']} MiB", flush=True)
            for rep in reps:
                order = levels[:]
                random.Random(SEED * 100 + rep).shuffle(order)
                for c in order:
                    position += 1
                    n = 4 if args.smoke else n_requests(c)
                    plan = envinfo.windows_info()["power_plan"]
                    run = run_level(name, prompt_set, c, rep, n, max_tokens)
                    s = run["summary"]
                    record = {"schema": 1, "server": name, "concurrency": c, "repeat": rep,
                              "block": block, "position": position, "n_requests": n,
                              "power_plan_at_start": plan,
                              "config": config, "env": env, "server_info": info, **run}
                    path = out / "runs" / f"{name}_c{c:02d}_r{rep}.json"
                    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
                    print(f"  {name} r{rep} c={c:2d}: {s['output_tok_per_s']:7.1f} tok/s  "
                          f"{s['requests_per_s']:.2f} req/s  ttft p50 "
                          f"{s['ttft_s']['p50'] * 1000:6.0f} ms  itl p50 "
                          f"{s['itl_s']['p50'] * 1000:5.1f} ms  errors {s['n_errors']}  "
                          f"peak {run['gpu']['mem_used_mib_max']} MiB  "
                          f"sm {run['gpu']['sm_clock_mhz_mean_busy']} MHz", flush=True)
                    time.sleep(3)
            info["details_after"] = server_details(name)
        finally:
            stop_server(proc)
            info["log_highlights"] = log_highlights(name, log_path)
            info["gpu_after_stop_mib"] = wait_gpu_released(base["mem_used_mib_max"] or 0)
            study["blocks"].append(info)
            study["finished_at"] = now_iso()
            (out / "study.json").write_text(json.dumps(study, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
