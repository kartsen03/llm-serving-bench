"""Which llama-server slot served each Ollama request, and was that request in the fast or slow
ITL mode? Writes results/ollama_slot_modes.md.

llama-server logs one "processing task" line per request, in arrival order. Within a server
launch the order is: the global warmup, then for each run its warmup wave of c requests and its
n measured requests, so the measured requests of a run are a known slice of those lines. Inside
a run, requests are matched to tasks in the order they were sent.

    python -m bench.ollama_slots [--results results]
"""

import argparse
import glob
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

TASK = re.compile(r"slot launch_slot_: id\s+(\d+) \| task \d+ \| processing task")


def read_log(logs: Path, block: int) -> str | None:
    for p in (logs / f"block{block}_ollama.log", logs / f"block{block}_ollama.log.gz"):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt", encoding="utf-8", errors="replace") as f:
                return f.read()
    return None


def slot_modes(results: Path, modes: dict) -> str:
    runs = sorted((json.loads(Path(p).read_text(encoding="utf-8"))
                   for p in glob.glob(str(results / "runs" / "ollama_*.json"))),
                  key=lambda r: r["position"])
    by_block = defaultdict(list)
    for r in runs:
        by_block[r["block"]].append(r)
    lines = ["Ollama, per run: each slot id with one letter per request it served at that "
             "concurrency, S = slow ITL mode, f = fast (threshold as in the ITL-modes table).", ""]
    for block, rs in sorted(by_block.items()):
        text = read_log(results / "logs", block)
        if text is None:
            lines.append(f"- launch {block}: log not found")
            continue
        slots = [int(m.group(1)) for m in TASK.finditer(text)]
        i = rs[0]["server_info"]["warmup"]["requests"]
        expected = i + sum(r["concurrency"] + r["n_requests"] for r in rs)
        if expected != len(slots):
            lines.append(f"- launch {block}: {len(slots)} tasks in the log, expected {expected}; skipped")
            continue
        for r in rs:
            c, n = r["concurrency"], r["n_requests"]
            i += c
            used = slots[i:i + n]
            i += n
            cut = modes.get(c, {}).get("threshold_ms")
            if c in (1, 16) or cut is None:
                continue
            reqs = sorted(r["requests"], key=lambda q: q["t_send"])
            per_slot = defaultdict(str)
            for q, s in zip(reqs, used):
                per_slot[s] += "S" if q["itl"] and q["itl"] * 1000 > cut else "f"
            cells = " ".join(f"{s}:{v}" for s, v in sorted(per_slot.items()))
            lines.append(f"- launch {block}, repeat {r['repeat']}, c={c}: {cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    modes = summary["analysis"]["ollama"]["itl_modes"]
    md = slot_modes(args.results, {int(k): v for k, v in modes.items()})
    (args.results / "ollama_slot_modes.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
