"""Seeded synthetic prompt set, identical for both servers.

Each prompt is a block of short "field notes" followed by an instruction to write a long
report about them, so that greedy decoding almost always runs to the max-token limit on
both servers (Ollama's OpenAI API has no ignore_eos switch).

Input length is stratified: every consecutive block of 16 prompts spans the whole range of
note counts, so a run that uses only the first 16 prompts (concurrency 1 and 2) sees the
same length mix as a run that uses all 64.

    python -m bench.prompts --seed 20261027 --n 64 --out data/prompts.jsonl
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

BLOCK = 16
MIN_NOTES, MAX_NOTES = 3, 30

SITES = ["Ashford", "Brennan", "Caldera", "Dunmore", "Elstree", "Fairview", "Glenrock",
         "Harlow", "Ironbridge", "Juniper", "Kestrel", "Linwood"]
ITEMS = ["steel brackets", "solar panels", "copper wire", "water filters", "lithium cells",
         "brake pads", "circuit boards", "glass panes", "seed packets", "pipe fittings",
         "medical gloves", "fan motors"]
UNITS = ["crates", "pallets", "boxes", "units", "rolls", "cartons"]
TRENDS = ["up sharply", "up slightly", "flat", "down slightly", "down sharply"]
CAUSES = ["a late supplier shipment", "a staffing shortage", "a new customer contract",
          "a forklift breakdown", "a holiday weekend", "a software outage",
          "a storm that closed the roads", "an inventory audit", "a price change",
          "a quality recall"]

TOPICS = [
    "regional supply chain performance", "inventory risk across all sites",
    "staffing and operations planning", "logistics bottlenecks and their causes",
    "quarterly warehouse throughput", "supplier reliability",
    "demand trends by product line", "operational incidents and lessons learned",
]
TASKS = [
    "Using the notes above, write a detailed report on {topic}. Cover every site mentioned, "
    "explain the causes behind each change, and end with ten numbered recommendations. "
    "Write at least 800 words.",
    "Based on these notes, write a long executive briefing on {topic}. Include an overview, "
    "a section per site, a risk assessment, and a step-by-step action plan. "
    "Write at least 800 words.",
    "Write a thorough analysis of {topic} from the notes above. Compare the sites with each "
    "other, discuss what might happen next month, and propose concrete fixes for each problem. "
    "Write at least 800 words.",
]

SYSTEM = "You are a helpful assistant that writes long, detailed, well-organized reports."


def make_note(rng: random.Random, k: int) -> str:
    return (f"Note {k}: On day {rng.randint(1, 90)}, the {rng.choice(SITES)} site handled "
            f"{rng.randint(10, 999)} {rng.choice(UNITS)} of {rng.choice(ITEMS)}, "
            f"{rng.choice(TRENDS)} from the previous week because of {rng.choice(CAUSES)}.")


def build(seed: int, n: int) -> list[dict]:
    assert n % BLOCK == 0, f"n must be a multiple of {BLOCK}"
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        stratum = i % BLOCK
        # One draw inside each of the 16 equal-width strata of [MIN_NOTES, MAX_NOTES].
        width = (MAX_NOTES - MIN_NOTES + 1) / BLOCK
        n_notes = MIN_NOTES + int((stratum + rng.random()) * width)
        notes = "\n".join(make_note(rng, k + 1) for k in range(n_notes))
        task = rng.choice(TASKS).format(topic=rng.choice(TOPICS))
        prompts.append({"id": i, "n_notes": n_notes, "system": SYSTEM,
                        "user": f"{notes}\n\n{task}"})
    # Shuffle within each block so prompt length does not rise with position in the block.
    for b in range(0, n, BLOCK):
        block = prompts[b:b + BLOCK]
        rng.shuffle(block)
        prompts[b:b + BLOCK] = block
    for i, p in enumerate(prompts):
        p["id"] = i
    return prompts


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("data/prompts.jsonl"))
    args = ap.parse_args()
    prompts = build(args.seed, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.out} sha256={file_sha256(args.out)}")


if __name__ == "__main__":
    main()
