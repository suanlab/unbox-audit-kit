#!/usr/bin/env python3
"""Sample 80 annotation pairs for G1 human evaluation.

Design (deterministic, seed=42):
  4 paradigms x 3 conditions x 6 ranks = 72 stratified pairs
  + 8 attention checks (predictions paired with WRONG paradigm's target)
  = 80 total

Outputs:
  annotation/pairs_master.csv      - all metadata (private; do NOT send to raters)
  annotation/pairs_for_raters.csv  - rater-facing (condition/rank hidden, shuffled)
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "annotation"

PARADIGMS_4 = ["transformer", "diffusion", "icl", "vit"]
CONDITIONS = [
    ("gpt4o_corpus",   "gpt4o_clean_prompt/{p}.json"),
    ("no_corpus",      "no_corpus_control/no_year_{p}.json"),
    ("claude_corpus",  "claude_clean_15paper/{p}.json"),
]
# 0-indexed: 1st, 3rd, 5th, 8th, 12th, 17th confidence-ranked predictions
RANKS = [0, 2, 4, 7, 11, 16]
SEED = 42

# Attention checks: predictions paired with a different paradigm's target.
# Should be obviously "1 / clearly different" - filters careless raters.
WRONG_PAIRS = [
    ("transformer", "vit"),
    ("diffusion", "icl"),
    ("icl", "transformer"),
    ("vit", "diffusion"),
]


def load_targets() -> dict:
    with open(DATA_DIR / "paradigm_shift_mapping.json") as f:
        return json.load(f)


def load_assumptions(path: Path) -> list:
    with open(path) as f:
        return json.load(f)["assumptions"]


def main() -> None:
    random.seed(SEED)
    targets = load_targets()
    OUTPUT_DIR.mkdir(exist_ok=True)

    pairs: list[dict] = []
    pid = 1

    # 1. Stratified pairs (72)
    for paradigm in PARADIGMS_4:
        target = targets[paradigm]
        target_text = target["broken_assumption"]
        target_aliases = "; ".join(target["aliases"])

        for cond_name, path_template in CONDITIONS:
            path = EXPERIMENTS_DIR / path_template.format(p=paradigm)
            assumptions = load_assumptions(path)
            for rank in RANKS:
                if rank >= len(assumptions):
                    continue
                a = assumptions[rank]
                pairs.append({
                    "pair_id": f"P{pid:03d}",
                    "is_attention_check": "false",
                    "paradigm": paradigm,
                    "condition": cond_name,
                    "rank": rank + 1,
                    "extracted_assumption": a["assumption"],
                    "target_assumption": target_text,
                    "target_aliases": target_aliases,
                    "source_paper": a.get("source_paper", ""),
                })
                pid += 1

    # 2. Attention checks (8)
    for pred_paradigm, target_paradigm in WRONG_PAIRS:
        wrong_target = targets[target_paradigm]
        path = EXPERIMENTS_DIR / "gpt4o_clean_prompt" / f"{pred_paradigm}.json"
        assumptions = load_assumptions(path)
        for rank in [0, 1]:
            a = assumptions[rank]
            pairs.append({
                "pair_id": f"A{pid:03d}",
                "is_attention_check": "true",
                "paradigm": f"{pred_paradigm}_target_{target_paradigm}",
                "condition": "attention_check",
                "rank": rank + 1,
                "extracted_assumption": a["assumption"],
                "target_assumption": wrong_target["broken_assumption"],
                "target_aliases": "; ".join(wrong_target["aliases"]),
                "source_paper": a.get("source_paper", ""),
            })
            pid += 1

    # Master CSV (full info)
    master_path = OUTPUT_DIR / "pairs_master.csv"
    with open(master_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "is_attention_check", "paradigm", "condition", "rank",
                      "extracted_assumption", "target_assumption", "target_aliases",
                      "source_paper"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pairs)

    # Rater-facing CSV (condition/rank hidden, shuffled)
    rater_pairs = pairs.copy()
    random.shuffle(rater_pairs)
    rater_path = OUTPUT_DIR / "pairs_for_raters.csv"
    with open(rater_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "extracted_assumption", "target_assumption",
                      "target_aliases", "score", "note"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in rater_pairs:
            w.writerow({
                "pair_id": p["pair_id"],
                "extracted_assumption": p["extracted_assumption"],
                "target_assumption": p["target_assumption"],
                "target_aliases": p["target_aliases"],
                "score": "",
                "note": "",
            })

    n_strat = sum(1 for p in pairs if p["is_attention_check"] == "false")
    n_ac = sum(1 for p in pairs if p["is_attention_check"] == "true")
    print(f"Wrote {len(pairs)} pairs")
    print(f"  stratified:       {n_strat} (4 paradigms x 3 conditions x 6 ranks)")
    print(f"  attention checks: {n_ac}")
    print(f"  master:    {master_path}")
    print(f"  raters:    {rater_path}")


if __name__ == "__main__":
    main()
