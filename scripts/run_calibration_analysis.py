#!/usr/bin/env python3
"""T2.1: Confidence calibration analysis.

Check if confidence scores from GPT-4o predict anything meaningful
about similarity to ground truth (target).
"""
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def spearman_correlation(x: list, y: list) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    def ranks(values):
        sorted_vals = sorted(enumerate(values), key=lambda p: p[1])
        r = [0] * len(values)
        for rank, (i, _) in enumerate(sorted_vals, 1):
            r[i] = rank
        return r
    rx = ranks(x)
    ry = ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
    den_y = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5
    return num / (den_x * den_y) if den_x * den_y > 0 else 0.0


def main():
    # For each paradigm, we need confidence-ordered assumptions and their similarity to target
    # We'll use the canonical_evaluation artifact if available
    canonical_path = PROJECT_ROOT / "experiments" / "canonical_evaluation.json"

    # Gather per-paradigm data
    paradigms = ["transformer", "diffusion", "icl", "vit", "gan", "batchnorm",
                 "resnet", "word2vec", "dropout", "bert"]

    # For each paradigm, load the GPT-4o extraction and analyze confidence distribution
    all_analysis = {}

    for paradigm in paradigms:
        path = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt" / f"{paradigm}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        assumptions = data.get("assumptions", [])
        if not assumptions:
            continue

        # Confidence statistics
        confidences = [a.get("confidence", 0) for a in assumptions]

        # Confidence-ordered position vs confidence value
        # (confidence-ordered because they're already sorted)
        all_analysis[paradigm] = {
            "num_assumptions": len(assumptions),
            "confidence_stats": {
                "mean": round(sum(confidences) / len(confidences), 3),
                "max": round(max(confidences), 3),
                "min": round(min(confidences), 3),
                "median": round(sorted(confidences)[len(confidences) // 2], 3),
            },
            "top_10_confidences": [round(c, 3) for c in confidences[:10]],
            "position_confidence_monotonic": all(
                confidences[i] >= confidences[i + 1] for i in range(min(9, len(confidences) - 1))
            ),
        }

    # Across all paradigms: overall confidence distribution
    all_confs = []
    for paradigm in paradigms:
        path = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt" / f"{paradigm}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        all_confs.extend([a.get("confidence", 0) for a in data.get("assumptions", [])])

    # Hit rate per confidence bin (using canonical results)
    # Target assumption rank vs confidence at target position
    # For 6 successes: conf at rank 1, 2, 3, 5, 7, 10
    if canonical_path.exists():
        canonical = json.load(open(canonical_path))
        hits = {r["paradigm"]: r for r in canonical["results"]}

        # Confidence at the hit rank (for successes)
        hit_confidences = []
        for paradigm in ["transformer", "diffusion", "icl", "vit", "batchnorm", "gan"]:
            res = hits.get(paradigm, {})
            rank = res.get("conf_rank")
            if rank is None:
                continue
            path = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt" / f"{paradigm}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            assumptions = data.get("assumptions", [])
            if rank <= len(assumptions):
                hit_confidences.append({
                    "paradigm": paradigm,
                    "rank": rank,
                    "confidence_at_rank": round(assumptions[rank - 1].get("confidence", 0), 3),
                    "top_confidence": round(assumptions[0].get("confidence", 0), 3),
                    "best_sim": res["best_sim"],
                })
        all_analysis["hit_confidence_analysis"] = hit_confidences

        # Is target-hit confidence notably different from the top confidence?
        if hit_confidences:
            avg_hit_conf = sum(h["confidence_at_rank"] for h in hit_confidences) / len(hit_confidences)
            avg_top_conf = sum(h["top_confidence"] for h in hit_confidences) / len(hit_confidences)
            all_analysis["summary"] = {
                "avg_confidence_at_hit_rank": round(avg_hit_conf, 3),
                "avg_top_confidence": round(avg_top_conf, 3),
                "confidence_gap_top_to_hit": round(avg_top_conf - avg_hit_conf, 3),
            }

    # Confidence-rank monotonicity across paradigms
    n_mono = sum(1 for p in all_analysis.values()
                 if isinstance(p, dict) and p.get("position_confidence_monotonic", False))
    n_total = sum(1 for p in all_analysis.values()
                  if isinstance(p, dict) and "position_confidence_monotonic" in p)
    all_analysis["monotonicity"] = {
        "monotonic_paradigms": f"{n_mono}/{n_total}",
        "interpretation": "Confidence-ordered lists are monotonic if confidences never increase with position.",
    }

    # Print summary
    print("=== Confidence Calibration Analysis ===\n")
    print(f"Total assumptions across all paradigms: {len(all_confs)}")
    print(f"Overall confidence range: [{min(all_confs):.2f}, {max(all_confs):.2f}], "
          f"mean {sum(all_confs)/len(all_confs):.3f}")
    print(f"Confidence-order monotonic in {n_mono}/{n_total} paradigms")

    if "summary" in all_analysis:
        s = all_analysis["summary"]
        print(f"\nHit analysis (successes only):")
        print(f"  avg top-1 confidence: {s['avg_top_confidence']}")
        print(f"  avg confidence at target rank: {s['avg_confidence_at_hit_rank']}")
        print(f"  gap (top - hit): {s['confidence_gap_top_to_hit']}")

    if "hit_confidence_analysis" in all_analysis:
        print(f"\nPer-success details:")
        for h in all_analysis["hit_confidence_analysis"]:
            print(f"  {h['paradigm']:12s}: rank {h['rank']:2d}  conf@rank={h['confidence_at_rank']}  top_conf={h['top_confidence']}")

    out_path = PROJECT_ROOT / "experiments" / "calibration" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
