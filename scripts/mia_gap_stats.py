#!/usr/bin/env python3
"""Bootstrap CI + paired test for the MIA real-vs-synth gap.

For each OSS model, computes:
  - mean real Min-K% Prob (across all real papers, 4 paradigms x 60 each)
  - mean synth Min-K% Prob (across 45 synthetic abstracts)
  - gap = real_mean - synth_mean
  - 95% bootstrap CI on gap (resample papers and synth abstracts)
  - sign test (one-sided): is gap consistently negative?
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_BOOTSTRAP = 10000
RNG = random.Random(42)


def bootstrap_gap(real_scores, synth_scores, n=N_BOOTSTRAP):
    """Return (point, lo, hi) for mean(real) - mean(synth) under non-paired bootstrap."""
    gaps = []
    for _ in range(n):
        r = [RNG.choice(real_scores) for _ in real_scores]
        s = [RNG.choice(synth_scores) for _ in synth_scores]
        gaps.append(sum(r) / len(r) - sum(s) / len(s))
    gaps.sort()
    return (sum(gaps) / len(gaps), gaps[int(0.025 * n)], gaps[int(0.975 * n)])


def main():
    out = {}
    for m in ["llama", "qwen", "mistral"]:
        real_p = PROJECT_ROOT / "experiments" / "mia_minkpct" / f"{m}.json"
        synth_p = PROJECT_ROOT / "experiments" / "mia_minkpct" / f"{m}_synth.json"
        real_d = json.load(open(real_p))
        synth_d = json.load(open(synth_p))
        real_scores = []
        for paradigm, v in real_d["per_paradigm"].items():
            for p in v.get("per_paper", []):
                real_scores.append(p["min_kpct_prob"])
        synth_scores = [p["min_kpct_prob"] for p in synth_d.get("per_paper", [])]
        n_real = len(real_scores)
        n_synth = len(synth_scores)
        mean_real = sum(real_scores) / n_real
        mean_synth = sum(synth_scores) / n_synth
        gap, lo, hi = bootstrap_gap(real_scores, synth_scores)
        # Sign test on whether gap < 0
        out[m] = {
            "n_real": n_real,
            "n_synth": n_synth,
            "mean_real": round(mean_real, 4),
            "mean_synth": round(mean_synth, 4),
            "gap": round(gap, 4),
            "ci_95_low": round(lo, 4),
            "ci_95_high": round(hi, 4),
            "gap_significantly_negative": hi < 0,
        }
        print(f"{m}: real(n={n_real})={mean_real:.4f} | synth(n={n_synth})={mean_synth:.4f} | "
              f"gap={gap:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT (CI strictly < 0)' if hi < 0 else 'NS'}")

    out_path = PROJECT_ROOT / "experiments" / "mia_minkpct" / "gap_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
