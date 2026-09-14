#!/usr/bin/env python3
"""Analyze the expert target-validation study (camera-ready deliverable).

Companion to scripts/build_target_validity_instrument.py. Where the earlier
10-rater study (scripts/compute_human_kappa.py) validated the similarity
*matcher*, this validates the *targets themselves*: are the author-defined
assumptions accurate descriptions of pre-shift field consensus, and did the
breakthrough actually overturn them?

Pre-registered analysis (frozen in annotation/expert_targets/freeze.json):
  * Primary    per-target median q1,q2; validated iff both medians >= 4
  * Reliability Krippendorff alpha (interval) on q1,q2; mean pairwise Cohen's
                kappa on the binarized validated-flag
  * Discriminant real vs decoy: Mann-Whitney U + 95% bootstrap CI on the mean
                gap (experts should rate decoys low; expect CI strictly > 0)

Agreement estimators are imported from compute_human_kappa.py so both studies
use identical implementations.

Inputs (--in-dir, default annotation/expert_targets/):
  instrument_master.csv   item_id, paradigm, type, candidate_assumption
  *_complete.csv          one per rater; item_id, q1_accuracy_1to5,
                          q2_overturned_1to5, q3_alternative_phrasing, q4_...

Output:
  experiments/target_validity/results.json  (+ human-readable summary on stdout)

Usage:
    python scripts/compute_target_validity.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_human_kappa import (  # noqa: E402  (path set above)
    cohens_kappa_binary,
    krippendorff_alpha_interval,
)

VALIDATED_MIN = 4  # pre-registered: q1>=4 and q2>=4


def load_master(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig") as fh:
        return {row["item_id"]: row for row in csv.DictReader(fh)}


def load_raters(in_dir: Path) -> dict[str, dict[str, dict[str, int]]]:
    """-> {rater: {item_id: {'q1': int, 'q2': int}}}, plus free-text captured separately."""
    raters: dict[str, dict[str, dict[str, int]]] = {}
    for path in sorted(in_dir.glob("*_complete.csv")):
        name = path.stem.removesuffix("_complete")
        scores: dict[str, dict[str, int]] = {}
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                item = row.get("item_id", "").strip()
                if not item:
                    continue
                rec: dict[str, int] = {}
                for key, col in (("q1", "q1_accuracy_1to5"), ("q2", "q2_overturned_1to5")):
                    raw = (row.get(col) or "").strip()
                    if raw:
                        try:
                            rec[key] = int(raw)
                        except ValueError:
                            print(f"WARN: non-int {col} in {path.name} item {item}: {raw!r}",
                                  file=sys.stderr)
                if rec:
                    scores[item] = rec
        raters[name] = scores
    return raters


def load_alternatives(in_dir: Path) -> dict[str, list[str]]:
    """Harvest Q3 free-text alternative phrasings, keyed by item_id."""
    alts: dict[str, list[str]] = defaultdict(list)
    for path in sorted(in_dir.glob("*_complete.csv")):
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                text = (row.get("q3_alternative_phrasing") or "").strip()
                if text:
                    alts[row.get("item_id", "").strip()].append(text)
    return dict(alts)


def mann_whitney_u(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    """Two-sided Mann-Whitney U with tie-corrected normal approximation.

    Returns (U, p). Normal approximation is adequate at the sample sizes here
    (10 real vs 10 decoy x n_raters); exact enumeration is unnecessary.
    """
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return None, None
    combined = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    tie_term = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t**3 - t
        i = j + 1

    r1 = sum(rank for rank, (_v, grp) in zip(ranks, combined) if grp == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = u1

    n = n1 + n2
    mu = n1 * n2 / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return u, None
    z = (u - mu) / var**0.5
    # two-sided p via erf-based normal CDF
    import math

    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, min(1.0, p)


def bootstrap_gap_ci(xs: list[float], ys: list[float], seed: int = 42,
                     iters: int = 10000) -> tuple[float, float, float] | None:
    """Mean(xs) - mean(ys) with percentile 95% CI, resampling groups independently."""
    if not xs or not ys:
        return None
    rng = random.Random(seed)
    point = statistics.fmean(xs) - statistics.fmean(ys)
    diffs = []
    for _ in range(iters):
        bx = statistics.fmean(rng.choices(xs, k=len(xs)))
        by = statistics.fmean(rng.choices(ys, k=len(ys)))
        diffs.append(bx - by)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return point, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="annotation/expert_targets")
    ap.add_argument("--out", default="experiments/target_validity/results.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = PROJECT_ROOT / args.in_dir
    master_path = in_dir / "instrument_master.csv"
    if not master_path.exists():
        print(f"ERROR: {master_path} not found. Run build_target_validity_instrument.py first.",
              file=sys.stderr)
        return 1
    master = load_master(master_path)
    raters = load_raters(in_dir)
    if not raters:
        print(f"ERROR: no *_complete.csv rater files in {in_dir}. "
              "Collect expert ratings before running this analysis.", file=sys.stderr)
        return 1

    real_items = [i for i, m in master.items() if m["type"] == "real"]
    decoy_items = [i for i, m in master.items() if m["type"] == "decoy"]
    att_items = [i for i, m in master.items() if m["type"] == "attention"]
    rater_names = sorted(raters)

    # ---- primary: per-target medians -------------------------------------
    per_target = {}
    for item in real_items:
        para = master[item]["paradigm"]
        q1 = [raters[r][item]["q1"] for r in rater_names
              if item in raters[r] and "q1" in raters[r][item]]
        q2 = [raters[r][item]["q2"] for r in rater_names
              if item in raters[r] and "q2" in raters[r][item]]
        if not q1 or not q2:
            continue
        m1, m2 = statistics.median(q1), statistics.median(q2)
        per_target[para] = {
            "item_id": item,
            "assumption": master[item]["candidate_assumption"],
            "n_ratings": len(q1),
            "median_q1_accuracy": m1,
            "median_q2_overturned": m2,
            "mean_q1": round(statistics.fmean(q1), 3),
            "mean_q2": round(statistics.fmean(q2), 3),
            "validated": bool(m1 >= VALIDATED_MIN and m2 >= VALIDATED_MIN),
        }
    n_validated = sum(v["validated"] for v in per_target.values())
    failed = [p for p, v in per_target.items() if not v["validated"]]

    # ---- reliability ------------------------------------------------------
    def matrix_for(q: str, items: list[str]) -> list[list[int | None]]:
        return [[raters[r].get(i, {}).get(q) for r in rater_names] for i in items]

    alpha_q1 = krippendorff_alpha_interval(matrix_for("q1", real_items + decoy_items))
    alpha_q2 = krippendorff_alpha_interval(matrix_for("q2", real_items + decoy_items))

    def validated_flags(rater: str) -> dict[str, int]:
        out = {}
        for item in real_items + decoy_items:
            rec = raters[rater].get(item, {})
            if "q1" in rec and "q2" in rec:
                out[item] = int(rec["q1"] >= VALIDATED_MIN and rec["q2"] >= VALIDATED_MIN)
        return out

    kappas = []
    for a in range(len(rater_names)):
        for b in range(a + 1, len(rater_names)):
            k, _n = cohens_kappa_binary(validated_flags(rater_names[a]),
                                        validated_flags(rater_names[b]))
            if k is not None:
                kappas.append(k)
    mean_kappa = round(statistics.fmean(kappas), 3) if kappas else None

    # ---- discriminant: real vs decoy -------------------------------------
    discriminant = {}
    for q in ("q1", "q2"):
        real_scores = [raters[r][i][q] for r in rater_names for i in real_items
                       if i in raters[r] and q in raters[r][i]]
        decoy_scores = [raters[r][i][q] for r in rater_names for i in decoy_items
                        if i in raters[r] and q in raters[r][i]]
        u, p = mann_whitney_u(real_scores, decoy_scores)
        ci = bootstrap_gap_ci(real_scores, decoy_scores, seed=args.seed)
        discriminant[q] = {
            "n_real": len(real_scores), "n_decoy": len(decoy_scores),
            "mean_real": round(statistics.fmean(real_scores), 3) if real_scores else None,
            "mean_decoy": round(statistics.fmean(decoy_scores), 3) if decoy_scores else None,
            "mann_whitney_u": u,
            "p_two_sided": round(p, 5) if p is not None else None,
            "bootstrap_gap": round(ci[0], 3) if ci else None,
            "bootstrap_ci95": [round(ci[1], 3), round(ci[2], 3)] if ci else None,
            "discriminates": bool(ci and ci[1] > 0),
        }

    # ---- attention checks -------------------------------------------------
    att_pass, att_total = 0, 0
    for item in att_items:
        expected_high = "high" in (master[item].get("expected") or "")
        for r in rater_names:
            rec = raters[r].get(item, {})
            if "q1" not in rec:
                continue
            att_total += 1
            ok = rec["q1"] >= 4 if expected_high else rec["q1"] <= 2
            att_pass += int(ok)

    results = {
        "study": "expert_target_validation",
        "n_raters": len(rater_names),
        "raters": rater_names,
        "n_real": len(real_items), "n_decoy": len(decoy_items),
        "validated_rule": f"median q1>={VALIDATED_MIN} and median q2>={VALIDATED_MIN}",
        "targets_validated": f"{n_validated}/{len(per_target)}",
        "targets_failed": failed,
        "per_target": per_target,
        "reliability": {
            "krippendorff_alpha_q1": round(alpha_q1, 3) if alpha_q1 is not None else None,
            "krippendorff_alpha_q2": round(alpha_q2, 3) if alpha_q2 is not None else None,
            "mean_pairwise_cohen_kappa_validated_flag": mean_kappa,
        },
        "discriminant": discriminant,
        "attention_checks": {
            "passed": att_pass, "total": att_total,
            "pass_rate": round(att_pass / att_total, 3) if att_total else None,
        },
        "alternative_phrasings": load_alternatives(in_dir),
    }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Expert target validation  ({len(rater_names)} raters)")
    print(f"  targets validated : {n_validated}/{len(per_target)}"
          + (f"   failed: {', '.join(failed)}" if failed else ""))
    print(f"  alpha q1/q2       : {results['reliability']['krippendorff_alpha_q1']} / "
          f"{results['reliability']['krippendorff_alpha_q2']}")
    print(f"  mean pairwise kappa: {mean_kappa}")
    for q, d in discriminant.items():
        print(f"  discriminant {q}    : real {d['mean_real']} vs decoy {d['mean_decoy']}, "
              f"gap {d['bootstrap_gap']} CI {d['bootstrap_ci95']} "
              f"{'OK' if d['discriminates'] else 'NOT SEPARATED'}")
    if att_total:
        print(f"  attention checks  : {att_pass}/{att_total}")
    try:
        shown = out_path.relative_to(PROJECT_ROOT)
    except ValueError:  # --out pointed outside the repo (e.g. a scratch dir)
        shown = out_path
    print(f"  -> {shown}")
    if failed:
        print("\nNOTE: re-run the canonical evaluator with the failed target(s) dropped to "
              "confirm the no-corpus vs wrong-corpus structure is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
