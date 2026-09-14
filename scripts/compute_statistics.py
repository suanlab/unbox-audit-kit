#!/usr/bin/env python3
"""Compute statistical tests on existing experiment results.

Uses ONLY stdlib — no scipy, numpy, or external packages.
"""

import json
import math
import os
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CATEGORIES = ["transformer", "diffusion", "icl", "vit"]

OUTPUT_PATH = BASE / "evidence" / "statistical_analysis.json"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def log_factorial(n: int) -> float:
    """Stirling-accurate log(n!) via summation."""
    return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0


def log_comb(n: int, k: int) -> float:
    """log(C(n, k))."""
    if k < 0 or k > n:
        return float("-inf")
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


def hypergeometric_pmf(k: int, N: int, K: int, n: int) -> float:
    """P(X = k) for hypergeometric distribution.
    N = population, K = successes in population, n = draws, k = observed successes.
    """
    log_p = log_comb(K, k) + log_comb(N - K, n - k) - log_comb(N, n)
    return math.exp(log_p)


def fisher_exact_one_sided(table):
    """One-sided Fisher's exact test for 2x2 contingency table.

    table = [[a, b], [c, d]]
    Tests if the odds ratio > 1 (i.e., method in row 0 is better).
    Returns p-value = P(X >= a) under the null.
    """
    a, b = table[0]
    c, d = table[1]
    N = a + b + c + d
    K = a + c  # total successes (column 0)
    n = a + b  # total in row 0

    # P(X >= a) where X ~ Hypergeometric(N, K, n)
    p_value = 0.0
    for x in range(a, min(K, n) + 1):
        p_value += hypergeometric_pmf(x, N, K, n)

    return p_value


def fisher_exact_two_sided(table):
    """Two-sided Fisher's exact test.

    Sum probabilities of all tables as extreme or more extreme than observed.
    """
    a, b = table[0]
    c, d = table[1]
    N = a + b + c + d
    K = a + c
    n = a + b

    p_observed = hypergeometric_pmf(a, N, K, n)

    p_value = 0.0
    for x in range(max(0, n - (N - K)), min(K, n) + 1):
        p_x = hypergeometric_pmf(x, N, K, n)
        if p_x <= p_observed + 1e-12:
            p_value += p_x

    return min(p_value, 1.0)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values, ddof=1):
    if len(values) <= ddof:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - ddof))


def cohens_d_paired(x, y):
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    diffs = [a - b for a, b in zip(x, y)]
    d_mean = mean(diffs)
    d_std = std(diffs, ddof=1)
    if d_std == 0:
        return float("inf") if d_mean != 0 else 0.0
    return d_mean / d_std


def cohens_d_independent(x, y):
    """Cohen's d for independent samples using pooled std."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0
    m1, m2 = mean(x), mean(y)
    s1, s2 = std(x, ddof=1), std(y, ddof=1)
    # Pooled std
    if n1 + n2 - 2 <= 0:
        return 0.0
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return float("inf") if m1 != m2 else 0.0
    return (m1 - m2) / sp


def permutation_test(x, y, n_perms=10000, seed=42):
    """Two-sided permutation test on difference of means."""
    random.seed(seed)
    observed_diff = mean(x) - mean(y)
    combined = list(x) + list(y)
    n_x = len(x)
    count_extreme = 0

    for _ in range(n_perms):
        random.shuffle(combined)
        perm_x = combined[:n_x]
        perm_y = combined[n_x:]
        perm_diff = mean(perm_x) - mean(perm_y)
        if abs(perm_diff) >= abs(observed_diff) - 1e-12:
            count_extreme += 1

    return count_extreme / n_perms


def power_analysis_two_proportions(n, alpha=0.05, power=0.80):
    """Approximate minimum detectable effect size for two-sample proportion test.

    Uses normal approximation: for given n per group, alpha, power,
    the detectable difference in proportions.

    h = (z_alpha + z_beta) / sqrt(n)  where h is Cohen's h.
    Then delta_p ~ h (approximately, for proportions around 0.5).
    """
    # z-values for common alpha and power
    # z_alpha/2 for two-sided test
    z_alpha = {0.01: 2.576, 0.05: 1.960, 0.10: 1.645}.get(alpha, 1.960)
    z_beta = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}.get(power, 0.842)

    h = (z_alpha + z_beta) / math.sqrt(n)

    return {
        "n_per_group": n,
        "alpha": alpha,
        "power": power,
        "cohens_h": round(h, 4),
        "interpretation": interpret_h(h),
        "note": (
            f"With N={n} per group, need Cohen's h >= {h:.2f} to detect at "
            f"alpha={alpha}, power={power}. This is a "
            f"{'very large' if h > 1.5 else 'large' if h > 0.8 else 'medium' if h > 0.5 else 'small'} "
            f"effect size."
        ),
    }


def interpret_h(h):
    if h >= 1.5:
        return "very large (only very large effects detectable)"
    elif h >= 0.8:
        return "large"
    elif h >= 0.5:
        return "medium"
    else:
        return "small"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_threshold_sweep():
    path = BASE / "evidence" / "threshold_sweep.json"
    return json.loads(path.read_text())


def load_ablation_summary():
    path = BASE / "experiments" / "ablation" / "summary.json"
    return json.loads(path.read_text())


def get_best_similarities_from_ablation():
    """Get best_similarity per category from ablation data."""
    abl = load_ablation_summary()
    results = {}
    for cond, data in abl["conditions"].items():
        results[cond] = {}
        for cat, cat_data in data["categories"].items():
            results[cond][cat] = cat_data["best_similarity"]
    return results


def get_recall_at_threshold(sweep_data, method, threshold="0.65"):
    """Get binary hit/miss per category from threshold sweep."""
    results = {}
    for cat in CATEGORIES:
        cat_data = sweep_data.get(method, {}).get(cat, {})
        recall = cat_data.get(threshold, 0.0)
        results[cat] = int(recall >= 0.5)  # 1 = hit, 0 = miss
    return results


# ---------------------------------------------------------------------------
# Main analyses
# ---------------------------------------------------------------------------

def analysis_1_fisher_unbox_vs_vanilla():
    """Fisher's exact test: Unbox vs Vanilla LLM at threshold 0.65."""
    sweep = load_threshold_sweep()

    unbox_hits = get_recall_at_threshold(sweep, "unbox", "0.65")
    vanilla_hits = get_recall_at_threshold(sweep, "vanilla_llm", "0.65")

    unbox_success = sum(unbox_hits.values())
    unbox_fail = len(unbox_hits) - unbox_success
    vanilla_success = sum(vanilla_hits.values())
    vanilla_fail = len(vanilla_hits) - vanilla_success

    # 2x2 table: [[unbox_hit, unbox_miss], [vanilla_hit, vanilla_miss]]
    table = [[unbox_success, unbox_fail], [vanilla_success, vanilla_fail]]

    p_one = fisher_exact_one_sided(table)
    p_two = fisher_exact_two_sided(table)

    return {
        "description": "Fisher's exact test: Unbox vs Vanilla LLM recall at threshold 0.65",
        "contingency_table": {
            "headers": ["Hit", "Miss"],
            "unbox": [unbox_success, unbox_fail],
            "vanilla_llm": [vanilla_success, vanilla_fail],
        },
        "per_category": {
            "unbox": unbox_hits,
            "vanilla_llm": vanilla_hits,
        },
        "p_value_one_sided": round(p_one, 6),
        "p_value_two_sided": round(p_two, 6),
        "significant_at_005": p_one < 0.05,
        "significant_at_010": p_one < 0.10,
        "interpretation": (
            f"One-sided p = {p_one:.4f}. "
            + ("Significant at alpha=0.05." if p_one < 0.05
               else "Not significant at alpha=0.05."
                    + (" Significant at alpha=0.10." if p_one < 0.10 else ""))
        ),
    }


def analysis_2_ablation_significance():
    """Fisher's exact tests on ablation conditions."""
    abl = load_ablation_summary()
    results = {}

    # Ablation uses 3 categories (transformer, diffusion, vit — no ICL)
    abl_cats = ["transformer", "diffusion", "vit"]

    def count_hits(cond, threshold=0.65):
        cats = abl["conditions"][cond]["categories"]
        hits = sum(1 for c in abl_cats if cats[c]["best_similarity"] >= threshold)
        return hits

    for comp_name, cond1, cond2 in [("B_vs_A", "B", "A"), ("B_vs_D", "B", "D")]:
        h1 = count_hits(cond1)
        m1 = len(abl_cats) - h1
        h2 = count_hits(cond2)
        m2 = len(abl_cats) - h2

        table = [[h1, m1], [h2, m2]]
        p_one = fisher_exact_one_sided(table)
        p_two = fisher_exact_two_sided(table)

        label1 = abl["conditions"][cond1]["label"]
        label2 = abl["conditions"][cond2]["label"]

        per_cat = {}
        for cat in abl_cats:
            per_cat[cat] = {
                cond1: abl["conditions"][cond1]["categories"][cat]["best_similarity"],
                cond2: abl["conditions"][cond2]["categories"][cat]["best_similarity"],
            }

        results[comp_name] = {
            "description": f"{label1} ({cond1}) vs {label2} ({cond2})",
            "n_categories": len(abl_cats),
            "contingency_table": {
                cond1: [h1, m1],
                cond2: [h2, m2],
            },
            "per_category_similarity": per_cat,
            "p_value_one_sided": round(p_one, 6),
            "p_value_two_sided": round(p_two, 6),
            "significant_at_005": p_one < 0.05,
        }

    return results


def analysis_3_effect_size():
    """Cohen's d on best_similarity scores: Unbox vs Vanilla."""
    abl = load_ablation_summary()
    abl_cats = ["transformer", "diffusion", "vit"]

    # Unbox (condition B = prompt-only with temporal corpus, best performer)
    # Using condition B as "Unbox" since it matches the main pipeline's field-wide prompt
    unbox_sims = [abl["conditions"]["B"]["categories"][c]["best_similarity"] for c in abl_cats]
    vanilla_sims = [abl["conditions"]["A"]["categories"][c]["best_similarity"] for c in abl_cats]

    d_paired = cohens_d_paired(unbox_sims, vanilla_sims)
    d_independent = cohens_d_independent(unbox_sims, vanilla_sims)

    # Also compute from threshold sweep if available — to get all 4 categories
    # We can infer best_similarity from the highest threshold where recall=1
    sweep = load_threshold_sweep()
    thresholds = sorted([float(t) for t in sweep["thresholds"]])

    def infer_best_similarity_range(method):
        """Infer approximate best_similarity from threshold sweep."""
        results = {}
        for cat in CATEGORIES:
            cat_data = sweep[method][cat]
            highest_hit = 0.0
            for t in thresholds:
                t_str = str(t)
                if cat_data.get(t_str, 0.0) >= 0.5:
                    highest_hit = t
            results[cat] = highest_hit
        return results

    unbox_inferred = infer_best_similarity_range("unbox")
    vanilla_inferred = infer_best_similarity_range("vanilla_llm")

    unbox_4 = [unbox_inferred[c] for c in CATEGORIES]
    vanilla_4 = [vanilla_inferred[c] for c in CATEGORIES]

    d_paired_4 = cohens_d_paired(unbox_4, vanilla_4)

    return {
        "description": "Cohen's d effect size on best_similarity scores",
        "from_ablation_3_categories": {
            "categories": abl_cats,
            "unbox_B_similarities": unbox_sims,
            "baseline_A_similarities": vanilla_sims,
            "mean_unbox": round(mean(unbox_sims), 4),
            "mean_baseline": round(mean(vanilla_sims), 4),
            "cohens_d_paired": round(d_paired, 4),
            "cohens_d_independent": round(d_independent, 4),
            "interpretation": (
                "very large" if abs(d_paired) > 1.2 else
                "large" if abs(d_paired) > 0.8 else
                "medium" if abs(d_paired) > 0.5 else "small"
            ),
        },
        "from_threshold_sweep_4_categories": {
            "categories": CATEGORIES,
            "unbox_inferred_thresholds": unbox_4,
            "vanilla_inferred_thresholds": vanilla_4,
            "mean_unbox": round(mean(unbox_4), 4),
            "mean_vanilla": round(mean(vanilla_4), 4),
            "cohens_d_paired": round(d_paired_4, 4),
            "note": "Inferred from highest threshold with recall=1; lower bound on true similarity",
        },
    }


def analysis_4_power():
    """Power analysis for N=4 categories."""
    result = power_analysis_two_proportions(n=4, alpha=0.05, power=0.80)

    # Also compute for one-sided Fisher's exact:
    # What's the minimum 2x2 table that gives p < 0.05 with n=4?
    fisher_min = {}
    for hits_a in range(5):
        for hits_b in range(5):
            if hits_a + hits_b == 0:
                continue
            table = [[hits_a, 4 - hits_a], [hits_b, 4 - hits_b]]
            p = fisher_exact_one_sided(table)
            key = f"{hits_a}v{hits_b}"
            fisher_min[key] = round(p, 4)

    result["fisher_exact_p_values_for_4_categories"] = fisher_min
    result["minimum_for_significance"] = (
        "With N=4, Fisher's exact one-sided test requires 4-vs-0 or 3-vs-0 to approach significance. "
        "3-vs-1 gives p=0.2429 (not significant). 4-vs-0 gives p=0.0143 (significant). "
        "This confirms the study is underpowered for moderate effects."
    )

    return result


def analysis_5_permutation():
    """Permutation test on best_similarity scores."""
    abl = load_ablation_summary()
    abl_cats = ["transformer", "diffusion", "vit"]

    unbox_sims = [abl["conditions"]["B"]["categories"][c]["best_similarity"] for c in abl_cats]
    baseline_sims = [abl["conditions"]["A"]["categories"][c]["best_similarity"] for c in abl_cats]

    p_perm = permutation_test(unbox_sims, baseline_sims, n_perms=10000, seed=42)

    # Also do with 4-category inferred data
    sweep = load_threshold_sweep()
    thresholds = sorted([float(t) for t in sweep["thresholds"]])

    def infer_sims(method):
        results = []
        for cat in CATEGORIES:
            cat_data = sweep[method][cat]
            highest_hit = 0.0
            for t in thresholds:
                if cat_data.get(str(t), 0.0) >= 0.5:
                    highest_hit = t
            results.append(highest_hit)
        return results

    unbox_4 = infer_sims("unbox")
    vanilla_4 = infer_sims("vanilla_llm")
    p_perm_4 = permutation_test(unbox_4, vanilla_4, n_perms=10000, seed=42)

    return {
        "description": "Permutation test (10000 permutations) on best_similarity scores",
        "from_ablation_3_categories": {
            "categories": abl_cats,
            "unbox_B": unbox_sims,
            "baseline_A": baseline_sims,
            "observed_mean_diff": round(mean(unbox_sims) - mean(baseline_sims), 4),
            "p_value": round(p_perm, 4),
            "significant_at_005": p_perm < 0.05,
        },
        "from_threshold_sweep_4_categories": {
            "categories": CATEGORIES,
            "unbox_inferred": unbox_4,
            "vanilla_inferred": vanilla_4,
            "observed_mean_diff": round(mean(unbox_4) - mean(vanilla_4), 4),
            "p_value": round(p_perm_4, 4),
            "significant_at_005": p_perm_4 < 0.05,
        },
    }


def main():
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    all_results = {}

    # 1. Fisher's exact: Unbox vs Vanilla
    print("\n--- 1. Fisher's Exact Test: Unbox vs Vanilla LLM ---")
    r1 = analysis_1_fisher_unbox_vs_vanilla()
    all_results["fisher_unbox_vs_vanilla"] = r1
    t = r1["contingency_table"]
    print(f"  Unbox:      {t['unbox'][0]} hits, {t['unbox'][1]} misses")
    print(f"  Vanilla LLM: {t['vanilla_llm'][0]} hits, {t['vanilla_llm'][1]} misses")
    print(f"  One-sided p = {r1['p_value_one_sided']:.4f}")
    print(f"  Two-sided p = {r1['p_value_two_sided']:.4f}")
    print(f"  {r1['interpretation']}")

    # 2. Ablation significance
    print("\n--- 2. Ablation Fisher's Exact Tests ---")
    r2 = analysis_2_ablation_significance()
    all_results["ablation_significance"] = r2
    for comp_name, comp in r2.items():
        print(f"  {comp['description']}:")
        for cond, vals in comp["contingency_table"].items():
            print(f"    {cond}: {vals[0]} hits, {vals[1]} misses")
        print(f"    One-sided p = {comp['p_value_one_sided']:.4f}")

    # 3. Effect size
    print("\n--- 3. Effect Size (Cohen's d) ---")
    r3 = analysis_3_effect_size()
    all_results["effect_size"] = r3
    abl3 = r3["from_ablation_3_categories"]
    print(f"  Ablation (3 cats): Unbox mean={abl3['mean_unbox']:.4f}, "
          f"Baseline mean={abl3['mean_baseline']:.4f}")
    print(f"    Cohen's d (paired) = {abl3['cohens_d_paired']:.4f} ({abl3['interpretation']})")
    sw4 = r3["from_threshold_sweep_4_categories"]
    print(f"  Threshold sweep (4 cats): Unbox mean={sw4['mean_unbox']:.4f}, "
          f"Vanilla mean={sw4['mean_vanilla']:.4f}")
    print(f"    Cohen's d (paired) = {sw4['cohens_d_paired']:.4f}")

    # 4. Power analysis
    print("\n--- 4. Power Analysis ---")
    r4 = analysis_4_power()
    all_results["power_analysis"] = r4
    print(f"  N=4 per group, alpha=0.05, power=0.80")
    print(f"  Required Cohen's h = {r4['cohens_h']:.4f} ({r4['interpretation']})")
    print(f"  {r4['note']}")
    # Show key Fisher p-values
    fp = r4["fisher_exact_p_values_for_4_categories"]
    print(f"  Fisher p-values with N=4: 4v0={fp.get('4v0','N/A')}, "
          f"3v0={fp.get('3v0','N/A')}, 3v1={fp.get('3v1','N/A')}, "
          f"4v1={fp.get('4v1','N/A')}")

    # 5. Permutation test
    print("\n--- 5. Permutation Test ---")
    r5 = analysis_5_permutation()
    all_results["permutation_test"] = r5
    a3 = r5["from_ablation_3_categories"]
    print(f"  Ablation (3 cats): mean diff = {a3['observed_mean_diff']:.4f}, "
          f"p = {a3['p_value']:.4f}")
    a4 = r5["from_threshold_sweep_4_categories"]
    print(f"  Threshold sweep (4 cats): mean diff = {a4['observed_mean_diff']:.4f}, "
          f"p = {a4['p_value']:.4f}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    os.chdir(BASE)
    main()
