#!/usr/bin/env python3
"""Analyze diversity differences between SCAM and Unbox assumption lists.

Uses ONLY existing data — no API calls.
"""

import json
import math
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CATEGORIES = ["transformer", "diffusion", "icl", "vit"]

SCAM_DIR = BASE / "experiments" / "baselines" / "scam"
UNBOX_DIR = BASE / "experiments" / "retrospective_full"
OUTPUT_PATH = BASE / "evidence" / "diversity_analysis.json"


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return normalize(text).split()


def bigrams(tokens: list) -> list:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def load_scam_top20(category: str) -> list:
    path = SCAM_DIR / f"{category}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("outputs", {}).get("top_20", [])


def load_unbox_top20(category: str) -> list:
    path = UNBOX_DIR / f"{category}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    items = data.get("outputs", {}).get("top_breakable_assumptions", [])
    # Extract assumption text from dicts
    result = []
    for item in items[:20]:
        if isinstance(item, dict):
            result.append(item.get("assumption", item.get("normalized_assumption", "")))
        else:
            result.append(str(item))
    return result


def compute_jaccard_per_category():
    results = {}
    for cat in CATEGORIES:
        scam = {normalize(a) for a in load_scam_top20(cat)}
        unbox = {normalize(a) for a in load_unbox_top20(cat)}
        j = jaccard(scam, unbox)
        results[cat] = {
            "jaccard_similarity": round(j, 4),
            "scam_count": len(scam),
            "unbox_count": len(unbox),
            "intersection_size": len(scam & unbox),
            "union_size": len(scam | unbox),
        }
    # Average
    vals = [r["jaccard_similarity"] for r in results.values()]
    results["average"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return results


def compute_lexical_diversity():
    """Count unique unigrams and bigrams across all assumptions per method."""
    methods = {"scam": load_scam_top20, "unbox": load_unbox_top20}
    results = {}

    for method_name, loader in methods.items():
        all_unigrams = Counter()
        all_bigrams = Counter()
        total_assumptions = 0

        for cat in CATEGORIES:
            assumptions = loader(cat)
            total_assumptions += len(assumptions)
            for a in assumptions:
                tokens = tokenize(a)
                all_unigrams.update(tokens)
                all_bigrams.update(bigrams(tokens))

        results[method_name] = {
            "total_assumptions": total_assumptions,
            "unique_unigrams": len(all_unigrams),
            "unique_bigrams": len(all_bigrams),
            "total_unigram_tokens": sum(all_unigrams.values()),
            "total_bigram_tokens": sum(all_bigrams.values()),
            "type_token_ratio_unigrams": round(
                len(all_unigrams) / max(sum(all_unigrams.values()), 1), 4
            ),
            "type_token_ratio_bigrams": round(
                len(all_bigrams) / max(sum(all_bigrams.values()), 1), 4
            ),
        }

    # Comparison
    results["comparison"] = {
        "more_diverse_unigrams": (
            "unbox" if results["unbox"]["unique_unigrams"] > results["scam"]["unique_unigrams"]
            else "scam" if results["scam"]["unique_unigrams"] > results["unbox"]["unique_unigrams"]
            else "tie"
        ),
        "more_diverse_bigrams": (
            "unbox" if results["unbox"]["unique_bigrams"] > results["scam"]["unique_bigrams"]
            else "scam" if results["scam"]["unique_bigrams"] > results["unbox"]["unique_bigrams"]
            else "tie"
        ),
        "unigram_ratio": round(
            results["unbox"]["unique_unigrams"] / max(results["scam"]["unique_unigrams"], 1), 4
        ),
        "bigram_ratio": round(
            results["unbox"]["unique_bigrams"] / max(results["scam"]["unique_bigrams"], 1), 4
        ),
    }
    return results


def compute_category_distribution():
    """Check if assumptions have taxonomy category labels and compute distribution."""
    results = {"scam": {}, "unbox": {}}

    # SCAM assumptions are plain strings — no category labels
    results["scam"]["note"] = "SCAM assumptions are plain strings without category labels"

    # Unbox assumptions may have fields from extraction
    unbox_cats = Counter()
    unbox_transformations = Counter()
    total = 0
    for cat in CATEGORIES:
        path = UNBOX_DIR / f"{cat}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        items = data.get("outputs", {}).get("top_breakable_assumptions", [])
        for item in items:
            if isinstance(item, dict):
                total += 1
                # Check for category/taxonomy field
                for key in ["category", "taxonomy", "type", "taxonomy_category"]:
                    if key in item:
                        unbox_cats[item[key]] += 1
                # Track transformation types
                if "best_transformation" in item:
                    unbox_transformations[item["best_transformation"]] += 1

    results["unbox"]["total_items"] = total
    results["unbox"]["transformation_distribution"] = dict(
        sorted(unbox_transformations.items(), key=lambda x: -x[1])
    )
    if unbox_cats:
        results["unbox"]["category_distribution"] = dict(
            sorted(unbox_cats.items(), key=lambda x: -x[1])
        )
    else:
        results["unbox"]["note"] = (
            "No explicit taxonomy category labels found in top_breakable_assumptions. "
            "Transformation distribution available instead."
        )

    return results


def compute_assumption_quality():
    """For each category, compute extraction stats from the retrospective_full data."""
    results = {}

    for cat in CATEGORIES:
        path = UNBOX_DIR / f"{cat}.json"
        if not path.exists():
            results[cat] = {"error": "file not found"}
            continue

        data = json.loads(path.read_text())
        config = data.get("config", {})
        inputs = data.get("inputs", {})
        outputs = data.get("outputs", {})

        num_papers = inputs.get("num_papers", config.get("max_papers", 0))
        max_assumptions = config.get("max_assumptions_per_paper", 5)
        raw_upper_bound = num_papers * max_assumptions
        num_unique = outputs.get("num_unique_assumptions", 0)
        num_ranked = outputs.get("num_ranked_hypotheses", 0)

        redundancy = 1.0 - (num_unique / raw_upper_bound) if raw_upper_bound > 0 else 0.0

        results[cat] = {
            "num_papers": num_papers,
            "max_assumptions_per_paper": max_assumptions,
            "raw_extraction_upper_bound": raw_upper_bound,
            "num_unique_assumptions": num_unique,
            "num_ranked_hypotheses": num_ranked,
            "redundancy_rate": round(redundancy, 4),
            "dedup_ratio": round(num_unique / raw_upper_bound, 4) if raw_upper_bound > 0 else 0.0,
        }

    # Averages
    cats_with_data = [r for r in results.values() if "error" not in r]
    if cats_with_data:
        results["average"] = {
            "avg_redundancy_rate": round(
                sum(r["redundancy_rate"] for r in cats_with_data) / len(cats_with_data), 4
            ),
            "avg_unique_assumptions": round(
                sum(r["num_unique_assumptions"] for r in cats_with_data) / len(cats_with_data), 1
            ),
            "total_unique_assumptions": sum(r["num_unique_assumptions"] for r in cats_with_data),
            "total_raw_upper_bound": sum(r["raw_extraction_upper_bound"] for r in cats_with_data),
        }

    return results


def main():
    print("=" * 60)
    print("DIVERSITY ANALYSIS: SCAM vs Unbox")
    print("=" * 60)

    all_results = {}

    # 1. Jaccard overlap
    print("\n--- 1. Jaccard Overlap (normalized text) ---")
    jaccard_results = compute_jaccard_per_category()
    all_results["jaccard_overlap"] = jaccard_results
    for cat in CATEGORIES:
        r = jaccard_results[cat]
        print(f"  {cat:>12}: Jaccard = {r['jaccard_similarity']:.4f}  "
              f"(intersection={r['intersection_size']}, union={r['union_size']})")
    print(f"  {'average':>12}: Jaccard = {jaccard_results['average']:.4f}")

    # 2. Lexical diversity
    print("\n--- 2. Lexical Diversity ---")
    lex_results = compute_lexical_diversity()
    all_results["lexical_diversity"] = lex_results
    for method in ["scam", "unbox"]:
        r = lex_results[method]
        print(f"  {method.upper():>8}: {r['total_assumptions']} assumptions, "
              f"{r['unique_unigrams']} unique unigrams (TTR={r['type_token_ratio_unigrams']:.4f}), "
              f"{r['unique_bigrams']} unique bigrams (TTR={r['type_token_ratio_bigrams']:.4f})")
    comp = lex_results["comparison"]
    print(f"  Winner (unigrams): {comp['more_diverse_unigrams']} "
          f"(ratio Unbox/SCAM = {comp['unigram_ratio']:.2f}x)")
    print(f"  Winner (bigrams):  {comp['more_diverse_bigrams']} "
          f"(ratio Unbox/SCAM = {comp['bigram_ratio']:.2f}x)")

    # 3. Category distribution
    print("\n--- 3. Category/Transformation Distribution ---")
    cat_dist = compute_category_distribution()
    all_results["category_distribution"] = cat_dist
    if "transformation_distribution" in cat_dist["unbox"]:
        print("  Unbox transformation types:")
        for t, count in cat_dist["unbox"]["transformation_distribution"].items():
            print(f"    {t:>12}: {count}")
    if "note" in cat_dist["unbox"]:
        print(f"  Note: {cat_dist['unbox']['note']}")
    if "note" in cat_dist["scam"]:
        print(f"  Note: {cat_dist['scam']['note']}")

    # 4. Assumption quality
    print("\n--- 4. Assumption Quality (Unbox) ---")
    quality = compute_assumption_quality()
    all_results["assumption_quality"] = quality
    for cat in CATEGORIES:
        if cat in quality and "error" not in quality[cat]:
            r = quality[cat]
            print(f"  {cat:>12}: {r['num_papers']} papers x {r['max_assumptions_per_paper']} = "
                  f"{r['raw_extraction_upper_bound']} raw -> "
                  f"{r['num_unique_assumptions']} unique "
                  f"(redundancy={r['redundancy_rate']:.1%})")
    if "average" in quality:
        avg = quality["average"]
        print(f"  {'average':>12}: redundancy={avg['avg_redundancy_rate']:.1%}, "
              f"total unique={avg['total_unique_assumptions']}/{avg['total_raw_upper_bound']}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    os.chdir(BASE)
    main()
