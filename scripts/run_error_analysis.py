#!/usr/bin/env python3
"""T1.4: Error analysis on 4 failure cases (ResNet, Word2Vec, Dropout, BERT).

For each failure, inspect top extracted assumptions and characterize
what went wrong.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_extraction(paradigm: str) -> list:
    """Load extracted assumptions in confidence order."""
    path = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt" / f"{paradigm}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("assumptions", [])


def main():
    gt_map = json.load(open(PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"))

    # Failure cases from main result
    failures = ["resnet", "word2vec", "dropout", "bert"]
    successes = ["transformer", "diffusion", "icl", "vit", "gan", "batchnorm"]

    analysis = {"failures": {}, "successes_summary": {}}

    # Failure analysis
    for paradigm in failures:
        assumptions = load_extraction(paradigm)
        gt = gt_map.get(paradigm, {})
        target = gt.get("broken_assumption", "")
        aliases = [target] + gt.get("aliases", [])

        top5 = assumptions[:5] if assumptions else []
        analysis["failures"][paradigm] = {
            "target": target,
            "aliases": aliases,
            "num_extracted": len(assumptions),
            "top_5": [
                {"rank": i + 1, "assumption": a["assumption"], "confidence": a.get("confidence", 0)}
                for i, a in enumerate(top5)
            ],
        }

    # Characterize failure patterns
    def count_pattern(assumptions: list, keywords: list) -> int:
        count = 0
        for a in assumptions:
            text = a.get("assumption", "").lower()
            if any(kw.lower() in text for kw in keywords):
                count += 1
        return count

    # For each failure, check if relevant keywords appear in top-10
    failure_keywords = {
        "resnet": ["shortcut", "skip", "sequential", "vanishing", "gradient flow"],
        "word2vec": ["symbolic", "hand-crafted", "feature", "linguistic", "knowledge base"],
        "dropout": ["active", "training", "all neurons", "random", "regularization"],
        "bert": ["left-to-right", "autoregressive", "unidirectional", "causal"],
    }

    keyword_analysis = {}
    for paradigm in failures:
        assumptions = load_extraction(paradigm)
        keywords = failure_keywords.get(paradigm, [])
        top10 = assumptions[:10]
        keyword_analysis[paradigm] = {
            "keywords": keywords,
            "count_in_top10": count_pattern(top10, keywords),
            "count_in_all": count_pattern(assumptions, keywords),
            "total_assumptions": len(assumptions),
        }

    analysis["keyword_analysis"] = keyword_analysis

    # Linguistic comparison: success vs failure average confidence, assumption length
    def avg_length(assumptions: list, k: int = 10) -> float:
        top = assumptions[:k]
        if not top:
            return 0
        return sum(len(a["assumption"].split()) for a in top) / len(top)

    def avg_confidence(assumptions: list, k: int = 10) -> float:
        top = assumptions[:k]
        if not top:
            return 0
        return sum(a.get("confidence", 0) for a in top) / len(top)

    linguistic = {
        "failures": {
            p: {
                "avg_assumption_length": round(avg_length(load_extraction(p)), 1),
                "avg_confidence_top10": round(avg_confidence(load_extraction(p)), 3),
            } for p in failures
        },
        "successes": {
            p: {
                "avg_assumption_length": round(avg_length(load_extraction(p)), 1),
                "avg_confidence_top10": round(avg_confidence(load_extraction(p)), 3),
            } for p in successes
        },
    }
    analysis["linguistic_comparison"] = linguistic

    # Summary findings
    failure_kw_total = sum(v["count_in_top10"] for v in keyword_analysis.values())
    analysis["summary"] = {
        "total_failures": 4,
        "failures_with_relevant_keywords_in_top10": sum(
            1 for v in keyword_analysis.values() if v["count_in_top10"] > 0
        ),
        "relevant_keywords_found_total": failure_kw_total,
        "interpretation": (
            "If relevant keywords appear in top-10 but target is still missed, "
            "failure is a ranking/phrasing problem. If keywords are absent, "
            "failure is an extraction-coverage problem."
        ),
    }

    print("=== Error Analysis on 4 Failure Cases ===\n")
    for p in failures:
        kw = keyword_analysis[p]
        ling = linguistic["failures"][p]
        print(f"{p}:")
        print(f"  target: {analysis['failures'][p]['target'][:70]}")
        print(f"  #extracted: {analysis['failures'][p]['num_extracted']}")
        print(f"  top-1: {analysis['failures'][p]['top_5'][0]['assumption'][:70] if analysis['failures'][p]['top_5'] else 'N/A'}")
        print(f"  relevant keywords in top-10: {kw['count_in_top10']}/{kw['total_assumptions']}")
        print(f"  avg top-10 length: {ling['avg_assumption_length']:.1f} words")
        print(f"  avg top-10 confidence: {ling['avg_confidence_top10']:.3f}")
        print()

    print("Linguistic comparison (avg top-10):")
    for p in successes:
        ling = linguistic["successes"][p]
        print(f"  SUCCESS {p:12s}: len={ling['avg_assumption_length']:5.1f}  conf={ling['avg_confidence_top10']:.3f}")
    for p in failures:
        ling = linguistic["failures"][p]
        print(f"  FAIL    {p:12s}: len={ling['avg_assumption_length']:5.1f}  conf={ling['avg_confidence_top10']:.3f}")

    out_path = PROJECT_ROOT / "experiments" / "error_analysis" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
