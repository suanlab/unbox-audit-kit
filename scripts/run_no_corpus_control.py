#!/usr/bin/env python3
"""No-Corpus Memorization Control Experiment.

Tests whether GPT-4o can identify paradigm-shifting assumptions from field name
alone (no paper text), which would indicate memorization rather than extraction.

Two conditions:
  1. Field name WITH year hints (e.g., "sequence modeling and neural machine translation (pre-2017)")
  2. Field name WITHOUT year hints (e.g., "sequence modeling")

If no-corpus achieves similar results to per-paper experiment -> memorization concern.
If no-corpus fails -> corpus genuinely contributes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI
from semantic_match import embed_texts, cosine_similarity, soft_rank, soft_recall_at_k

# --- Configuration ---

PARADIGMS = ["transformer", "diffusion", "icl", "vit"]

FIELD_NAMES_WITH_YEAR = {
    "transformer": "sequence modeling and neural machine translation (pre-2017)",
    "diffusion": "generative modeling and image synthesis (pre-2020)",
    "icl": "transfer learning and task adaptation in NLP (pre-2020)",
    "vit": "visual recognition and image classification (pre-2020)",
}

FIELD_NAMES_NO_YEAR = {
    "transformer": "sequence modeling",
    "diffusion": "generative modeling",
    "icl": "transfer learning in NLP",
    "vit": "visual recognition",
}

PROMPT_TEMPLATE = """You are an expert at identifying the foundational, field-wide assumptions
that academic papers inherit from their broader research community.

Identify the **paradigmatic assumptions** — beliefs shared across the
entire subfield that researchers take for granted without questioning.
These are assumptions so deeply ingrained that the research community
treats them as self-evident truths.

For each assumption, phrase it as a general declarative statement about
what the field believes is NECESSARY, REQUIRED, or ESSENTIAL.
Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories: architectural, training, data, theoretical, evaluation

Return ONLY valid JSON. Limit to top 10 assumptions.

Format: {{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "string"}}]}}

Field: {field_name}"""

NUM_CALLS_PER_FIELD = 15
MODEL = "gpt-4o"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "no_corpus_control"
SOFT_THRESHOLD = 0.65


def load_ground_truth() -> dict:
    gt_path = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
    with open(gt_path) as f:
        return json.load(f)


def call_gpt4o(client: OpenAI, field_name: str, call_idx: int) -> list[dict]:
    """Make a single GPT-4o call and parse assumptions."""
    prompt = PROMPT_TEMPLATE.format(field_name=field_name)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
            seed=42,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        parsed = json.loads(content)
        assumptions = parsed.get("assumptions", [])
        for a in assumptions:
            a["call_idx"] = call_idx
        return assumptions

    except Exception as e:
        print(f"  [WARN] Call {call_idx} failed: {e}")
        return []


def deduplicate_assumptions(all_assumptions: list[dict]) -> list[dict]:
    """Deduplicate by exact string match, keeping first occurrence."""
    seen = set()
    unique = []
    for a in all_assumptions:
        text = a["assumption"].strip().lower()
        if text not in seen:
            seen.add(text)
            unique.append(a)
    return unique


def evaluate_paradigm(
    unique_assumptions: list[dict],
    ground_truth_entry: dict,
) -> dict:
    """Compute soft matching metrics against ground truth."""
    gt_text = ground_truth_entry["broken_assumption"]
    gt_aliases = [gt_text] + ground_truth_entry.get("aliases", [])

    predicted = [a["assumption"] for a in unique_assumptions]

    # Compute soft rank and recall
    rank, rank_details = soft_rank(predicted, gt_aliases, threshold=SOFT_THRESHOLD)
    recall_5, _ = soft_recall_at_k(predicted, gt_aliases, k=5, threshold=SOFT_THRESHOLD)
    recall_10, _ = soft_recall_at_k(predicted, gt_aliases, k=10, threshold=SOFT_THRESHOLD)
    recall_20, _ = soft_recall_at_k(predicted, gt_aliases, k=20, threshold=SOFT_THRESHOLD)

    # Find best similarity across all assumptions
    best_sim = 0.0
    best_assumption = ""
    for d in rank_details:
        if d["best_similarity"] > best_sim:
            best_sim = d["best_similarity"]
            best_assumption = d["assumption"]

    # Top 5 by similarity
    sorted_details = sorted(rank_details, key=lambda x: x["best_similarity"], reverse=True)
    top_5 = [
        {"assumption": d["assumption"], "similarity": d["best_similarity"]}
        for d in sorted_details[:5]
    ]

    return {
        "num_unique_assumptions": len(unique_assumptions),
        "best_similarity": best_sim,
        "best_matching_assumption": best_assumption,
        "soft_rank": int(rank) if rank != float("inf") else len(predicted) + 1,
        "soft_recall_at_5": recall_5,
        "soft_recall_at_10": recall_10,
        "soft_recall_at_20": recall_20,
        "top_5_assumptions": top_5,
    }


def run_condition(
    client: OpenAI,
    field_names: dict[str, str],
    condition_name: str,
    ground_truth: dict,
) -> dict:
    """Run one experimental condition (with or without year hints)."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name}")
    print(f"{'='*60}")

    results = {}

    for paradigm in PARADIGMS:
        field_name = field_names[paradigm]
        gt_entry = ground_truth[paradigm]

        print(f"\n--- {paradigm} (field: '{field_name}') ---")

        all_assumptions = []
        for i in range(NUM_CALLS_PER_FIELD):
            print(f"  Call {i+1}/{NUM_CALLS_PER_FIELD}...", end=" ", flush=True)
            assumptions = call_gpt4o(client, field_name, i)
            print(f"got {len(assumptions)} assumptions")
            all_assumptions.extend(assumptions)
            time.sleep(0.5)  # Rate limiting

        unique = deduplicate_assumptions(all_assumptions)
        print(f"  Total raw: {len(all_assumptions)}, Unique: {len(unique)}")

        metrics = evaluate_paradigm(unique, gt_entry)
        metrics["paradigm"] = paradigm
        metrics["field_name"] = field_name
        metrics["num_raw_assumptions"] = len(all_assumptions)

        print(f"  Best similarity: {metrics['best_similarity']:.4f}")
        print(f"  Soft rank: {metrics['soft_rank']}")
        print(f"  Recall@5: {metrics['soft_recall_at_5']}")

        results[paradigm] = metrics

        # Save per-paradigm results
        per_paradigm_path = OUTPUT_DIR / f"{condition_name}_{paradigm}.json"
        with open(per_paradigm_path, "w") as f:
            json.dump(
                {
                    "condition": condition_name,
                    "paradigm": paradigm,
                    "field_name": field_name,
                    "ground_truth": gt_entry["broken_assumption"],
                    "num_calls": NUM_CALLS_PER_FIELD,
                    "num_raw_assumptions": len(all_assumptions),
                    "num_unique_assumptions": len(unique),
                    "assumptions": unique,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

    return results


def load_corpus_baseline() -> dict:
    """Load the per-paper (corpus-based) results for comparison."""
    summary_path = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt" / "summary.json"
    if not summary_path.exists():
        print(f"[WARN] Corpus baseline not found at {summary_path}")
        return {}

    with open(summary_path) as f:
        data = json.load(f)

    baseline = {}
    for entry in data["results"]:
        p = entry["paradigm"]
        if p in PARADIGMS:
            baseline[p] = {
                "best_similarity": entry["best_similarity"],
                "soft_rank": entry["soft_rank"],
                "soft_recall_at_5": entry["soft_recall_at_5"],
                "soft_recall_at_10": entry["soft_recall_at_10"],
                "num_unique_assumptions": entry["num_unique_assumptions"],
            }
    return baseline


def build_summary(
    with_year_results: dict,
    no_year_results: dict,
    corpus_baseline: dict,
) -> dict:
    """Build the final comparison summary."""
    comparison = []

    for paradigm in PARADIGMS:
        entry = {"paradigm": paradigm}

        # Corpus baseline
        if paradigm in corpus_baseline:
            cb = corpus_baseline[paradigm]
            entry["corpus_based"] = {
                "best_similarity": cb["best_similarity"],
                "soft_rank": cb["soft_rank"],
                "recall_at_5": cb["soft_recall_at_5"],
                "num_unique": cb["num_unique_assumptions"],
            }

        # With year hints
        wy = with_year_results[paradigm]
        entry["no_corpus_with_year"] = {
            "best_similarity": wy["best_similarity"],
            "soft_rank": wy["soft_rank"],
            "recall_at_5": wy["soft_recall_at_5"],
            "num_unique": wy["num_unique_assumptions"],
            "best_matching_assumption": wy["best_matching_assumption"],
            "top_5": wy["top_5_assumptions"],
        }

        # No year hints
        ny = no_year_results[paradigm]
        entry["no_corpus_no_year"] = {
            "best_similarity": ny["best_similarity"],
            "soft_rank": ny["soft_rank"],
            "recall_at_5": ny["soft_recall_at_5"],
            "num_unique": ny["num_unique_assumptions"],
            "best_matching_assumption": ny["best_matching_assumption"],
            "top_5": ny["top_5_assumptions"],
        }

        comparison.append(entry)

    # Aggregate stats
    def avg_metric(results, key):
        vals = [results[p][key] for p in PARADIGMS]
        return round(sum(vals) / len(vals), 4)

    def count_hits(results, key="soft_recall_at_5"):
        return sum(1 for p in PARADIGMS if results[p][key] > 0)

    aggregate = {
        "corpus_based": {},
        "no_corpus_with_year": {
            "avg_best_similarity": avg_metric(with_year_results, "best_similarity"),
            "avg_rank": avg_metric(with_year_results, "soft_rank"),
            "hits_at_5": count_hits(with_year_results),
            "hits_at_10": count_hits(with_year_results, "soft_recall_at_10"),
        },
        "no_corpus_no_year": {
            "avg_best_similarity": avg_metric(no_year_results, "best_similarity"),
            "avg_rank": avg_metric(no_year_results, "soft_rank"),
            "hits_at_5": count_hits(no_year_results),
            "hits_at_10": count_hits(no_year_results, "soft_recall_at_10"),
        },
    }

    if corpus_baseline:
        cb_sims = [corpus_baseline[p]["best_similarity"] for p in PARADIGMS if p in corpus_baseline]
        cb_ranks = [corpus_baseline[p]["soft_rank"] for p in PARADIGMS if p in corpus_baseline]
        cb_hits = sum(1 for p in PARADIGMS if p in corpus_baseline and corpus_baseline[p]["soft_recall_at_5"] > 0)
        aggregate["corpus_based"] = {
            "avg_best_similarity": round(sum(cb_sims) / len(cb_sims), 4) if cb_sims else None,
            "avg_rank": round(sum(cb_ranks) / len(cb_ranks), 4) if cb_ranks else None,
            "hits_at_5": cb_hits,
        }

    # Interpretation
    wy_hits = aggregate["no_corpus_with_year"]["hits_at_5"]
    ny_hits = aggregate["no_corpus_no_year"]["hits_at_5"]
    cb_hits = aggregate["corpus_based"].get("hits_at_5", 0)

    if wy_hits >= cb_hits and cb_hits > 0:
        interpretation = (
            "WARNING: No-corpus control achieves comparable hit rate to corpus-based extraction. "
            "This suggests the model may be retrieving paradigm-shift knowledge from pretraining memory "
            "rather than genuinely extracting assumptions from the paper corpus."
        )
    elif wy_hits > 0 and wy_hits < cb_hits:
        interpretation = (
            "MIXED: No-corpus control achieves some hits but fewer than corpus-based extraction. "
            "The corpus provides partial added value, but memorization is a contributing factor."
        )
    else:
        interpretation = (
            "CLEAN: No-corpus control fails to identify paradigm shifts, while corpus-based extraction succeeds. "
            "This supports the claim that the corpus genuinely contributes to assumption extraction."
        )

    if ny_hits < wy_hits:
        interpretation += (
            f" Year hints matter: removing them drops hits from {wy_hits} to {ny_hits}, "
            "suggesting the model uses temporal context to narrow down to specific breakthroughs."
        )
    elif ny_hits == wy_hits:
        interpretation += (
            " Year hints do NOT matter: the model achieves the same hits without them, "
            "indicating strong memorization of field-level paradigm shifts."
        )

    return {
        "experiment": "no_corpus_memorization_control",
        "model": MODEL,
        "num_calls_per_field": NUM_CALLS_PER_FIELD,
        "soft_threshold": SOFT_THRESHOLD,
        "conditions": {
            "with_year": "Field name with year hint (e.g., 'pre-2017')",
            "no_year": "Field name only, no year hint",
            "corpus_based": "Original per-paper extraction (from gpt4o_clean_prompt)",
        },
        "comparison": comparison,
        "aggregate": aggregate,
        "interpretation": interpretation,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    ground_truth = load_ground_truth()

    # Condition 1: Field name WITH year hints
    with_year_results = run_condition(
        client, FIELD_NAMES_WITH_YEAR, "with_year", ground_truth
    )

    # Condition 2: Field name WITHOUT year hints
    no_year_results = run_condition(
        client, FIELD_NAMES_NO_YEAR, "no_year", ground_truth
    )

    # Load corpus baseline for comparison
    corpus_baseline = load_corpus_baseline()

    # Build and save summary
    summary = build_summary(with_year_results, no_year_results, corpus_baseline)

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\nSaved to: {summary_path}")
    print(f"\nInterpretation: {summary['interpretation']}")

    print("\n--- Per-paradigm comparison ---")
    for entry in summary["comparison"]:
        p = entry["paradigm"]
        cb = entry.get("corpus_based", {})
        wy = entry["no_corpus_with_year"]
        ny = entry["no_corpus_no_year"]
        print(f"\n{p}:")
        if cb:
            print(f"  Corpus-based:       sim={cb['best_similarity']:.4f}  rank={cb['soft_rank']}  R@5={cb['recall_at_5']}")
        print(f"  No-corpus+year:     sim={wy['best_similarity']:.4f}  rank={wy['soft_rank']}  R@5={wy['recall_at_5']}")
        print(f"  No-corpus (no year): sim={ny['best_similarity']:.4f}  rank={ny['soft_rank']}  R@5={ny['recall_at_5']}")


if __name__ == "__main__":
    main()
