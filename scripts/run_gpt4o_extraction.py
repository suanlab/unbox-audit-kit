#!/usr/bin/env python3
"""
M3 Experiment: GPT-4o Cross-Validation for assumption extraction.

Tests whether the extraction pipeline works with a different LLM (GPT-4o instead
of Claude) to address circular evaluation concerns. Uses the SAME field-wide
extraction prompt but calls OpenAI GPT-4o.

Runs on the 4 primary paradigm shifts (transformer, diffusion, icl, vit),
using first 15 papers from each.

Output: experiments/gpt4o_extraction/
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction import EXTRACTION_PROMPT
from src.semantic_match import embed_texts, cosine_similarity

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "gpt4o_extraction"
DATA_DIR = PROJECT_ROOT / "data"
MAPPING_PATH = DATA_DIR / "paradigm_shift_mapping.json"

MAX_PAPERS = 15
MAX_ASSUMPTIONS_PER_PAPER = 5
SOFT_THRESHOLD = 0.65
MAX_RETRIES = 3

PRIMARY_PARADIGMS = ["transformer", "diffusion", "icl", "vit"]


# ── GPT-4o extraction ─────────────────────────────────────────────────

def extract_assumptions_gpt4o(
    paper_text: str,
    openai_client: Any,
) -> list[dict[str, Any]]:
    """
    Extract assumptions from paper text using GPT-4o.
    Uses the same prompt as Claude extraction but calls OpenAI API.
    Retries up to MAX_RETRIES times on malformed JSON.
    """
    prompt = EXTRACTION_PROMPT.format(paper_text=paper_text)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0 if attempt > 1 else 0.3,  # Lower temp on retry
                seed=42,
            )

            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from GPT-4o")

            # Parse JSON from response (same approach as Claude extraction)
            decoder = json.JSONDecoder()
            response_data = None
            for idx, char in enumerate(response_text):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(response_text[idx:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    response_data = parsed
                    break

            if response_data is None:
                raise ValueError(
                    f"Failed to extract JSON from GPT-4o response (attempt {attempt}): "
                    f"{response_text[:200]}"
                )

            if "assumptions" not in response_data:
                raise ValueError(f"Response missing 'assumptions' key (attempt {attempt})")

            assumptions = response_data["assumptions"]

            # Validate
            validated = []
            for a in assumptions:
                if not isinstance(a, dict):
                    continue
                if "assumption" not in a:
                    continue
                conf = a.get("confidence", 0.5)
                if not isinstance(conf, (int, float)):
                    conf = 0.5
                conf = max(0.0, min(1.0, float(conf)))
                cat = a.get("category", "")
                if cat not in ("architectural", "training", "data", "theoretical", "evaluation"):
                    cat = "training"  # default fallback
                validated.append({
                    "assumption": str(a["assumption"]),
                    "confidence": conf,
                    "category": cat,
                })

            return validated[:10]

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"      Retry {attempt}/{MAX_RETRIES}: {e}")
                time.sleep(2 * attempt)

    raise RuntimeError(f"GPT-4o extraction failed after {MAX_RETRIES} retries: {last_error}")


# ── Paper loading ─────────────────────────────────────────────────────

def load_papers(category: str, max_papers: int = MAX_PAPERS) -> list[dict[str, Any]]:
    """Load papers from data/{category}/papers.jsonl, taking first max_papers."""
    papers_path = DATA_DIR / category / "papers.jsonl"
    if not papers_path.exists():
        print(f"  WARNING: {papers_path} does not exist")
        return []

    papers = []
    with papers_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
                if len(papers) >= max_papers:
                    break

    return papers


# ── Deduplication ─────────────────────────────────────────────────────

def deduplicate_assumptions(
    assumptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate assumptions by normalized text, keeping highest confidence."""

    def normalize(text: str) -> str:
        lowered = text.strip().lower()
        return " ".join(re.sub(r"[^a-z0-9]+", " ", lowered).split())

    best: dict[str, dict[str, Any]] = {}
    for a in assumptions:
        text = str(a.get("assumption", "")).strip()
        if not text:
            continue
        key = normalize(text)
        if not key:
            continue
        existing = best.get(key)
        if existing is None or float(a.get("confidence", 0)) > float(existing.get("confidence", 0)):
            best[key] = {**a, "_normalized": key}

    ranked = sorted(best.values(), key=lambda x: float(x.get("confidence", 0)), reverse=True)
    return ranked


# ── Soft matching ─────────────────────────────────────────────────────

def compute_soft_metrics(
    predicted_assumptions: list[str],
    ground_truth_aliases: list[str],
    threshold: float = SOFT_THRESHOLD,
) -> dict[str, Any]:
    """Compute soft rank, recall@k, best similarity using embeddings."""
    if not predicted_assumptions or not ground_truth_aliases:
        return {
            "best_similarity": 0.0,
            "soft_rank": float(len(predicted_assumptions) + 1),
            "soft_recall_at_5": 0.0,
            "soft_recall_at_10": 0.0,
            "soft_recall_at_20": 0.0,
            "details": [],
        }

    all_texts = predicted_assumptions + ground_truth_aliases
    embeddings = embed_texts(all_texts)
    pred_embeddings = embeddings[: len(predicted_assumptions)]
    alias_embeddings = embeddings[len(predicted_assumptions) :]

    details: list[dict[str, Any]] = []
    best_sim_overall = 0.0
    soft_rank_val = float(len(predicted_assumptions) + 1)

    for idx, (cand, cand_emb) in enumerate(zip(predicted_assumptions, pred_embeddings)):
        sim = max(cosine_similarity(cand_emb, ae) for ae in alias_embeddings)
        is_match = sim >= threshold
        rank = float(idx + 1)
        if is_match and rank < soft_rank_val:
            soft_rank_val = rank
        if sim > best_sim_overall:
            best_sim_overall = sim
        details.append({
            "rank": idx + 1,
            "assumption": cand,
            "best_similarity": round(sim, 4),
            "is_match": is_match,
        })

    def soft_recall_at_k(k: int) -> float:
        for d in details:
            if d["rank"] <= k and d["is_match"]:
                return 1.0
        return 0.0

    return {
        "best_similarity": round(best_sim_overall, 4),
        "soft_rank": soft_rank_val,
        "soft_recall_at_5": soft_recall_at_k(5),
        "soft_recall_at_10": soft_recall_at_k(10),
        "soft_recall_at_20": soft_recall_at_k(20),
        "details": sorted(details, key=lambda d: d["best_similarity"], reverse=True),
    }


# ── Per-paradigm runner ──────────────────────────────────────────────

def run_paradigm(
    category: str,
    ground_truth: dict[str, Any],
    openai_client: Any,
) -> dict[str, Any]:
    """Run GPT-4o extraction for a single paradigm."""
    display = category.upper()
    print(f"\n{'='*60}")
    print(f"  Paradigm: {display} (year {ground_truth['year']})")
    print(f"{'='*60}")

    paradigm_dir = OUTPUT_DIR / category
    paradigm_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load papers
    papers = load_papers(category, max_papers=MAX_PAPERS)
    if not papers:
        return {"paradigm": category, "error": "no_papers"}

    print(f"  Loaded {len(papers)} papers from data/{category}/papers.jsonl")

    # Step 2: Extract assumptions with GPT-4o (or load if already done)
    assumptions_path = paradigm_dir / "assumptions.json"
    if assumptions_path.exists():
        print(f"  Loading existing GPT-4o assumptions from {assumptions_path}")
        with assumptions_path.open("r") as f:
            raw_assumptions = json.load(f)
    else:
        print(f"  Extracting assumptions with GPT-4o from {len(papers)} papers...")
        raw_assumptions: list[dict[str, Any]] = []

        for i, paper in enumerate(papers):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            paper_text = f"{title}\n\n{abstract}".strip()
            if not paper_text:
                continue

            print(f"    [{i+1}/{len(papers)}] {title[:60]}...")
            try:
                assumptions = extract_assumptions_gpt4o(paper_text, openai_client)
                for a in assumptions[:MAX_ASSUMPTIONS_PER_PAPER]:
                    raw_assumptions.append({
                        "paper_id": paper.get("paperId"),
                        "paper_title": title,
                        "assumption": a.get("assumption", ""),
                        "confidence": a.get("confidence", 0.5),
                        "category": a.get("category", ""),
                    })
            except Exception as e:
                print(f"      WARNING: extraction failed: {e}")
                time.sleep(2)
                continue

            # Delay between API calls
            time.sleep(1.5)

        # Save intermediate results
        with assumptions_path.open("w") as f:
            json.dump(raw_assumptions, f, indent=2, ensure_ascii=False)

    # Step 3: Deduplicate
    unique_assumptions = deduplicate_assumptions(raw_assumptions)
    predicted = [str(a["assumption"]) for a in unique_assumptions]

    print(f"  Total raw assumptions: {len(raw_assumptions)}")
    print(f"  Unique assumptions: {len(unique_assumptions)}")

    # Step 4: Compute soft metrics
    aliases = [ground_truth["broken_assumption"]] + ground_truth.get("aliases", [])

    print(f"  Computing soft matching against ground truth...")
    metrics = compute_soft_metrics(
        predicted_assumptions=predicted,
        ground_truth_aliases=aliases,
        threshold=SOFT_THRESHOLD,
    )

    top_10 = [
        {
            "rank": d["rank"],
            "assumption": d["assumption"],
            "similarity": d["best_similarity"],
        }
        for d in metrics["details"][:10]
    ]

    result = {
        "paradigm": category,
        "year": ground_truth["year"],
        "ground_truth": ground_truth["broken_assumption"],
        "ground_truth_aliases": aliases,
        "extraction_model": "gpt-4o",
        "num_papers": len(papers),
        "num_raw_assumptions": len(raw_assumptions),
        "num_unique_assumptions": len(unique_assumptions),
        "best_similarity": metrics["best_similarity"],
        "soft_rank": metrics["soft_rank"],
        "soft_recall_at_5": metrics["soft_recall_at_5"],
        "soft_recall_at_10": metrics["soft_recall_at_10"],
        "soft_recall_at_20": metrics["soft_recall_at_20"],
        "top_10_assumptions": top_10,
        "ran_at": datetime.now(UTC).isoformat(),
    }

    # Save individual result
    result_path = paradigm_dir / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved result to {result_path}")

    # Print summary
    print(f"  Results for {display} (GPT-4o):")
    print(f"    Papers: {len(papers)}")
    print(f"    Unique assumptions: {len(unique_assumptions)}")
    print(f"    Best similarity: {metrics['best_similarity']:.4f}")
    print(f"    Soft rank: {metrics['soft_rank']}")
    print(f"    Soft recall@5: {metrics['soft_recall_at_5']:.1f}")
    print(f"    Soft recall@10: {metrics['soft_recall_at_10']:.1f}")
    print(f"    Soft recall@20: {metrics['soft_recall_at_20']:.1f}")

    return result


# ── Summary ──────────────────────────────────────────────────────────

def _save_summary(results: list[dict[str, Any]]) -> None:
    """Save summary.json with all results so far."""
    successful = [r for r in results if "error" not in r]

    avg_best_sim = 0.0
    avg_soft_rank = 0.0
    avg_recall_5 = 0.0
    avg_recall_10 = 0.0
    avg_recall_20 = 0.0
    if successful:
        avg_best_sim = sum(r["best_similarity"] for r in successful) / len(successful)
        avg_soft_rank = sum(r["soft_rank"] for r in successful) / len(successful)
        avg_recall_5 = sum(r["soft_recall_at_5"] for r in successful) / len(successful)
        avg_recall_10 = sum(r["soft_recall_at_10"] for r in successful) / len(successful)
        avg_recall_20 = sum(r["soft_recall_at_20"] for r in successful) / len(successful)

    summary = {
        "experiment": "M3_gpt4o_cross_validation",
        "description": "GPT-4o cross-validation extraction on 4 primary paradigm shifts",
        "extraction_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "ran_at": datetime.now(UTC).isoformat(),
        "config": {
            "max_papers": MAX_PAPERS,
            "max_assumptions_per_paper": MAX_ASSUMPTIONS_PER_PAPER,
            "soft_threshold": SOFT_THRESHOLD,
        },
        "num_paradigms_attempted": len(results),
        "num_paradigms_successful": len(successful),
        "paradigms": [r.get("paradigm", "unknown") for r in results],
        "aggregate_metrics": {
            "avg_best_similarity": round(avg_best_sim, 4),
            "avg_soft_rank": round(avg_soft_rank, 2),
            "avg_soft_recall_at_5": round(avg_recall_5, 3),
            "avg_soft_recall_at_10": round(avg_recall_10, 3),
            "avg_soft_recall_at_20": round(avg_recall_20, 3),
        },
        "results": results,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_final_summary(results: list[dict[str, Any]]) -> None:
    """Print a readable final summary."""
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY - GPT-4o Cross-Validation Extraction")
    print(f"{'='*60}")

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n  Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"  Failed: {', '.join(r['paradigm'] for r in failed)}")

    if successful:
        print(f"\n  {'Paradigm':<15} {'Papers':>6} {'Unique':>6} {'BestSim':>8} {'SoftRank':>9} {'R@5':>5} {'R@10':>5} {'R@20':>5}")
        print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*5} {'-'*5} {'-'*5}")

        for r in successful:
            print(
                f"  {r['paradigm']:<15} "
                f"{r['num_papers']:>6} "
                f"{r['num_unique_assumptions']:>6} "
                f"{r['best_similarity']:>8.4f} "
                f"{r['soft_rank']:>9.1f} "
                f"{r['soft_recall_at_5']:>5.1f} "
                f"{r['soft_recall_at_10']:>5.1f} "
                f"{r['soft_recall_at_20']:>5.1f}"
            )

        avg_sim = sum(r["best_similarity"] for r in successful) / len(successful)
        avg_rank = sum(r["soft_rank"] for r in successful) / len(successful)
        avg_r5 = sum(r["soft_recall_at_5"] for r in successful) / len(successful)
        avg_r10 = sum(r["soft_recall_at_10"] for r in successful) / len(successful)
        avg_r20 = sum(r["soft_recall_at_20"] for r in successful) / len(successful)

        print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*5} {'-'*5} {'-'*5}")
        print(
            f"  {'AVERAGE':<15} {'':>6} {'':>6} "
            f"{avg_sim:>8.4f} "
            f"{avg_rank:>9.1f} "
            f"{avg_r5:>5.3f} "
            f"{avg_r10:>5.3f} "
            f"{avg_r20:>5.3f}"
        )

    print(f"\n  Results saved to: {OUTPUT_DIR}/summary.json")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set. Load .env first.")
        sys.exit(1)

    # Load ground truth mapping
    with MAPPING_PATH.open("r") as f:
        mapping = json.load(f)

    # Initialize OpenAI client
    from openai import OpenAI
    openai_client = OpenAI()

    all_results: list[dict[str, Any]] = []

    for category in PRIMARY_PARADIGMS:
        if category not in mapping:
            print(f"  WARNING: {category} not found in paradigm_shift_mapping.json, skipping.")
            all_results.append({
                "paradigm": category,
                "error": "not_in_mapping",
                "ran_at": datetime.now(UTC).isoformat(),
            })
            continue

        try:
            result = run_paradigm(
                category=category,
                ground_truth=mapping[category],
                openai_client=openai_client,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR running {category}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "paradigm": category,
                "error": str(e),
                "ran_at": datetime.now(UTC).isoformat(),
            })

        # Save intermediate summary after each paradigm
        _save_summary(all_results)

    # Final summary
    _save_summary(all_results)
    _print_final_summary(all_results)


if __name__ == "__main__":
    main()
