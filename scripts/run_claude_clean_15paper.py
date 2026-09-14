#!/usr/bin/env python3
"""Experiment 1: Fair Comparison — Claude clean prompt on 15 papers x 4 primary shifts.

Uses the same clean prompt (no few-shot examples) as the GPT-4o clean prompt experiment,
but with Claude Sonnet 4. This enables a fair model-vs-model comparison:
same prompt, same corpus, same papers — only the model differs.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_papers = getattr(importlib.import_module("src.pipeline"), "load_papers")
embed_texts = getattr(importlib.import_module("src.semantic_match"), "embed_texts")
cosine_similarity = getattr(importlib.import_module("src.semantic_match"), "cosine_similarity")

CATEGORIES = ("transformer", "diffusion", "icl", "vit")
MAPPING_PATH = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "claude_clean_15paper"
MODEL = "claude-sonnet-4-20250514"
MAX_PAPERS = 15
MAX_RETRIES = 3
RETRY_DELAY = 5
SOFT_THRESHOLD = 0.65

CLEAN_PROMPT_TEMPLATE = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

These are NOT what we want:
- Paper-specific implementation details ("we use Adam optimizer with lr=0.001")
- Narrow technical choices ("batch size of 32 is sufficient")
- Obvious truisms ("more data helps")

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories:
- architectural: Structural requirements
- training: Learning procedure requirements
- data: Data requirements
- theoretical: Theoretical constraints
- evaluation: Evaluation paradigm assumptions

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how foundational they are to the entire field (not just this paper).

Format:
{{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "architectural|training|data|theoretical|evaluation"}}]}}

Paper text:
{paper_text}"""


def load_mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def ground_truth_aliases(category: str) -> list[str]:
    mapping = load_mapping()
    entry = mapping[category]
    aliases = [entry["broken_assumption"]] + entry.get("aliases", [])
    return [a.strip() for a in aliases if a.strip()]


def call_claude(prompt: str, api_key: str) -> list[dict]:
    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Retry {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(RETRY_DELAY * attempt)

    text = message.content[0].text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                if isinstance(parsed, dict) and "assumptions" in parsed:
                    return parsed["assumptions"]
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from response: {text[:300]}")


def compute_soft_metrics(assumptions: list[str], aliases: list[str]) -> dict:
    """Compute soft matching metrics using OpenAI embeddings."""
    all_texts = assumptions + aliases
    embeddings = embed_texts(all_texts)
    cand_embs = embeddings[:len(assumptions)]
    alias_embs = embeddings[len(assumptions):]

    details = []
    best_rank = len(assumptions) + 1
    for idx, (assump, emb) in enumerate(zip(assumptions, cand_embs)):
        sim = max(cosine_similarity(emb, ae) for ae in alias_embs)
        is_match = sim >= SOFT_THRESHOLD
        if is_match and (idx + 1) < best_rank:
            best_rank = idx + 1
        details.append({
            "rank": idx + 1,
            "assumption": assump,
            "best_similarity": round(sim, 4),
            "is_match": is_match,
        })

    details_sorted = sorted(details, key=lambda d: d["best_similarity"], reverse=True)
    best_sim = details_sorted[0]["best_similarity"] if details_sorted else 0.0

    # Recall@K
    def recall_at(k):
        return 1.0 if any(d["is_match"] for d in details if d["rank"] <= k) else 0.0

    return {
        "best_similarity": best_sim,
        "soft_rank": best_rank,
        "soft_recall_at_5": recall_at(5),
        "soft_recall_at_10": recall_at(10),
        "soft_recall_at_20": recall_at(20),
        "top_5_assumptions": [
            {"assumption": d["assumption"], "similarity": d["best_similarity"]}
            for d in details[:5]
        ],
        "all_details": details_sorted,
    }


def run_category(category: str, api_key: str) -> dict:
    paper_path = PROJECT_ROOT / "data" / category / "papers.jsonl"
    papers = load_papers(str(paper_path))[:MAX_PAPERS]
    aliases = ground_truth_aliases(category)

    print(f"\n{'='*60}")
    print(f"Category: {category} ({len(papers)} papers)")
    print(f"Ground truth: {aliases[0]}")
    print(f"{'='*60}")

    all_assumptions = []
    for i, paper in enumerate(papers):
        title = str(paper.get("title", ""))
        abstract = str(paper.get("abstract", ""))
        paper_text = f"Title: {title}\nAbstract: {abstract}"
        prompt = CLEAN_PROMPT_TEMPLATE.format(paper_text=paper_text)

        print(f"  [{i+1}/{len(papers)}] {title[:60]}...", flush=True)
        extracted = call_claude(prompt, api_key)
        for item in extracted:
            if isinstance(item, dict):
                all_assumptions.append({
                    "assumption": item.get("assumption", ""),
                    "confidence": item.get("confidence", 0.0),
                    "category": item.get("category", ""),
                    "source_paper": title,
                })
            elif isinstance(item, str):
                all_assumptions.append({
                    "assumption": item,
                    "confidence": 0.5,
                    "category": "unknown",
                    "source_paper": title,
                })
        time.sleep(1.5)

    # Deduplicate by normalized text
    seen = set()
    unique = []
    for a in all_assumptions:
        key = a["assumption"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    assumption_texts = [a["assumption"] for a in unique]
    print(f"  Raw: {len(all_assumptions)}, Unique: {len(unique)}")

    # Soft matching
    print(f"  Computing soft matching...", flush=True)
    metrics = compute_soft_metrics(assumption_texts, aliases)

    result = {
        "paradigm": category,
        "num_papers": len(papers),
        "num_raw_assumptions": len(all_assumptions),
        "num_unique_assumptions": len(unique),
        "ground_truth": aliases[0],
        "assumptions": unique,
        **{k: v for k, v in metrics.items() if k != "all_details"},
        "all_details": metrics["all_details"],
    }
    return result


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY required")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for soft matching")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for category in CATEGORIES:
        result = run_category(category, api_key)
        # Save per-category
        out_path = OUTPUT_DIR / f"{category}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  Saved: {out_path}")
        all_results.append(result)

    # Summary
    summary = {
        "experiment": "claude_clean_15paper",
        "prompt_type": "clean_no_examples",
        "model": MODEL,
        "total_api_calls": sum(r["num_papers"] for r in all_results),
        "soft_threshold": SOFT_THRESHOLD,
        "results": [
            {
                "paradigm": r["paradigm"],
                "num_papers": r["num_papers"],
                "num_unique_assumptions": r["num_unique_assumptions"],
                "best_similarity": r["best_similarity"],
                "soft_rank": r["soft_rank"],
                "soft_recall_at_5": r["soft_recall_at_5"],
                "soft_recall_at_10": r["soft_recall_at_10"],
                "soft_recall_at_20": r["soft_recall_at_20"],
                "top_5_assumptions": r["top_5_assumptions"],
            }
            for r in all_results
        ],
        "aggregate": {
            "total_paradigms": len(all_results),
            "hits_at_5": sum(1 for r in all_results if r["soft_recall_at_5"] > 0),
            "hits_at_10": sum(1 for r in all_results if r["soft_recall_at_10"] > 0),
            "hits_at_20": sum(1 for r in all_results if r["soft_recall_at_20"] > 0),
            "avg_best_similarity": round(
                sum(r["best_similarity"] for r in all_results) / len(all_results), 4
            ),
            "avg_rank_when_hit": round(
                sum(r["soft_rank"] for r in all_results if r["soft_rank"] <= len(r.get("all_details", [])))
                / max(1, sum(1 for r in all_results if r["soft_rank"] <= len(r.get("all_details", [])))),
                1,
            ),
        },
        "ran_at": datetime.now(UTC).isoformat(),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in summary["results"]:
        hit = "HIT" if r["soft_recall_at_5"] > 0 else "MISS"
        print(f"  {r['paradigm']:12s}  best_sim={r['best_similarity']:.4f}  rank={r['soft_rank']:3d}  [{hit}]")
    agg = summary["aggregate"]
    print(f"\n  Hits@5: {agg['hits_at_5']}/{agg['total_paradigms']}")
    print(f"  Avg best similarity: {agg['avg_best_similarity']:.4f}")
    print(f"  Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
