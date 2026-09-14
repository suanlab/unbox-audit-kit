#!/usr/bin/env python3
"""Run GPT-4o extraction with a CLEAN prompt (no few-shot examples) on all 10 paradigm shifts."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Clean prompt — NO few-shot examples
# ---------------------------------------------------------------------------
CLEAN_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

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

# ---------------------------------------------------------------------------
# Paradigm shift configuration
# ---------------------------------------------------------------------------
PRIMARY_PARADIGMS = ["transformer", "diffusion", "icl", "vit"]
ADDITIONAL_PARADIGMS = ["resnet", "gan", "word2vec", "dropout", "batchnorm", "bert"]
ALL_PARADIGMS = PRIMARY_PARADIGMS + ADDITIONAL_PARADIGMS

PAPERS_PER_PARADIGM = 15

EMBEDDING_MODEL = "text-embedding-3-small"


def load_papers(paradigm: str) -> list[dict]:
    """Load papers JSONL for a paradigm."""
    if paradigm in PRIMARY_PARADIGMS:
        path = PROJECT_ROOT / "data" / paradigm / "papers.jsonl"
    else:
        path = PROJECT_ROOT / "experiments" / "additional_paradigms_60" / paradigm / "papers.jsonl"

    papers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers[:PAPERS_PER_PARADIGM]


def format_paper_text(paper: dict) -> str:
    """Format paper metadata into text for the prompt."""
    parts = []
    parts.append(f"Title: {paper.get('title', 'Unknown')}")
    if paper.get("year"):
        parts.append(f"Year: {paper['year']}")
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    return "\n".join(parts)


def extract_assumptions_gpt4o(client: OpenAI, paper_text: str, max_retries: int = 3) -> list[dict]:
    """Call GPT-4o with the clean prompt and parse assumptions."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": CLEAN_PROMPT.format(paper_text=paper_text)}],
                temperature=0.0,
                seed=42,
            )
            content = response.choices[0].message.content.strip()

            # Try to extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()

            parsed = json.loads(content)
            assumptions = parsed.get("assumptions", [])
            return assumptions

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"    [Retry {attempt + 1}/{max_retries}] JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"    [Retry {attempt + 1}/{max_retries}] API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return []


def deduplicate_assumptions(all_assumptions: list[dict], client: OpenAI) -> list[dict]:
    """Deduplicate assumptions by exact text (case-insensitive). Keep highest confidence."""
    seen: dict[str, dict] = {}
    for a in all_assumptions:
        key = a["assumption"].strip().lower()
        if key not in seen or a.get("confidence", 0) > seen[key].get("confidence", 0):
            seen[key] = a
    # Sort by confidence descending
    deduped = sorted(seen.values(), key=lambda x: x.get("confidence", 0), reverse=True)
    return deduped


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI embeddings API."""
    if not texts:
        return []
    # Batch in groups of 100 to stay under limits
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def evaluate_paradigm(
    client: OpenAI,
    assumptions: list[dict],
    ground_truth: dict,
) -> dict:
    """Compute soft matching metrics for one paradigm."""
    if not assumptions:
        return {
            "best_similarity": 0.0,
            "soft_rank": len(assumptions) + 1,
            "soft_recall_at_5": 0.0,
            "soft_recall_at_10": 0.0,
            "soft_recall_at_20": 0.0,
        }

    # Build ground truth alias list
    gt_aliases = [ground_truth["broken_assumption"]] + ground_truth.get("aliases", [])

    # Embed all assumptions + aliases together
    assumption_texts = [a["assumption"] for a in assumptions]
    all_texts = assumption_texts + gt_aliases
    all_embeddings = embed_texts(client, all_texts)

    assumption_embeddings = all_embeddings[:len(assumption_texts)]
    alias_embeddings = all_embeddings[len(assumption_texts):]

    # Compute similarities
    sims = []
    for i, emb in enumerate(assumption_embeddings):
        best_sim = max(cosine_similarity(emb, ae) for ae in alias_embeddings)
        sims.append((i, best_sim))

    # Keep original confidence order (assumptions are already sorted by confidence)
    # Do NOT re-sort by similarity — rank must reflect the model's own confidence ordering
    best_sim_val = max(sim for _, sim in sims) if sims else 0.0

    # Confidence-based rank: position in original order where first sim >= threshold
    threshold = 0.65
    soft_rank_val = len(assumptions) + 1
    for orig_idx, sim in sims:  # sims is in original confidence order
        if sim >= threshold:
            soft_rank_val = orig_idx + 1  # 1-indexed position in confidence list
            break

    # Recall@K: 1 if any in top-K (by confidence) exceeds threshold
    def recall_at_k(k):
        for orig_idx, sim in sims[:k]:
            if sim >= threshold:
                return 1.0
        return 0.0

    # Top-5 with similarities (by original rank order, taking top 5 by confidence)
    top_5 = []
    for i in range(min(5, len(assumptions))):
        sim_for_i = sims[i][1]
        top_5.append({
            "assumption": assumptions[i]["assumption"],
            "similarity": round(sim_for_i, 4),
        })

    return {
        "best_similarity": round(best_sim_val, 4),
        "soft_rank": soft_rank_val,
        "soft_recall_at_5": recall_at_k(5),
        "soft_recall_at_10": recall_at_k(10),
        "soft_recall_at_20": recall_at_k(20),
        "top_5_assumptions": top_5,
    }


def main():
    # Load environment
    client = OpenAI()

    # Load ground truth
    gt_path = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    # Output directory
    output_dir = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_calls = 0

    for paradigm in ALL_PARADIGMS:
        print(f"\n{'='*60}")
        print(f"Processing paradigm: {paradigm}")
        print(f"{'='*60}")

        papers = load_papers(paradigm)
        print(f"  Loaded {len(papers)} papers")

        all_assumptions = []

        for i, paper in enumerate(papers):
            paper_text = format_paper_text(paper)
            print(f"  [{i+1}/{len(papers)}] Extracting from: {paper.get('title', 'Unknown')[:70]}...")

            assumptions = extract_assumptions_gpt4o(client, paper_text)
            total_calls += 1
            print(f"    -> {len(assumptions)} assumptions extracted")

            for a in assumptions:
                a["source_paper"] = paper.get("title", "Unknown")
            all_assumptions.extend(assumptions)

            time.sleep(1)  # Rate limit

        print(f"  Total raw assumptions: {len(all_assumptions)}")

        # Deduplicate
        deduped = deduplicate_assumptions(all_assumptions, client)
        print(f"  After deduplication: {len(deduped)}")

        # Save per-paradigm results
        paradigm_output = {
            "paradigm": paradigm,
            "num_papers": len(papers),
            "num_raw_assumptions": len(all_assumptions),
            "num_unique_assumptions": len(deduped),
            "ground_truth": ground_truth[paradigm]["broken_assumption"],
            "assumptions": deduped,
        }

        paradigm_path = output_dir / f"{paradigm}.json"
        with open(paradigm_path, "w") as f:
            json.dump(paradigm_output, f, indent=2)
        print(f"  Saved to {paradigm_path}")

        # Evaluate
        print(f"  Computing soft matching metrics...")
        metrics = evaluate_paradigm(client, deduped, ground_truth[paradigm])
        print(f"  best_similarity: {metrics['best_similarity']}")
        print(f"  soft_rank: {metrics['soft_rank']}")
        print(f"  soft_recall@5: {metrics['soft_recall_at_5']}")
        print(f"  soft_recall@10: {metrics['soft_recall_at_10']}")
        print(f"  soft_recall@20: {metrics['soft_recall_at_20']}")

        result_entry = {
            "paradigm": paradigm,
            "num_papers": len(papers),
            "num_unique_assumptions": len(deduped),
            "best_similarity": metrics["best_similarity"],
            "soft_rank": metrics["soft_rank"],
            "soft_recall_at_5": metrics["soft_recall_at_5"],
            "soft_recall_at_10": metrics["soft_recall_at_10"],
            "soft_recall_at_20": metrics["soft_recall_at_20"],
            "top_5_assumptions": metrics.get("top_5_assumptions", []),
        }
        results.append(result_entry)

    # Compute aggregate metrics
    print(f"\n{'='*60}")
    print("Computing aggregate metrics...")
    print(f"{'='*60}")

    hits_at_5 = sum(1 for r in results if r["soft_recall_at_5"] > 0)
    hits_at_10 = sum(1 for r in results if r["soft_recall_at_10"] > 0)
    hits_at_20 = sum(1 for r in results if r["soft_recall_at_20"] > 0)
    avg_best_sim = sum(r["best_similarity"] for r in results) / len(results) if results else 0

    # Average rank only for paradigms where there was a hit (rank <= num_assumptions)
    hit_ranks = [r["soft_rank"] for r in results if r["soft_rank"] <= r["num_unique_assumptions"]]
    avg_rank_when_hit = sum(hit_ranks) / len(hit_ranks) if hit_ranks else float("inf")

    summary = {
        "experiment": "gpt4o_clean_prompt_all_10",
        "prompt_type": "clean_no_examples",
        "model": "gpt-4o",
        "total_api_calls": total_calls,
        "results": results,
        "aggregate": {
            "total_paradigms": len(ALL_PARADIGMS),
            "hits_at_5": hits_at_5,
            "hits_at_10": hits_at_10,
            "hits_at_20": hits_at_20,
            "avg_best_similarity": round(avg_best_sim, 4),
            "avg_rank_when_hit": round(avg_rank_when_hit, 2) if avg_rank_when_hit != float("inf") else None,
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total paradigms: {len(ALL_PARADIGMS)}")
    print(f"Hits@5:  {hits_at_5}/{len(ALL_PARADIGMS)}")
    print(f"Hits@10: {hits_at_10}/{len(ALL_PARADIGMS)}")
    print(f"Hits@20: {hits_at_20}/{len(ALL_PARADIGMS)}")
    print(f"Avg best similarity: {avg_best_sim:.4f}")
    print(f"Avg rank (when hit): {avg_rank_when_hit:.2f}" if avg_rank_when_hit != float("inf") else "Avg rank (when hit): N/A")
    print()
    for r in results:
        marker = "HIT" if r["soft_recall_at_5"] > 0 else ("hit" if r["soft_recall_at_20"] > 0 else "---")
        print(f"  [{marker}] {r['paradigm']:12s}  best_sim={r['best_similarity']:.4f}  rank={r['soft_rank']:3}  "
              f"R@5={r['soft_recall_at_5']:.0f}  R@10={r['soft_recall_at_10']:.0f}  R@20={r['soft_recall_at_20']:.0f}  "
              f"({r['num_unique_assumptions']} assumptions)")


if __name__ == "__main__":
    main()
