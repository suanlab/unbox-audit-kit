#!/usr/bin/env python3
"""Experiment 2: Per-Paper Extraction vs Multi-Call Aggregation.

Tests whether per-paper extraction is essential, or if multi-call sampling
achieves the same result. Three conditions on the "transformer" category:

1. Single-call (Vanilla LLM style): One GPT-4o call with ALL 15 paper abstracts
   concatenated. Ask for top 20 assumptions.
2. Multi-call no-context: 15 independent GPT-4o calls, each with the clean prompt
   but NO paper text — just a field description. Deduplicate.
3. Per-paper extraction (Unbox style): 15 calls, each with one paper's text.
   Deduplicate.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_papers = getattr(importlib.import_module("src.pipeline"), "load_papers")
embed_texts = getattr(importlib.import_module("src.semantic_match"), "embed_texts")
cosine_similarity = getattr(importlib.import_module("src.semantic_match"), "cosine_similarity")

CATEGORY = "transformer"
MAPPING_PATH = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multicall_ablation"
MODEL = "gpt-4o"
MAX_PAPERS = 15
SOFT_THRESHOLD = 0.65
MAX_RETRIES = 3
RETRY_DELAY = 5

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

SINGLE_CALL_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Below are abstracts from {num_papers} papers in the sequence modeling / neural machine translation subfield. Identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that these papers take for granted without questioning.

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Return ONLY valid JSON with the top 20 most foundational assumptions.

Format:
{{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "architectural|training|data|theoretical|evaluation"}}]}}

Papers:

{papers_block}"""

NO_CONTEXT_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions in the sequence modeling / neural machine translation research community (circa 2015-2016).

Identify the **paradigmatic assumptions** — beliefs shared across this entire subfield that researchers take for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how foundational they are.

Format:
{{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "architectural|training|data|theoretical|evaluation"}}]}}"""


def ground_truth_aliases() -> list[str]:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    entry = mapping[CATEGORY]
    return [entry["broken_assumption"]] + entry.get("aliases", [])


def call_gpt4o(prompt: str, api_key: str, temperature: float = 0.0) -> list[dict]:
    openai = importlib.import_module("openai")
    client = openai.OpenAI(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                seed=42,
            )
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Retry {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(RETRY_DELAY * attempt)

    text = response.choices[0].message.content.strip()
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
    raise ValueError(f"Could not parse JSON: {text[:300]}")


def deduplicate(assumptions: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for a in assumptions:
        text = a.get("assumption", "") if isinstance(a, dict) else str(a)
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def compute_soft_metrics(assumptions: list[str], aliases: list[str]) -> dict:
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

    def recall_at(k):
        return 1.0 if any(d["is_match"] for d in details if d["rank"] <= k) else 0.0

    return {
        "best_similarity": best_sim,
        "soft_rank": best_rank,
        "soft_recall_at_5": recall_at(5),
        "soft_recall_at_10": recall_at(10),
        "soft_recall_at_20": recall_at(20),
        "details": details_sorted,
    }


def condition_1_single_call(papers: list[dict], api_key: str) -> dict:
    """One call with all 15 papers concatenated."""
    print("\n--- Condition 1: Single-call (Vanilla LLM style) ---")
    papers_block = "\n\n".join(
        f"Paper {i+1}: {p.get('title', '')}\n{p.get('abstract', '')}"
        for i, p in enumerate(papers)
    )
    prompt = SINGLE_CALL_PROMPT.format(num_papers=len(papers), papers_block=papers_block)
    raw = call_gpt4o(prompt, api_key)
    assumptions = []
    for item in raw:
        if isinstance(item, dict):
            assumptions.append(item)
        else:
            assumptions.append({"assumption": str(item), "confidence": 0.5, "category": "unknown"})
    unique = deduplicate(assumptions)
    texts = [a["assumption"] for a in unique]
    print(f"  Got {len(raw)} raw, {len(unique)} unique assumptions")
    return {"condition": "single_call", "raw_count": len(raw), "unique": unique, "texts": texts}


def condition_2_no_context(api_key: str) -> dict:
    """15 independent calls with no paper text."""
    print("\n--- Condition 2: Multi-call no-context ---")
    all_assumptions = []
    for i in range(15):
        print(f"  [{i+1}/15] No-context call...", flush=True)
        # Use slightly different temperature per call to get diversity
        raw = call_gpt4o(NO_CONTEXT_PROMPT, api_key, temperature=0.3)
        for item in raw:
            if isinstance(item, dict):
                all_assumptions.append(item)
            else:
                all_assumptions.append({"assumption": str(item), "confidence": 0.5, "category": "unknown"})
        time.sleep(1.5)
    unique = deduplicate(all_assumptions)
    texts = [a["assumption"] for a in unique]
    print(f"  Got {len(all_assumptions)} raw, {len(unique)} unique assumptions")
    return {"condition": "multi_call_no_context", "raw_count": len(all_assumptions), "unique": unique, "texts": texts}


def condition_3_per_paper(papers: list[dict], api_key: str) -> dict:
    """15 calls, each with one paper's text (Unbox style)."""
    print("\n--- Condition 3: Per-paper extraction (Unbox style) ---")
    all_assumptions = []
    for i, paper in enumerate(papers):
        title = str(paper.get("title", ""))
        abstract = str(paper.get("abstract", ""))
        paper_text = f"Title: {title}\nAbstract: {abstract}"
        prompt = CLEAN_PROMPT_TEMPLATE.format(paper_text=paper_text)
        print(f"  [{i+1}/{len(papers)}] {title[:50]}...", flush=True)
        raw = call_gpt4o(prompt, api_key)
        for item in raw:
            if isinstance(item, dict):
                item["source_paper"] = title
                all_assumptions.append(item)
            else:
                all_assumptions.append({"assumption": str(item), "confidence": 0.5, "category": "unknown", "source_paper": title})
        time.sleep(1.5)
    unique = deduplicate(all_assumptions)
    texts = [a["assumption"] for a in unique]
    print(f"  Got {len(all_assumptions)} raw, {len(unique)} unique assumptions")
    return {"condition": "per_paper", "raw_count": len(all_assumptions), "unique": unique, "texts": texts}


def main():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY required")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_path = PROJECT_ROOT / "data" / CATEGORY / "papers.jsonl"
    papers = load_papers(str(paper_path))[:MAX_PAPERS]
    aliases = ground_truth_aliases()
    print(f"Category: {CATEGORY}")
    print(f"Papers: {len(papers)}")
    print(f"Ground truth: {aliases[0]}")

    # Run all 3 conditions
    c1 = condition_1_single_call(papers, openai_key)
    c2 = condition_2_no_context(openai_key)
    c3 = condition_3_per_paper(papers, openai_key)

    # Compute metrics for each
    results = []
    for cond in [c1, c2, c3]:
        print(f"\nComputing soft metrics for {cond['condition']}...")
        metrics = compute_soft_metrics(cond["texts"], aliases)
        result = {
            "condition": cond["condition"],
            "raw_count": cond["raw_count"],
            "unique_count": len(cond["unique"]),
            "redundancy_rate": round(1.0 - len(cond["unique"]) / max(1, cond["raw_count"]), 4),
            **metrics,
            "assumptions": cond["unique"],
        }
        results.append(result)

        # Save per-condition
        out_path = OUTPUT_DIR / f"{cond['condition']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  Saved: {out_path}")

    # Summary
    summary = {
        "experiment": "multicall_ablation",
        "category": CATEGORY,
        "model": MODEL,
        "soft_threshold": SOFT_THRESHOLD,
        "ground_truth": aliases,
        "conditions": [
            {
                "condition": r["condition"],
                "raw_count": r["raw_count"],
                "unique_count": r["unique_count"],
                "redundancy_rate": r["redundancy_rate"],
                "best_similarity": r["best_similarity"],
                "soft_rank": r["soft_rank"],
                "soft_recall_at_5": r["soft_recall_at_5"],
                "soft_recall_at_10": r["soft_recall_at_10"],
                "soft_recall_at_20": r["soft_recall_at_20"],
            }
            for r in results
        ],
        "ran_at": datetime.now(UTC).isoformat(),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("MULTICALL ABLATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        hit = "HIT" if r["soft_recall_at_5"] > 0 else "MISS"
        print(f"  {r['condition']:25s}  raw={r['raw_count']:3d}  unique={r['unique_count']:3d}  "
              f"redundancy={r['redundancy_rate']:.2%}  best_sim={r['best_similarity']:.4f}  "
              f"rank={r['soft_rank']:3d}  [{hit}]")
    print(f"\n  Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
