#!/usr/bin/env python3
"""Self-Consistent Assumption Miner (SCAM): same prompt+corpus as Unbox, no TRIZ, 5-sample majority vote."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction_prompts import FIELD_WIDE_PROMPT
from src.semantic_match import soft_recall_at_k, soft_rank

CATEGORIES = ("transformer", "diffusion", "icl", "vit")
DEFAULT_MAPPING_PATH = "data/paradigm_shift_mapping.json"
MODEL = "claude-sonnet-4-20250514"
MAX_PAPERS = 60
MAX_ASSUMPTIONS = 5
N_SAMPLES = 5
TOP_K_VALUES = [5, 10, 20]
THRESHOLD = 0.65


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    alnum = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(alnum.split())


def _load_papers(category: str) -> list[dict[str, object]]:
    load_papers_fn = getattr(importlib.import_module("src.pipeline"), "load_papers")
    path = PROJECT_ROOT / "data" / category / "papers.jsonl"
    return cast(list[dict[str, object]], load_papers_fn(str(path)))[:MAX_PAPERS]


def _load_aliases(mapping_path: str, category: str) -> list[str]:
    data = json.loads((PROJECT_ROOT / mapping_path).read_text())
    entry = data.get(category, {})
    aliases = [str(entry.get("broken_assumption", "")).strip()]
    for alias in entry.get("aliases", []):
        aliases.append(str(alias).strip())
    return [a for a in aliases if a]


def _call_once(paper_text: str, api_key: str) -> list[str]:
    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    prompt = FIELD_WIDE_PROMPT.format(paper_text=paper_text)
    message = None
    for attempt in range(1, 4):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception:
            if attempt == 3:
                return []
            time.sleep(5 * attempt)
    if message is None:
        return []
    text = getattr(getattr(message, "content", [None])[0], "text", "")
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict) and "assumptions" in parsed:
                return [
                    str(a.get("assumption", "")).strip()
                    for a in parsed["assumptions"]
                    if a.get("assumption")
                ]
        except json.JSONDecodeError:
            continue
    return []


def run_scam(category: str, mapping_path: str, api_key: str) -> dict[str, object]:
    papers = _load_papers(category)
    aliases = _load_aliases(mapping_path, category)

    per_paper_vote_counts: Counter[str] = Counter()
    per_paper_assumptions: dict[str, list[str]] = {}

    for paper in papers:
        title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        paper_text = f"{title}. {abstract}".strip()
        if not paper_text:
            continue

        paper_key = title[:60]
        assumption_votes: Counter[str] = Counter()

        for _ in range(N_SAMPLES):
            assumptions = _call_once(paper_text, api_key)
            for assumption in assumptions[:MAX_ASSUMPTIONS]:
                normalized = _normalize(assumption)
                if normalized:
                    assumption_votes[normalized] += 1

        top_for_paper = [
            norm for norm, _ in assumption_votes.most_common(MAX_ASSUMPTIONS)
        ]
        per_paper_assumptions[paper_key] = top_for_paper
        for norm in top_for_paper:
            per_paper_vote_counts[norm] += 1

    ranked_assumptions = [
        assumption for assumption, _ in per_paper_vote_counts.most_common(50)
    ]

    rank_value, _ = soft_rank(
        ranked_assumptions, aliases, threshold=THRESHOLD, api_key=None
    )
    metrics: dict[str, object] = {"soft_rank": rank_value}
    for k in TOP_K_VALUES:
        recall_val, _ = soft_recall_at_k(
            ranked_assumptions, aliases, k, threshold=THRESHOLD, api_key=None
        )
        metrics[f"soft_recall_at_{k}"] = recall_val

    return {
        "baseline": "scam",
        "category": category,
        "ran_at": datetime.now(UTC).isoformat(),
        "config": {
            "n_samples": N_SAMPLES,
            "max_papers": MAX_PAPERS,
            "max_assumptions_per_sample": MAX_ASSUMPTIONS,
            "threshold": THRESHOLD,
            "prompt": "field_wide",
        },
        "inputs": {
            "num_papers": len(papers),
            "ground_truth_aliases": aliases,
        },
        "outputs": {
            "num_unique_assumptions": len(ranked_assumptions),
            "top_20": ranked_assumptions[:20],
        },
        "metrics": metrics,
    }


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY required")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY required for semantic matching")
    os.environ["OPENAI_API_KEY"] = openai_key

    output_dir = PROJECT_ROOT / "experiments" / "baselines" / "scam"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for category in CATEGORIES:
        print(
            f"Running SCAM on {category} ({N_SAMPLES} samples × {MAX_PAPERS} papers)..."
        )
        result = run_scam(category, DEFAULT_MAPPING_PATH, api_key)
        results.append(result)
        path = output_dir / f"{category}.json"
        path.write_text(json.dumps(result, indent=2))
        m = cast(dict[str, object], result["metrics"])
        print(f"  rank={m['soft_rank']} recall@10={m.get('soft_recall_at_10')}")

    def _to_f(v: object) -> float:
        return float(v) if isinstance(v, (int, float)) else 0.0

    overall_recall_10 = sum(
        _to_f(cast(dict[str, object], r["metrics"]).get("soft_recall_at_10", 0.0))
        for r in results
    ) / len(results)
    avg_rank = sum(
        _to_f(cast(dict[str, object], r["metrics"])["soft_rank"]) for r in results
    ) / len(results)

    summary = {
        "ran_at": datetime.now(UTC).isoformat(),
        "baseline": "scam",
        "overall_soft_recall_at_10": round(overall_recall_10, 3),
        "avg_soft_rank": round(avg_rank, 2),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSCAM: overall_recall@10={overall_recall_10:.3f} avg_rank={avg_rank:.1f}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
