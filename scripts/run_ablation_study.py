#!/usr/bin/env python3
"""2x2 Frozen Ablation: prompt-type x corpus-type on 3 primary paradigm shifts."""

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

from src.extraction_prompts import PAPER_SPECIFIC_PROMPT, FIELD_WIDE_PROMPT
from src.semantic_match import soft_recall_at_k, soft_rank

CATEGORIES = ["transformer", "diffusion", "vit"]
DEFAULT_MAPPING_PATH = "data/paradigm_shift_mapping.json"
MODEL = "claude-sonnet-4-20250514"
MAX_PAPERS = 15
MAX_ASSUMPTIONS = 5
TOP_K_VALUES = [5, 10, 20]
THRESHOLD = 0.65

CORPUS_PATHS: dict[str, dict[str, str]] = {
    "transformer": {
        "temporal_only": "data/transformer/papers.jsonl",
        "curated_representative": "data/transformer/papers.jsonl",
    },
    "diffusion": {
        "temporal_only": "data/diffusion/papers.jsonl",
        "curated_representative": "data/diffusion/papers.jsonl",
    },
    "vit": {
        "temporal_only": "data/vit/papers.jsonl",
        "curated_representative": "data/vit/papers.jsonl",
    },
}

PROMPTS: dict[str, str] = {
    "paper_specific": PAPER_SPECIFIC_PROMPT,
    "field_wide": FIELD_WIDE_PROMPT,
}

CONDITIONS: dict[str, dict[str, str]] = {
    "A": {"prompt": "paper_specific", "corpus": "temporal_only", "label": "Baseline"},
    "B": {"prompt": "field_wide", "corpus": "temporal_only", "label": "Prompt-only"},
    "C": {
        "prompt": "paper_specific",
        "corpus": "curated_representative",
        "label": "Corpus-only",
    },
    "D": {
        "prompt": "field_wide",
        "corpus": "curated_representative",
        "label": "Full (Unbox)",
    },
}


def _normalize_text(text: str) -> str:
    import re

    lowered = text.strip().lower()
    alnum = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(alnum.split())


def _load_papers(path: str) -> list[dict[str, object]]:
    load_papers_fn = getattr(importlib.import_module("src.pipeline"), "load_papers")
    return cast(list[dict[str, object]], load_papers_fn(str(PROJECT_ROOT / path)))


def _load_ground_truth_aliases(mapping_path: str, category: str) -> list[str]:
    data = json.loads((PROJECT_ROOT / mapping_path).read_text())
    entry = data.get(category, {})
    aliases = [str(entry.get("broken_assumption", "")).strip()]
    for alias in entry.get("aliases", []):
        aliases.append(str(alias).strip())
    return [a for a in aliases if a]


def _extract_assumptions_with_prompt(
    paper_text: str, prompt_template: str, api_key: str
) -> list[dict[str, object]]:
    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    prompt = prompt_template.format(paper_text=paper_text)
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
                raise
            time.sleep(5 * attempt)
    if message is None:
        raise RuntimeError("API call failed")
    text = getattr(getattr(message, "content", [None])[0], "text", "")
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict) and "assumptions" in parsed:
                return cast(list[dict[str, object]], parsed["assumptions"])
        except json.JSONDecodeError:
            continue
    return []


def _paper_text(paper: dict[str, object]) -> str:
    title = str(paper.get("title", "")).strip()
    abstract = str(paper.get("abstract", "")).strip()
    return f"{title}. {abstract}".strip()


def run_condition(
    condition_key: str,
    category: str,
    mapping_path: str,
    api_key: str,
) -> dict[str, object]:
    cond = CONDITIONS[condition_key]
    prompt_type = cond["prompt"]
    corpus_type = cond["corpus"]
    prompt_template = PROMPTS[prompt_type]
    corpus_path = CORPUS_PATHS[category][corpus_type]

    papers = _load_papers(corpus_path)[:MAX_PAPERS]
    aliases = _load_ground_truth_aliases(mapping_path, category)

    all_assumptions: list[str] = []
    seen_normalized: set[str] = set()

    for paper in papers:
        text = _paper_text(paper)
        if not text:
            continue
        assumptions = _extract_assumptions_with_prompt(text, prompt_template, api_key)
        for item in assumptions[:MAX_ASSUMPTIONS]:
            assumption = str(item.get("assumption", "")).strip()
            normalized = _normalize_text(assumption)
            if assumption and normalized not in seen_normalized:
                seen_normalized.add(normalized)
                all_assumptions.append(assumption)

    all_assumptions = all_assumptions[:200]

    rank_value, rank_details = soft_rank(
        all_assumptions, aliases, threshold=THRESHOLD, api_key=None
    )
    metrics: dict[str, object] = {"soft_rank": rank_value}
    for k in TOP_K_VALUES:
        recall_val, _ = soft_recall_at_k(
            all_assumptions, aliases, k, threshold=THRESHOLD, api_key=None
        )
        metrics[f"soft_recall_at_{k}"] = recall_val

    def _safe_float_val(v: object) -> float:
        return float(v) if isinstance(v, (int, float)) else 0.0

    best_sim = max(
        (
            _safe_float_val(d["best_similarity"])
            for d in rank_details
            if isinstance(d.get("best_similarity"), (int, float))
        ),
        default=0.0,
    )

    return {
        "condition": condition_key,
        "label": cond["label"],
        "category": category,
        "prompt_type": prompt_type,
        "corpus_type": corpus_type,
        "num_papers": len(papers),
        "num_unique_assumptions": len(all_assumptions),
        "best_similarity": round(best_sim, 4),
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
    output_dir = PROJECT_ROOT / "experiments" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for condition_key in ["A", "B", "C", "D"]:
        for category in CATEGORIES:
            out_path = output_dir / f"{condition_key}_{category}.json"
            if out_path.exists():
                print(f"Condition {condition_key}/{category}: already done, loading...")
                result = cast(dict[str, object], json.loads(out_path.read_text()))
                results.append(result)
                continue
            print(
                f"Condition {condition_key} ({CONDITIONS[condition_key]['label']}) / {category}..."
            )
            result = run_condition(
                condition_key=condition_key,
                category=category,
                mapping_path=DEFAULT_MAPPING_PATH,
                api_key=api_key,
            )
            results.append(result)
            path = output_dir / f"{condition_key}_{category}.json"
            path.write_text(json.dumps(result, indent=2))
            m = cast(dict[str, object], result["metrics"])
            print(
                f"  rank={m['soft_rank']} recall@10={m.get('soft_recall_at_10')} best_sim={result['best_similarity']}"
            )

    matrix: dict[str, dict[str, object]] = {}
    for r in results:
        cond_key = str(r["condition"])
        cat = str(r["category"])
        m = cast(dict[str, object], r["metrics"])
        if cond_key not in matrix:
            matrix[cond_key] = {
                "label": r["label"],
                "prompt_type": r["prompt_type"],
                "corpus_type": r["corpus_type"],
                "avg_recall_at_10": 0.0,
                "categories": {},
            }
        cast(dict[str, object], matrix[cond_key]["categories"])[cat] = {
            "soft_rank": m["soft_rank"],
            "soft_recall_at_10": m.get("soft_recall_at_10"),
            "best_similarity": r["best_similarity"],
        }

    for cond_key, cond_data in matrix.items():
        cats = cast(dict[str, object], cond_data["categories"])

        def _to_float(x: object) -> float:
            return float(x) if isinstance(x, (int, float)) else 0.0

        recalls = [
            _to_float(cast(dict[str, object], v).get("soft_recall_at_10", 0.0))
            for v in cats.values()
        ]
        cond_data["avg_recall_at_10"] = (
            round(sum(recalls) / len(recalls), 3) if recalls else 0.0
        )

    summary = {
        "ran_at": datetime.now(UTC).isoformat(),
        "freeze_file": "evidence/protocol_ablation_freeze.json",
        "conditions": matrix,
        "raw_results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== ABLATION RESULTS ===")
    print(f"{'Cond':<5} {'Label':<20} {'Prompt':<18} {'Corpus':<25} {'Avg R@10'}")
    for cond_key in ["A", "B", "C", "D"]:
        cond_data = matrix.get(cond_key, {})
        print(
            f"{cond_key:<5} {str(cond_data.get('label', '')):<20} {str(cond_data.get('prompt_type', '')):<18} {str(cond_data.get('corpus_type', '')):<25} {cond_data.get('avg_recall_at_10', 0.0):.3f}"
        )
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
