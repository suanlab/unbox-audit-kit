#!/usr/bin/env python3
"""Vanilla LLM baseline: directly ask Claude to identify breakable assumptions."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_papers = cast(
    Callable[[str], list[dict[str, object]]],
    getattr(importlib.import_module("src.pipeline"), "load_papers"),
)
mean_rank = cast(
    Callable[[list[str], str], float],
    getattr(importlib.import_module("src.metrics"), "mean_rank"),
)
recall_at_k = cast(
    Callable[[list[str], str, int], float],
    getattr(importlib.import_module("src.metrics"), "recall_at_k"),
)
precision_at_k = cast(
    Callable[[list[str], str, int], float],
    getattr(importlib.import_module("src.metrics"), "precision_at_k"),
)
summarize_results = cast(
    Callable[[list[dict[str, object]], list[int]], dict[str, object]],
    getattr(
        importlib.import_module("scripts.run_retrospective_experiment"),
        "summarize_results",
    ),
)

CATEGORIES = ("transformer", "diffusion", "icl", "vit")
DEFAULT_MAPPING_PATH = "data/paradigm_shift_mapping.json"
DEFAULT_OUTPUT_DIR = "experiments/baselines/vanilla_llm"
MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_DELAY = 5


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    alnum_spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(alnum_spaced.split())


def _paper_path_for_category(category: str) -> Path:
    return PROJECT_ROOT / "data" / category / "papers.jsonl"


def _load_mapping(mapping_path: str) -> dict[str, dict[str, object]]:
    path = PROJECT_ROOT / mapping_path
    payload_obj = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload_obj, dict):
        raise ValueError("Invalid mapping payload")
    mapping: dict[str, dict[str, object]] = {}
    payload = cast(dict[object, object], payload_obj)
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            value_dict = cast(dict[object, object], value)
            mapping[key] = {
                str(inner_key): inner_value
                for inner_key, inner_value in value_dict.items()
            }
    return mapping


def _ground_truth_aliases(mapping_path: str, category: str) -> list[str]:
    entry = _load_mapping(mapping_path).get(category)
    if not entry:
        raise ValueError(f"Missing category mapping: {category}")
    aliases = [str(entry.get("broken_assumption", "")).strip()]
    alias_values = entry.get("aliases")
    if isinstance(alias_values, list):
        aliases.extend(str(alias).strip() for alias in alias_values)
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = _normalize_text(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(alias)
    return deduped


def _build_prompt(papers: list[dict[str, object]], limit: int) -> str:
    paper_block_parts: list[str] = []
    for i, paper in enumerate(papers, 1):
        title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        paper_block_parts.append(f"Paper {i}: {title}\n{abstract}")
    paper_block = "\n\n".join(paper_block_parts)

    return f"""You are an expert AI researcher. Below are abstracts from recent papers in a specific subfield.

Your task: Identify the most important **hidden assumptions** that these papers collectively take for granted. These are beliefs so deeply embedded that authors never question them — yet breaking them could lead to paradigm-shifting breakthroughs.

For each assumption, phrase it as a concise declarative statement (e.g., "Sequential processing requires recurrence", "High-quality generation requires adversarial training").

Return exactly a JSON object with a single key "assumptions" containing a list of strings. Return the {limit} most important breakable assumptions, ranked by how transformative it would be to violate them.

Papers:

{paper_block}

Return JSON only, no markdown fences:
{{"assumptions": ["assumption 1", "assumption 2", ...]}}"""


def _call_llm(prompt: str, api_key: str) -> list[str]:
    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic(api_key=api_key)

    message = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Retry {attempt}/{MAX_RETRIES} after error: {exc}")
            time.sleep(RETRY_DELAY * attempt)

    if message is None:
        raise RuntimeError("LLM call failed after all retries")
    response_content = getattr(message, "content", None)
    if not isinstance(response_content, list) or not response_content:
        raise ValueError("Empty response from LLM")
    response_text = getattr(response_content[0], "text", "")
    if not isinstance(response_text, str):
        raise ValueError("Non-text response from LLM")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    decoder = json.JSONDecoder()
    data = None
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        raise ValueError(
            f"Could not extract JSON object from response: {cleaned[:300]}"
        )
    if "assumptions" not in data:
        raise ValueError(f"Unexpected JSON structure: {data}")
    assumptions_raw = data["assumptions"]
    if not isinstance(assumptions_raw, list):
        raise ValueError(f"assumptions is not a list: {assumptions_raw}")
    return [str(item).strip() for item in assumptions_raw if str(item).strip()]


def _metric_bundle(
    predicted: list[str], aliases: list[str], top_ks: list[int]
) -> dict[str, object]:
    normalized_predicted = [_normalize_text(item) for item in predicted]
    normalized_aliases = [_normalize_text(alias) for alias in aliases]
    best_rank = float(len(predicted) + 1)
    for alias in normalized_aliases:
        best_rank = min(best_rank, mean_rank(normalized_predicted, alias))

    metrics: dict[str, object] = {"rank": best_rank}
    for k in top_ks:
        hit = any(
            recall_at_k(normalized_predicted, alias, k) > 0
            for alias in normalized_aliases
        )
        metrics[f"recall_at_{k}"] = 1.0 if hit else 0.0
        metrics[f"precision_at_{k}"] = (1.0 / k) if hit else 0.0
    return metrics


def run_vanilla_baseline(
    *,
    category: str,
    mapping_path: str,
    top_ks: list[int],
    max_papers: int | None,
    assumption_limit: int,
    api_key: str,
) -> dict[str, object]:
    papers = load_papers(str(_paper_path_for_category(category)))
    if max_papers is not None:
        papers = papers[:max_papers]
    aliases = _ground_truth_aliases(mapping_path, category)

    prompt = _build_prompt(papers, assumption_limit)
    predicted = _call_llm(prompt, api_key)[:assumption_limit]
    metrics = _metric_bundle(predicted, aliases, top_ks)

    return {
        "baseline": "vanilla_llm",
        "category": category,
        "ran_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "num_papers": len(papers),
            "ground_truth_aliases": aliases,
        },
        "config": {
            "model": MODEL,
            "top_ks": top_ks,
            "max_papers": max_papers,
            "assumption_limit": assumption_limit,
        },
        "outputs": {
            "num_candidates": len(predicted),
            "candidates": predicted,
        },
        "metrics": metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vanilla LLM baseline for retrospective comparison"
    )
    _ = parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
        choices=list(CATEGORIES),
    )
    _ = parser.add_argument("--mapping", default=DEFAULT_MAPPING_PATH)
    _ = parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    _ = parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 10, 20])
    _ = parser.add_argument("--max-papers", type=int, default=15)
    _ = parser.add_argument("--assumption-limit", type=int, default=20)
    _ = parser.add_argument("--api-key", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    categories = cast(list[str], list(args.categories))
    top_ks = sorted({v for v in cast(list[int], list(args.top_ks)) if v > 0})
    api_key = cast(str | None, args.api_key) or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY required")

    output_root = PROJECT_ROOT / cast(str, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for category in categories:
        print(f"Running vanilla LLM baseline for {category}...")
        result = run_vanilla_baseline(
            category=category,
            mapping_path=cast(str, args.mapping),
            top_ks=top_ks,
            max_papers=cast(int | None, args.max_papers),
            assumption_limit=cast(int, args.assumption_limit),
            api_key=api_key,
        )
        results.append(result)
        path = output_root / f"{category}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = summarize_results(results, top_ks)
    aggregate = {
        "ran_at": datetime.now(UTC).isoformat(),
        "baseline": "vanilla_llm",
        "summary": summary,
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2))
    print(f"Wrote vanilla LLM baseline report: {summary_path}")


if __name__ == "__main__":
    main()
