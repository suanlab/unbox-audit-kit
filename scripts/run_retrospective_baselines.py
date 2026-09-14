#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
import random
import re
import sys
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
summarize_results = cast(
    Callable[[list[dict[str, object]], list[int]], dict[str, object]],
    getattr(
        importlib.import_module("scripts.run_retrospective_experiment"),
        "summarize_results",
    ),
)


CATEGORIES = ("transformer", "diffusion", "icl", "vit")
DEFAULT_MAPPING_PATH = "data/paradigm_shift_mapping.json"
DEFAULT_OUTPUT_DIR = "experiments/baselines"
KEYWORD_PATTERNS = (
    "necessary",
    "requires",
    "required",
    "must",
    "depends on",
    "assume",
    "assumption",
    "sufficient",
    "crucial",
    "important",
)


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


def _candidate_sentences(papers: list[dict[str, object]]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for paper in papers:
        title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        text = f"{title}. {abstract}".strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            cleaned = sentence.strip()
            normalized = _normalize_text(cleaned)
            if len(normalized.split()) < 5 or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(cleaned)
    return candidates


def _heuristic_candidates(papers: list[dict[str, object]], limit: int) -> list[str]:
    scored: list[tuple[tuple[int, int], str]] = []
    for sentence in _candidate_sentences(papers):
        normalized = _normalize_text(sentence)
        keyword_hits = sum(1 for keyword in KEYWORD_PATTERNS if keyword in normalized)
        if keyword_hits == 0:
            continue
        length_score = -abs(len(normalized.split()) - 12)
        scored.append(((keyword_hits, length_score), sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in scored[:limit]]


def _random_candidates(
    papers: list[dict[str, object]],
    limit: int,
    seed: int,
) -> list[str]:
    candidates = _candidate_sentences(papers)
    rng = random.Random(seed)
    if len(candidates) <= limit:
        return candidates
    return rng.sample(candidates, limit)


def _metric_bundle(
    predicted: list[str], aliases: list[str], top_ks: list[int]
) -> dict[str, object]:
    normalized_predicted = [_normalize_text(item) for item in predicted]
    normalized_aliases = [_normalize_text(alias) for alias in aliases]
    best_rank = float(len(predicted) + 1)
    for alias in normalized_aliases:
        best_rank = min(best_rank, mean_rank(normalized_predicted, alias))

    metrics: dict[str, object] = {"rank": best_rank}
    for top_k in top_ks:
        hit = any(
            recall_at_k(normalized_predicted, alias, top_k) > 0
            for alias in normalized_aliases
        )
        metrics[f"recall_at_{top_k}"] = 1.0 if hit else 0.0
        metrics[f"precision_at_{top_k}"] = (1.0 / top_k) if hit else 0.0
    return metrics


def run_baseline(
    *,
    baseline_name: str,
    category: str,
    mapping_path: str,
    top_ks: list[int],
    candidate_limit: int,
    seed: int,
) -> dict[str, object]:
    papers = load_papers(str(_paper_path_for_category(category)))
    aliases = _ground_truth_aliases(mapping_path, category)

    if baseline_name == "random":
        predicted = _random_candidates(papers, max(top_ks + [candidate_limit]), seed)
    elif baseline_name == "heuristic":
        predicted = _heuristic_candidates(papers, max(top_ks + [candidate_limit]))
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    metrics = _metric_bundle(predicted, aliases, top_ks)
    return {
        "baseline": baseline_name,
        "category": category,
        "ran_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "num_papers": len(papers),
            "ground_truth_aliases": aliases,
        },
        "config": {
            "top_ks": top_ks,
            "candidate_limit": candidate_limit,
            "seed": seed,
        },
        "outputs": {
            "num_candidates": len(predicted),
            "top_candidates": predicted[: max(top_ks)],
        },
        "metrics": metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrospective baselines for Unbox"
    )
    _ = parser.add_argument(
        "--baselines",
        nargs="+",
        default=["random", "heuristic"],
        choices=["random", "heuristic"],
    )
    _ = parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
        choices=list(CATEGORIES),
    )
    _ = parser.add_argument("--mapping", default=DEFAULT_MAPPING_PATH)
    _ = parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    _ = parser.add_argument("--candidate-limit", type=int, default=20)
    _ = parser.add_argument("--seed", type=int, default=7)
    _ = parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 10, 20])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    baselines = cast(list[str], list(args.baselines))
    categories = cast(list[str], list(args.categories))
    top_ks = sorted(
        {value for value in cast(list[int], list(args.top_ks)) if value > 0}
    )
    if not top_ks:
        raise ValueError("top_ks must contain at least one positive integer")

    output_root = PROJECT_ROOT / cast(str, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, object] = {
        "ran_at": datetime.now(UTC).isoformat(),
        "results": {},
    }
    aggregate_results: dict[str, object] = {}

    for baseline_name in baselines:
        baseline_results: list[dict[str, object]] = []
        for category in categories:
            result = run_baseline(
                baseline_name=baseline_name,
                category=category,
                mapping_path=cast(str, args.mapping),
                top_ks=top_ks,
                candidate_limit=cast(int, args.candidate_limit),
                seed=cast(int, args.seed),
            )
            baseline_results.append(result)
            path = output_root / f"{baseline_name}_{category}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        aggregate_results[baseline_name] = {
            "summary": summarize_results(baseline_results, top_ks),
            "categories": baseline_results,
        }

    aggregate["results"] = aggregate_results

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote baseline report: {summary_path}")


if __name__ == "__main__":
    main()
