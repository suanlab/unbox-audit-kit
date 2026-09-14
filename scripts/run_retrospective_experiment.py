#!/usr/bin/env python3
"""Run retrospective validation across the four target paradigm shifts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
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
run_pipeline = cast(
    Callable[..., list[dict[str, object]]],
    getattr(importlib.import_module("src.pipeline"), "run_pipeline"),
)
recall_at_k = cast(
    Callable[[list[str], str, int], float],
    getattr(importlib.import_module("src.metrics"), "recall_at_k"),
)
mean_rank = cast(
    Callable[[list[str], str], float],
    getattr(importlib.import_module("src.metrics"), "mean_rank"),
)
precision_at_k = cast(
    Callable[[list[str], str, int], float],
    getattr(importlib.import_module("src.metrics"), "precision_at_k"),
)


CATEGORIES = ("transformer", "diffusion", "icl", "vit")
DEFAULT_MAPPING_PATH = "data/paradigm_shift_mapping.json"
DEFAULT_OUTPUT_DIR = "experiments/retrospective"
MAX_ASSUMPTIONS_PER_PAPER_DEFAULT = 5
TRANSFORMATIONS_PER_ASSUMPTION = 3
MODEL_CALLS_PER_HYPOTHESIS = 10


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    alnum_spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(alnum_spaced.split())


def _load_mapping(mapping_path: str) -> dict[str, dict[str, object]]:
    mapping_file = PROJECT_ROOT / mapping_path
    with mapping_file.open("r", encoding="utf-8") as handle:
        payload_obj = cast(object, json.load(handle))

    if not isinstance(payload_obj, dict):
        raise ValueError("Invalid mapping payload: expected top-level object")

    mapping: dict[str, dict[str, object]] = {}
    payload = cast(dict[object, object], payload_obj)
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            mapping[key] = {
                str(inner_key): inner_value for inner_key, inner_value in value.items()
            }
    return mapping


def _mapping_entry(mapping_path: str, category: str) -> dict[str, object]:
    mapping = _load_mapping(mapping_path)
    category_entry = mapping.get(category)
    if not category_entry:
        raise ValueError(f"Missing mapping entry for category: {category}")
    return category_entry


def _ground_truth_aliases(mapping_path: str, category: str) -> list[str]:
    category_entry = _mapping_entry(mapping_path, category)
    ground_truth = str(category_entry.get("broken_assumption", "")).strip()
    if not ground_truth:
        raise ValueError(f"Missing broken_assumption for category: {category}")

    aliases = [ground_truth]
    alias_value = category_entry.get("aliases")
    if isinstance(alias_value, list):
        for alias in alias_value:
            alias_text = str(alias).strip()
            if alias_text:
                aliases.append(alias_text)

    seen: set[str] = set()
    deduped: list[str] = []
    for alias in aliases:
        normalized = _normalize_text(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(alias)
    return deduped


def _unique_assumptions_from_ranked_hypotheses(
    ranked_hypotheses: list[dict[str, object]],
) -> list[str]:
    seen: set[str] = set()
    assumptions: list[str] = []

    for item in ranked_hypotheses:
        raw_assumption = str(item.get("assumption", "")).strip()
        if not raw_assumption:
            continue
        normalized = _normalize_text(raw_assumption)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        assumptions.append(raw_assumption)

    return assumptions


def _paper_path_for_category(category: str) -> str:
    return f"data/{category}/papers.jsonl"


def _output_path_for_category(output_dir: str, category: str) -> Path:
    return PROJECT_ROOT / output_dir / f"{category}.json"


def _has_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _parseable_paper_count(papers_path: Path) -> int:
    try:
        return len(load_papers(str(papers_path)))
    except (FileNotFoundError, ValueError):
        return 0


def _estimated_call_volume(
    num_papers: int, max_assumptions_per_paper: int
) -> dict[str, int]:
    extraction_calls = num_papers
    hypotheses = num_papers * max_assumptions_per_paper * TRANSFORMATIONS_PER_ASSUMPTION
    evaluator_calls = hypotheses * MODEL_CALLS_PER_HYPOTHESIS
    return {
        "extraction_calls_upper_bound": extraction_calls,
        "hypothesis_count_upper_bound": hypotheses,
        "evaluator_calls_upper_bound": evaluator_calls,
        "total_model_calls_upper_bound": extraction_calls + evaluator_calls,
    }


def _best_assumption_candidates(
    ranked_hypotheses: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}

    for item in ranked_hypotheses:
        raw_assumption = str(item.get("assumption", "")).strip()
        normalized = _normalize_text(raw_assumption)
        if not normalized:
            continue

        scores_obj = item.get("scores")
        composite = 0.0
        novelty = 0.0
        if isinstance(scores_obj, dict):
            score_map = cast(dict[str, object], scores_obj)
            composite = _as_float(score_map.get("composite"))
            novelty = _as_float(score_map.get("novelty"))

        candidate = {
            "assumption": raw_assumption,
            "normalized_assumption": normalized,
            "best_composite": composite,
            "best_novelty": novelty,
            "paper_id": item.get("paper_id"),
            "paper_title": item.get("paper_title"),
            "best_hypothesis": item.get("hypothesis"),
            "best_transformation": item.get("transformation"),
        }

        existing = grouped.get(normalized)
        if existing is None:
            grouped[normalized] = candidate
            continue

        existing_composite = _as_float(existing.get("best_composite"))
        existing_novelty = _as_float(existing.get("best_novelty"))
        if composite > existing_composite or (
            composite == existing_composite and novelty > existing_novelty
        ):
            grouped[normalized] = candidate

    return sorted(
        grouped.values(),
        key=lambda item: (
            _as_float(item.get("best_composite")),
            _as_float(item.get("best_novelty")),
        ),
        reverse=True,
    )


def build_preflight(
    categories: list[str],
    mapping_path: str,
    evaluator_provider: str,
    extraction_api_key: str | None,
    evaluator_api_key: str | None,
    max_assumptions_per_paper: int,
    max_papers: int | None,
) -> dict[str, object]:
    provider_module = "anthropic" if evaluator_provider == "anthropic" else "openai"
    provider_key_name = (
        "ANTHROPIC_API_KEY" if evaluator_provider == "anthropic" else "OPENAI_API_KEY"
    )
    extraction_key = extraction_api_key or os.getenv("ANTHROPIC_API_KEY")
    evaluator_key = evaluator_api_key or os.getenv(provider_key_name)
    mapping_file = PROJECT_ROOT / mapping_path
    mapping_exists = mapping_file.exists()
    mapping = _load_mapping(mapping_path) if mapping_exists else {}

    category_checks: dict[str, object] = {}
    for category in categories:
        papers_path = PROJECT_ROOT / _paper_path_for_category(category)
        parseable_paper_count = (
            _parseable_paper_count(papers_path) if papers_path.exists() else 0
        )
        effective_paper_count = (
            min(parseable_paper_count, max_papers)
            if max_papers is not None and parseable_paper_count > 0
            else parseable_paper_count
        )
        mapping_entry = mapping.get(category)
        ground_truth_aliases = []
        if mapping_entry is not None:
            try:
                ground_truth_aliases = _ground_truth_aliases(mapping_path, category)
            except ValueError:
                ground_truth_aliases = []
        category_checks[category] = {
            "papers_path": str(papers_path),
            "papers_exists": papers_path.exists(),
            "parseable_paper_count": parseable_paper_count,
            "effective_paper_count": effective_paper_count,
            "mapping_entry_exists": mapping_entry is not None,
            "ground_truth_alias_count": len(ground_truth_aliases),
            "ground_truth_aliases": ground_truth_aliases,
            "estimated_call_volume": _estimated_call_volume(
                effective_paper_count,
                max_assumptions_per_paper,
            ),
        }

    ready = (
        mapping_exists
        and _has_import("anthropic")
        and _has_import(provider_module)
        and bool(extraction_key)
        and bool(evaluator_key)
        and all(
            isinstance(check, dict)
            and bool(check.get("papers_exists"))
            and bool(check.get("mapping_entry_exists"))
            and _as_int(check.get("parseable_paper_count")) > 0
            and _as_int(check.get("ground_truth_alias_count")) > 0
            for check in category_checks.values()
        )
    )

    return {
        "ready": ready,
        "mapping_path": mapping_path,
        "mapping_exists": mapping_exists,
        "anthropic_installed": _has_import("anthropic"),
        "evaluator_provider": evaluator_provider,
        "evaluator_dependency_installed": _has_import(provider_module),
        "extraction_api_key_present": bool(extraction_key),
        "evaluator_api_key_present": bool(evaluator_key),
        "max_assumptions_per_paper": max_assumptions_per_paper,
        "max_papers": max_papers,
        "categories": category_checks,
    }


def run_single_retrospective_experiment(
    *,
    category: str,
    mapping_path: str,
    top_ks: list[int],
    pipeline_top_k: int,
    extraction_api_key: str | None,
    evaluator_provider: str,
    evaluator_model: str | None,
    evaluator_api_key: str | None,
    max_papers: int | None,
    max_assumptions_per_paper: int,
    skip_evaluation: bool = False,
) -> dict[str, object]:
    papers_path = _paper_path_for_category(category)
    papers = load_papers(str(PROJECT_ROOT / papers_path))
    if max_papers is not None:
        papers = papers[:max_papers]
    ranked_hypotheses = run_pipeline(
        paper_path=str(PROJECT_ROOT / papers_path),
        top_k=pipeline_top_k,
        extraction_api_key=extraction_api_key,
        evaluator_provider=evaluator_provider,
        evaluator_model=evaluator_model,
        evaluator_api_key=evaluator_api_key,
        max_papers=max_papers,
        max_assumptions_per_paper=max_assumptions_per_paper,
        skip_evaluation=skip_evaluation,
    )

    ranked_assumptions = _best_assumption_candidates(ranked_hypotheses)
    predicted_assumptions = [str(item["assumption"]) for item in ranked_assumptions]
    ground_truth_aliases = _ground_truth_aliases(mapping_path, category)
    ground_truth = ground_truth_aliases[0]
    normalized_predicted = [_normalize_text(item) for item in predicted_assumptions]
    normalized_ground_truth_aliases = {
        _normalize_text(alias)
        for alias in ground_truth_aliases
        if _normalize_text(alias)
    }

    best_rank = float(len(normalized_predicted) + 1)
    for alias in normalized_ground_truth_aliases:
        best_rank = min(best_rank, mean_rank(normalized_predicted, alias))

    def _recall_for_k(k: int) -> float:
        if any(
            recall_at_k(normalized_predicted, alias, k) > 0
            for alias in normalized_ground_truth_aliases
        ):
            return 1.0
        return 0.0

    def _precision_for_k(k: int) -> float:
        if any(
            precision_at_k(normalized_predicted, alias, k) > 0
            for alias in normalized_ground_truth_aliases
        ):
            return 1.0 / k
        return 0.0

    metrics: dict[str, object] = {
        "rank": best_rank,
    }
    for top_k in top_ks:
        metrics[f"recall_at_{top_k}"] = _recall_for_k(top_k)
        metrics[f"precision_at_{top_k}"] = _precision_for_k(top_k)

    ground_truth_rank = cast(float, metrics["rank"])
    top_assumptions = (
        ranked_assumptions[: max(top_ks)] if top_ks else ranked_assumptions
    )
    ground_truth_hits = {
        f"top_{top_k}": any(
            alias in normalized_predicted[:top_k]
            for alias in normalized_ground_truth_aliases
        )
        for top_k in top_ks
    }

    return {
        "category": category,
        "ran_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "papers_path": papers_path,
            "num_papers": len(papers),
            "ground_truth_assumption": ground_truth,
            "ground_truth_aliases": ground_truth_aliases,
        },
        "config": {
            "top_ks": top_ks,
            "pipeline_top_k": pipeline_top_k,
            "max_papers": max_papers,
            "max_assumptions_per_paper": max_assumptions_per_paper,
            "evaluator_provider": evaluator_provider,
            "evaluator_model": evaluator_model,
        },
        "outputs": {
            "num_ranked_hypotheses": len(ranked_hypotheses),
            "num_unique_assumptions": len(predicted_assumptions),
            "top_breakable_assumptions": top_assumptions,
            "ground_truth_hits": ground_truth_hits,
            "ground_truth_rank": ground_truth_rank,
        },
        "metrics": metrics,
    }


def summarize_results(
    results: list[dict[str, object]], top_ks: list[int]
) -> dict[str, object]:
    summary: dict[str, object] = {
        "num_categories": len(results),
        "categories": [result["category"] for result in results],
        "average_rank": 0.0,
    }

    if not results:
        for top_k in top_ks:
            summary[f"overall_recall_at_{top_k}"] = 0.0
        return summary

    ranks = [
        _as_float(cast(dict[str, object], result["metrics"])["rank"])
        for result in results
    ]
    summary["average_rank"] = round(sum(ranks) / len(ranks), 3)

    for top_k in top_ks:
        recalls = [
            _as_float(cast(dict[str, object], result["metrics"])[f"recall_at_{top_k}"])
            for result in results
        ]
        summary[f"overall_recall_at_{top_k}"] = round(sum(recalls) / len(recalls), 3)

    return summary


def write_results(
    *,
    output_dir: str,
    preflight: dict[str, object],
    results: list[dict[str, object]],
    summary: dict[str, object],
) -> Path:
    root = PROJECT_ROOT / output_dir
    root.mkdir(parents=True, exist_ok=True)

    for result in results:
        category = str(result["category"])
        path = _output_path_for_category(output_dir, category)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary_path = root / "summary.json"
    payload = {
        "ran_at": datetime.now(UTC).isoformat(),
        "preflight": preflight,
        "summary": summary,
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrospective validation on the four Unbox paradigm shifts"
    )
    _ = parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
        choices=list(CATEGORIES),
        help="Subset of categories to evaluate",
    )
    _ = parser.add_argument("--mapping", default=DEFAULT_MAPPING_PATH)
    _ = parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    _ = parser.add_argument(
        "--top-ks",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="K values for Recall@K and Precision@K",
    )
    _ = parser.add_argument(
        "--pipeline-top-k",
        type=int,
        default=150,
        help="Number of hypotheses retained from the full pipeline before deduping assumptions",
    )
    _ = parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap on papers processed per category to control API cost",
    )
    _ = parser.add_argument(
        "--max-assumptions-per-paper",
        type=int,
        default=MAX_ASSUMPTIONS_PER_PAPER_DEFAULT,
        help="Cap assumptions extracted per paper before constraint breaking",
    )
    _ = parser.add_argument("--extraction-api-key", default=None)
    _ = parser.add_argument(
        "--evaluator-provider",
        choices=["anthropic", "openai"],
        default="anthropic",
    )
    _ = parser.add_argument("--evaluator-model", default=None)
    _ = parser.add_argument("--evaluator-api-key", default=None)
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run environment/data preflight and write summary",
    )
    _ = parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip multi-agent debate; rank by extraction confidence only (fast mode)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    categories = cast(list[str], args.categories)
    top_ks = sorted({value for value in cast(list[int], args.top_ks) if value > 0})
    if not top_ks:
        raise ValueError("top_ks must contain at least one positive integer")

    preflight = build_preflight(
        categories=categories,
        mapping_path=cast(str, args.mapping),
        evaluator_provider=cast(str, args.evaluator_provider),
        extraction_api_key=cast(str | None, args.extraction_api_key),
        evaluator_api_key=cast(str | None, args.evaluator_api_key),
        max_assumptions_per_paper=cast(int, args.max_assumptions_per_paper),
        max_papers=cast(int | None, args.max_papers),
    )

    results: list[dict[str, object]] = []
    if not cast(bool, args.dry_run):
        if not bool(preflight["ready"]):
            summary = summarize_results(results, top_ks)
            summary_path = write_results(
                output_dir=cast(str, args.output_dir),
                preflight=preflight,
                results=results,
                summary=summary,
            )
            print(json.dumps({"preflight": preflight, "summary": summary}, indent=2))
            raise SystemExit(
                "Preflight failed. See "
                f"{summary_path} or re-run with --dry-run after installing dependencies and setting API keys."
            )

        for category in categories:
            print(f"Running retrospective experiment for {category}...")
            results.append(
                run_single_retrospective_experiment(
                    category=category,
                    mapping_path=cast(str, args.mapping),
                    top_ks=top_ks,
                    pipeline_top_k=cast(int, args.pipeline_top_k),
                    extraction_api_key=cast(str | None, args.extraction_api_key),
                    evaluator_provider=cast(str, args.evaluator_provider),
                    evaluator_model=cast(str | None, args.evaluator_model),
                    evaluator_api_key=cast(str | None, args.evaluator_api_key),
                    max_papers=cast(int | None, args.max_papers),
                    max_assumptions_per_paper=cast(int, args.max_assumptions_per_paper),
                    skip_evaluation=cast(bool, args.skip_evaluation),
                )
            )

    summary = summarize_results(results, top_ks)
    summary_path = write_results(
        output_dir=cast(str, args.output_dir),
        preflight=preflight,
        results=results,
        summary=summary,
    )
    print(json.dumps({"preflight": preflight, "summary": summary}, indent=2))
    print(f"Wrote retrospective report: {summary_path}")


if __name__ == "__main__":
    main()
