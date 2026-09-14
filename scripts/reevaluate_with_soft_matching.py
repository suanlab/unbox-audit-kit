#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_match import soft_recall_at_k, soft_rank

CATEGORIES = ("transformer", "diffusion", "icl", "vit")
DEFAULT_THRESHOLD = 0.75


def _load_result(result_dir: Path, category: str) -> dict[str, object] | None:
    path = result_dir / f"{category}.json"
    if not path.exists():
        return None
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _extract_assumptions_from_result(result: dict[str, object]) -> list[str]:
    outputs = cast(dict[str, object], result.get("outputs", {}))
    top_items = outputs.get("top_breakable_assumptions")
    if isinstance(top_items, list):
        return [
            str(item.get("assumption", ""))
            for item in top_items
            if isinstance(item, dict)
        ]
    return []


def _extract_candidates_from_baseline(result: dict[str, object]) -> list[str]:
    outputs = cast(dict[str, object], result.get("outputs", {}))
    candidates = outputs.get("candidates")
    if isinstance(candidates, list):
        return [str(c) for c in candidates if str(c).strip()]
    return _extract_assumptions_from_result(result)


def _extract_aliases(result: dict[str, object]) -> list[str]:
    inputs = cast(dict[str, object], result.get("inputs", {}))
    aliases = inputs.get("ground_truth_aliases")
    if isinstance(aliases, list):
        return [str(a) for a in aliases if str(a).strip()]
    gt = inputs.get("ground_truth_assumption")
    if isinstance(gt, str) and gt.strip():
        return [gt]
    return []


def reevaluate_category(
    result: dict[str, object],
    top_ks: list[int],
    threshold: float,
    api_key: str,
    is_baseline: bool = False,
) -> dict[str, object]:
    if is_baseline:
        predicted = _extract_candidates_from_baseline(result)
    else:
        predicted = _extract_assumptions_from_result(result)
    aliases = _extract_aliases(result)
    category = str(result.get("category", "unknown"))

    rank_value, rank_details = soft_rank(
        predicted, aliases, threshold=threshold, api_key=api_key
    )

    soft_metrics: dict[str, object] = {"soft_rank": rank_value}
    recall_details: dict[str, object] = {}
    for k in top_ks:
        recall_value, details = soft_recall_at_k(
            predicted,
            aliases,
            k,
            threshold=threshold,
            api_key=api_key,
        )
        soft_metrics[f"soft_recall_at_{k}"] = recall_value
        recall_details[f"top_{k}"] = details

    top_10_similarities = sorted(
        rank_details[:20],
        key=lambda d: cast(float, d["best_similarity"]),
        reverse=True,
    )[:10]

    return {
        "category": category,
        "threshold": threshold,
        "num_predicted": len(predicted),
        "num_aliases": len(aliases),
        "aliases": aliases,
        "soft_metrics": soft_metrics,
        "top_10_by_similarity": top_10_similarities,
        "original_metrics": result.get("metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--result-dir", required=True)
    _ = parser.add_argument("--output", required=True)
    _ = parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 10, 20])
    _ = parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    _ = parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    _ = parser.add_argument("--baseline", action="store_true")
    _ = parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = cast(str | None, args.api_key) or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY required for embedding-based evaluation")

    result_dir = Path(cast(str, args.result_dir))
    top_ks = sorted({v for v in cast(list[int], args.top_ks) if v > 0})
    categories = cast(list[str], args.categories)
    threshold = cast(float, args.threshold)

    all_results: list[dict[str, object]] = []
    for category in categories:
        result = _load_result(result_dir, category)
        if result is None:
            print(f"  [{category}] no result file found, skipping")
            continue
        print(
            f"  [{category}] re-evaluating with soft matching (threshold={threshold})..."
        )
        reeval = reevaluate_category(
            result,
            top_ks,
            threshold,
            api_key,
            is_baseline=cast(bool, args.baseline),
        )
        all_results.append(reeval)

        soft_m = cast(dict[str, object], reeval["soft_metrics"])
        print(
            f"    soft_rank={soft_m['soft_rank']}  soft_recall@10={soft_m.get('soft_recall_at_10', 'N/A')}"
        )
        top_sim = reeval["top_10_by_similarity"]
        if isinstance(top_sim, list) and top_sim:
            best = top_sim[0]
            print(
                f'    best match: sim={best["best_similarity"]}  "{str(best.get("assumption", ""))[:80]}"'
            )

    summary: dict[str, object] = {
        "ran_at": datetime.now(UTC).isoformat(),
        "threshold": threshold,
        "source": str(result_dir),
        "categories": {},
    }

    if all_results:
        avg_rank = sum(
            cast(float, cast(dict[str, object], r["soft_metrics"])["soft_rank"])
            for r in all_results
        ) / len(all_results)
        summary["average_soft_rank"] = round(avg_rank, 2)
        for k in top_ks:
            recalls = [
                cast(
                    float,
                    cast(dict[str, object], r["soft_metrics"]).get(
                        f"soft_recall_at_{k}", 0.0
                    ),
                )
                for r in all_results
            ]
            summary[f"overall_soft_recall_at_{k}"] = round(
                sum(recalls) / len(recalls), 3
            )

    for r in all_results:
        cat_name = str(r["category"])
        summary["categories"] = cast(dict[str, object], summary["categories"])
        cast(dict[str, object], summary["categories"])[cat_name] = r

    output_path = Path(cast(str, args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: avg_soft_rank={summary.get('average_soft_rank')}")
    for k in top_ks:
        print(f"  overall_soft_recall@{k}={summary.get(f'overall_soft_recall_at_{k}')}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
