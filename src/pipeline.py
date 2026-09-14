from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import importlib
import json
from pathlib import Path
from typing import Protocol, cast


ConstraintTransform = Callable[[str], str]
JSONObject = dict[str, object]


class ConstraintBreakerModule(Protocol):
    def negate_assumption(self, assumption: str) -> str: ...

    def relax_assumption(self, assumption: str) -> str: ...

    def invert_assumption(self, assumption: str) -> str: ...


def _constraint_transform(name: str) -> ConstraintTransform:
    module = cast(
        ConstraintBreakerModule,
        cast(object, importlib.import_module("src.constraint_breaker")),
    )
    transforms: dict[str, ConstraintTransform] = {
        "negate": module.negate_assumption,
        "relax": module.relax_assumption,
        "invert": module.invert_assumption,
    }
    if name not in transforms:
        raise ValueError(f"Unknown transformation: {name}")
    return transforms[name]


def load_papers(paper_path: str) -> list[JSONObject]:
    path = Path(paper_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper file not found: {paper_path}")

    papers: list[JSONObject] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = cast(object, json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {paper_path}: {exc}"
                ) from exc
            if isinstance(record, dict):
                record_dict = cast(dict[object, object], record)
                papers.append({str(key): value for key, value in record_dict.items()})

    return papers


def _paper_text(record: Mapping[str, object]) -> str:
    title = str(record.get("title", "")).strip()
    abstract = str(record.get("abstract", "")).strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def _transformed_hypotheses(assumption: str) -> list[tuple[str, str]]:
    negate = _constraint_transform("negate")
    relax = _constraint_transform("relax")
    invert = _constraint_transform("invert")
    return [
        ("negate", negate(assumption)),
        ("relax", relax(assumption)),
        ("invert", invert(assumption)),
    ]


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


def _scores_from_evaluation(
    evaluation: Mapping[str, object],
) -> tuple[float, float, float]:
    return (
        _as_float(evaluation.get("novelty")),
        _as_float(evaluation.get("feasibility")),
        _as_float(evaluation.get("transformational_potential")),
    )


def _sort_key(item: Mapping[str, object]) -> tuple[float, float]:
    scores_obj = item.get("scores")
    if not isinstance(scores_obj, Mapping):
        return (0.0, 0.0)
    scores = cast(Mapping[str, object], scores_obj)
    return (_as_float(scores.get("composite")), _as_float(scores.get("novelty")))


def run_pipeline(
    paper_path: str,
    top_k: int = 10,
    extraction_api_key: str | None = None,
    evaluator_provider: str = "anthropic",
    evaluator_model: str | None = None,
    evaluator_api_key: str | None = None,
    max_papers: int | None = None,
    max_assumptions_per_paper: int | None = None,
    evaluator_rounds: int = 2,
    skip_evaluation: bool = False,
) -> list[JSONObject]:
    from .extraction import extract_assumptions

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_papers is not None and max_papers <= 0:
        raise ValueError("max_papers must be greater than 0 when provided")
    if max_assumptions_per_paper is not None and max_assumptions_per_paper <= 0:
        raise ValueError(
            "max_assumptions_per_paper must be greater than 0 when provided"
        )

    papers = load_papers(paper_path)
    if max_papers is not None:
        papers = papers[:max_papers]

    evaluator = None
    if not skip_evaluation:
        from .multi_agent_evaluator import MultiAgentEvaluator

        evaluator = MultiAgentEvaluator(
            provider=evaluator_provider,
            model=evaluator_model,
            api_key=evaluator_api_key,
            rounds=evaluator_rounds,
        )

    scored: list[JSONObject] = []

    for record in papers:
        paper_text = _paper_text(record)
        if not paper_text:
            continue

        assumptions = extract_assumptions(paper_text, api_key=extraction_api_key)
        if max_assumptions_per_paper is not None:
            assumptions = assumptions[:max_assumptions_per_paper]
        for assumption_item in assumptions:
            assumption = str(assumption_item.get("assumption", "")).strip()
            if not assumption:
                continue
            confidence = _as_float(assumption_item.get("confidence", 0.5))

            for transformation, new_hypothesis in _transformed_hypotheses(assumption):
                if evaluator is not None:
                    hypothesis = {
                        "original_assumption": assumption,
                        "new_hypothesis": new_hypothesis,
                        "transformation": transformation,
                    }
                    evaluation = evaluator.evaluate_hypothesis(hypothesis)
                    novelty, feasibility, transformational = _scores_from_evaluation(
                        evaluation
                    )
                    composite_score = round(
                        (novelty + feasibility + transformational) / 3,
                        2,
                    )
                    debate_summary = evaluation.get("debate_summary", "")
                    rationale = evaluation.get("rationale", "")
                    consensus = evaluation.get("consensus_reached", False)
                else:
                    novelty = confidence * 10.0
                    feasibility = 5.0
                    transformational = confidence * 10.0
                    composite_score = round(
                        (novelty + feasibility + transformational) / 3, 2
                    )
                    debate_summary = "skipped"
                    rationale = "evaluation skipped — confidence-based scoring"
                    consensus = True

                scored.append(
                    {
                        "paper_id": record.get("paperId"),
                        "paper_title": record.get("title"),
                        "assumption": assumption,
                        "assumption_confidence": assumption_item.get("confidence"),
                        "assumption_category": assumption_item.get("category"),
                        "transformation": transformation,
                        "hypothesis": new_hypothesis,
                        "scores": {
                            "novelty": novelty,
                            "feasibility": feasibility,
                            "transformational_potential": transformational,
                            "composite": composite_score,
                        },
                        "debate_summary": debate_summary,
                        "rationale": rationale,
                        "consensus_reached": consensus,
                    }
                )

    ranked = sorted(
        scored,
        key=_sort_key,
        reverse=True,
    )
    return ranked[:top_k]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Unbox hypothesis pipeline")
    _ = parser.add_argument(
        "--paper",
        required=True,
        help="Path to paper JSONL file (e.g., data/transformer/papers.jsonl)",
    )
    _ = parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-ranked hypotheses to return",
    )
    _ = parser.add_argument(
        "--extraction-api-key",
        default=None,
        help="Anthropic API key for assumption extraction (optional)",
    )
    _ = parser.add_argument(
        "--evaluator-provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Provider for multi-agent evaluator",
    )
    _ = parser.add_argument(
        "--evaluator-model",
        default=None,
        help="Model name for evaluator backend (optional)",
    )
    _ = parser.add_argument(
        "--evaluator-api-key",
        default=None,
        help="API key for evaluator backend (optional)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    paper_path = cast(str, getattr(args, "paper"))
    top_k = cast(int, getattr(args, "top_k"))
    extraction_api_key = cast(str | None, getattr(args, "extraction_api_key"))
    evaluator_provider = cast(str, getattr(args, "evaluator_provider"))
    evaluator_model = cast(str | None, getattr(args, "evaluator_model"))
    evaluator_api_key = cast(str | None, getattr(args, "evaluator_api_key"))
    ranked_hypotheses = run_pipeline(
        paper_path=paper_path,
        top_k=top_k,
        extraction_api_key=extraction_api_key,
        evaluator_provider=evaluator_provider,
        evaluator_model=evaluator_model,
        evaluator_api_key=evaluator_api_key,
    )
    print(json.dumps(ranked_hypotheses, indent=2))


if __name__ == "__main__":
    main()
