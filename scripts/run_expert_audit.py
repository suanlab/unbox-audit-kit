#!/usr/bin/env python3
"""Multi-persona expert audit of prospective hypotheses."""

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

MODEL = "claude-sonnet-4-20250514"

PERSONAS = {
    "nlp_researcher": "You are a senior NLP researcher at a top university. You value linguistic grounding, evaluation rigor, and practical impact on language understanding tasks. You are skeptical of ideas that lack clear evaluation methodology.",
    "cv_researcher": "You are a senior computer vision researcher. You value geometric reasoning, visual understanding, and computational efficiency. You judge ideas by whether they could produce measurable improvements on standard benchmarks.",
    "ml_theorist": "You are a machine learning theorist. You value mathematical rigor, generalization guarantees, and principled frameworks. You are skeptical of empirical-only claims and look for theoretical justification.",
    "skeptical_reviewer": "You are a famously tough NeurIPS reviewer (top-5% in review quality). You actively look for flaws, overclaims, and lack of novelty. You only give high scores to ideas that are genuinely new AND well-motivated.",
    "industry_practitioner": "You are a senior ML engineer at a major tech company. You value scalability, practical deployment, and real-world impact. You are skeptical of ideas that only work in toy settings.",
}

JUDGE_PROMPT = """{persona}

Rate each of these research hypotheses on four dimensions (1-10 scale):
- novelty: How genuinely new is this? (1=well-known, 5=minor twist, 10=paradigm-shifting)
- feasibility: Can this be tested within 1-2 years? (1=impossible, 5=challenging, 10=straightforward)
- impact: If validated, how significant? (1=trivial, 5=useful, 10=field-changing)
- clarity: Is the hypothesis precise enough to test? (1=vague, 5=needs work, 10=ready to implement)

Be honest and calibrated. Not every idea deserves high scores.

Hypotheses:
{hypotheses_text}

Return JSON only:
{{"ratings": [
  {{"index": 0, "novelty": N, "feasibility": N, "impact": N, "clarity": N, "comment": "brief"}}
]}}"""


def _safe_int(v: object) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return -1


def _safe_float(v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _call_judge(prompt: str, api_key: str) -> dict[str, object]:
    anthropic = importlib.import_module("anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    message = None
    for attempt in range(1, 4):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    if message is None:
        raise RuntimeError("Judge call failed")
    text = getattr(getattr(message, "content", [None])[0], "text", "")
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast(dict[str, object], parsed)
    raise ValueError(f"Could not parse JSON: {text[:300]}")


def _format_hypotheses(hypotheses: list[dict[str, object]]) -> str:
    parts = []
    for i, h in enumerate(hypotheses):
        parts.append(
            f"[{i}] Paradigm: {h.get('paradigm', '?')}\n"
            f'    Breaks: "{h.get("original_assumption", "")}"\n'
            f"    Via: {h.get('transformation', '?')}\n"
            f"    Hypothesis: {h.get('hypothesis', '')}\n"
            f"    Rationale: {h.get('rationale', '')}"
        )
    return "\n\n".join(parts)


def run_audit(hypotheses: list[dict[str, object]], api_key: str) -> dict[str, object]:
    hypotheses_text = _format_hypotheses(hypotheses)
    all_ratings: dict[str, list[dict[str, object]]] = {}

    for persona_name, persona_desc in PERSONAS.items():
        print(f"  Judge: {persona_name}...")
        prompt = JUDGE_PROMPT.format(
            persona=persona_desc, hypotheses_text=hypotheses_text
        )
        result = _call_judge(prompt, api_key)
        ratings = result.get("ratings", [])
        if not isinstance(ratings, list):
            ratings = []
        all_ratings[persona_name] = [r for r in ratings if isinstance(r, dict)]

    dims = ["novelty", "feasibility", "impact", "clarity"]
    hypothesis_scores: list[dict[str, object]] = []

    for h_idx, h in enumerate(hypotheses):
        per_judge: dict[str, dict[str, float]] = {}
        for judge_name, ratings in all_ratings.items():
            matched = [r for r in ratings if _safe_int(r.get("index", -1)) == h_idx]
            if matched:
                r = matched[0]
                per_judge[judge_name] = {d: _safe_float(r.get(d, 5)) for d in dims}

        avg_scores: dict[str, float] = {}
        for d in dims:
            vals = [scores[d] for scores in per_judge.values() if d in scores]
            avg_scores[d] = round(sum(vals) / len(vals), 2) if vals else 0.0
        avg_scores["composite"] = round(sum(avg_scores.values()) / len(dims), 2)

        std_scores: dict[str, float] = {}
        for d in dims:
            vals = [scores[d] for scores in per_judge.values() if d in scores]
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                std_scores[d] = round(var**0.5, 2)
            else:
                std_scores[d] = 0.0

        hypothesis_scores.append(
            {
                **h,
                "avg_scores": avg_scores,
                "std_scores": std_scores,
                "per_judge": per_judge,
                "num_judges": len(per_judge),
            }
        )

    hypothesis_scores.sort(
        key=lambda x: _safe_float(
            cast(dict[str, object], x["avg_scores"]).get("composite", 0)
        ),
        reverse=True,
    )

    overall_avgs: dict[str, float] = {}
    for d in dims + ["composite"]:
        vals = [
            _safe_float(cast(dict[str, object], h["avg_scores"]).get(d, 0))
            for h in hypothesis_scores
        ]
        overall_avgs[d] = round(sum(vals) / len(vals), 2) if vals else 0.0

    agreement_per_dim: dict[str, float] = {}
    for d in dims:
        judge_names = list(PERSONAS.keys())
        ratings_matrix: list[list[float]] = []
        for jn in judge_names:
            row = []
            for h_idx in range(len(hypotheses)):
                matched_h = [
                    hs
                    for hs in hypothesis_scores
                    if isinstance(hs.get("per_judge"), dict)
                    and cast(dict[str, object], hs["per_judge"]).get(jn) is not None
                ]
                if h_idx < len(matched_h):
                    per_j = cast(
                        dict[str, dict[str, float]],
                        matched_h[h_idx].get("per_judge", {}),
                    )
                    judge_scores = per_j.get(jn, {})
                    row.append(judge_scores.get(d, 5.0))
                else:
                    row.append(5.0)
            ratings_matrix.append(row)

        from src.metrics import inter_rater_agreement

        agreement_per_dim[d] = round(inter_rater_agreement(ratings_matrix), 3)

    return {
        "ran_at": datetime.now(UTC).isoformat(),
        "num_hypotheses": len(hypotheses),
        "num_judges": len(PERSONAS),
        "judge_personas": list(PERSONAS.keys()),
        "overall_averages": overall_avgs,
        "inter_rater_agreement": agreement_per_dim,
        "evaluations": hypothesis_scores,
    }


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY required")

    summary_path = PROJECT_ROOT / "experiments" / "prospective" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    top_hypotheses = cast(list[dict[str, object]], summary.get("top_12_hypotheses", []))
    if not top_hypotheses:
        raise SystemExit("No hypotheses to audit")

    print(
        f"Running expert audit with {len(PERSONAS)} judge personas on {len(top_hypotheses)} hypotheses..."
    )
    result = run_audit(top_hypotheses, api_key)

    out_path = PROJECT_ROOT / "experiments" / "prospective" / "expert_audit.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nOverall averages (5 judges × {len(top_hypotheses)} hypotheses):")
    for dim, val in cast(dict[str, float], result["overall_averages"]).items():
        print(f"  {dim}: {val}")
    print(f"\nInter-rater agreement (Krippendorff's alpha):")
    for dim, val in cast(dict[str, float], result["inter_rater_agreement"]).items():
        print(f"  {dim}: {val}")
    print(f"\nTop 5 by consensus:")
    evals = cast(list[dict[str, object]], result["evaluations"])
    for i, e in enumerate(evals[:5], 1):
        avg = cast(dict[str, float], e["avg_scores"])
        std = cast(dict[str, float], e["std_scores"])
        print(
            f"  {i}. [{avg['composite']:.1f} ±{std.get('novelty', 0):.1f}] {str(e.get('hypothesis', ''))[:80]}"
        )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
