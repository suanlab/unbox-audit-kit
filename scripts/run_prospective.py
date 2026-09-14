#!/usr/bin/env python3
"""Generate prospective breakable-assumption hypotheses for current AI paradigms."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CURRENT_PARADIGMS = {
    "autoregressive_llm": {
        "description": "Autoregressive language modeling with next-token prediction",
        "seed_assumptions": [
            "Next-token prediction is the optimal training objective for language understanding",
            "Autoregressive generation is necessary for coherent long-form text",
            "Tokenization into discrete subword units is necessary for language modeling",
        ],
    },
    "rlhf_alignment": {
        "description": "Reinforcement Learning from Human Feedback for AI alignment",
        "seed_assumptions": [
            "Human preference labels are necessary for aligning language models",
            "Reward models can accurately capture human values and preferences",
            "RLHF-style training is necessary to make language models safe and helpful",
        ],
    },
    "scaling_laws": {
        "description": "Neural scaling laws and compute-optimal training",
        "seed_assumptions": [
            "Scaling model parameters and data jointly is the primary path to capability improvement",
            "Power-law scaling relationships are fundamental rather than artifact of current architectures",
            "Compute-optimal training requires balancing parameters and tokens according to fixed ratios",
        ],
    },
    "attention_mechanism": {
        "description": "Self-attention as the core building block of modern architectures",
        "seed_assumptions": [
            "Self-attention over all token pairs is necessary for capturing long-range dependencies",
            "Quadratic attention complexity is an acceptable cost for sequence modeling quality",
            "Attention patterns learned from data are superior to structured or fixed connectivity",
        ],
    },
}

MODEL = "claude-sonnet-4-20250514"

PROSPECTIVE_PROMPT = """You are an expert AI researcher analyzing the current dominant paradigms in AI/ML.

Given this paradigm description and seed assumptions, generate novel research hypotheses by systematically breaking each assumption using TRIZ-inspired transformations:
- NEGATE: What if this assumption is simply wrong?
- RELAX: What if this assumption only holds partially?
- INVERT: What if the opposite is true?

For each generated hypothesis, assess:
- novelty (1-10): How different is this from existing published work?
- feasibility (1-10): Could this be tested with current resources?
- transformational_potential (1-10): If validated, how much would this change the field?

Paradigm: {paradigm_name}
Description: {paradigm_description}

Seed assumptions to break:
{seed_assumptions_text}

Return JSON only:
{{"hypotheses": [
  {{
    "original_assumption": "the assumption being broken",
    "transformation": "negate|relax|invert",
    "hypothesis": "the new research hypothesis",
    "rationale": "why this could be transformative (2-3 sentences)",
    "novelty": 1-10,
    "feasibility": 1-10,
    "transformational_potential": 1-10
  }}
]}}

Generate 6-9 hypotheses (2-3 per seed assumption). Be bold but grounded."""


def _call_llm(prompt: str, api_key: str) -> dict[str, object]:
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
        except Exception as exc:
            if attempt == 3:
                raise
            time.sleep(5 * attempt)

    if message is None:
        raise RuntimeError("LLM call failed")

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
    raise ValueError(f"Could not parse JSON from response: {text[:300]}")


def generate_prospective(
    paradigm_name: str, config: Mapping[str, object], api_key: str
) -> dict[str, object]:
    seeds = cast(list[str], config["seed_assumptions"])
    seed_text = "\n".join(f"  {i}. {s}" for i, s in enumerate(seeds, 1))
    prompt = PROSPECTIVE_PROMPT.format(
        paradigm_name=paradigm_name,
        paradigm_description=str(config["description"]),
        seed_assumptions_text=seed_text,
    )
    result = _call_llm(prompt, api_key)
    hypotheses = result.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        hypotheses = []

    scored = []
    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        n = float(h.get("novelty", 5))
        f = float(h.get("feasibility", 5))
        t = float(h.get("transformational_potential", 5))
        h["composite"] = round((n + f + t) / 3, 2)
        scored.append(h)

    scored.sort(key=lambda x: float(x.get("composite", 0)), reverse=True)

    return {
        "paradigm": paradigm_name,
        "description": str(config["description"]),
        "seed_assumptions": seeds,
        "ran_at": datetime.now(UTC).isoformat(),
        "num_hypotheses": len(scored),
        "hypotheses": scored,
    }


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY required")

    output_dir = PROJECT_ROOT / "experiments" / "prospective"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    for paradigm_name, config in CURRENT_PARADIGMS.items():
        print(f"Generating hypotheses for {paradigm_name}...")
        result = generate_prospective(paradigm_name, config, api_key)
        all_results.append(result)

        path = output_dir / f"{paradigm_name}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  → {result['num_hypotheses']} hypotheses generated")

    total = sum(cast(int, r["num_hypotheses"]) for r in all_results)
    top_hypotheses = []
    for r in all_results:
        hyps = cast(list[dict[str, object]], r["hypotheses"])
        for h in hyps[:3]:
            h["paradigm"] = r["paradigm"]
            top_hypotheses.append(h)
    top_hypotheses.sort(key=lambda x: float(x.get("composite", 0)), reverse=True)

    summary = {
        "ran_at": datetime.now(UTC).isoformat(),
        "total_hypotheses": total,
        "paradigms": [r["paradigm"] for r in all_results],
        "top_12_hypotheses": top_hypotheses[:12],
        "results": all_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTotal: {total} hypotheses across {len(all_results)} paradigms")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
