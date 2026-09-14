"""
TRIZ-inspired assumption transformations for hypothesis generation.

This module operationalizes a small AI-relevant subset of TRIZ principles and
uses three transformation types inspired by constraint-breaking workflows from
AutoTRIZ (arXiv:2403.13002): negate, relax, and invert.
"""

from __future__ import annotations

import json
import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, TypedDict, cast


TransformationType = Literal["negate", "relax", "invert"]


class TrizPrinciple(TypedDict):
    id: int
    name: str
    description: str
    ai_mapping: str


class HypothesisResult(TypedDict):
    original: str
    transformation: TransformationType
    transformed_assumption: str
    triz_principles: list[str]
    new_hypothesis: str
    rationale: str
    template: str


class HypothesisPatch(TypedDict, total=False):
    new_hypothesis: str
    rationale: str
    template: str


class _AnthropicTextBlock(Protocol):
    text: str


class _AnthropicMessage(Protocol):
    content: Sequence[_AnthropicTextBlock]


class _AnthropicMessagesAPI(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, str]],
    ) -> _AnthropicMessage: ...


class _AnthropicClient(Protocol):
    messages: _AnthropicMessagesAPI


class _AnthropicModule(Protocol):
    Anthropic: Callable[..., _AnthropicClient]


AI_TRIZ_PRINCIPLES: list[TrizPrinciple] = [
    {
        "id": 1,
        "name": "Segmentation",
        "description": "Split a system into independent parts",
        "ai_mapping": "Decompose monolithic models into modular or expert sub-systems",
    },
    {
        "id": 5,
        "name": "Merging",
        "description": "Combine related operations or objects",
        "ai_mapping": "Fuse retrieval, reasoning, and generation into unified model loops",
    },
    {
        "id": 6,
        "name": "Universality",
        "description": "Make one element perform multiple functions",
        "ai_mapping": "Use shared representations for multitask and multimodal learning",
    },
    {
        "id": 7,
        "name": "Nesting",
        "description": "Place one system inside another",
        "ai_mapping": "Embed specialist tools/agents inside larger planning architectures",
    },
    {
        "id": 10,
        "name": "Prior Action",
        "description": "Perform a useful action in advance",
        "ai_mapping": "Pre-compute memory, features, or synthetic curricula before training",
    },
    {
        "id": 13,
        "name": "The Other Way Around",
        "description": "Invert action direction or process logic",
        "ai_mapping": "Train from constraints/objectives backward to architecture design",
    },
    {
        "id": 15,
        "name": "Dynamics",
        "description": "Allow properties to adapt during operation",
        "ai_mapping": "Use dynamic routing, adaptive depth, and test-time compute scaling",
    },
    {
        "id": 17,
        "name": "Another Dimension",
        "description": "Move into a new dimension",
        "ai_mapping": "Expand from token-space optimization to trajectory-space planning",
    },
    {
        "id": 19,
        "name": "Periodic Action",
        "description": "Replace continuous action with periodic action",
        "ai_mapping": "Alternate optimization phases (e.g., reasoning, critique, distillation)",
    },
    {
        "id": 24,
        "name": "Intermediary",
        "description": "Use an intermediary carrier/process",
        "ai_mapping": "Use latent scratchpads, tool APIs, or symbolic buffers",
    },
    {
        "id": 25,
        "name": "Self-Service",
        "description": "Make a system maintain or improve itself",
        "ai_mapping": "Use self-reflection and self-generated data for iterative improvement",
    },
    {
        "id": 35,
        "name": "Parameter Changes",
        "description": "Change flexibility, concentration, or operating conditions",
        "ai_mapping": "Adapt precision, sparsity, context length, and temperature dynamically",
    },
]


def _extract_assumption_text(assumption: str | Mapping[str, object]) -> str:
    if isinstance(assumption, str):
        text = assumption
    else:
        text = str(assumption.get("assumption", "")).strip()

    if not text:
        raise ValueError("assumption must contain non-empty text")
    return text


def negate_assumption(assumption: str | Mapping[str, object]) -> str:
    """Transform assumption by negation: "X is necessary" -> "X is NOT necessary"."""
    text = _extract_assumption_text(assumption)
    if " is necessary" in text:
        return text.replace(" is necessary", " is NOT necessary", 1)
    return f"It is NOT necessary that {text}"


def relax_assumption(assumption: str | Mapping[str, object]) -> str:
    """Relax assumption strength: "X is necessary" -> "X is partially sufficient"."""
    text = _extract_assumption_text(assumption)
    if " is necessary" in text:
        return text.replace(" is necessary", " is partially sufficient", 1)
    return f"{text} is partially sufficient in some regimes"


def invert_assumption(assumption: str | Mapping[str, object]) -> str:
    """Invert assumption direction: "X is necessary" -> "opposite of X is better"."""
    text = _extract_assumption_text(assumption)
    if " is necessary" in text:
        subject = text.split(" is necessary", 1)[0].strip()
        return f"opposite of {subject} is better"
    return f"opposite of ({text}) is better"


def _fallback_hypothesis(
    transformed: str, transformation: TransformationType
) -> HypothesisPatch:
    if transformation == "negate":
        possibility = "alternative mechanisms can replace the assumed requirement"
        benefit = "broader architectural search and fewer inherited constraints"
    elif transformation == "relax":
        possibility = "hybrid or conditional mechanisms can satisfy task needs"
        benefit = "better efficiency-accuracy trade-offs across settings"
    else:
        possibility = "an opposite design choice can outperform the status quo"
        benefit = "discovery of non-obvious paradigms with stronger generalization"

    template = f"If {transformed}, then {possibility}, enabling {benefit}."
    return {
        "new_hypothesis": template,
        "rationale": (
            f"The {transformation} transformation weakens a hidden design lock-in and "
            "opens a plausible alternative research path."
        ),
        "template": template,
    }


def _load_anthropic_module() -> _AnthropicModule | None:
    try:
        module = importlib.import_module("anthropic")
        return cast(_AnthropicModule, cast(object, module))
    except ModuleNotFoundError:
        return None


def generate_hypothesis(
    assumption: str | Mapping[str, object],
    transformation: TransformationType,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> HypothesisResult:
    """
    Apply a TRIZ-style transformation and produce a concrete research hypothesis.

    Uses the hypothesis template:
    "If [transformed assumption], then [new possibility], enabling [benefit]".

    If Anthropic API is configured, this function asks an LLM to draft a concrete
    AI research hypothesis informed by AutoTRIZ-style contradiction reframing.
    Otherwise, it returns a deterministic fallback hypothesis.
    """
    original_text = _extract_assumption_text(assumption)

    transformers = {
        "negate": negate_assumption,
        "relax": relax_assumption,
        "invert": invert_assumption,
    }
    if transformation not in transformers:
        raise ValueError(
            f"Unsupported transformation '{transformation}'. "
            + "Use one of: negate, relax, invert."
        )

    transformed_assumption = transformers[transformation](original_text)
    selected_principles = [
        p["name"]
        for p in AI_TRIZ_PRINCIPLES
        if p["name"]
        in {"Dynamics", "Segmentation", "The Other Way Around", "Parameter Changes"}
    ]

    result: HypothesisResult = {
        "original": original_text,
        "transformation": transformation,
        "transformed_assumption": transformed_assumption,
        "triz_principles": selected_principles,
        "new_hypothesis": "",
        "rationale": "",
        "template": "",
    }

    anthropic_module = _load_anthropic_module()
    if anthropic_module is None:
        fallback = _fallback_hypothesis(transformed_assumption, transformation)
        result["new_hypothesis"] = fallback.get("new_hypothesis", "")
        result["rationale"] = fallback.get("rationale", "")
        result["template"] = fallback.get("template", "")
        return result

    prompt = (
        "You are generating one AI research hypothesis using TRIZ-style assumption breaking. "
        + "Follow AutoTRIZ methodology (arXiv:2403.13002): reformulate contradiction, "
        + "propose a concrete experimental direction, and state expected benefit.\n\n"
        + f"Original assumption: {original_text}\n"
        + f"Transformation: {transformation}\n"
        + f"Transformed assumption: {transformed_assumption}\n"
        + "Candidate TRIZ principles: "
        + ", ".join(selected_principles)
        + "\n\n"
        + "Return JSON only with keys: new_hypothesis, rationale.\n"
        + "new_hypothesis must use this exact template structure: "
        + "'If [transformed assumption], then [new possibility], enabling [benefit]'."
    )

    try:
        client = anthropic_module.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        parsed_obj_raw: object = json.loads(response_text)  # pyright: ignore[reportAny]
        if not isinstance(parsed_obj_raw, dict):
            raise ValueError("LLM response must be a JSON object")
        parsed_obj = cast(dict[str, object], parsed_obj_raw)

        hypothesis_text = str(parsed_obj.get("new_hypothesis", "")).strip()
        rationale_text = str(parsed_obj.get("rationale", "")).strip()
        if not hypothesis_text:
            raise ValueError("LLM returned empty new_hypothesis")

        result.update(
            {
                "new_hypothesis": hypothesis_text,
                "rationale": rationale_text
                or "Generated from transformed assumption via LLM.",
                "template": hypothesis_text,
            }
        )
        return result
    except Exception:
        fallback = _fallback_hypothesis(transformed_assumption, transformation)
        result["new_hypothesis"] = fallback.get("new_hypothesis", "")
        result["rationale"] = fallback.get("rationale", "")
        result["template"] = fallback.get("template", "")
        return result
