from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import os
from typing import Protocol, cast


JSONDict = dict[str, object]
ScoreDict = dict[str, float]

SCORE_DIMENSIONS = (
    "novelty",
    "feasibility",
    "transformational_potential",
)

ADVOCATE_SYSTEM_PROMPT = """You are an enthusiastic researcher who sees potential in novel ideas.

Your job is to argue for the upside of a scientific hypothesis, explain why the
idea could matter, and show how criticism can be addressed. Be ambitious but not
reckless: ground your optimism in plausible mechanisms, experiments, or research
paths.

Always return valid JSON only.
"""

CRITIC_SYSTEM_PROMPT = """You are a skeptical reviewer who identifies flaws and risks.

Your job is to stress-test a scientific hypothesis, expose hidden assumptions,
point out failure modes, and explain why the idea may be infeasible,
incremental, or misleading. Be rigorous and fair rather than dismissive.

Always return valid JSON only.
"""

JUDGE_SYSTEM_PROMPT = """You are a neutral evaluator who scores ideas objectively.

Your job is to synthesize debate between an Advocate and a Critic, identify what
is genuinely novel versus merely restated, decide whether the idea is feasible
to test with current methods, and assess whether it could transform a field if
validated.

Use the debate to drive convergence across rounds. In intermediate rounds, ask
clarifying questions that reduce uncertainty. In the final round, assign scores
on a 1-10 scale for novelty, feasibility, and transformational potential.

Always return valid JSON only.
"""

DEFAULT_CALIBRATION_CASES = [
    {
        "name": "transformer_breakthrough",
        "label": "breakthrough",
        "hypothesis": {
            "original_assumption": "Recurrence is necessary for sequence modeling.",
            "new_hypothesis": "Replace recurrence with self-attention and parallel processing for sequence modeling.",
            "transformation": "negate",
        },
    },
    {
        "name": "vision_transformer_breakthrough",
        "label": "breakthrough",
        "hypothesis": {
            "original_assumption": "Vision tasks require convolutional inductive biases.",
            "new_hypothesis": "Model images as patch sequences and use transformer architectures instead of convolutions.",
            "transformation": "negate",
        },
    },
    {
        "name": "incremental_lstm_baseline",
        "label": "incremental",
        "hypothesis": {
            "original_assumption": "LSTM capacity limits sequence performance.",
            "new_hypothesis": "Add one more LSTM layer and tune dropout for better sequence accuracy.",
            "transformation": "relax",
        },
    },
]


class ResponseBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class _AnthropicMessageProtocol(Protocol):
    content: object


class _AnthropicMessagesProtocol(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, str]],
    ) -> _AnthropicMessageProtocol: ...


class _AnthropicClientProtocol(Protocol):
    messages: _AnthropicMessagesProtocol


class _AnthropicConstructorProtocol(Protocol):
    def __call__(self, *, api_key: str | None = None) -> _AnthropicClientProtocol: ...


class _OpenAIMessageProtocol(Protocol):
    content: str | None


class _OpenAIChoiceProtocol(Protocol):
    message: _OpenAIMessageProtocol


class _OpenAIResponseProtocol(Protocol):
    choices: object


class _OpenAICompletionsProtocol(Protocol):
    def create(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        messages: list[dict[str, str]],
    ) -> _OpenAIResponseProtocol: ...


class _OpenAIChatProtocol(Protocol):
    completions: _OpenAICompletionsProtocol


class _OpenAIClientProtocol(Protocol):
    chat: _OpenAIChatProtocol


class _OpenAIConstructorProtocol(Protocol):
    def __call__(self, *, api_key: str | None = None) -> _OpenAIClientProtocol: ...


@dataclass(slots=True)
class DebateTurn:
    round_number: int
    agent: str
    argument: str
    key_points: list[str] = field(default_factory=list)
    score_estimate: ScoreDict = field(default_factory=dict)
    clarifying_questions: list[str] = field(default_factory=list)
    convergence_note: str = ""


class APIModelBackend:
    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> None:
        self.provider: str = provider.lower()
        self.model: str = model or self._default_model(self.provider)
        self.api_key: str | None = api_key or self._default_api_key(self.provider)
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

        if self.provider not in {"anthropic", "openai"}:
            raise ValueError("provider must be 'anthropic' or 'openai'")
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass api_key explicitly."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        return self._generate_openai(system_prompt, user_prompt)

    @staticmethod
    def _default_model(provider: str) -> str:
        defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4.1",
        }
        return defaults[provider]

    @staticmethod
    def _default_api_key(provider: str) -> str | None:
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        return os.getenv(env_var)

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for provider='anthropic'"
            ) from exc

        anthropic_constructor = cast(
            _AnthropicConstructorProtocol,
            getattr(anthropic_module, "Anthropic"),
        )
        client = anthropic_constructor(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content_blocks = cast(list[object], message.content)
        return "".join(str(getattr(block, "text", "")) for block in content_blocks)

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            raise ImportError(
                "openai package is required for provider='openai'"
            ) from exc

        openai_constructor = cast(
            _OpenAIConstructorProtocol,
            getattr(openai_module, "OpenAI"),
        )
        client = openai_constructor(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choices = cast(list[_OpenAIChoiceProtocol], response.choices)
        content = choices[0].message.content if choices else None
        return content or ""


class BaseDebateAgent:
    def __init__(self, name: str, system_prompt: str, backend: ResponseBackend) -> None:
        self.name: str = name
        self.system_prompt: str = system_prompt
        self.backend: ResponseBackend = backend

    def _generate(self, prompt: str) -> JSONDict:
        response_text = self.backend.generate(self.system_prompt, prompt)
        return _extract_json_object(response_text)


class AdvocateAgent(BaseDebateAgent):
    def __init__(self, backend: ResponseBackend) -> None:
        super().__init__("Advocate", ADVOCATE_SYSTEM_PROMPT, backend)

    def argue(
        self,
        hypothesis: dict[str, str],
        transcript: list[DebateTurn],
        round_number: int,
        total_rounds: int,
    ) -> DebateTurn:
        prompt = f"""
Evaluate this generated scientific hypothesis in round {round_number} of {total_rounds}.

Hypothesis:
{json.dumps(hypothesis, indent=2)}

Debate transcript so far:
{_format_transcript(transcript)}

Instructions:
- Round 1: focus on why the idea could matter and what paradigm it may break.
- Later rounds: directly answer the Critic's strongest concerns and the Judge's questions.
- Estimate scores on a 1-10 scale from the perspective of an optimistic but credible advocate.

Return only JSON with this schema:
{{
  "argument": "short paragraph",
  "key_points": ["point 1", "point 2", "point 3"],
  "score_estimate": {{
    "novelty": 0,
    "feasibility": 0,
    "transformational_potential": 0
  }}
}}
"""
        payload = self._generate(prompt)
        return DebateTurn(
            round_number=round_number,
            agent=self.name,
            argument=str(payload.get("argument", "")).strip(),
            key_points=_coerce_string_list(payload.get("key_points")),
            score_estimate=_normalize_scores(payload.get("score_estimate")),
        )


class CriticAgent(BaseDebateAgent):
    def __init__(self, backend: ResponseBackend) -> None:
        super().__init__("Critic", CRITIC_SYSTEM_PROMPT, backend)

    def argue(
        self,
        hypothesis: dict[str, str],
        transcript: list[DebateTurn],
        round_number: int,
        total_rounds: int,
    ) -> DebateTurn:
        prompt = f"""
Stress-test this generated scientific hypothesis in round {round_number} of {total_rounds}.

Hypothesis:
{json.dumps(hypothesis, indent=2)}

Debate transcript so far:
{_format_transcript(transcript)}

Instructions:
- Round 1: identify the main flaws, empirical risks, and hidden assumptions.
- Later rounds: counter the Advocate's rebuttals and answer the Judge's probes.
- Estimate scores on a 1-10 scale from the perspective of a skeptical reviewer.

Return only JSON with this schema:
{{
  "argument": "short paragraph",
  "key_points": ["risk 1", "risk 2", "risk 3"],
  "score_estimate": {{
    "novelty": 0,
    "feasibility": 0,
    "transformational_potential": 0
  }}
}}
"""
        payload = self._generate(prompt)
        return DebateTurn(
            round_number=round_number,
            agent=self.name,
            argument=str(payload.get("argument", "")).strip(),
            key_points=_coerce_string_list(payload.get("key_points")),
            score_estimate=_normalize_scores(payload.get("score_estimate")),
        )


class JudgeAgent(BaseDebateAgent):
    def __init__(self, backend: ResponseBackend) -> None:
        super().__init__("Judge", JUDGE_SYSTEM_PROMPT, backend)

    def review_round(
        self,
        hypothesis: dict[str, str],
        transcript: list[DebateTurn],
        round_number: int,
        total_rounds: int,
    ) -> DebateTurn:
        prompt = f"""
You are reviewing round {round_number} of {total_rounds} for a 3-agent debate.

Hypothesis:
{json.dumps(hypothesis, indent=2)}

Debate transcript so far:
{_format_transcript(transcript)}

Instructions:
- Summarize what is resolved versus still uncertain.
- Ask 1-3 clarifying questions that should reduce disagreement in the next round.
- Provide interim 1-10 scores.
- Add a short note on whether the scores are converging.

Return only JSON with this schema:
{{
  "argument": "judge assessment",
  "clarifying_questions": ["question 1"],
  "interim_scores": {{
    "novelty": 0,
    "feasibility": 0,
    "transformational_potential": 0
  }},
  "convergence_note": "short note"
}}
"""
        payload = self._generate(prompt)
        return DebateTurn(
            round_number=round_number,
            agent=self.name,
            argument=str(payload.get("argument", "")).strip(),
            clarifying_questions=_coerce_string_list(
                payload.get("clarifying_questions")
            ),
            score_estimate=_normalize_scores(payload.get("interim_scores")),
            convergence_note=str(payload.get("convergence_note", "")).strip(),
        )

    def render_verdict(
        self,
        hypothesis: dict[str, str],
        transcript: list[DebateTurn],
        consensus_scores: ScoreDict,
    ) -> JSONDict:
        prompt = f"""
Render the final verdict for this 3-agent debate.

Hypothesis:
{json.dumps(hypothesis, indent=2)}

Debate transcript:
{_format_transcript(transcript)}

Consensus seed scores derived from prior rounds:
{json.dumps(consensus_scores, indent=2)}

Instructions:
- Produce final 1-10 scores for novelty, feasibility, and transformational potential.
- Respect the debate evidence and the consensus seed, but adjust if the evidence warrants it.
- Explain the verdict concisely.

Return only JSON with this schema:
{{
  "novelty": 0,
  "feasibility": 0,
  "transformational_potential": 0,
  "debate_summary": "2-3 sentence summary",
  "rationale": {{
    "novelty": "brief reason",
    "feasibility": "brief reason",
    "transformational_potential": "brief reason"
  }}
}}
"""
        payload = self._generate(prompt)
        verdict_scores = _normalize_scores(payload)
        rationale = payload.get("rationale", {})
        if not isinstance(rationale, dict):
            rationale = {}
        rationale_dict = cast(JSONDict, rationale)
        return {
            **verdict_scores,
            "debate_summary": str(payload.get("debate_summary", "")).strip(),
            "rationale": {
                dimension: str(rationale_dict.get(dimension, "")).strip()
                for dimension in SCORE_DIMENSIONS
            },
        }


class MultiAgentEvaluator:
    def __init__(
        self,
        backend: ResponseBackend | None = None,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        rounds: int = 3,
        convergence_threshold: float = 1.25,
    ) -> None:
        if rounds not in {2, 3}:
            raise ValueError("rounds must be 2 or 3")

        self.rounds: int = rounds
        self.convergence_threshold: float = convergence_threshold
        self.backend: ResponseBackend = backend or APIModelBackend(
            provider=provider,
            model=model,
            api_key=api_key,
        )
        self.advocate: AdvocateAgent = AdvocateAgent(self.backend)
        self.critic: CriticAgent = CriticAgent(self.backend)
        self.judge: JudgeAgent = JudgeAgent(self.backend)

    def evaluate_hypothesis(self, hypothesis: dict[str, str]) -> JSONDict:
        normalized_hypothesis = self._validate_hypothesis(hypothesis)
        transcript: list[DebateTurn] = []
        round_scores: list[ScoreDict] = []

        for round_number in range(1, self.rounds + 1):
            advocate_turn = self.advocate.argue(
                normalized_hypothesis,
                transcript,
                round_number,
                self.rounds,
            )
            transcript.append(advocate_turn)

            critic_turn = self.critic.argue(
                normalized_hypothesis,
                transcript,
                round_number,
                self.rounds,
            )
            transcript.append(critic_turn)

            judge_turn = self.judge.review_round(
                normalized_hypothesis,
                transcript,
                round_number,
                self.rounds,
            )
            transcript.append(judge_turn)
            round_scores.append(judge_turn.score_estimate)

        consensus_scores = self._derive_consensus_scores(transcript, round_scores)
        verdict = self.judge.render_verdict(
            normalized_hypothesis,
            transcript,
            consensus_scores,
        )
        final_scores = self._stabilize_final_scores(consensus_scores, verdict)

        return {
            **final_scores,
            "debate_summary": verdict["debate_summary"],
            "rationale": verdict["rationale"],
            "consensus_reached": self._scores_converged(round_scores + [final_scores]),
            "rounds": self.rounds,
            "transcript": [self._serialize_turn(turn) for turn in transcript],
        }

    def calibrate(self, cases: list[JSONDict] | None = None) -> JSONDict:
        calibration_cases = cases or DEFAULT_CALIBRATION_CASES
        results: list[JSONDict] = []

        for case in calibration_cases:
            hypothesis_value = case.get("hypothesis")
            if not isinstance(hypothesis_value, dict):
                raise ValueError("Calibration case is missing a valid hypothesis")
            hypothesis_dict = cast(JSONDict, hypothesis_value)
            hypothesis = {
                str(key): str(value).strip() for key, value in hypothesis_dict.items()
            }
            evaluation = self.evaluate_hypothesis(hypothesis)
            composite = round(
                sum(
                    _clamp_score(evaluation[dimension])
                    for dimension in SCORE_DIMENSIONS
                )
                / len(SCORE_DIMENSIONS),
                2,
            )
            results.append(
                {
                    "name": str(case.get("name", "unknown")),
                    "label": str(case.get("label", "unknown")),
                    "composite_score": composite,
                    "scores": {
                        dimension: evaluation[dimension]
                        for dimension in SCORE_DIMENSIONS
                    },
                    "debate_summary": evaluation["debate_summary"],
                }
            )

        breakthrough_scores = [
            _clamp_score(item["composite_score"])
            for item in results
            if item["label"] == "breakthrough"
        ]
        incremental_scores = [
            _clamp_score(item["composite_score"])
            for item in results
            if item["label"] == "incremental"
        ]
        breakthrough_average = _safe_mean(breakthrough_scores)
        incremental_average = _safe_mean(incremental_scores)

        return {
            "results": results,
            "breakthrough_average": breakthrough_average,
            "incremental_average": incremental_average,
            "passes": breakthrough_average > incremental_average,
        }

    @staticmethod
    def _validate_hypothesis(hypothesis: dict[str, str]) -> dict[str, str]:
        required_fields = {
            "original_assumption",
            "new_hypothesis",
            "transformation",
        }
        missing = sorted(required_fields - set(hypothesis))
        if missing:
            raise ValueError(
                f"Hypothesis missing required fields: {', '.join(missing)}"
            )
        return {key: str(hypothesis[key]).strip() for key in required_fields}

    def _derive_consensus_scores(
        self,
        transcript: list[DebateTurn],
        round_scores: list[ScoreDict],
    ) -> ScoreDict:
        advocate_scores = [
            turn.score_estimate for turn in transcript if turn.agent == "Advocate"
        ]
        critic_scores = [
            turn.score_estimate for turn in transcript if turn.agent == "Critic"
        ]

        last_advocate = advocate_scores[-1] if advocate_scores else _default_scores()
        last_critic = critic_scores[-1] if critic_scores else _default_scores()
        last_judge = round_scores[-1] if round_scores else _default_scores()

        consensus: ScoreDict = {}
        for dimension in SCORE_DIMENSIONS:
            weighted_value = (
                0.5 * last_judge[dimension]
                + 0.25 * last_advocate[dimension]
                + 0.25 * last_critic[dimension]
            )
            consensus[dimension] = round(_clamp_score(weighted_value), 1)
        return consensus

    def _stabilize_final_scores(
        self,
        consensus_scores: ScoreDict,
        verdict: JSONDict,
    ) -> ScoreDict:
        stabilized: ScoreDict = {}
        for dimension in SCORE_DIMENSIONS:
            final_value = _clamp_score(verdict[dimension])
            gap = abs(final_value - consensus_scores[dimension])
            if gap > self.convergence_threshold:
                final_value = (2 * final_value + consensus_scores[dimension]) / 3
            stabilized[dimension] = round(_clamp_score(final_value), 1)
        return stabilized

    def _scores_converged(self, score_history: list[ScoreDict]) -> bool:
        if len(score_history) < 2:
            return True
        latest = score_history[-1]
        previous = score_history[-2]
        max_gap = max(
            abs(latest[dimension] - previous[dimension])
            for dimension in SCORE_DIMENSIONS
        )
        return max_gap <= self.convergence_threshold

    @staticmethod
    def _serialize_turn(turn: DebateTurn) -> JSONDict:
        return {
            "round": turn.round_number,
            "agent": turn.agent,
            "argument": turn.argument,
            "key_points": turn.key_points,
            "score_estimate": turn.score_estimate,
            "clarifying_questions": turn.clarifying_questions,
            "convergence_note": turn.convergence_note,
        }


def _extract_json_object(text: str) -> JSONDict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload = cast(object, decoder.raw_decode(text[index:])[0])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload_dict = cast(dict[object, object], payload)
            return {str(key): value for key, value in payload_dict.items()}
    raise ValueError(f"Unable to parse JSON object from response: {text}")


def _normalize_scores(candidate: object) -> ScoreDict:
    if not isinstance(candidate, dict):
        return _default_scores()
    candidate_dict = cast(JSONDict, candidate)
    normalized: ScoreDict = {}
    for dimension in SCORE_DIMENSIONS:
        value = candidate_dict.get(dimension, 5.0)
        normalized[dimension] = round(_clamp_score(value), 1)
    return normalized


def _clamp_score(value: object) -> float:
    if isinstance(value, bool):
        numeric_value = 1.0 if value else 5.0
    elif isinstance(value, (int, float)):
        numeric_value = float(value)
    elif isinstance(value, str):
        try:
            numeric_value = float(value)
        except ValueError:
            numeric_value = 5.0
    else:
        numeric_value = 5.0
    return min(10.0, max(1.0, numeric_value))


def _default_scores() -> ScoreDict:
    return {dimension: 5.0 for dimension in SCORE_DIMENSIONS}


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    values = cast(list[object], value)
    result: list[str] = []
    for item in values:
        item_text = str(item).strip()
        if item_text:
            result.append(item_text)
    return result


def _format_transcript(transcript: list[DebateTurn]) -> str:
    if not transcript:
        return "[]"
    serialized: list[JSONDict] = []
    for turn in transcript:
        serialized.append(
            {
                "round": turn.round_number,
                "agent": turn.agent,
                "argument": turn.argument,
                "key_points": turn.key_points,
                "score_estimate": turn.score_estimate,
                "clarifying_questions": turn.clarifying_questions,
                "convergence_note": turn.convergence_note,
            }
        )
    return json.dumps(serialized, indent=2)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


__all__ = [
    "AdvocateAgent",
    "CriticAgent",
    "JudgeAgent",
    "MultiAgentEvaluator",
    "APIModelBackend",
    "DEFAULT_CALIBRATION_CASES",
]
