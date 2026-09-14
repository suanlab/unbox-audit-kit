"""
Extract implicit assumptions from academic papers using LLM API.
"""

import json
import importlib
from typing import List, Dict, Optional


class _AnthropicPlaceholder:
    Anthropic = None


try:
    anthropic = importlib.import_module("anthropic")
except ImportError:
    anthropic = _AnthropicPlaceholder()


# Clean prompt (no few-shot examples) — used for main results.
# The variant with few-shot examples is in extraction_prompts.py (FIELD_WIDE_PROMPT).
EXTRACTION_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

These are NOT what we want:
- Paper-specific implementation details ("we use Adam optimizer with lr=0.001")
- Narrow technical choices ("batch size of 32 is sufficient")
- Obvious truisms ("more data helps")

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories:
- architectural: Structural requirements (e.g., "recurrence is needed for sequences")
- training: Learning procedure requirements (e.g., "adversarial training is needed for generation")
- data: Data requirements (e.g., "labeled data is essential for classification")
- theoretical: Theoretical constraints (e.g., "bias-variance tradeoff always holds")
- evaluation: Evaluation paradigm assumptions (e.g., "accuracy is the right metric")

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how foundational they are to the entire field (not just this paper).

Format:
{{
  "assumptions": [
    {{
      "assumption": "string describing the field-wide assumption",
      "confidence": 0.0-1.0,
      "category": "architectural|training|data|theoretical|evaluation"
    }}
  ]
}}

Paper text:
{paper_text}
"""


def extract_assumptions(
    paper_text: str,
    api_key: Optional[str] = None,
) -> List[Dict[str, float | str]]:
    """
    Extract implicit assumptions from a paper's text using Claude API.

    Args:
        paper_text: The full text of the academic paper
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)

    Returns:
        List of dicts with keys:
        - assumption: str - Description of the assumption
        - confidence: float - Confidence score (0.0-1.0)
        - category: str - Category of assumption

    Raises:
        RuntimeError: If Anthropic SDK is not installed
        ValueError: If API response is invalid or cannot be parsed
        Exception: If API call fails
    """
    anthropic_client = getattr(anthropic, "Anthropic", None)
    if anthropic_client is None:
        raise RuntimeError(
            "Anthropic SDK is not installed. Install dependency `anthropic` to run extraction."
        )

    client = anthropic_client(api_key=api_key)

    prompt = EXTRACTION_PROMPT.format(paper_text=paper_text)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    except Exception:
        raise

    response_content = getattr(message, "content", None)
    if not isinstance(response_content, list) or not response_content:
        raise ValueError("Anthropic response did not contain any content blocks")

    first_block = response_content[0]
    response_text = getattr(first_block, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Anthropic response did not contain a text block")

    decoder = json.JSONDecoder()
    response_data = None
    for idx, char in enumerate(response_text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(response_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            response_data = parsed
            break

    if response_data is None:
        raise ValueError(
            f"Failed to extract JSON object from API response.\n"
            f"Response was: {response_text[:500]}"
        )

    if "assumptions" not in response_data:
        raise ValueError(f"Response missing 'assumptions' key. Got: {response_data}")

    assumptions = response_data["assumptions"]

    # Validate each assumption has required fields
    for assumption in assumptions:
        if not isinstance(assumption, dict):
            raise ValueError(f"Assumption is not a dict: {assumption}")

        required_fields = {"assumption", "confidence", "category"}
        if not required_fields.issubset(assumption.keys()):
            raise ValueError(f"Assumption missing required fields. Got: {assumption}")

        # Validate confidence is a float between 0 and 1
        if not isinstance(assumption["confidence"], (int, float)):
            raise ValueError(
                f"Confidence must be numeric, got: {assumption['confidence']}"
            )

        if not (0.0 <= assumption["confidence"] <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got: {assumption['confidence']}"
            )

    # Limit to top 10 assumptions
    return assumptions[:10]
