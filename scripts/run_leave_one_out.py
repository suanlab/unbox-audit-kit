#!/usr/bin/env python3
"""Leave-one-out experiment to test whether few-shot examples cause data leakage.

Tests THREE conditions:
1. Leave-One-Out (LOO): Remove the corresponding few-shot example for each paradigm shift
2. No-Examples: Remove ALL few-shot examples
3. Cross-Domain Examples: Replace AI examples with examples from other fields

Compares against the original prompt as a control.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib

anthropic = importlib.import_module("anthropic")

from src.pipeline import load_papers
from src.semantic_match import embed_texts, cosine_similarity, soft_recall_at_k, soft_rank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = ("transformer", "diffusion", "icl", "vit")
MAX_PAPERS = 15
SOFT_THRESHOLD = 0.65
MODEL = "claude-sonnet-4-20250514"

MAPPING_PATH = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "leave_one_out"

# The 4 few-shot examples in the original prompt, keyed by which paradigm they correspond to
FEWSHOT_EXAMPLES = {
    "transformer": '"Sequential processing requires recurrence" (pre-Transformer seq2seq field)',
    "diffusion": '"High-quality image generation requires adversarial training" (pre-Diffusion generative modeling)',
    "icl": '"Task adaptation requires fine-tuning model weights" (pre-GPT-3 transfer learning)',
    "vit": '"Visual feature learning requires convolutional inductive biases" (pre-ViT computer vision)',
}

CROSS_DOMAIN_EXAMPLES = [
    '"Newtonian mechanics is necessary for predicting planetary motion" (pre-Einstein physics)',
    '"Chemical reactions require physical contact between molecules" (pre-quantum chemistry)',
    '"Genetic information flows from DNA to protein, never reverse" (pre-reverse transcriptase molecular biology)',
    '"Continental positions are fixed and permanent" (pre-plate tectonics geology)',
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt(example_lines: list[str] | None) -> str:
    """Build the extraction prompt with the given examples, or no examples."""

    examples_block = ""
    if example_lines:
        bullet_list = "\n".join(f"- {ex}" for ex in example_lines)
        examples_block = f"""
Examples of paradigmatic assumptions (the kind you should find):
{bullet_list}

"""

    return f"""You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.
{examples_block}These are NOT what we want:
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
{{{{
  "assumptions": [
    {{{{
      "assumption": "string describing the field-wide assumption",
      "confidence": 0.0-1.0,
      "category": "architectural|training|data|theoretical|evaluation"
    }}}}
  ]
}}}}

Paper text:
{{paper_text}}"""


def build_original_prompt() -> str:
    """The original prompt with all 4 AI few-shot examples."""
    return _build_prompt(list(FEWSHOT_EXAMPLES.values()))


def build_loo_prompt(excluded_category: str) -> str:
    """Leave-one-out: remove the example for the given category."""
    examples = [
        ex for cat, ex in FEWSHOT_EXAMPLES.items() if cat != excluded_category
    ]
    return _build_prompt(examples)


def build_no_examples_prompt() -> str:
    """No examples at all."""
    return _build_prompt(None)


def build_cross_domain_prompt() -> str:
    """Cross-domain examples only (non-AI fields)."""
    return _build_prompt(CROSS_DOMAIN_EXAMPLES)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_with_prompt(
    paper_text: str,
    prompt_template: str,
    client: object,
) -> list[dict]:
    """Call Claude API with the given prompt template and parse assumptions."""
    prompt = prompt_template.format(paper_text=paper_text)

    create_fn = getattr(getattr(client, "messages"), "create")
    message = create_fn(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    response_content = getattr(message, "content", None)
    if not isinstance(response_content, list) or not response_content:
        return []

    response_text = getattr(response_content[0], "text", "")
    if not isinstance(response_text, str) or not response_text.strip():
        return []

    # Parse JSON from response
    decoder = json.JSONDecoder()
    for idx, char in enumerate(response_text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(response_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "assumptions" in parsed:
            return parsed["assumptions"][:10]

    return []


def paper_text_from_record(record: dict) -> str:
    title = str(record.get("title", "")).strip()
    abstract = str(record.get("abstract", "")).strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_ground_truth() -> dict[str, dict]:
    with open(MAPPING_PATH, "r") as f:
        return json.load(f)


def get_aliases(mapping: dict, category: str) -> list[str]:
    entry = mapping[category]
    aliases = [entry["broken_assumption"]]
    if "aliases" in entry:
        aliases.extend(entry["aliases"])
    return aliases


def evaluate_assumptions(
    assumptions: list[str],
    ground_truth_aliases: list[str],
    openai_api_key: str | None,
) -> dict:
    """Compute soft matching metrics for a list of extracted assumptions."""
    if not assumptions:
        return {
            "best_similarity": 0.0,
            "soft_rank": float("inf"),
            "soft_recall_at_10": 0.0,
            "num_assumptions": 0,
            "top_5_assumptions": [],
            "best_match": {"assumption": "", "similarity": 0.0},
        }

    # Embed all assumptions and ground truth aliases together
    all_texts = assumptions + ground_truth_aliases
    embeddings = embed_texts(all_texts, api_key=openai_api_key)
    assumption_embeddings = embeddings[: len(assumptions)]
    alias_embeddings = embeddings[len(assumptions) :]

    # Find best similarity across all assumptions
    best_sim = 0.0
    best_match_text = ""
    similarities = []
    for i, (a, a_emb) in enumerate(zip(assumptions, assumption_embeddings)):
        sim = max(cosine_similarity(a_emb, ae) for ae in alias_embeddings)
        similarities.append(sim)
        if sim > best_sim:
            best_sim = sim
            best_match_text = a

    # Soft rank: rank of first assumption above threshold
    sr = float(len(assumptions) + 1)
    for i, sim in enumerate(similarities):
        if sim >= SOFT_THRESHOLD:
            sr = float(i + 1)
            break

    # Soft recall@10: is any of top-10 above threshold?
    top_10_hit = any(s >= SOFT_THRESHOLD for s in similarities[:10])

    return {
        "best_similarity": round(best_sim, 4),
        "soft_rank": sr,
        "soft_recall_at_10": 1.0 if top_10_hit else 0.0,
        "num_assumptions": len(assumptions),
        "top_5_assumptions": [
            {"assumption": a, "similarity": round(s, 4)}
            for a, s in sorted(
                zip(assumptions, similarities), key=lambda x: x[1], reverse=True
            )[:5]
        ],
        "best_match": {"assumption": best_match_text, "similarity": round(best_sim, 4)},
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    prompt_builder,
    categories: tuple[str, ...],
    mapping: dict,
    client: object,
    openai_api_key: str | None,
) -> dict:
    """Run one experimental condition across all categories."""
    results = {}

    for category in categories:
        print(f"  [{condition_name}] Processing {category}...")
        start_time = time.time()

        # Build the prompt for this condition/category
        if condition_name == "leave_one_out":
            prompt_template = prompt_builder(category)
        else:
            prompt_template = prompt_builder()

        # Load papers
        papers_path = PROJECT_ROOT / "data" / category / "papers.jsonl"
        papers = load_papers(str(papers_path))[:MAX_PAPERS]

        # Extract assumptions from each paper
        all_assumptions = []
        extraction_errors = 0
        for i, paper in enumerate(papers):
            text = paper_text_from_record(paper)
            if not text:
                continue
            try:
                assumptions = extract_with_prompt(text, prompt_template, client)
                for a in assumptions:
                    assumption_text = str(a.get("assumption", "")).strip()
                    if assumption_text:
                        all_assumptions.append(assumption_text)
            except Exception as e:
                extraction_errors += 1
                print(f"    Error on paper {i}: {e}")
                continue

        # Deduplicate while preserving order
        seen = set()
        unique_assumptions = []
        for a in all_assumptions:
            normalized = a.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_assumptions.append(a)

        # Evaluate against ground truth
        aliases = get_aliases(mapping, category)
        eval_result = evaluate_assumptions(unique_assumptions, aliases, openai_api_key)
        eval_result["num_papers"] = len(papers)
        eval_result["extraction_errors"] = extraction_errors
        eval_result["elapsed_seconds"] = round(time.time() - start_time, 1)

        results[category] = eval_result
        print(
            f"    {category}: best_sim={eval_result['best_similarity']:.4f}, "
            f"soft_rank={eval_result['soft_rank']}, "
            f"recall@10={eval_result['soft_recall_at_10']}, "
            f"n_assumptions={eval_result['num_assumptions']}"
        )

    return results


def main() -> None:
    # Load env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    AnthropicClass = getattr(anthropic, "Anthropic")
    client = AnthropicClass(api_key=anthropic_key)
    mapping = load_ground_truth()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_conditions = {}
    total_start = time.time()

    # --- Condition 0: Original (control) ---
    print("\n=== CONDITION: ORIGINAL (control) ===")
    all_conditions["original"] = run_condition(
        "original", build_original_prompt, CATEGORIES, mapping, client, openai_key
    )

    # --- Condition 1: Leave-One-Out ---
    print("\n=== CONDITION: LEAVE-ONE-OUT ===")
    all_conditions["leave_one_out"] = run_condition(
        "leave_one_out", build_loo_prompt, CATEGORIES, mapping, client, openai_key
    )

    # --- Condition 2: No Examples ---
    print("\n=== CONDITION: NO EXAMPLES ===")
    all_conditions["no_examples"] = run_condition(
        "no_examples", build_no_examples_prompt, CATEGORIES, mapping, client, openai_key
    )

    # --- Condition 3: Cross-Domain Examples ---
    print("\n=== CONDITION: CROSS-DOMAIN EXAMPLES ===")
    all_conditions["cross_domain_examples"] = run_condition(
        "cross_domain_examples",
        build_cross_domain_prompt,
        CATEGORIES,
        mapping,
        client,
        openai_key,
    )

    # --- Summary ---
    def avg_metric(condition_results: dict, metric: str) -> float:
        values = [condition_results[cat][metric] for cat in CATEGORIES]
        return round(sum(values) / len(values), 4)

    comparison = {}
    for cond_name, cond_results in all_conditions.items():
        comparison[f"{cond_name}_avg_best_similarity"] = avg_metric(cond_results, "best_similarity")
        comparison[f"{cond_name}_avg_soft_rank"] = avg_metric(cond_results, "soft_rank")
        comparison[f"{cond_name}_avg_recall_at_10"] = avg_metric(cond_results, "soft_recall_at_10")

    # Determine conclusion
    orig_recall = comparison["original_avg_recall_at_10"]
    loo_recall = comparison["leave_one_out_avg_recall_at_10"]
    no_ex_recall = comparison["no_examples_avg_recall_at_10"]
    cross_recall = comparison["cross_domain_examples_avg_recall_at_10"]

    orig_sim = comparison["original_avg_best_similarity"]
    loo_sim = comparison["leave_one_out_avg_best_similarity"]
    no_ex_sim = comparison["no_examples_avg_best_similarity"]
    cross_sim = comparison["cross_domain_examples_avg_best_similarity"]

    if no_ex_recall >= 0.75 and cross_recall >= 0.75:
        conclusion = (
            "VALIDATED: Extraction works WITHOUT few-shot examples. "
            f"No-examples recall={no_ex_recall}, cross-domain recall={cross_recall} "
            f"vs original={orig_recall}. The model genuinely extracts paradigmatic assumptions "
            "from paper content, not from prompt examples."
        )
    elif loo_recall >= orig_recall * 0.8 and no_ex_recall < 0.5:
        conclusion = (
            "PARTIAL LEAKAGE: LOO still works but no-examples fails. "
            f"LOO recall={loo_recall}, no-examples recall={no_ex_recall}. "
            "The remaining 3 examples may provide enough signal for cross-contamination."
        )
    elif no_ex_recall < 0.5 and cross_recall < 0.5:
        conclusion = (
            "DATA LEAKAGE CONFIRMED: Performance drops drastically without AI examples. "
            f"Original recall={orig_recall}, no-examples={no_ex_recall}, "
            f"cross-domain={cross_recall}. The model was copying/paraphrasing prompt examples."
        )
    else:
        conclusion = (
            f"MIXED RESULTS: Original recall={orig_recall}, LOO={loo_recall}, "
            f"no-examples={no_ex_recall}, cross-domain={cross_recall}. "
            f"Similarity: original={orig_sim}, LOO={loo_sim}, "
            f"no-examples={no_ex_sim}, cross-domain={cross_sim}. "
            "Requires manual inspection of extracted assumptions."
        )

    comparison["conclusion"] = conclusion

    total_elapsed = round(time.time() - total_start, 1)

    output = {
        "experiment": "leave_one_out_data_leakage_test",
        "ran_at": datetime.now(UTC).isoformat(),
        "config": {
            "max_papers_per_category": MAX_PAPERS,
            "model": MODEL,
            "soft_threshold": SOFT_THRESHOLD,
            "categories": list(CATEGORIES),
        },
        "conditions": all_conditions,
        "comparison": comparison,
        "total_elapsed_seconds": total_elapsed,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Results saved to: {summary_path}")
    print(f"Total time: {total_elapsed}s")
    print(f"\n--- COMPARISON ---")
    for key, val in comparison.items():
        print(f"  {key}: {val}")
    print(f"\n--- CONCLUSION ---")
    print(f"  {conclusion}")


if __name__ == "__main__":
    main()
