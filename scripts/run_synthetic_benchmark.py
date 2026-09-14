#!/usr/bin/env python3
"""
Experiment B: Synthetic Benchmark — Eliminate Pretraining Memorization Entirely.

Creates fictional research fields with planted ground-truth assumptions, generates
synthetic paper abstracts using GPT-4o, then tests whether assumption extraction
can recover the planted assumption from the corpus.

Key insight: Since these fields are entirely fictional, the model CANNOT have
memorized the ground-truth assumption. If with-corpus extraction succeeds but
the no-corpus control fails, this is the cleanest possible evidence that
corpus grounding works.

Three scenarios:
  1. "Chromatic Signal Processing" — planted: frequency decomposition necessity
  2. "Neuromorphic Memory Architecture" — planted: hierarchical addressing necessity
  3. "Adaptive Topology Networks" — planted: static topology before training

Output: experiments/synthetic_benchmark/summary.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction import EXTRACTION_PROMPT
from src.semantic_match import embed_texts, cosine_similarity

# ── Configuration ────────────────────────────────────────────────────

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "synthetic_benchmark"
SOFT_THRESHOLD = 0.65
MAX_RETRIES = 3
NUM_ABSTRACTS = 15
GENERATION_TEMP = 0.7
EXTRACTION_TEMP = 0.0

SCENARIOS = [
    {
        "field": "Chromatic Signal Processing",
        "description": "A fictional field studying color-based data encoding",
        "planted_assumption": "Frequency decomposition is necessary for chromatic signal analysis",
        "aliases": [
            "Frequency decomposition is necessary for chromatic signal analysis",
            "Frequency-domain analysis is essential for processing chromatic signals",
            "Chromatic signal processing requires frequency decomposition",
            "Spectral decomposition is fundamental to chromatic signal analysis",
            "Frequency-based methods are required for chromatic data processing",
        ],
        "abstract_prompt": (
            "Write a 150-word abstract for a research paper in the fictional field of "
            "chromatic signal processing, which studies color-based data encoding and "
            "transmission. The paper should implicitly assume that frequency decomposition "
            "is fundamental to the field — every method in this field relies on decomposing "
            "chromatic signals into frequency components. Do NOT explicitly state this "
            "assumption; rather, the paper should take it for granted as an unquestioned "
            "foundation (e.g., by describing methods that build on frequency analysis, "
            "referencing spectral components, etc.). "
            "Make the abstract sound like a real academic paper with concrete (fictional) results. "
            "Vary the specific topic: paper #{paper_num} of 15."
        ),
    },
    {
        "field": "Neuromorphic Memory Architecture",
        "description": "A fictional field about brain-inspired storage systems",
        "planted_assumption": "Hierarchical addressing is necessary for neuromorphic memory retrieval",
        "aliases": [
            "Hierarchical addressing is necessary for neuromorphic memory retrieval",
            "Neuromorphic memory retrieval requires hierarchical addressing schemes",
            "Hierarchical address structures are essential for brain-inspired memory systems",
            "Memory retrieval in neuromorphic architectures depends on hierarchical addressing",
            "Hierarchical organization of addresses is fundamental to neuromorphic memory",
        ],
        "abstract_prompt": (
            "Write a 150-word abstract for a research paper in the fictional field of "
            "neuromorphic memory architecture, which studies brain-inspired data storage "
            "and retrieval systems. The paper should implicitly assume that hierarchical "
            "addressing is necessary for neuromorphic memory retrieval — every system in "
            "this field uses hierarchical address structures to locate stored patterns. "
            "Do NOT explicitly state this assumption; rather, the paper should take it "
            "for granted (e.g., by describing multi-level address resolution, tree-structured "
            "memory banks, etc.). "
            "Make the abstract sound like a real academic paper with concrete (fictional) results. "
            "Vary the specific topic: paper #{paper_num} of 15."
        ),
    },
    {
        "field": "Adaptive Topology Networks",
        "description": "A fictional field about self-modifying graph structures",
        "planted_assumption": "Static topology must be defined before training begins",
        "aliases": [
            "Static topology must be defined before training begins",
            "Network topology must be fixed prior to the training phase",
            "A static graph structure is required before training can start",
            "The topology of the network must be predetermined before learning",
            "Fixed topology definition before training is essential for adaptive networks",
            "Graph structure must be established before the training process",
        ],
        "abstract_prompt": (
            "Write a 150-word abstract for a research paper in the fictional field of "
            "adaptive topology networks, which studies self-modifying graph structures "
            "for computation. The paper should implicitly assume that the network topology "
            "must be statically defined before training begins — every method in this "
            "field first fixes the graph structure, then trains on it. Do NOT explicitly "
            "state this assumption; rather, the paper should take it for granted (e.g., "
            "by describing topology design phases that precede training, architecture "
            "search that happens before learning, etc.). "
            "Make the abstract sound like a real academic paper with concrete (fictional) results. "
            "Vary the specific topic: paper #{paper_num} of 15."
        ),
    },
]

# No-corpus control prompt: just the field name, no papers
NO_CORPUS_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions
that researchers in a given field take for granted without questioning.

Identify the **paradigmatic assumptions** — beliefs shared across the
entire subfield that researchers treat as self-evident truths.

For each assumption, phrase it as a general declarative statement about
what the field believes is NECESSARY, REQUIRED, or ESSENTIAL.
Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories: architectural, training, data, theoretical, evaluation

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how
foundational they are.

Format: {{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "string"}}]}}

Field: {field_name}"""


# ── GPT-4o API helpers ───────────────────────────────────────────────

def _get_openai_client() -> Any:
    from openai import OpenAI
    return OpenAI()


def _call_gpt4o(
    client: Any,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Call GPT-4o and return the response text."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from GPT-4o")
            return content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"      Retry {attempt}/{MAX_RETRIES}: {e}")
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"GPT-4o call failed after {MAX_RETRIES} retries: {e}")
    return ""  # unreachable


def _parse_json_response(text: str) -> dict:
    """Extract JSON object from a GPT-4o response, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Scan for first JSON object
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Failed to extract JSON from response: {text[:300]}")


# ── Abstract generation ──────────────────────────────────────────────

def generate_abstracts(
    client: Any,
    scenario: dict,
    num_abstracts: int = NUM_ABSTRACTS,
) -> list[dict]:
    """Generate synthetic paper abstracts for a fictional field using GPT-4o."""
    field = scenario["field"]
    abstracts = []

    print(f"  Generating {num_abstracts} synthetic abstracts for '{field}'...")

    for i in range(1, num_abstracts + 1):
        prompt = scenario["abstract_prompt"].format(paper_num=i)
        print(f"    [{i}/{num_abstracts}] Generating abstract...", end=" ", flush=True)

        try:
            text = _call_gpt4o(client, prompt, temperature=GENERATION_TEMP, max_tokens=500)
            # Create a synthetic paper record
            paper = {
                "paperId": f"synthetic_{field.lower().replace(' ', '_')}_{i:03d}",
                "title": f"Synthetic Paper #{i} in {field}",
                "abstract": text,
                "year": 2025,
                "venue": f"Fictional {field} Conference",
                "authors": [],
            }
            abstracts.append(paper)
            print(f"OK ({len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")

        time.sleep(1.0)  # rate limiting

    return abstracts


# ── Assumption extraction ────────────────────────────────────────────

def extract_assumptions_gpt4o(
    paper_text: str,
    client: Any,
) -> list[dict]:
    """Extract assumptions from paper text using GPT-4o with the CLEAN prompt."""
    prompt = EXTRACTION_PROMPT.format(paper_text=paper_text)

    text = _call_gpt4o(client, prompt, temperature=EXTRACTION_TEMP)
    data = _parse_json_response(text)

    if "assumptions" not in data:
        raise ValueError(f"Response missing 'assumptions' key. Got: {list(data.keys())}")

    validated = []
    for a in data["assumptions"]:
        if not isinstance(a, dict) or "assumption" not in a:
            continue
        conf = a.get("confidence", 0.5)
        if not isinstance(conf, (int, float)):
            conf = 0.5
        conf = max(0.0, min(1.0, float(conf)))
        cat = a.get("category", "")
        if cat not in ("architectural", "training", "data", "theoretical", "evaluation"):
            cat = "architectural"
        validated.append({
            "assumption": str(a["assumption"]),
            "confidence": conf,
            "category": cat,
        })

    return validated[:10]


def extract_from_corpus(
    client: Any,
    abstracts: list[dict],
    max_per_paper: int = 5,
) -> list[dict]:
    """Run extraction on each abstract and collect all assumptions."""
    all_assumptions = []

    for i, paper in enumerate(abstracts):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        paper_text = f"{title}\n\n{abstract}".strip()
        if not paper_text:
            continue

        print(f"    [{i+1}/{len(abstracts)}] Extracting from: {title[:50]}...", end=" ", flush=True)

        try:
            assumptions = extract_assumptions_gpt4o(paper_text, client)
            for a in assumptions[:max_per_paper]:
                all_assumptions.append({
                    "paper_id": paper.get("paperId", ""),
                    "paper_title": title,
                    **a,
                })
            print(f"got {min(len(assumptions), max_per_paper)} assumptions")
        except Exception as e:
            print(f"FAILED: {e}")

        time.sleep(1.5)

    return all_assumptions


def extract_no_corpus(
    client: Any,
    field_name: str,
    num_calls: int = 15,
) -> list[dict]:
    """No-corpus control: extract assumptions from just the field name."""
    all_assumptions = []

    for i in range(1, num_calls + 1):
        prompt = NO_CORPUS_PROMPT.format(field_name=field_name)
        print(f"    [Call {i}/{num_calls}]", end=" ", flush=True)

        try:
            text = _call_gpt4o(client, prompt, temperature=0.0)
            data = _parse_json_response(text)
            assumptions = data.get("assumptions", [])

            for a in assumptions:
                if isinstance(a, dict) and "assumption" in a:
                    conf = a.get("confidence", 0.5)
                    if not isinstance(conf, (int, float)):
                        conf = 0.5
                    all_assumptions.append({
                        "assumption": str(a["assumption"]),
                        "confidence": max(0.0, min(1.0, float(conf))),
                        "category": a.get("category", ""),
                        "call_idx": i,
                    })
            print(f"got {len(assumptions)} assumptions")
        except Exception as e:
            print(f"FAILED: {e}")

        time.sleep(0.5)

    return all_assumptions


# ── Deduplication ────────────────────────────────────────────────────

def deduplicate_assumptions(assumptions: list[dict]) -> list[dict]:
    """Deduplicate by normalized text, keeping highest confidence."""
    def normalize(text: str) -> str:
        lowered = text.strip().lower()
        return " ".join(re.sub(r"[^a-z0-9]+", " ", lowered).split())

    best: dict[str, dict] = {}
    for a in assumptions:
        text = str(a.get("assumption", "")).strip()
        if not text:
            continue
        key = normalize(text)
        if not key:
            continue
        existing = best.get(key)
        if existing is None or float(a.get("confidence", 0)) > float(existing.get("confidence", 0)):
            best[key] = {**a, "_normalized": key}

    ranked = sorted(best.values(), key=lambda x: float(x.get("confidence", 0)), reverse=True)
    # Remove internal key
    for item in ranked:
        item.pop("_normalized", None)
    return ranked


# ── Soft matching ────────────────────────────────────────────────────

def compute_soft_metrics(
    predicted_assumptions: list[str],
    ground_truth_aliases: list[str],
    threshold: float = SOFT_THRESHOLD,
) -> dict:
    """Compute confidence-based rank, best similarity, recall@K using embeddings."""
    if not predicted_assumptions or not ground_truth_aliases:
        return {
            "best_similarity": 0.0,
            "conf_rank": float(len(predicted_assumptions) + 1),
            "r_at_5": 0.0,
            "r_at_10": 0.0,
            "details": [],
        }

    all_texts = predicted_assumptions + ground_truth_aliases
    embeddings = embed_texts(all_texts)
    pred_embeddings = embeddings[:len(predicted_assumptions)]
    alias_embeddings = embeddings[len(predicted_assumptions):]

    details = []
    best_sim_overall = 0.0
    conf_rank = float(len(predicted_assumptions) + 1)

    for idx, (cand, cand_emb) in enumerate(zip(predicted_assumptions, pred_embeddings)):
        sim = max(cosine_similarity(cand_emb, ae) for ae in alias_embeddings)
        is_match = sim >= threshold
        rank = float(idx + 1)
        if is_match and rank < conf_rank:
            conf_rank = rank
        if sim > best_sim_overall:
            best_sim_overall = sim
        details.append({
            "rank": idx + 1,
            "assumption": cand,
            "best_similarity": round(sim, 4),
            "is_match": is_match,
        })

    def recall_at_k(k: int) -> float:
        for d in details:
            if d["rank"] <= k and d["is_match"]:
                return 1.0
        return 0.0

    return {
        "best_similarity": round(best_sim_overall, 4),
        "conf_rank": conf_rank,
        "r_at_5": recall_at_k(5),
        "r_at_10": recall_at_k(10),
        "details": sorted(details, key=lambda d: d["best_similarity"], reverse=True),
    }


# ── Per-scenario runner ──────────────────────────────────────────────

def run_scenario(
    client: Any,
    scenario: dict,
) -> dict:
    """Run a single synthetic scenario: generate abstracts, extract, evaluate."""
    field = scenario["field"]
    planted = scenario["planted_assumption"]
    aliases = scenario["aliases"]

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {field}")
    print(f"  Planted assumption: {planted}")
    print(f"{'='*70}")

    scenario_dir = OUTPUT_DIR / field.lower().replace(" ", "_")
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate synthetic abstracts ──────────────────────
    abstracts_path = scenario_dir / "abstracts.json"
    if abstracts_path.exists():
        print(f"\n  Loading existing abstracts from {abstracts_path}")
        with abstracts_path.open("r") as f:
            abstracts = json.load(f)
    else:
        print(f"\n  Step 1: Generate synthetic abstracts")
        abstracts = generate_abstracts(client, scenario, NUM_ABSTRACTS)
        with abstracts_path.open("w") as f:
            json.dump(abstracts, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(abstracts)} abstracts to {abstracts_path}")

    # ── Step 2: With-corpus extraction ────────────────────────────
    corpus_assumptions_path = scenario_dir / "corpus_assumptions.json"
    if corpus_assumptions_path.exists():
        print(f"\n  Loading existing corpus assumptions from {corpus_assumptions_path}")
        with corpus_assumptions_path.open("r") as f:
            raw_corpus_assumptions = json.load(f)
    else:
        print(f"\n  Step 2: Extract assumptions WITH corpus ({len(abstracts)} abstracts)")
        raw_corpus_assumptions = extract_from_corpus(client, abstracts)
        with corpus_assumptions_path.open("w") as f:
            json.dump(raw_corpus_assumptions, f, indent=2, ensure_ascii=False)

    unique_corpus = deduplicate_assumptions(raw_corpus_assumptions)
    corpus_predicted = [str(a["assumption"]) for a in unique_corpus]

    print(f"  With-corpus: {len(raw_corpus_assumptions)} raw -> {len(unique_corpus)} unique")

    # ── Step 3: No-corpus control ─────────────────────────────────
    nocorpus_assumptions_path = scenario_dir / "nocorpus_assumptions.json"
    if nocorpus_assumptions_path.exists():
        print(f"\n  Loading existing no-corpus assumptions from {nocorpus_assumptions_path}")
        with nocorpus_assumptions_path.open("r") as f:
            raw_nocorpus_assumptions = json.load(f)
    else:
        print(f"\n  Step 3: No-corpus control (field name only: '{field}')")
        raw_nocorpus_assumptions = extract_no_corpus(client, field, num_calls=15)
        with nocorpus_assumptions_path.open("w") as f:
            json.dump(raw_nocorpus_assumptions, f, indent=2, ensure_ascii=False)

    unique_nocorpus = deduplicate_assumptions(raw_nocorpus_assumptions)
    nocorpus_predicted = [str(a["assumption"]) for a in unique_nocorpus]

    print(f"  No-corpus: {len(raw_nocorpus_assumptions)} raw -> {len(unique_nocorpus)} unique")

    # ── Step 4: Compute soft metrics ──────────────────────────────
    print(f"\n  Step 4: Compute soft matching metrics")

    print(f"  Computing with-corpus metrics...")
    corpus_metrics = compute_soft_metrics(corpus_predicted, aliases)

    print(f"  Computing no-corpus metrics...")
    nocorpus_metrics = compute_soft_metrics(nocorpus_predicted, aliases)

    # ── Build result ──────────────────────────────────────────────
    result = {
        "field": field,
        "planted_assumption": planted,
        "with_corpus": {
            "conf_rank": corpus_metrics["conf_rank"],
            "best_sim": corpus_metrics["best_similarity"],
            "r_at_5": corpus_metrics["r_at_5"],
            "r_at_10": corpus_metrics["r_at_10"],
            "top_5": corpus_metrics["details"][:5],
        },
        "no_corpus": {
            "conf_rank": nocorpus_metrics["conf_rank"],
            "best_sim": nocorpus_metrics["best_similarity"],
            "r_at_5": nocorpus_metrics["r_at_5"],
            "r_at_10": nocorpus_metrics["r_at_10"],
            "top_5": nocorpus_metrics["details"][:5],
        },
        "num_abstracts": len(abstracts),
        "num_unique_assumptions_corpus": len(unique_corpus),
        "num_unique_assumptions_nocorpus": len(unique_nocorpus),
    }

    # Save individual scenario result
    result_path = scenario_dir / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Results for '{field}':")
    print(f"    WITH CORPUS:")
    print(f"      Conf rank:       {corpus_metrics['conf_rank']}")
    print(f"      Best similarity: {corpus_metrics['best_similarity']:.4f}")
    print(f"      Recall@5:        {corpus_metrics['r_at_5']:.1f}")
    print(f"      Recall@10:       {corpus_metrics['r_at_10']:.1f}")
    print(f"    NO CORPUS (control):")
    print(f"      Conf rank:       {nocorpus_metrics['conf_rank']}")
    print(f"      Best similarity: {nocorpus_metrics['best_similarity']:.4f}")
    print(f"      Recall@5:        {nocorpus_metrics['r_at_5']:.1f}")
    print(f"      Recall@10:       {nocorpus_metrics['r_at_10']:.1f}")

    corpus_wins = corpus_metrics["best_similarity"] > nocorpus_metrics["best_similarity"]
    print(f"    Corpus advantage:  {'YES' if corpus_wins else 'NO'} "
          f"(delta={corpus_metrics['best_similarity'] - nocorpus_metrics['best_similarity']:+.4f})")

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
        sys.exit(1)

    client = _get_openai_client()

    print(f"Synthetic Benchmark — Experiment B")
    print(f"Eliminates pretraining memorization with fictional fields")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print(f"Output: {OUTPUT_DIR}")

    all_results = []

    for scenario in SCENARIOS:
        try:
            result = run_scenario(client, scenario)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR in scenario '{scenario['field']}': {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "field": scenario["field"],
                "planted_assumption": scenario["planted_assumption"],
                "error": str(e),
            })

        # Save intermediate summary after each scenario
        _save_summary(all_results)

    # Final summary
    _save_summary(all_results)
    _print_final_summary(all_results)


def _save_summary(results: list[dict]) -> None:
    """Save summary.json with all results so far."""
    successful = [r for r in results if "error" not in r]

    with_corpus_r10_hits = sum(
        1 for r in successful if r["with_corpus"]["r_at_10"] > 0
    )
    no_corpus_r10_hits = sum(
        1 for r in successful if r["no_corpus"]["r_at_10"] > 0
    )

    # Build interpretation
    if successful:
        corpus_wins = sum(
            1 for r in successful
            if r["with_corpus"]["best_sim"] > r["no_corpus"]["best_sim"]
        )
        if with_corpus_r10_hits > no_corpus_r10_hits:
            interpretation = (
                f"CLEAN EVIDENCE: With-corpus extraction hits {with_corpus_r10_hits}/{len(successful)} "
                f"scenarios at R@10, while no-corpus control hits only {no_corpus_r10_hits}/{len(successful)}. "
                f"Since these are fictional fields with no pretraining data, the corpus genuinely contributes "
                f"to assumption discovery."
            )
        elif with_corpus_r10_hits == no_corpus_r10_hits and with_corpus_r10_hits > 0:
            interpretation = (
                f"INCONCLUSIVE: Both conditions hit {with_corpus_r10_hits}/{len(successful)} at R@10. "
                f"The model may be inferring assumptions from the field name structure."
            )
        else:
            interpretation = (
                f"With-corpus: {with_corpus_r10_hits}/{len(successful)} hits, "
                f"no-corpus: {no_corpus_r10_hits}/{len(successful)} hits at R@10."
            )
    else:
        interpretation = "No successful scenarios yet."

    summary = {
        "experiment": "synthetic_benchmark",
        "description": (
            "Experiment B: Synthetic mini-literatures for fictional fields. "
            "Eliminates pretraining memorization entirely."
        ),
        "model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "soft_threshold": SOFT_THRESHOLD,
        "ran_at": datetime.now(UTC).isoformat(),
        "scenarios": [
            {
                "field": r["field"],
                "planted_assumption": r["planted_assumption"],
                **(
                    {
                        "with_corpus": {
                            "conf_rank": r["with_corpus"]["conf_rank"],
                            "best_sim": r["with_corpus"]["best_sim"],
                            "r_at_10": r["with_corpus"]["r_at_10"],
                        },
                        "no_corpus": {
                            "conf_rank": r["no_corpus"]["conf_rank"],
                            "best_sim": r["no_corpus"]["best_sim"],
                            "r_at_10": r["no_corpus"]["r_at_10"],
                        },
                        "num_abstracts": r["num_abstracts"],
                        "num_unique_assumptions": r["num_unique_assumptions_corpus"],
                    }
                    if "error" not in r
                    else {"error": r["error"]}
                ),
            }
            for r in results
        ],
        "aggregate": {
            "with_corpus_r10": f"{with_corpus_r10_hits}/{len(successful)}",
            "no_corpus_r10": f"{no_corpus_r10_hits}/{len(successful)}",
        },
        "interpretation": interpretation,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_final_summary(results: list[dict]) -> None:
    """Print a readable final summary."""
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — Synthetic Benchmark (Experiment B)")
    print(f"{'='*70}")

    print(f"\n  Successful: {len(successful)}/{len(results)} scenarios")
    if failed:
        print(f"  Failed: {', '.join(r['field'] for r in failed)}")

    if successful:
        print(f"\n  {'Field':<30} {'Condition':<12} {'ConfRank':>8} {'BestSim':>8} {'R@5':>5} {'R@10':>5}")
        print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*8} {'-'*5} {'-'*5}")

        for r in successful:
            wc = r["with_corpus"]
            nc = r["no_corpus"]
            print(
                f"  {r['field']:<30} {'with_corpus':<12} "
                f"{wc['conf_rank']:>8.1f} {wc['best_sim']:>8.4f} "
                f"{wc['r_at_5']:>5.1f} {wc['r_at_10']:>5.1f}"
            )
            print(
                f"  {'':<30} {'no_corpus':<12} "
                f"{nc['conf_rank']:>8.1f} {nc['best_sim']:>8.4f} "
                f"{nc['r_at_5']:>5.1f} {nc['r_at_10']:>5.1f}"
            )
            delta = wc["best_sim"] - nc["best_sim"]
            print(f"  {'':<30} {'delta':<12} {'':>8} {delta:>+8.4f}")
            print()

        wc_r10 = sum(1 for r in successful if r["with_corpus"]["r_at_10"] > 0)
        nc_r10 = sum(1 for r in successful if r["no_corpus"]["r_at_10"] > 0)
        print(f"  AGGREGATE:")
        print(f"    With-corpus R@10 hits:  {wc_r10}/{len(successful)}")
        print(f"    No-corpus R@10 hits:    {nc_r10}/{len(successful)}")
        print(f"    Corpus advantage:       {wc_r10 - nc_r10} additional hits")

    # Load and print interpretation
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        with summary_path.open("r") as f:
            summary = json.load(f)
        print(f"\n  Interpretation: {summary.get('interpretation', 'N/A')}")

    print(f"\n  Results saved to: {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
