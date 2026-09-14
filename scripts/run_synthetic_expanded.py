#!/usr/bin/env python3
"""
T2.2: Expanded Synthetic Benchmark — 8 Fictional Fields + Adversarial Distractors.

Extends the 3-field synthetic benchmark (experiments/synthetic_benchmark/) to 8
fictional fields by adding 5 new scenarios. Each scenario now also tests for
"adversarial distractor" false positives: a similar-sounding but incorrect
assumption that should NOT be ranked higher than the planted target.

Pipeline per field:
  1. Generate 15 synthetic paper abstracts with GPT-4o (temperature=0.7) that
     implicitly rely on the PLANTED assumption.
  2. Run CLEAN-prompt extraction on each abstract (GPT-4o, temperature=0.0).
  3. Deduplicate and compute CONFIDENCE-BASED rank (NOT similarity-reranked).
  4. Compute similarity of best match to (a) target aliases, (b) distractor aliases.
  5. Run no-corpus control (field name only) with the same metrics.
  6. Flag false positives where the distractor out-ranks the target.

Existing 3 fields reuse abstracts / extractions from
experiments/synthetic_benchmark/<field_slug>/ if present (saves API cost);
5 new fields are generated fresh.

Output: experiments/synthetic_expanded/summary.json
"""

from __future__ import annotations

import json
import math
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

LEGACY_DIR = PROJECT_ROOT / "experiments" / "synthetic_benchmark"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "synthetic_expanded"
SOFT_THRESHOLD = 0.65
MAX_RETRIES = 3
NUM_ABSTRACTS = 15
GENERATION_TEMP = 0.7
EXTRACTION_TEMP = 0.0
API_DELAY = 1.0  # seconds between API calls


# Generic abstract-generation prompt template for new fields.  Mirrors the
# phrasing used in run_synthetic_benchmark.py so distribution is comparable.
ABSTRACT_PROMPT_TEMPLATE = (
    "Write a 150-word abstract for a research paper in the fictional field of "
    "{field_name}. {field_description} "
    "The paper should implicitly rely on the assumption that "
    "{planted_short}. Do NOT explicitly state this assumption as a "
    "standalone sentence; rather, the paper should take it for granted as an "
    "unquestioned foundation (e.g., by describing methods that build on it, "
    "referencing results that presuppose it, etc.). "
    "Make the abstract sound like a real academic paper with concrete "
    "(fictional) results. Vary the specific topic: paper #{paper_num} of 15."
)


# No-corpus control prompt: just the field name, no papers.
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


# ── Scenarios ────────────────────────────────────────────────────────
#
# 3 existing fields (reuse data from LEGACY_DIR when available) + 5 new fields.
# Each scenario carries a planted assumption and an adversarial distractor.
# Aliases are multiple paraphrases used for soft matching (threshold 0.65).

SCENARIOS: list[dict[str, Any]] = [
    # ── Existing fields ──────────────────────────────────────────────
    {
        "field": "Chromatic Signal Processing",
        "slug": "chromatic_signal_processing",
        "legacy_slug": "chromatic_signal_processing",
        "description": "A fictional field studying color-based data encoding.",
        "planted_assumption": "Frequency decomposition is necessary for chromatic signal analysis",
        "planted_short": (
            "frequency decomposition is fundamental to the field — every method "
            "relies on decomposing chromatic signals into frequency components"
        ),
        "target_aliases": [
            "Frequency decomposition is necessary for chromatic signal analysis",
            "Frequency-domain analysis is essential for processing chromatic signals",
            "Chromatic signal processing requires frequency decomposition",
            "Spectral decomposition is fundamental to chromatic signal analysis",
            "Frequency-based methods are required for chromatic data processing",
        ],
        "distractor_assumption": "Amplitude quantization is essential for chromatic signal fidelity",
        "distractor_aliases": [
            "Amplitude quantization is essential for chromatic signal fidelity",
            "Discrete amplitude encoding is required for chromatic signals",
            "Chromatic signal fidelity depends on amplitude quantization",
        ],
    },
    {
        "field": "Neuromorphic Memory Architecture",
        "slug": "neuromorphic_memory_architecture",
        "legacy_slug": "neuromorphic_memory_architecture",
        "description": "A fictional field about brain-inspired storage systems.",
        "planted_assumption": "Hierarchical addressing is necessary for neuromorphic memory retrieval",
        "planted_short": (
            "hierarchical addressing is necessary for neuromorphic memory retrieval — "
            "every system uses hierarchical address structures to locate stored patterns"
        ),
        "target_aliases": [
            "Hierarchical addressing is necessary for neuromorphic memory retrieval",
            "Neuromorphic memory retrieval requires hierarchical addressing schemes",
            "Hierarchical address structures are essential for brain-inspired memory systems",
            "Memory retrieval in neuromorphic architectures depends on hierarchical addressing",
            "Hierarchical organization of addresses is fundamental to neuromorphic memory",
        ],
        "distractor_assumption": "Content-addressable lookup is essential for neuromorphic memory",
        "distractor_aliases": [
            "Content-addressable lookup is essential for neuromorphic memory",
            "Associative retrieval is required for neuromorphic memory systems",
            "Neuromorphic memory depends on content-addressable access",
        ],
    },
    {
        "field": "Adaptive Topology Networks",
        "slug": "adaptive_topology_networks",
        "legacy_slug": "adaptive_topology_networks",
        "description": "A fictional field about self-modifying graph structures.",
        "planted_assumption": "Static topology must be defined before training begins",
        "planted_short": (
            "the network topology must be statically defined before training begins — "
            "every method first fixes the graph structure, then trains on it"
        ),
        "target_aliases": [
            "Static topology must be defined before training begins",
            "Network topology must be fixed prior to the training phase",
            "A static graph structure is required before training can start",
            "The topology of the network must be predetermined before learning",
            "Fixed topology definition before training is essential for adaptive networks",
            "Graph structure must be established before the training process",
        ],
        "distractor_assumption": "Edge weights must be initialized randomly before training",
        "distractor_aliases": [
            "Edge weights must be initialized randomly before training",
            "Random weight initialization is required for adaptive topology networks",
            "Adaptive topology training requires random edge initialization",
        ],
    },
    # ── New fields ───────────────────────────────────────────────────
    {
        "field": "Temporal Coherence Networks",
        "slug": "temporal_coherence_networks",
        "description": (
            "A fictional field studying time-locked neural oscillator models "
            "for sequential information processing."
        ),
        "planted_assumption": "Phase synchronization is necessary for temporal coherence networks",
        "planted_short": (
            "phase synchronization is the fundamental requirement for temporal "
            "coherence networks — every method aligns oscillator phases to maintain "
            "coherent temporal representations"
        ),
        "target_aliases": [
            "Phase synchronization is necessary for temporal coherence networks",
            "Phase-locked coupling is required for temporal coherence",
            "Temporal coherence depends on phase synchronization",
            "Synchronized phases are essential for temporal coherence networks",
            "Temporal coherence networks require phase alignment",
        ],
        "distractor_assumption": "Amplitude modulation is essential for temporal coherence",
        "distractor_aliases": [
            "Amplitude modulation is essential for temporal coherence",
            "Temporal coherence requires amplitude-based encoding",
            "Amplitude modulation is necessary for temporal coherence networks",
        ],
    },
    {
        "field": "Quantum-Inspired Optimization Fields",
        "slug": "quantum_inspired_optimization_fields",
        "description": (
            "A fictional field studying optimization algorithms that borrow "
            "formal structures from quantum mechanics (without physical quantum hardware)."
        ),
        "planted_assumption": "Superposition states are required for quantum-inspired optimization",
        "planted_short": (
            "superposition states are the required representational substrate — "
            "every quantum-inspired optimizer maintains a superposition over "
            "candidate solutions before collapse"
        ),
        "target_aliases": [
            "Superposition states are required for quantum-inspired optimization",
            "Superposition representations are necessary for quantum-inspired optimization",
            "Quantum-inspired optimization requires superposition states",
            "Candidate solutions must be represented as superpositions in quantum-inspired optimization",
            "Superposition encoding is essential for quantum-inspired optimizers",
        ],
        "distractor_assumption": "Entanglement operators are essential for optimization",
        "distractor_aliases": [
            "Entanglement operators are essential for optimization",
            "Quantum-inspired optimization requires entanglement operators",
            "Entanglement is necessary for quantum-inspired optimization",
        ],
    },
    {
        "field": "Morphogenetic Pattern Encoding",
        "slug": "morphogenetic_pattern_encoding",
        "description": (
            "A fictional field studying developmental-biology-inspired encoders "
            "that represent spatial patterns through simulated morphogen gradients."
        ),
        "planted_assumption": "Gradient-based signaling is necessary for morphogenetic encoding",
        "planted_short": (
            "gradient-based signaling is the fundamental mechanism — every method "
            "in this field encodes spatial structure via simulated morphogen "
            "concentration gradients"
        ),
        "target_aliases": [
            "Gradient-based signaling is necessary for morphogenetic encoding",
            "Morphogenetic encoding requires gradient signaling",
            "Gradient signaling is essential for morphogenetic pattern encoding",
            "Morphogen gradients are required for morphogenetic encoding",
            "Concentration gradients are necessary for morphogenetic pattern formation",
        ],
        "distractor_assumption": "Diffusion-reaction dynamics are essential for pattern formation",
        "distractor_aliases": [
            "Diffusion-reaction dynamics are essential for pattern formation",
            "Reaction-diffusion equations are required for morphogenetic encoding",
            "Pattern formation requires diffusion-reaction dynamics",
        ],
    },
    {
        "field": "Dynamic Ensemble Routing Systems",
        "slug": "dynamic_ensemble_routing_systems",
        "description": (
            "A fictional field studying dispatch mechanisms that coordinate large "
            "ensembles of expert models at inference time."
        ),
        "planted_assumption": "Centralized routing tables are necessary for dynamic ensemble coordination",
        "planted_short": (
            "centralized routing tables are the required coordination mechanism — "
            "every ensemble system dispatches inputs through a single shared "
            "routing table"
        ),
        "target_aliases": [
            "Centralized routing tables are necessary for dynamic ensemble coordination",
            "Dynamic ensemble coordination requires centralized routing tables",
            "Centralized dispatch tables are essential for ensemble routing",
            "Ensemble coordination depends on centralized routing tables",
            "A centralized routing table is required for dynamic ensemble routing",
        ],
        "distractor_assumption": "Distributed consensus is required for ensemble coordination",
        "distractor_aliases": [
            "Distributed consensus is required for ensemble coordination",
            "Dynamic ensembles need distributed consensus protocols",
            "Ensemble coordination requires distributed consensus",
        ],
    },
    {
        "field": "Sparse Predictive Coding Architectures",
        "slug": "sparse_predictive_coding_architectures",
        "description": (
            "A fictional field studying hierarchical generative models that use "
            "sparse codes to predict sensory inputs across layers."
        ),
        "planted_assumption": "Top-down error propagation is necessary for sparse predictive coding",
        "planted_short": (
            "top-down error propagation is the required learning signal — every "
            "method in this field propagates prediction errors from higher to "
            "lower layers"
        ),
        "target_aliases": [
            "Top-down error propagation is necessary for sparse predictive coding",
            "Sparse predictive coding requires top-down error propagation",
            "Top-down prediction errors are essential for sparse predictive coding",
            "Predictive coding architectures depend on top-down error signals",
            "Top-down error signals are required for sparse predictive coding",
        ],
        "distractor_assumption": "Bottom-up feature aggregation is essential for predictive coding",
        "distractor_aliases": [
            "Bottom-up feature aggregation is essential for predictive coding",
            "Predictive coding requires bottom-up feature aggregation",
            "Bottom-up aggregation is necessary for sparse predictive coding",
        ],
    },
]


# ── GPT-4o helpers ───────────────────────────────────────────────────

def _get_openai_client() -> Any:
    from openai import OpenAI
    return OpenAI()


def _call_gpt4o(
    client: Any,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
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
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

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
    field = scenario["field"]
    abstracts: list[dict] = []

    print(f"  Generating {num_abstracts} synthetic abstracts for '{field}'...")

    for i in range(1, num_abstracts + 1):
        prompt = ABSTRACT_PROMPT_TEMPLATE.format(
            field_name=field,
            field_description=scenario["description"],
            planted_short=scenario["planted_short"],
            paper_num=i,
        )
        print(f"    [{i}/{num_abstracts}] Generating abstract...", end=" ", flush=True)

        try:
            text = _call_gpt4o(client, prompt, temperature=GENERATION_TEMP, max_tokens=500)
            paper = {
                "paperId": f"synthetic_{scenario['slug']}_{i:03d}",
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

        time.sleep(API_DELAY)

    return abstracts


# ── Extraction ───────────────────────────────────────────────────────

def extract_assumptions_gpt4o(paper_text: str, client: Any) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(paper_text=paper_text)
    text = _call_gpt4o(client, prompt, temperature=EXTRACTION_TEMP)
    data = _parse_json_response(text)

    if "assumptions" not in data:
        raise ValueError(f"Response missing 'assumptions' key. Got: {list(data.keys())}")

    validated: list[dict] = []
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
    all_assumptions: list[dict] = []
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
        time.sleep(API_DELAY)
    return all_assumptions


def extract_no_corpus(
    client: Any,
    field_name: str,
    num_calls: int = 15,
) -> list[dict]:
    all_assumptions: list[dict] = []
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
        time.sleep(API_DELAY)
    return all_assumptions


# ── Deduplication ────────────────────────────────────────────────────

def deduplicate_assumptions(assumptions: list[dict]) -> list[dict]:
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
    for item in ranked:
        item.pop("_normalized", None)
    return ranked


# ── Metrics (confidence-based rank + distractor check) ───────────────

def compute_metrics(
    predicted: list[str],
    target_aliases: list[str],
    distractor_aliases: list[str],
    threshold: float = SOFT_THRESHOLD,
) -> dict:
    """Compute CONFIDENCE-based rank + similarity-to-target + similarity-to-distractor.

    - predicted is assumed to already be sorted by confidence (highest first).
    - conf_rank = 1-indexed position of first target hit in the confidence list.
    - best_sim_to_target = max cosine similarity to any target alias across all preds.
    - best_sim_to_distractor = max cosine similarity to any distractor alias across all preds.
    - distractor_rank = 1-indexed position of first distractor hit (threshold 0.65).
    - r_at_k = 1 if target hit falls within first k (confidence-ranked) predictions.
    """
    if not predicted:
        return {
            "conf_rank": float(len(predicted) + 1),
            "best_sim_to_target": 0.0,
            "best_sim_to_distractor": 0.0,
            "distractor_rank": float(len(predicted) + 1),
            "r_at_5": 0.0,
            "r_at_10": 0.0,
            "distractor_beats_target": False,
            "top_5": [],
        }

    all_texts = list(predicted) + list(target_aliases) + list(distractor_aliases)
    embeddings = embed_texts(all_texts)
    n_pred = len(predicted)
    n_tgt = len(target_aliases)
    pred_emb = embeddings[:n_pred]
    tgt_emb = embeddings[n_pred : n_pred + n_tgt]
    dist_emb = embeddings[n_pred + n_tgt :]

    total = len(predicted)
    conf_rank = float(total + 1)
    dist_rank = float(total + 1)
    best_sim_tgt = 0.0
    best_sim_dist = 0.0
    details = []

    for idx, (cand, emb) in enumerate(zip(predicted, pred_emb)):
        sim_tgt = max(cosine_similarity(emb, ae) for ae in tgt_emb) if tgt_emb else 0.0
        sim_dist = max(cosine_similarity(emb, ae) for ae in dist_emb) if dist_emb else 0.0
        rank = float(idx + 1)
        is_tgt = sim_tgt >= threshold
        is_dist = sim_dist >= threshold
        if is_tgt and rank < conf_rank:
            conf_rank = rank
        if is_dist and rank < dist_rank:
            dist_rank = rank
        if sim_tgt > best_sim_tgt:
            best_sim_tgt = sim_tgt
        if sim_dist > best_sim_dist:
            best_sim_dist = sim_dist
        details.append({
            "rank": idx + 1,
            "assumption": cand,
            "sim_to_target": round(sim_tgt, 4),
            "sim_to_distractor": round(sim_dist, 4),
            "is_target_match": is_tgt,
            "is_distractor_match": is_dist,
        })

    def recall_at_k(k: int) -> float:
        return 1.0 if conf_rank <= k else 0.0

    distractor_beats_target = dist_rank < conf_rank

    # top_5 by confidence (first 5 of the already-sorted list)
    top_5 = details[:5]

    return {
        "conf_rank": conf_rank,
        "best_sim_to_target": round(best_sim_tgt, 4),
        "best_sim_to_distractor": round(best_sim_dist, 4),
        "distractor_rank": dist_rank,
        "r_at_5": recall_at_k(5),
        "r_at_10": recall_at_k(10),
        "distractor_beats_target": distractor_beats_target,
        "top_5": top_5,
    }


# ── Fisher exact (stdlib-only) ───────────────────────────────────────

def _log_factorial(n: int) -> float:
    return sum(math.log(i) for i in range(1, n + 1)) if n > 0 else 0.0


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def _hyp_pmf(k: int, N: int, K: int, n: int) -> float:
    return math.exp(_log_comb(K, k) + _log_comb(N - K, n - k) - _log_comb(N, n))


def fisher_exact_one_sided(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher's exact for table [[a,b],[c,d]]: P(X>=a)."""
    N, K, n = a + b + c + d, a + c, a + b
    p = 0.0
    for x in range(a, min(K, n) + 1):
        p += _hyp_pmf(x, N, K, n)
    return p


# ── Scenario orchestration ───────────────────────────────────────────

def _load_or_generate_abstracts(
    client: Any,
    scenario: dict,
    scenario_dir: Path,
) -> list[dict]:
    """Load abstracts from scenario dir; fall back to legacy dir; else generate."""
    local_path = scenario_dir / "abstracts.json"
    if local_path.exists():
        print(f"  Loading abstracts from {local_path}")
        with local_path.open("r") as f:
            return json.load(f)

    legacy_slug = scenario.get("legacy_slug")
    if legacy_slug:
        legacy_path = LEGACY_DIR / legacy_slug / "abstracts.json"
        if legacy_path.exists():
            print(f"  Reusing legacy abstracts from {legacy_path}")
            with legacy_path.open("r") as f:
                abstracts = json.load(f)
            with local_path.open("w") as f:
                json.dump(abstracts, f, indent=2, ensure_ascii=False)
            return abstracts

    print(f"  Step 1: Generate synthetic abstracts")
    abstracts = generate_abstracts(client, scenario, NUM_ABSTRACTS)
    with local_path.open("w") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)
    return abstracts


def _load_or_extract_corpus(
    client: Any,
    scenario: dict,
    scenario_dir: Path,
    abstracts: list[dict],
) -> list[dict]:
    local_path = scenario_dir / "corpus_assumptions.json"
    if local_path.exists():
        print(f"  Loading corpus assumptions from {local_path}")
        with local_path.open("r") as f:
            return json.load(f)

    legacy_slug = scenario.get("legacy_slug")
    if legacy_slug:
        legacy_path = LEGACY_DIR / legacy_slug / "corpus_assumptions.json"
        if legacy_path.exists():
            print(f"  Reusing legacy corpus_assumptions from {legacy_path}")
            with legacy_path.open("r") as f:
                data = json.load(f)
            with local_path.open("w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data

    print(f"  Step 2: Extract assumptions WITH corpus ({len(abstracts)} abstracts)")
    data = extract_from_corpus(client, abstracts)
    with local_path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def _load_or_extract_nocorpus(
    client: Any,
    scenario: dict,
    scenario_dir: Path,
) -> list[dict]:
    local_path = scenario_dir / "nocorpus_assumptions.json"
    if local_path.exists():
        print(f"  Loading nocorpus assumptions from {local_path}")
        with local_path.open("r") as f:
            return json.load(f)

    legacy_slug = scenario.get("legacy_slug")
    if legacy_slug:
        legacy_path = LEGACY_DIR / legacy_slug / "nocorpus_assumptions.json"
        if legacy_path.exists():
            print(f"  Reusing legacy nocorpus_assumptions from {legacy_path}")
            with legacy_path.open("r") as f:
                data = json.load(f)
            with local_path.open("w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return data

    print(f"  Step 3: No-corpus control (field name only: '{scenario['field']}')")
    data = extract_no_corpus(client, scenario["field"], num_calls=15)
    with local_path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def run_scenario(client: Any, scenario: dict) -> dict:
    field = scenario["field"]
    planted = scenario["planted_assumption"]
    distractor = scenario["distractor_assumption"]
    target_aliases = scenario["target_aliases"]
    distractor_aliases = scenario["distractor_aliases"]

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {field}")
    print(f"  Planted:    {planted}")
    print(f"  Distractor: {distractor}")
    print(f"{'='*70}")

    scenario_dir = OUTPUT_DIR / scenario["slug"]
    scenario_dir.mkdir(parents=True, exist_ok=True)

    abstracts = _load_or_generate_abstracts(client, scenario, scenario_dir)
    raw_corpus = _load_or_extract_corpus(client, scenario, scenario_dir, abstracts)
    raw_nocorpus = _load_or_extract_nocorpus(client, scenario, scenario_dir)

    unique_corpus = deduplicate_assumptions(raw_corpus)
    unique_nocorpus = deduplicate_assumptions(raw_nocorpus)
    corpus_preds = [str(a["assumption"]) for a in unique_corpus]
    nocorpus_preds = [str(a["assumption"]) for a in unique_nocorpus]

    print(f"  With-corpus: {len(raw_corpus)} raw -> {len(unique_corpus)} unique")
    print(f"  No-corpus:   {len(raw_nocorpus)} raw -> {len(unique_nocorpus)} unique")

    print(f"  Computing metrics (with-corpus)...")
    corpus_metrics = compute_metrics(corpus_preds, target_aliases, distractor_aliases)
    print(f"  Computing metrics (no-corpus)...")
    nocorpus_metrics = compute_metrics(nocorpus_preds, target_aliases, distractor_aliases)

    result = {
        "field": field,
        "planted_assumption": planted,
        "distractor_assumption": distractor,
        "with_corpus": {
            "conf_rank": corpus_metrics["conf_rank"],
            "best_sim_to_target": corpus_metrics["best_sim_to_target"],
            "best_sim_to_distractor": corpus_metrics["best_sim_to_distractor"],
            "distractor_rank": corpus_metrics["distractor_rank"],
            "distractor_beats_target": corpus_metrics["distractor_beats_target"],
            "r_at_5": corpus_metrics["r_at_5"],
            "r_at_10": corpus_metrics["r_at_10"],
            "top_5": corpus_metrics["top_5"],
        },
        "no_corpus": {
            "conf_rank": nocorpus_metrics["conf_rank"],
            "best_sim_to_target": nocorpus_metrics["best_sim_to_target"],
            "best_sim_to_distractor": nocorpus_metrics["best_sim_to_distractor"],
            "distractor_rank": nocorpus_metrics["distractor_rank"],
            "distractor_beats_target": nocorpus_metrics["distractor_beats_target"],
            "r_at_5": nocorpus_metrics["r_at_5"],
            "r_at_10": nocorpus_metrics["r_at_10"],
            "top_5": nocorpus_metrics["top_5"],
        },
        "num_abstracts": len(abstracts),
        "num_unique_assumptions": len(unique_corpus),
        "num_unique_assumptions_nocorpus": len(unique_nocorpus),
    }

    with (scenario_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Summary print
    print(f"\n  Results for '{field}':")
    for label, m in [("WITH CORPUS", corpus_metrics), ("NO CORPUS", nocorpus_metrics)]:
        print(f"    {label}:")
        print(f"      Conf rank:            {m['conf_rank']}")
        print(f"      Best sim to target:   {m['best_sim_to_target']:.4f}")
        print(f"      Best sim to distract: {m['best_sim_to_distractor']:.4f}")
        print(f"      Distractor rank:      {m['distractor_rank']}")
        print(f"      R@5 / R@10:           {m['r_at_5']:.1f} / {m['r_at_10']:.1f}")
        print(f"      Distractor > target:  {m['distractor_beats_target']}")

    return result


# ── Summary / aggregate ──────────────────────────────────────────────

def build_summary(results: list[dict]) -> dict:
    successful = [r for r in results if "error" not in r]

    wc_hits = sum(1 for r in successful if r["with_corpus"]["r_at_10"] > 0)
    nc_hits = sum(1 for r in successful if r["no_corpus"]["r_at_10"] > 0)
    n = len(successful)

    # Fisher exact: is with_corpus hit rate > no_corpus hit rate?
    # Table: [[wc_hit, wc_miss], [nc_hit, nc_miss]]
    p_one = fisher_exact_one_sided(wc_hits, n - wc_hits, nc_hits, n - nc_hits) if n > 0 else 1.0

    fp_wc = sum(1 for r in successful if r["with_corpus"]["distractor_beats_target"])
    fp_nc = sum(1 for r in successful if r["no_corpus"]["distractor_beats_target"])

    return {
        "experiment": "synthetic_benchmark_expanded",
        "description": (
            "T2.2 — 8 fictional fields (3 reused + 5 new) with adversarial distractors. "
            "Confidence-based rank (no similarity reranking)."
        ),
        "model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "soft_threshold": SOFT_THRESHOLD,
        "ran_at": datetime.now(UTC).isoformat(),
        "num_fields": n,
        "scenarios": [
            {
                "field": r["field"],
                "planted_assumption": r["planted_assumption"],
                "distractor_assumption": r["distractor_assumption"],
                "with_corpus": {
                    "conf_rank": r["with_corpus"]["conf_rank"],
                    "best_sim_to_target": r["with_corpus"]["best_sim_to_target"],
                    "best_sim_to_distractor": r["with_corpus"]["best_sim_to_distractor"],
                    "distractor_rank": r["with_corpus"]["distractor_rank"],
                    "distractor_beats_target": r["with_corpus"]["distractor_beats_target"],
                    "r_at_5": r["with_corpus"]["r_at_5"],
                    "r_at_10": r["with_corpus"]["r_at_10"],
                },
                "no_corpus": {
                    "conf_rank": r["no_corpus"]["conf_rank"],
                    "best_sim_to_target": r["no_corpus"]["best_sim_to_target"],
                    "best_sim_to_distractor": r["no_corpus"]["best_sim_to_distractor"],
                    "distractor_rank": r["no_corpus"]["distractor_rank"],
                    "distractor_beats_target": r["no_corpus"]["distractor_beats_target"],
                    "r_at_5": r["no_corpus"]["r_at_5"],
                    "r_at_10": r["no_corpus"]["r_at_10"],
                },
                "num_unique_assumptions": r["num_unique_assumptions"],
            }
            for r in successful
        ]
        + [
            {"field": r["field"], "error": r["error"]}
            for r in results if "error" in r
        ],
        "aggregate": {
            "with_corpus_r10": f"{wc_hits}/{n}",
            "no_corpus_r10": f"{nc_hits}/{n}",
            "distractor_false_positives_with_corpus": f"{fp_wc}/{n}",
            "distractor_false_positives_no_corpus": f"{fp_nc}/{n}",
            "fisher_p_one_sided": round(p_one, 4),
            "corpus_helps": (wc_hits > nc_hits) and (p_one < 0.05),
        },
    }


def _save_summary(results: list[dict]) -> dict:
    summary = build_summary(results)
    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _print_final(summary: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — Synthetic Benchmark Expanded (T2.2)")
    print(f"{'='*70}")
    agg = summary["aggregate"]
    print(f"  N fields:                     {summary['num_fields']}")
    print(f"  With-corpus R@10:             {agg['with_corpus_r10']}")
    print(f"  No-corpus   R@10:             {agg['no_corpus_r10']}")
    print(f"  Fisher one-sided p:           {agg['fisher_p_one_sided']}")
    print(f"  Corpus helps (p<0.05 & wc>nc): {agg['corpus_helps']}")
    print(f"  Distractor FP (with-corpus):  {agg['distractor_false_positives_with_corpus']}")
    print(f"  Distractor FP (no-corpus):    {agg['distractor_false_positives_no_corpus']}")
    print()
    print(f"  {'Field':<38} {'WC:conf':>8} {'WC:R@10':>8} {'NC:conf':>8} {'NC:R@10':>8}")
    print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for s in summary["scenarios"]:
        if "error" in s:
            print(f"  {s['field']:<38} ERROR: {s['error']}")
            continue
        wc, nc = s["with_corpus"], s["no_corpus"]
        print(
            f"  {s['field']:<38} {wc['conf_rank']:>8.1f} {wc['r_at_10']:>8.1f} "
            f"{nc['conf_rank']:>8.1f} {nc['r_at_10']:>8.1f}"
        )
    print(f"\n  Saved: {OUTPUT_DIR / 'summary.json'}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
        sys.exit(1)

    client = _get_openai_client()

    print(f"Synthetic Benchmark Expanded — T2.2")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print(f"Legacy dir: {LEGACY_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Fields: {len(SCENARIOS)}")

    all_results: list[dict] = []
    for scenario in SCENARIOS:
        try:
            all_results.append(run_scenario(client, scenario))
        except Exception as e:
            print(f"\n  ERROR in scenario '{scenario['field']}': {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "field": scenario["field"],
                "planted_assumption": scenario["planted_assumption"],
                "distractor_assumption": scenario.get("distractor_assumption", ""),
                "error": str(e),
            })
        _save_summary(all_results)

    summary = _save_summary(all_results)
    _print_final(summary)


if __name__ == "__main__":
    main()
