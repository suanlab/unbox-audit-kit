#!/usr/bin/env python3
"""Niche Subfield Benchmark (Experiment A).

Tests whether the memorization confound holds for less well-known breakthroughs.
For each niche paradigm shift:
  1. Collects 15 papers via OpenAlex API (keywords + year filter)
  2. Extracts assumptions using GPT-4o with the CLEAN prompt
  3. Computes soft matching (OpenAI embeddings) against ground truth
  4. Reports confidence-based rank (NOT similarity-reranked)
  5. Runs no-corpus control (field name only, no papers)

KEY hypothesis: if these are truly less memorized, no-corpus should fail
while with-corpus should succeed (at least sometimes).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.4
MAX_RETRIES = 4
PAPERS_PER_PARADIGM = 15
EMBEDDING_MODEL = "text-embedding-3-small"
EXTRACTION_MODEL = "gpt-4o"
SOFT_THRESHOLD = 0.65
NUM_NO_CORPUS_CALLS = 15  # Match per-paper call count

# ---------------------------------------------------------------------------
# Clean extraction prompt (no few-shot examples)
# ---------------------------------------------------------------------------
CLEAN_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

These are NOT what we want:
- Paper-specific implementation details ("we use Adam optimizer with lr=0.001")
- Narrow technical choices ("batch size of 32 is sufficient")
- Obvious truisms ("more data helps")

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories:
- architectural: Structural requirements
- training: Learning procedure requirements
- data: Data requirements
- theoretical: Theoretical constraints
- evaluation: Evaluation paradigm assumptions

Return ONLY valid JSON. Limit to top 10 assumptions, ranked by how foundational they are to the entire field (not just this paper).

Format:
{{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "architectural|training|data|theoretical|evaluation"}}]}}

Paper text:
{paper_text}"""

NO_CORPUS_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Identify the **paradigmatic assumptions** — beliefs shared across the entire subfield that researchers take for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

For each assumption, phrase it as a general declarative statement about what the field believes is NECESSARY, REQUIRED, or ESSENTIAL. Use the pattern: "[X] is necessary/required/essential for [Y]".

Categories: architectural, training, data, theoretical, evaluation

Return ONLY valid JSON. Limit to top 10 assumptions.

Format: {{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "string"}}]}}

Field: {field_name}"""

# ---------------------------------------------------------------------------
# Niche paradigm shift definitions
# ---------------------------------------------------------------------------
NICHE_PARADIGMS = {
    "mixture_of_experts": {
        "name": "Mixture of Experts routing",
        "year": 2017,
        "authors": "Shazeer et al.",
        "broken_assumption": "All parameters must be activated for every input",
        "aliases": [
            "all model parameters should be used for every input",
            "dense computation is necessary for neural network inference",
        ],
        "pre_shift_queries": [
            "mixture of experts neural network gating",
            "conditional computation sparse neural network",
            "sparse gating mixture experts deep learning",
            "expert selection routing neural network",
            "dense neural network computation scaling",
            "modular neural network expert specialization",
        ],
        "max_year": 2016,
        "field_name": "neural network efficiency and conditional computation",
    },
    "neural_ode": {
        "name": "Neural ODE",
        "year": 2018,
        "authors": "Chen et al.",
        "broken_assumption": "Neural networks require discrete layers",
        "aliases": [
            "neural networks must have a fixed number of discrete layers",
            "depth must be a discrete integer in neural networks",
        ],
        "pre_shift_queries": [
            "residual network depth continuous transformation",
            "ordinary differential equations neural network layers",
            "continuous depth neural network dynamics",
            "residual blocks discrete layers deep network",
            "neural network differential equation dynamical system",
            "depth parameterization residual network",
        ],
        "max_year": 2017,
        "field_name": "residual networks and network depth analysis",
    },
    "lottery_ticket": {
        "name": "Lottery Ticket Hypothesis",
        "year": 2019,
        "authors": "Frankle & Carlin",
        "broken_assumption": "Dense networks are necessary before pruning can work",
        "aliases": [
            "pruning requires training a full dense network first",
            "sparse subnetworks cannot be trained from scratch",
        ],
        "pre_shift_queries": [
            "neural network pruning weight sparsity",
            "network pruning compression deep learning",
            "sparse neural network training from scratch",
            "weight magnitude pruning convolutional network",
            "structured pruning neural network efficiency",
            "sparse subnetwork deep neural network",
        ],
        "max_year": 2018,
        "field_name": "network pruning and neural network compression",
    },
    "simclr": {
        "name": "Contrastive Learning / SimCLR",
        "year": 2020,
        "authors": "Chen et al.",
        "broken_assumption": "Labels are necessary for learning visual representations",
        "aliases": [
            "supervised training is required for good visual features",
            "visual representation learning requires labeled data",
        ],
        "pre_shift_queries": [
            "self-supervised visual representation learning contrastive",
            "unsupervised feature learning image representation",
            "pretext task visual self-supervised learning",
            "contrastive learning visual features unsupervised",
            "instance discrimination self-supervised representation",
            "unsupervised pretraining visual recognition",
        ],
        "max_year": 2019,
        "field_name": "self-supervised and unsupervised visual representation learning",
    },
    "lora": {
        "name": "LoRA",
        "year": 2021,
        "authors": "Hu et al.",
        "broken_assumption": "Fine-tuning requires updating all model parameters",
        "aliases": [
            "full parameter updates are necessary for task adaptation",
            "all weights must be modified during fine-tuning",
        ],
        "pre_shift_queries": [
            "parameter efficient fine-tuning pretrained language model",
            "adapter layers fine-tuning transformer",
            "efficient adaptation large pretrained model",
            "low rank approximation weight update fine-tuning",
            "prefix tuning prompt tuning language model",
            "fine-tuning all parameters pretrained model downstream task",
        ],
        "max_year": 2020,
        "field_name": "parameter efficient fine-tuning and adapter methods",
    },
    "flash_attention": {
        "name": "FlashAttention",
        "year": 2022,
        "authors": "Dao et al.",
        "broken_assumption": "Attention computation must materialize the full attention matrix",
        "aliases": [
            "the full N×N attention matrix must be computed explicitly",
            "attention requires quadratic memory",
        ],
        "pre_shift_queries": [
            "efficient attention mechanism transformer memory",
            "linear attention approximation transformer",
            "sparse attention long sequence transformer",
            "memory efficient self-attention computation",
            "attention matrix quadratic complexity transformer",
            "IO-aware attention kernel GPU optimization",
        ],
        "max_year": 2021,
        "field_name": "efficient attention mechanisms for transformers",
    },
    "dpo": {
        "name": "DPO / Direct Preference Optimization",
        "year": 2023,
        "authors": "Rafailov et al.",
        "broken_assumption": "Preference alignment requires reinforcement learning",
        "aliases": [
            "RLHF requires a separate reward model and RL training loop",
            "preference learning needs reinforcement learning optimization",
        ],
        "pre_shift_queries": [
            "reinforcement learning human feedback language model RLHF",
            "reward model training preference alignment",
            "preference learning language model optimization",
            "human preference alignment reinforcement learning",
            "PPO language model reward optimization",
            "RLHF reward model policy gradient",
        ],
        "max_year": 2022,
        "field_name": "RLHF and preference-based language model alignment",
    },
    "kan": {
        "name": "Kolmogorov-Arnold Networks / KAN",
        "year": 2024,
        "authors": "Liu et al.",
        "broken_assumption": "Activation functions must be fixed and applied to nodes",
        "aliases": [
            "neural network activation functions should be fixed nonlinearities on nodes",
            "learnable activations on edges are impractical",
        ],
        "pre_shift_queries": [
            "MLP activation function design neural network architecture",
            "neural network universal function approximation theorem",
            "Kolmogorov superposition representation theorem neural",
            "activation function architecture multilayer perceptron",
            "learnable activation function neural network",
            "spline activation function neural network approximation",
        ],
        "max_year": 2023,
        "field_name": "MLP activation function design and neural network function approximation",
    },
}

# ---------------------------------------------------------------------------
# OpenAlex paper collection
# ---------------------------------------------------------------------------
def _request_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    """Make a request to OpenAlex with retry logic."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                wait = min(30, 2 * attempt)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} retries: {last_error}")


def _reconstruct_abstract(abstract_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not abstract_index:
        return ""
    position_to_token: dict[int, str] = {}
    for token, positions in abstract_index.items():
        for position in positions:
            position_to_token[position] = token
    if not position_to_token:
        return ""
    return " ".join(position_to_token[p] for p in sorted(position_to_token))


def _venue_name(work: dict[str, Any]) -> str:
    primary_location = work.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            name = source.get("display_name")
            if isinstance(name, str):
                return name
    return ""


def _author_list(work: dict[str, Any]) -> list[dict[str, str]]:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return []
    result = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        display_name = author.get("display_name")
        if isinstance(display_name, str):
            result.append({"authorId": str(author.get("id", "")), "name": display_name})
    return result


def _relevance_keywords(config: dict) -> list[str]:
    """Extract relevance keywords from the paradigm config queries."""
    # Collect all unique words from queries as potential relevance keywords
    keywords = set()
    for query in config["pre_shift_queries"]:
        for word in query.lower().split():
            if len(word) >= 4:  # Skip short words
                keywords.add(word)
    # Also add key terms from the field name
    for word in config["field_name"].lower().split():
        if len(word) >= 4:
            keywords.add(word)
    return list(keywords)


def _abstract_relevance_score(abstract: str, title: str, keywords: list[str]) -> float:
    """Score how relevant a paper is based on keyword overlap with abstract+title."""
    text = (abstract + " " + title).lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits / max(len(keywords), 1)


def collect_papers(paradigm_key: str, config: dict) -> list[dict]:
    """Collect papers from OpenAlex for a niche paradigm.

    Uses relevance_score sorting (NOT citation count) and filters papers
    by keyword overlap to ensure topical relevance.
    """
    max_year = config["max_year"]
    # Use a wider window: max_year-4 to max_year
    start_year = max_year - 4
    queries = config["pre_shift_queries"]
    keywords = _relevance_keywords(config)

    all_works: dict[str, dict] = {}  # keyed by openalex ID

    for query in queries:
        print(f"    Query: '{query}' ({start_year}-{max_year})")
        # Use relevance_score sorting instead of cited_by_count
        params = {
            "search": query,
            "filter": (
                f"from_publication_date:{start_year}-01-01,"
                f"to_publication_date:{max_year}-12-31,"
                "type:article|proceedings-article,language:en"
            ),
            "per-page": "50",
            "cursor": "*",
            "sort": "relevance_score:desc",
        }

        try:
            payload = _request_json(OPENALEX_WORKS_URL, params)
        except RuntimeError as e:
            print(f"    [WARN] Failed to fetch: {e}")
            continue

        results = payload.get("results", [])
        for work in results:
            work_id = work.get("id", "")
            if work_id in all_works:
                continue

            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 50:
                continue

            year = work.get("publication_year")
            if not year or year > max_year:
                continue

            title = work.get("display_name", "Unknown")

            # Filter: require at least 2 keyword hits in abstract+title
            relevance = _abstract_relevance_score(abstract, title, keywords)
            keyword_hits = int(relevance * len(keywords))
            if keyword_hits < 2:
                continue

            all_works[work_id] = {
                "paperId": work_id,
                "title": title,
                "authors": _author_list(work),
                "year": year,
                "abstract": abstract,
                "venue": _venue_name(work),
                "citationCount": work.get("cited_by_count", 0),
                "_relevance": relevance,
                "_keyword_hits": keyword_hits,
            }

        time.sleep(REQUEST_SLEEP)

    # Score candidates: combine relevance (primary) with citation bonus (secondary)
    for paper in all_works.values():
        cite_bonus = math.log10(paper.get("citationCount", 0) + 1)
        paper["_score"] = paper["_relevance"] * 10 + cite_bonus

    # Sort by combined score and take top N
    papers = sorted(all_works.values(), key=lambda p: p["_score"], reverse=True)
    papers = papers[:PAPERS_PER_PARADIGM]

    # Clean up internal scoring fields
    for p in papers:
        p.pop("_relevance", None)
        p.pop("_keyword_hits", None)
        p.pop("_score", None)

    print(f"    Collected {len(papers)} papers (from {len(all_works)} candidates)")
    if papers:
        print(f"    Sample titles:")
        for p in papers[:3]:
            print(f"      - {p['title'][:75]}")
    return papers


# ---------------------------------------------------------------------------
# GPT-4o extraction
# ---------------------------------------------------------------------------
def format_paper_text(paper: dict) -> str:
    parts = []
    parts.append(f"Title: {paper.get('title', 'Unknown')}")
    if paper.get("year"):
        parts.append(f"Year: {paper['year']}")
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    return "\n".join(parts)


def extract_assumptions_gpt4o(client: OpenAI, paper_text: str, max_retries: int = 3) -> list[dict]:
    """Call GPT-4o with the clean prompt and parse assumptions."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": CLEAN_PROMPT.format(paper_text=paper_text)}],
                temperature=0.0,
                seed=42,
            )
            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()

            parsed = json.loads(content)
            return parsed.get("assumptions", [])

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"      [Retry {attempt + 1}/{max_retries}] JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"      [Retry {attempt + 1}/{max_retries}] API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return []


def extract_no_corpus(client: OpenAI, field_name: str, num_calls: int) -> list[dict]:
    """Extract assumptions from field name alone (no papers)."""
    all_assumptions = []
    for i in range(num_calls):
        try:
            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": NO_CORPUS_PROMPT.format(field_name=field_name)}],
                temperature=0.0,
                max_tokens=2000,
                seed=42,
            )
            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines)

            parsed = json.loads(content)
            assumptions = parsed.get("assumptions", [])
            for a in assumptions:
                a["call_idx"] = i
            all_assumptions.extend(assumptions)

        except Exception as e:
            print(f"      [WARN] No-corpus call {i+1} failed: {e}")

        time.sleep(0.5)

    return all_assumptions


# ---------------------------------------------------------------------------
# Deduplication and evaluation
# ---------------------------------------------------------------------------
def deduplicate_assumptions(all_assumptions: list[dict]) -> list[dict]:
    """Deduplicate by text (case-insensitive), keeping highest confidence."""
    seen: dict[str, dict] = {}
    for a in all_assumptions:
        key = a["assumption"].strip().lower()
        if key not in seen or a.get("confidence", 0) > seen[key].get("confidence", 0):
            seen[key] = a
    # Sort by confidence descending — this is the CANONICAL order
    return sorted(seen.values(), key=lambda x: x.get("confidence", 0), reverse=True)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI embeddings API."""
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def evaluate_confidence_rank(
    client: OpenAI,
    assumptions: list[dict],
    ground_truth: str,
    aliases: list[str],
) -> dict:
    """Compute metrics using CONFIDENCE-sorted rank (not similarity-reranked).

    The assumptions list must already be sorted by confidence (descending).
    We compute similarity for each assumption against ground truth aliases,
    then find the rank of the first assumption exceeding the threshold
    IN THE CONFIDENCE ORDER.
    """
    if not assumptions:
        return {
            "conf_rank": None,
            "best_sim": 0.0,
            "r_at_10": 0,
            "num_assumptions": 0,
            "details": [],
        }

    gt_aliases = [ground_truth] + aliases
    assumption_texts = [a["assumption"] for a in assumptions]

    all_texts = assumption_texts + gt_aliases
    all_embeddings = embed_texts(client, all_texts)

    assumption_embeddings = all_embeddings[: len(assumption_texts)]
    alias_embeddings = all_embeddings[len(assumption_texts) :]

    # Compute similarity for each assumption (in confidence order)
    details = []
    best_sim = 0.0
    conf_rank = None  # rank in confidence-sorted list where first match >= threshold

    for idx, (a, emb) in enumerate(zip(assumptions, assumption_embeddings)):
        sim = max(cosine_similarity(emb, ae) for ae in alias_embeddings)
        is_match = sim >= SOFT_THRESHOLD

        if sim > best_sim:
            best_sim = sim

        if is_match and conf_rank is None:
            conf_rank = idx + 1  # 1-indexed

        details.append({
            "rank": idx + 1,
            "assumption": a["assumption"],
            "confidence": a.get("confidence", 0),
            "similarity": round(sim, 4),
            "is_match": is_match,
        })

    # R@10: 1 if any assumption in top 10 (by confidence) matches
    r_at_10 = 0
    for d in details[:10]:
        if d["is_match"]:
            r_at_10 = 1
            break

    return {
        "conf_rank": conf_rank,
        "best_sim": round(best_sim, 4),
        "r_at_10": r_at_10,
        "num_assumptions": len(assumptions),
        "details": details[:20],  # Save top 20 for inspection
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client = OpenAI()

    output_dir = PROJECT_ROOT / "experiments" / "niche_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_api_calls = 0

    for paradigm_key, config in NICHE_PARADIGMS.items():
        print(f"\n{'='*70}")
        print(f"PARADIGM: {config['name']} ({config['year']}, {config['authors']})")
        print(f"  Broken assumption: {config['broken_assumption']}")
        print(f"{'='*70}")

        # ----- Step 1: Collect papers from OpenAlex -----
        print(f"\n  [1/4] Collecting papers (max_year={config['max_year']})...")
        papers = collect_papers(paradigm_key, config)

        if not papers:
            print(f"  [WARN] No papers found for {paradigm_key}, skipping.")
            results.append({
                "paradigm": paradigm_key,
                "name": config["name"],
                "with_corpus": {"conf_rank": None, "best_sim": 0.0, "r_at_10": 0},
                "no_corpus": {"conf_rank": None, "best_sim": 0.0, "r_at_10": 0},
                "error": "no papers found",
            })
            continue

        # Save papers
        papers_path = output_dir / f"{paradigm_key}_papers.jsonl"
        with open(papers_path, "w") as f:
            for p in papers:
                f.write(json.dumps(p) + "\n")

        # ----- Step 2: Extract assumptions WITH corpus -----
        print(f"\n  [2/4] Extracting assumptions from {len(papers)} papers (GPT-4o)...")
        all_assumptions = []
        for i, paper in enumerate(papers):
            paper_text = format_paper_text(paper)
            title_short = paper.get("title", "Unknown")[:60]
            print(f"    [{i+1}/{len(papers)}] {title_short}...")

            assumptions = extract_assumptions_gpt4o(client, paper_text)
            total_api_calls += 1
            print(f"      -> {len(assumptions)} assumptions")

            for a in assumptions:
                a["source_paper"] = paper.get("title", "Unknown")
            all_assumptions.extend(assumptions)
            time.sleep(1)  # Rate limiting

        print(f"    Total raw: {len(all_assumptions)}")
        deduped = deduplicate_assumptions(all_assumptions)
        print(f"    After dedup (sorted by confidence): {len(deduped)}")

        # Save per-paradigm assumptions
        assumptions_path = output_dir / f"{paradigm_key}_assumptions.json"
        with open(assumptions_path, "w") as f:
            json.dump({
                "paradigm": paradigm_key,
                "name": config["name"],
                "num_papers": len(papers),
                "num_raw": len(all_assumptions),
                "num_unique": len(deduped),
                "ground_truth": config["broken_assumption"],
                "aliases": config["aliases"],
                "assumptions": deduped,
            }, f, indent=2)

        # Evaluate with corpus
        print(f"\n  [3/4] Evaluating with-corpus (confidence-based rank)...")
        with_corpus_metrics = evaluate_confidence_rank(
            client, deduped, config["broken_assumption"], config["aliases"]
        )
        print(f"    conf_rank: {with_corpus_metrics['conf_rank']}")
        print(f"    best_sim:  {with_corpus_metrics['best_sim']}")
        print(f"    R@10:      {with_corpus_metrics['r_at_10']}")

        # ----- Step 3: No-corpus control -----
        print(f"\n  [4/4] Running no-corpus control (field: '{config['field_name']}')...")
        no_corpus_raw = extract_no_corpus(client, config["field_name"], NUM_NO_CORPUS_CALLS)
        total_api_calls += NUM_NO_CORPUS_CALLS
        no_corpus_deduped = deduplicate_assumptions(no_corpus_raw)
        print(f"    No-corpus: {len(no_corpus_raw)} raw -> {len(no_corpus_deduped)} unique")

        # Save no-corpus assumptions
        no_corpus_path = output_dir / f"{paradigm_key}_no_corpus.json"
        with open(no_corpus_path, "w") as f:
            json.dump({
                "paradigm": paradigm_key,
                "field_name": config["field_name"],
                "num_calls": NUM_NO_CORPUS_CALLS,
                "num_raw": len(no_corpus_raw),
                "num_unique": len(no_corpus_deduped),
                "assumptions": no_corpus_deduped,
            }, f, indent=2)

        # Evaluate no-corpus
        no_corpus_metrics = evaluate_confidence_rank(
            client, no_corpus_deduped, config["broken_assumption"], config["aliases"]
        )
        print(f"    conf_rank: {no_corpus_metrics['conf_rank']}")
        print(f"    best_sim:  {no_corpus_metrics['best_sim']}")
        print(f"    R@10:      {no_corpus_metrics['r_at_10']}")

        results.append({
            "paradigm": paradigm_key,
            "name": config["name"],
            "year": config["year"],
            "ground_truth": config["broken_assumption"],
            "num_papers": len(papers),
            "with_corpus": {
                "conf_rank": with_corpus_metrics["conf_rank"],
                "best_sim": with_corpus_metrics["best_sim"],
                "r_at_10": with_corpus_metrics["r_at_10"],
                "num_assumptions": with_corpus_metrics["num_assumptions"],
                "top_matches": [
                    d for d in with_corpus_metrics["details"] if d["similarity"] >= 0.5
                ][:5],
            },
            "no_corpus": {
                "conf_rank": no_corpus_metrics["conf_rank"],
                "best_sim": no_corpus_metrics["best_sim"],
                "r_at_10": no_corpus_metrics["r_at_10"],
                "num_assumptions": no_corpus_metrics["num_assumptions"],
                "top_matches": [
                    d for d in no_corpus_metrics["details"] if d["similarity"] >= 0.5
                ][:5],
            },
        })

    # ----- Aggregate -----
    n_paradigms = len(NICHE_PARADIGMS)
    with_corpus_r10 = sum(1 for r in results if r["with_corpus"]["r_at_10"] == 1)
    no_corpus_r10 = sum(1 for r in results if r["no_corpus"]["r_at_10"] == 1)

    # Corpus helps if it gets more hits OR gets hits where no-corpus doesn't
    corpus_helps_cases = sum(
        1 for r in results
        if r["with_corpus"]["r_at_10"] == 1 and r["no_corpus"]["r_at_10"] == 0
    )
    corpus_helps = corpus_helps_cases > 0

    # Average best similarity
    avg_with_sim = (
        sum(r["with_corpus"]["best_sim"] for r in results) / len(results)
        if results else 0
    )
    avg_no_sim = (
        sum(r["no_corpus"]["best_sim"] for r in results) / len(results)
        if results else 0
    )

    summary = {
        "experiment": "niche_subfield_benchmark",
        "model": EXTRACTION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "soft_threshold": SOFT_THRESHOLD,
        "total_api_calls": total_api_calls,
        "results": results,
        "aggregate": {
            "with_corpus_r10": f"{with_corpus_r10}/{n_paradigms}",
            "no_corpus_r10": f"{no_corpus_r10}/{n_paradigms}",
            "corpus_helps": corpus_helps,
            "corpus_exclusive_hits": corpus_helps_cases,
            "avg_with_corpus_sim": round(avg_with_sim, 4),
            "avg_no_corpus_sim": round(avg_no_sim, 4),
        },
        "interpretation": _build_interpretation(
            with_corpus_r10, no_corpus_r10, n_paradigms, corpus_helps_cases
        ),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ----- Print final report -----
    print(f"\n{'='*70}")
    print("NICHE BENCHMARK — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nWith-corpus R@10:  {with_corpus_r10}/{n_paradigms}")
    print(f"No-corpus R@10:    {no_corpus_r10}/{n_paradigms}")
    print(f"Corpus helps:      {corpus_helps} ({corpus_helps_cases} exclusive hits)")
    print(f"Avg best sim (corpus):    {avg_with_sim:.4f}")
    print(f"Avg best sim (no-corpus): {avg_no_sim:.4f}")

    print(f"\n{'─'*70}")
    print(f"{'Paradigm':<22} {'Corpus rank':>12} {'Corpus sim':>11} {'No-corp rank':>13} {'No-corp sim':>12}")
    print(f"{'─'*70}")
    for r in results:
        cr = r["with_corpus"]["conf_rank"]
        cs = r["with_corpus"]["best_sim"]
        nr = r["no_corpus"]["conf_rank"]
        ns = r["no_corpus"]["best_sim"]
        cr_str = str(cr) if cr else "miss"
        nr_str = str(nr) if nr else "miss"
        marker = ""
        if r["with_corpus"]["r_at_10"] == 1 and r["no_corpus"]["r_at_10"] == 0:
            marker = " *CORPUS HELPS*"
        elif r["with_corpus"]["r_at_10"] == 0 and r["no_corpus"]["r_at_10"] == 1:
            marker = " *NO-CORPUS BETTER*"
        elif r["with_corpus"]["r_at_10"] == 1 and r["no_corpus"]["r_at_10"] == 1:
            marker = " (both hit)"
        print(f"  {r['paradigm']:<20} {cr_str:>12} {cs:>11.4f} {nr_str:>13} {ns:>12.4f}{marker}")

    print(f"\nInterpretation: {summary['interpretation']}")
    print(f"\nSaved to: {summary_path}")


def _build_interpretation(
    with_r10: int, no_r10: int, total: int, exclusive_hits: int
) -> str:
    if with_r10 > no_r10 and exclusive_hits > 0:
        return (
            f"POSITIVE: With-corpus ({with_r10}/{total}) outperforms no-corpus ({no_r10}/{total}) "
            f"with {exclusive_hits} paradigm(s) found ONLY with corpus grounding. "
            "This supports the hypothesis that for niche breakthroughs, "
            "corpus grounding genuinely contributes beyond memorization."
        )
    elif with_r10 == no_r10 and with_r10 > 0:
        return (
            f"MEMORIZATION CONCERN: Both conditions achieve {with_r10}/{total} hits. "
            "Even for niche breakthroughs, the model may be retrieving knowledge from pretraining. "
            "The memorization confound persists."
        )
    elif no_r10 > with_r10:
        return (
            f"UNEXPECTED: No-corpus ({no_r10}/{total}) outperforms with-corpus ({with_r10}/{total}). "
            "The corpus may be introducing noise that hurts extraction."
        )
    elif with_r10 == 0 and no_r10 == 0:
        return (
            f"NULL RESULT: Neither condition finds the broken assumptions ({with_r10}/{total}). "
            "These niche paradigm shifts may be genuinely hard to extract, "
            "or the ground truth formulations may need refinement."
        )
    else:
        return (
            f"MIXED: With-corpus {with_r10}/{total}, no-corpus {no_r10}/{total}, "
            f"{exclusive_hits} exclusive corpus hits. Partial evidence for corpus value."
        )


if __name__ == "__main__":
    main()
