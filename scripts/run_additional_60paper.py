#!/usr/bin/env python3
"""
F2 Experiment: Re-run additional paradigms with 60-paper corpora.

Previous runs used only 20 papers per paradigm. This script re-collects larger
corpora (target 60 papers, minimum 40) for the 6 additional paradigms and
re-runs extraction + soft matching.

Output: experiments/additional_paradigms_60/
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

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction import extract_assumptions
from src.semantic_match import embed_texts, cosine_similarity

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "additional_paradigms_60"
MAX_PAPERS_PER_PARADIGM = 60
MIN_PAPERS_PER_PARADIGM = 40
MAX_ASSUMPTIONS_PER_PAPER = 5
SOFT_THRESHOLD = 0.65

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 30
REQUEST_SLEEP = 0.5
MAX_RETRIES = 4

# ── Paradigm definitions (expanded queries to reach 60 papers) ────────

PARADIGMS: list[dict[str, Any]] = [
    {
        "name": "resnet",
        "display": "ResNet / Skip Connections",
        "year": 2015,
        "year_guard": 2014,
        "ground_truth": "Deep networks must process information through sequential layers without shortcuts",
        "aliases": [
            "Deep networks must process information through sequential layers without shortcuts",
            "deep networks require sequential layer-by-layer processing",
            "adding more layers always increases training difficulty",
        ],
        "category": "Architectural",
        "subcategory": "Model Structure",
        "queries": [
            "deep neural network training vanishing gradients",
            "very deep convolutional neural networks image recognition",
            "training deep networks optimization",
            "convolutional neural network depth image classification",
            "deep learning network architecture layers",
            "deep convolutional network image features",
            "vanishing gradient problem deep learning",
            "deep neural network degradation problem",
            "highway networks deep architectures",
            "stacking layers deep convolutional networks",
            "deep networks feature hierarchy",
            "image recognition deep convolutional layers",
        ],
        "topic_keywords": [
            "deep", "convolutional", "layers", "network", "training",
            "gradient", "image", "recognition", "classification",
        ],
        "start_year": 2012,
        "end_year": 2014,
    },
    {
        "name": "gan",
        "display": "GANs / Generative Adversarial Networks",
        "year": 2014,
        "year_guard": 2013,
        "ground_truth": "Generative modeling requires explicit density estimation or variational inference",
        "aliases": [
            "Generative modeling requires explicit density estimation or variational inference",
            "generative models must define an explicit likelihood function",
            "learning to generate data requires modeling the probability distribution explicitly",
        ],
        "category": "Training",
        "subcategory": "Learning Procedure",
        "queries": [
            "deep generative models restricted boltzmann machines",
            "variational inference deep learning",
            "deep belief networks generative pretraining",
            "density estimation neural networks",
            "generative models maximum likelihood",
            "autoencoder representation learning",
            "restricted boltzmann machine learning",
            "deep boltzmann machines generative",
            "variational autoencoder latent space",
            "generative stochastic networks",
            "deep energy models training",
            "probabilistic generative models neural",
        ],
        "topic_keywords": [
            "generative", "boltzmann", "density", "likelihood", "variational",
            "autoencoder", "deep belief", "representation",
        ],
        "start_year": 2010,
        "end_year": 2013,
    },
    {
        "name": "word2vec",
        "display": "Word2Vec / Distributed Word Representations",
        "year": 2013,
        "year_guard": 2012,
        "ground_truth": "Word meaning requires symbolic or hand-crafted feature representations",
        "aliases": [
            "Word meaning requires symbolic or hand-crafted feature representations",
            "word representations should be based on linguistic features",
            "semantic similarity requires structured knowledge bases",
        ],
        "category": "Data",
        "subcategory": "Data Requirements",
        "queries": [
            "distributional semantics word similarity",
            "latent semantic analysis word representations",
            "word sense disambiguation computational linguistics",
            "bag of words text classification",
            "feature engineering natural language processing",
            "semantic similarity knowledge base",
            "word co-occurrence matrix representations",
            "pointwise mutual information word similarity",
            "neural language model word embeddings",
            "latent dirichlet allocation topic models",
            "vector space model text retrieval",
            "distributional hypothesis semantic representations",
        ],
        "topic_keywords": [
            "word", "semantic", "distributional", "lexical", "text",
            "language", "representation", "features", "similarity",
        ],
        "start_year": 2008,
        "end_year": 2012,
    },
    {
        "name": "dropout",
        "display": "Dropout Regularization",
        "year": 2014,
        "year_guard": 2013,
        "ground_truth": "All neurons should be active during training for optimal learning",
        "aliases": [
            "All neurons should be active during training for optimal learning",
            "training neural networks requires using all available parameters",
            "randomly disabling neurons during training would harm learning",
        ],
        "category": "Training",
        "subcategory": "Learning Procedure",
        "queries": [
            "neural network regularization overfitting",
            "deep learning generalization weight decay",
            "training neural networks preventing overfitting",
            "neural network ensemble methods",
            "regularization techniques deep neural networks",
            "model selection neural networks generalization",
            "overfitting deep learning large datasets",
            "L2 regularization neural network training",
            "Bayesian neural networks regularization",
            "early stopping neural network training",
            "data augmentation regularization deep learning",
            "model averaging ensemble deep learning",
        ],
        "topic_keywords": [
            "neural", "regularization", "overfitting", "generalization",
            "training", "weight", "ensemble", "network",
        ],
        "start_year": 2010,
        "end_year": 2013,
    },
    {
        "name": "batchnorm",
        "display": "Batch Normalization",
        "year": 2015,
        "year_guard": 2014,
        "ground_truth": "Network training requires careful weight initialization and low learning rates",
        "aliases": [
            "Network training requires careful weight initialization and low learning rates",
            "training deep networks requires careful initialization schemes",
            "high learning rates cause training instability",
        ],
        "category": "Training",
        "subcategory": "Learning Procedure",
        "queries": [
            "deep learning weight initialization training",
            "learning rate optimization deep neural networks",
            "training deep networks convergence",
            "neural network initialization strategies",
            "stochastic gradient descent deep learning",
            "covariate shift deep learning",
            "Xavier initialization deep networks",
            "gradient flow deep networks training",
            "learning rate scheduling neural networks",
            "momentum SGD deep learning optimization",
            "deep network training instability",
            "input normalization neural network training",
        ],
        "topic_keywords": [
            "initialization", "learning rate", "training", "convergence",
            "gradient", "optimization", "deep", "network",
        ],
        "start_year": 2012,
        "end_year": 2014,
    },
    {
        "name": "bert",
        "display": "BERT / Bidirectional Pre-training",
        "year": 2018,
        "year_guard": 2017,
        "ground_truth": "Language model pre-training must be left-to-right (autoregressive)",
        "aliases": [
            "Language model pre-training must be left-to-right (autoregressive)",
            "language models should generate text left-to-right",
            "autoregressive training is necessary for language model pre-training",
        ],
        "category": "Training",
        "subcategory": "Learning Procedure",
        "queries": [
            "language model pretraining recurrent neural networks",
            "neural language model perplexity LSTM",
            "autoregressive language modeling",
            "word embeddings language model pretraining",
            "language model transfer learning NLP",
            "sequence modeling language understanding",
            "recurrent language model next word prediction",
            "ELMo contextual word representations",
            "semi-supervised learning language models",
            "language model fine-tuning downstream tasks",
            "LSTM language model text generation",
            "unsupervised pretraining NLP transfer",
        ],
        "topic_keywords": [
            "language model", "pretraining", "recurrent", "lstm",
            "autoregressive", "word embedding", "transfer", "sequence",
        ],
        "start_year": 2015,
        "end_year": 2017,
    },
]


# ── OpenAlex paper collection ─────────────────────────────────────────

def _openalex_request(params: dict[str, str]) -> dict[str, Any]:
    """Make a request to OpenAlex with retry logic."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                OPENALEX_WORKS_URL, params=params, timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 429:
                wait = min(30, 2 * attempt)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as e:
            last_error = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"OpenAlex request failed after {MAX_RETRIES} retries: {last_error}")


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
    result: list[dict[str, str]] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        display_name = author.get("display_name")
        if isinstance(display_name, str):
            result.append({
                "authorId": str(author.get("id", "")),
                "name": display_name,
            })
    return result


def _relevance_score(
    work: dict[str, Any], abstract: str, topic_keywords: list[str]
) -> float:
    """Score paper relevance using keyword matching and citation count."""
    title = work.get("display_name", "")
    text = f"{title} {abstract}".lower()
    keyword_hits = sum(1 for kw in topic_keywords if kw.lower() in text)
    citation_count = work.get("cited_by_count", 0)
    if not isinstance(citation_count, int):
        citation_count = 0
    citation_bonus = math.log10(citation_count + 1)
    abstract_bonus = 1.0 if abstract else -0.5
    return (2.0 * keyword_hits) + citation_bonus + abstract_bonus


def collect_papers_openalex(paradigm: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect pre-shift papers via OpenAlex API with target of 60 papers."""
    name = paradigm["name"]
    queries = paradigm["queries"]
    start_year = paradigm["start_year"]
    end_year = paradigm["end_year"]
    topic_keywords = paradigm.get("topic_keywords", [])
    target = MAX_PAPERS_PER_PARADIGM

    print(f"  Collecting papers for {name} via OpenAlex (years {start_year}-{end_year}, target={target})...")

    # Gather candidates from all queries, fetching multiple pages if needed
    candidates: dict[str, tuple[float, dict[str, Any]]] = {}

    for query in queries:
        print(f"    Query: '{query}'...")
        try:
            # Fetch up to 2 pages per query to get more candidates
            cursor = "*"
            for page in range(2):
                params: dict[str, str] = {
                    "search": query,
                    "filter": (
                        f"from_publication_date:{start_year}-01-01,"
                        f"to_publication_date:{end_year}-12-31,"
                        "type:article|proceedings-article,language:en"
                    ),
                    "per-page": "100",
                    "cursor": cursor,
                    "sort": "cited_by_count:desc",
                }
                payload = _openalex_request(params)
                page_results = payload.get("results", [])
                if not isinstance(page_results, list):
                    page_results = []

                if not page_results:
                    break

                for work in page_results:
                    if not isinstance(work, dict):
                        continue
                    paper_id = str(work.get("id", ""))
                    if not paper_id:
                        continue

                    year = work.get("publication_year")
                    if not isinstance(year, int) or year > end_year:
                        continue

                    abstract = _reconstruct_abstract(
                        work.get("abstract_inverted_index")
                        if isinstance(work.get("abstract_inverted_index"), dict)
                        else None
                    )
                    if not abstract.strip():
                        continue

                    title = work.get("display_name", "")
                    if not isinstance(title, str) or not title.strip():
                        continue

                    score = _relevance_score(work, abstract, topic_keywords)

                    existing = candidates.get(paper_id)
                    if existing is None or score > existing[0]:
                        candidates[paper_id] = (score, {
                            "paperId": paper_id,
                            "title": title.strip(),
                            "authors": _author_list(work),
                            "year": year,
                            "abstract": abstract,
                            "venue": _venue_name(work),
                            "citationCount": int(work.get("cited_by_count", 0) or 0),
                            "source": "openalex",
                            "category": name,
                        })

                # Get next cursor
                meta = payload.get("meta", {})
                next_cursor = meta.get("next_cursor")
                if not next_cursor or page_results.__len__() < 100:
                    break
                cursor = next_cursor
                time.sleep(REQUEST_SLEEP)

        except Exception as e:
            print(f"      Warning: query failed: {e}")

        time.sleep(REQUEST_SLEEP)

    # Sort by relevance score and take top N
    ranked = sorted(candidates.values(), key=lambda x: x[0], reverse=True)
    papers = [paper for _, paper in ranked[:target]]

    print(f"    Collected {len(papers)} papers for {name} (from {len(candidates)} candidates)")
    return papers


def save_papers_jsonl(papers: list[dict[str, Any]], path: Path) -> None:
    """Save papers as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")


# ── Extraction ────────────────────────────────────────────────────────

def extract_all_assumptions(
    papers: list[dict[str, Any]],
    max_per_paper: int = MAX_ASSUMPTIONS_PER_PAPER,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Extract assumptions from all papers, returning flat list with source info."""
    all_assumptions: list[dict[str, Any]] = []

    for i, paper in enumerate(papers):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        paper_text = f"{title}\n\n{abstract}".strip()
        if not paper_text:
            continue

        print(f"    Extracting from paper {i+1}/{len(papers)}: {title[:60]}...")
        try:
            assumptions = extract_assumptions(paper_text, api_key=api_key)
            for a in assumptions[:max_per_paper]:
                all_assumptions.append({
                    "paper_id": paper.get("paperId"),
                    "paper_title": title,
                    "assumption": a.get("assumption", ""),
                    "confidence": a.get("confidence", 0.5),
                    "category": a.get("category", ""),
                })
        except Exception as e:
            print(f"      Warning: extraction failed for '{title[:40]}': {e}")
            time.sleep(2)
            continue

        # Delay for API rate limits
        time.sleep(2.0)

    return all_assumptions


# ── Deduplication & ranking ───────────────────────────────────────────

def deduplicate_assumptions(
    assumptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate assumptions by normalized text, keeping highest confidence."""

    def normalize(text: str) -> str:
        lowered = text.strip().lower()
        return " ".join(re.sub(r"[^a-z0-9]+", " ", lowered).split())

    best: dict[str, dict[str, Any]] = {}
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

    # Sort by confidence descending
    ranked = sorted(best.values(), key=lambda x: float(x.get("confidence", 0)), reverse=True)
    return ranked


# ── Soft matching (semantic similarity) ───────────────────────────────

def compute_soft_metrics(
    predicted_assumptions: list[str],
    ground_truth_aliases: list[str],
    threshold: float = SOFT_THRESHOLD,
) -> dict[str, Any]:
    """Compute soft rank, recall@k, best similarity using embeddings."""
    if not predicted_assumptions or not ground_truth_aliases:
        return {
            "best_similarity": 0.0,
            "soft_rank": float(len(predicted_assumptions) + 1),
            "soft_recall_at_5": 0.0,
            "soft_recall_at_10": 0.0,
            "soft_recall_at_20": 0.0,
            "details": [],
        }

    # Embed all texts in one batch
    all_texts = predicted_assumptions + ground_truth_aliases
    embeddings = embed_texts(all_texts)
    pred_embeddings = embeddings[: len(predicted_assumptions)]
    alias_embeddings = embeddings[len(predicted_assumptions) :]

    # Compute similarity for each predicted assumption
    details: list[dict[str, Any]] = []
    best_sim_overall = 0.0
    soft_rank_val = float(len(predicted_assumptions) + 1)

    for idx, (cand, cand_emb) in enumerate(zip(predicted_assumptions, pred_embeddings)):
        sim = max(cosine_similarity(cand_emb, ae) for ae in alias_embeddings)
        is_match = sim >= threshold
        rank = float(idx + 1)
        if is_match and rank < soft_rank_val:
            soft_rank_val = rank
        if sim > best_sim_overall:
            best_sim_overall = sim
        details.append({
            "rank": idx + 1,
            "assumption": cand,
            "best_similarity": round(sim, 4),
            "is_match": is_match,
        })

    def soft_recall_at_k(k: int) -> float:
        for d in details:
            if d["rank"] <= k and d["is_match"]:
                return 1.0
        return 0.0

    return {
        "best_similarity": round(best_sim_overall, 4),
        "soft_rank": soft_rank_val,
        "soft_recall_at_5": soft_recall_at_k(5),
        "soft_recall_at_10": soft_recall_at_k(10),
        "soft_recall_at_20": soft_recall_at_k(20),
        "details": sorted(details, key=lambda d: d["best_similarity"], reverse=True),
    }


# ── Main experiment runner ────────────────────────────────────────────

def run_paradigm_experiment(
    paradigm: dict[str, Any],
    anthropic_api_key: str | None = None,
) -> dict[str, Any]:
    """Run full experiment for a single paradigm shift."""
    name = paradigm["name"]
    display = paradigm["display"]
    print(f"\n{'='*60}")
    print(f"  Paradigm: {display} ({paradigm['year']})")
    print(f"{'='*60}")

    paradigm_dir = OUTPUT_DIR / name
    paradigm_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect papers (or load if already collected)
    papers_path = paradigm_dir / "papers.jsonl"
    if papers_path.exists():
        print(f"  Loading existing papers from {papers_path}")
        papers = []
        with papers_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(json.loads(line))
    else:
        papers = collect_papers_openalex(paradigm)
        save_papers_jsonl(papers, papers_path)

    if len(papers) < MIN_PAPERS_PER_PARADIGM:
        print(f"  WARNING: Only {len(papers)} papers collected for {name} (minimum {MIN_PAPERS_PER_PARADIGM}).")
        if not papers:
            print(f"  ERROR: No papers collected for {name}, skipping.")
            return {"paradigm": name, "error": "no_papers"}

    # Step 2: Extract assumptions (or load if already extracted)
    assumptions_path = paradigm_dir / "assumptions.json"
    if assumptions_path.exists():
        print(f"  Loading existing assumptions from {assumptions_path}")
        with assumptions_path.open("r") as f:
            raw_assumptions = json.load(f)
    else:
        print(f"  Extracting assumptions from {len(papers)} papers...")
        raw_assumptions = extract_all_assumptions(papers, api_key=anthropic_api_key)
        with assumptions_path.open("w") as f:
            json.dump(raw_assumptions, f, indent=2, ensure_ascii=False)

    # Step 3: Deduplicate and rank
    unique_assumptions = deduplicate_assumptions(raw_assumptions)
    predicted = [str(a["assumption"]) for a in unique_assumptions]

    print(f"  Total raw assumptions: {len(raw_assumptions)}")
    print(f"  Unique assumptions: {len(unique_assumptions)}")

    # Step 4: Compute soft metrics
    print(f"  Computing soft matching metrics...")
    metrics = compute_soft_metrics(
        predicted_assumptions=predicted,
        ground_truth_aliases=paradigm["aliases"],
        threshold=SOFT_THRESHOLD,
    )

    # Build result
    top_10 = [
        {
            "rank": d["rank"],
            "assumption": d["assumption"],
            "similarity": d["best_similarity"],
        }
        for d in metrics["details"][:10]
    ]

    result = {
        "paradigm": name,
        "display": display,
        "year": paradigm["year"],
        "ground_truth": paradigm["ground_truth"],
        "ground_truth_aliases": paradigm["aliases"],
        "num_papers": len(papers),
        "num_raw_assumptions": len(raw_assumptions),
        "num_unique_assumptions": len(unique_assumptions),
        "best_similarity": metrics["best_similarity"],
        "soft_rank": metrics["soft_rank"],
        "soft_recall_at_5": metrics["soft_recall_at_5"],
        "soft_recall_at_10": metrics["soft_recall_at_10"],
        "soft_recall_at_20": metrics["soft_recall_at_20"],
        "top_10_assumptions": top_10,
        "ran_at": datetime.now(UTC).isoformat(),
    }

    # Save individual result
    result_path = paradigm_dir / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved result to {result_path}")

    # Print summary
    print(f"  Results for {display}:")
    print(f"    Papers: {len(papers)}")
    print(f"    Unique assumptions: {len(unique_assumptions)}")
    print(f"    Best similarity: {metrics['best_similarity']:.4f}")
    print(f"    Soft rank: {metrics['soft_rank']}")
    print(f"    Soft recall@5: {metrics['soft_recall_at_5']:.1f}")
    print(f"    Soft recall@10: {metrics['soft_recall_at_10']:.1f}")
    print(f"    Soft recall@20: {metrics['soft_recall_at_20']:.1f}")

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Load .env first.")
        sys.exit(1)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set. Load .env first.")
        sys.exit(1)

    all_results: list[dict[str, Any]] = []

    for paradigm in PARADIGMS:
        try:
            result = run_paradigm_experiment(
                paradigm=paradigm,
                anthropic_api_key=anthropic_key,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR running {paradigm['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "paradigm": paradigm["name"],
                "error": str(e),
                "ran_at": datetime.now(UTC).isoformat(),
            })

        # Save intermediate summary after each paradigm
        _save_summary(all_results)

    # Final summary
    _save_summary(all_results)
    _print_final_summary(all_results)


def _save_summary(results: list[dict[str, Any]]) -> None:
    """Save summary.json with all results so far."""
    successful = [r for r in results if "error" not in r]

    avg_best_sim = 0.0
    avg_soft_rank = 0.0
    avg_recall_5 = 0.0
    avg_recall_10 = 0.0
    avg_recall_20 = 0.0
    if successful:
        avg_best_sim = sum(r["best_similarity"] for r in successful) / len(successful)
        avg_soft_rank = sum(r["soft_rank"] for r in successful) / len(successful)
        avg_recall_5 = sum(r["soft_recall_at_5"] for r in successful) / len(successful)
        avg_recall_10 = sum(r["soft_recall_at_10"] for r in successful) / len(successful)
        avg_recall_20 = sum(r["soft_recall_at_20"] for r in successful) / len(successful)

    summary = {
        "experiment": "F2_additional_paradigms_60paper",
        "description": "Re-run additional paradigms with 60-paper corpora (up from 20)",
        "ran_at": datetime.now(UTC).isoformat(),
        "config": {
            "max_papers_per_paradigm": MAX_PAPERS_PER_PARADIGM,
            "min_papers_per_paradigm": MIN_PAPERS_PER_PARADIGM,
            "max_assumptions_per_paper": MAX_ASSUMPTIONS_PER_PAPER,
            "soft_threshold": SOFT_THRESHOLD,
        },
        "num_paradigms_attempted": len(results),
        "num_paradigms_successful": len(successful),
        "paradigms": [r.get("paradigm", "unknown") for r in results],
        "aggregate_metrics": {
            "avg_best_similarity": round(avg_best_sim, 4),
            "avg_soft_rank": round(avg_soft_rank, 2),
            "avg_soft_recall_at_5": round(avg_recall_5, 3),
            "avg_soft_recall_at_10": round(avg_recall_10, 3),
            "avg_soft_recall_at_20": round(avg_recall_20, 3),
        },
        "results": results,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_final_summary(results: list[dict[str, Any]]) -> None:
    """Print a readable final summary."""
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY - Additional Paradigm Shifts (60 papers)")
    print(f"{'='*60}")

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n  Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"  Failed: {', '.join(r['paradigm'] for r in failed)}")

    if successful:
        print(f"\n  {'Paradigm':<20} {'Papers':>6} {'Unique':>6} {'BestSim':>8} {'SoftRank':>9} {'R@5':>5} {'R@10':>5} {'R@20':>5}")
        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*5} {'-'*5} {'-'*5}")

        for r in successful:
            print(
                f"  {r['paradigm']:<20} "
                f"{r['num_papers']:>6} "
                f"{r['num_unique_assumptions']:>6} "
                f"{r['best_similarity']:>8.4f} "
                f"{r['soft_rank']:>9.1f} "
                f"{r['soft_recall_at_5']:>5.1f} "
                f"{r['soft_recall_at_10']:>5.1f} "
                f"{r['soft_recall_at_20']:>5.1f}"
            )

        avg_sim = sum(r["best_similarity"] for r in successful) / len(successful)
        avg_rank = sum(r["soft_rank"] for r in successful) / len(successful)
        avg_r5 = sum(r["soft_recall_at_5"] for r in successful) / len(successful)
        avg_r10 = sum(r["soft_recall_at_10"] for r in successful) / len(successful)
        avg_r20 = sum(r["soft_recall_at_20"] for r in successful) / len(successful)

        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*5} {'-'*5} {'-'*5}")
        print(
            f"  {'AVERAGE':<20} {'':>6} {'':>6} "
            f"{avg_sim:>8.4f} "
            f"{avg_rank:>9.1f} "
            f"{avg_r5:>5.3f} "
            f"{avg_r10:>5.3f} "
            f"{avg_r20:>5.3f}"
        )

    print(f"\n  Results saved to: {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
