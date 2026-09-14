"""Semantic similarity matching for assumption evaluation using OpenAI embeddings."""

from __future__ import annotations

import importlib
import math
import os
from typing import cast


EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_THRESHOLD = 0.65  # Paper uses 0.65 (conservative, zero false positives at this threshold)


def _get_client(api_key: str | None = None) -> object:
    openai = importlib.import_module("openai")
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is required for semantic matching")
    return openai.OpenAI(api_key=resolved_key)


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    client = _get_client(api_key)
    create_fn = getattr(getattr(client, "embeddings"), "create")
    response = create_fn(model=EMBEDDING_MODEL, input=texts)
    data = getattr(response, "data", [])
    return [getattr(item, "embedding") for item in data]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def best_similarity(
    candidate: str,
    ground_truth_aliases: list[str],
    candidate_embedding: list[float] | None = None,
    alias_embeddings: list[list[float]] | None = None,
    api_key: str | None = None,
) -> float:
    if candidate_embedding is not None and alias_embeddings is not None:
        return max(
            cosine_similarity(candidate_embedding, ae) for ae in alias_embeddings
        )
    all_texts = [candidate] + ground_truth_aliases
    embeddings = embed_texts(all_texts, api_key=api_key)
    cand_emb = embeddings[0]
    alias_embs = embeddings[1:]
    return max(cosine_similarity(cand_emb, ae) for ae in alias_embs)


def soft_recall_at_k(
    predicted: list[str],
    ground_truth_aliases: list[str],
    k: int,
    threshold: float = DEFAULT_THRESHOLD,
    api_key: str | None = None,
) -> tuple[float, list[dict[str, object]]]:
    if k <= 0 or not predicted or not ground_truth_aliases:
        return 0.0, []

    top_k = predicted[:k]
    all_texts = top_k + ground_truth_aliases
    embeddings = embed_texts(all_texts, api_key=api_key)
    candidate_embeddings = embeddings[: len(top_k)]
    alias_embeddings = embeddings[len(top_k) :]

    details: list[dict[str, object]] = []
    hit = False
    for idx, (cand, cand_emb) in enumerate(zip(top_k, candidate_embeddings)):
        sim = max(cosine_similarity(cand_emb, ae) for ae in alias_embeddings)
        is_match = sim >= threshold
        if is_match:
            hit = True
        details.append(
            {
                "rank": idx + 1,
                "assumption": cand,
                "best_similarity": round(sim, 4),
                "is_match": is_match,
            }
        )

    details.sort(key=lambda d: cast(float, d["best_similarity"]), reverse=True)
    return (1.0 if hit else 0.0), details


def soft_rank(
    predicted: list[str],
    ground_truth_aliases: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    api_key: str | None = None,
) -> tuple[float, list[dict[str, object]]]:
    if not predicted or not ground_truth_aliases:
        return float(len(predicted) + 1), []

    all_texts = predicted + ground_truth_aliases
    embeddings = embed_texts(all_texts, api_key=api_key)
    candidate_embeddings = embeddings[: len(predicted)]
    alias_embeddings = embeddings[len(predicted) :]

    details: list[dict[str, object]] = []
    best_matching_rank = float(len(predicted) + 1)
    for idx, (cand, cand_emb) in enumerate(zip(predicted, candidate_embeddings)):
        sim = max(cosine_similarity(cand_emb, ae) for ae in alias_embeddings)
        is_match = sim >= threshold
        rank = float(idx + 1)
        if is_match and rank < best_matching_rank:
            best_matching_rank = rank
        details.append(
            {
                "rank": idx + 1,
                "assumption": cand,
                "best_similarity": round(sim, 4),
                "is_match": is_match,
            }
        )

    return best_matching_rank, details
