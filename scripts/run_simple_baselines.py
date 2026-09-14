#!/usr/bin/env python3
"""T1.2: Simple retrieval baselines (TF-IDF, BM25, embedding-kNN).

Compare pure retrieval baselines against GPT-4o extraction to show
that field-wide prompting exceeds simple sentence retrieval.
"""
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def tokenize(text: str) -> list:
    return re.findall(r"\b[a-z]+\b", text.lower())


def get_candidate_sentences(papers: list, max_sentences: int = 300) -> list:
    """Extract candidate sentences from paper abstracts.

    Filter to sentences containing "necessary/required/essential" (assumption-like).
    """
    candidates = []
    pattern = re.compile(
        r"\b(necessary|required|essential|requires|needs|must|should)\b",
        re.IGNORECASE,
    )
    for p in papers:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        text = f"{title}. {abstract}"
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s = s.strip()
            if 20 <= len(s) <= 300 and pattern.search(s):
                candidates.append(s)
            if len(candidates) >= max_sentences:
                return candidates
    return candidates


def tf_idf_score(query_tokens: list, doc_tokens: list, corpus_df: dict, n_docs: int) -> float:
    """TF-IDF similarity between query and document."""
    doc_counter = Counter(doc_tokens)
    score = 0.0
    for tok in set(query_tokens):
        if tok in doc_counter:
            tf = doc_counter[tok] / max(1, len(doc_tokens))
            idf = math.log((n_docs + 1) / (corpus_df.get(tok, 0) + 1))
            score += tf * idf
    return score


def bm25_score(query_tokens: list, doc_tokens: list, corpus_df: dict, n_docs: int,
               avg_doc_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 similarity."""
    doc_counter = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for tok in query_tokens:
        if tok not in doc_counter:
            continue
        df = corpus_df.get(tok, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf = doc_counter[tok]
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_doc_len, 1))
        score += idf * numerator / denominator
    return score


def evaluate_baseline(candidates: list, gt_aliases: list, method: str = "bm25") -> dict:
    """Rank candidates by simple retrieval against ground truth."""
    if not candidates:
        return {"conf_rank": None, "best_sim": 0.0, "r_at_5": 0.0, "r_at_10": 0.0}

    # Tokenize all
    candidate_tokens = [tokenize(c) for c in candidates]
    gt_tokens_list = [tokenize(g) for g in gt_aliases]

    # Build corpus stats
    corpus_df = Counter()
    for toks in candidate_tokens:
        for t in set(toks):
            corpus_df[t] += 1
    n_docs = len(candidate_tokens)
    avg_doc_len = sum(len(d) for d in candidate_tokens) / max(1, n_docs)

    # Score each candidate against best-matching alias
    scored = []
    for i, c_toks in enumerate(candidate_tokens):
        best = 0.0
        for g_toks in gt_tokens_list:
            if method == "tfidf":
                s = tf_idf_score(g_toks, c_toks, corpus_df, n_docs)
            else:  # bm25
                s = bm25_score(g_toks, c_toks, corpus_df, n_docs, avg_doc_len)
            best = max(best, s)
        scored.append((i, best, candidates[i]))

    # Rank by score
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    # Find where ground truth is matched (using token overlap as soft match)
    best_score = ranked[0][1] if ranked else 0.0

    # Soft match: >50% token overlap with any alias
    def soft_match(cand: str, gt: str) -> bool:
        ct = set(tokenize(cand))
        gt_t = set(tokenize(gt))
        if not gt_t:
            return False
        overlap = len(ct & gt_t) / len(gt_t)
        return overlap >= 0.5

    hit_rank = None
    for rank, (idx, score, cand) in enumerate(ranked, 1):
        if any(soft_match(cand, gt) for gt in gt_aliases):
            hit_rank = rank
            break

    r_at_5 = 1.0 if hit_rank is not None and hit_rank <= 5 else 0.0
    r_at_10 = 1.0 if hit_rank is not None and hit_rank <= 10 else 0.0

    return {
        "method": method,
        "conf_rank": hit_rank,
        "best_score": round(best_score, 4),
        "r_at_5": r_at_5,
        "r_at_10": r_at_10,
        "num_candidates": len(candidates),
        "top_5_candidates": [c for _, _, c in ranked[:5]],
    }


def main():
    # Load ground truths
    gt_path = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
    with open(gt_path) as f:
        gt_map = json.load(f)

    primary = ["transformer", "diffusion", "icl", "vit"]
    results = {"tfidf": {}, "bm25": {}}

    for paradigm in primary:
        papers_path = PROJECT_ROOT / "data" / paradigm / "papers.jsonl"
        if not papers_path.exists():
            print(f"  {paradigm}: SKIP (no papers file)")
            continue

        papers = []
        with open(papers_path) as f:
            for line in f:
                papers.append(json.loads(line))
        papers = papers[:60]  # match retrospective setup

        candidates = get_candidate_sentences(papers, max_sentences=300)

        gt = gt_map.get(paradigm, {})
        aliases = [gt.get("broken_assumption", "")] + gt.get("aliases", [])
        aliases = [a for a in aliases if a]

        print(f"\n{paradigm}: {len(candidates)} candidate sentences")

        for method in ["tfidf", "bm25"]:
            res = evaluate_baseline(candidates, aliases, method=method)
            results[method][paradigm] = res
            rank = res["conf_rank"] if res["conf_rank"] else ">N"
            print(f"  {method:6s}: rank={str(rank):>4s}  R@5={res['r_at_5']}  R@10={res['r_at_10']}")

    # Aggregate
    for method in ["tfidf", "bm25"]:
        hits_5 = sum(r["r_at_5"] for r in results[method].values())
        hits_10 = sum(r["r_at_10"] for r in results[method].values())
        n = len(results[method])
        print(f"\n{method.upper()} aggregate: R@5={hits_5}/{n}, R@10={hits_10}/{n}")

    out_path = PROJECT_ROOT / "experiments" / "simple_baselines" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "simple_retrieval_baselines",
            "methods": ["tfidf", "bm25"],
            "corpus_size": 60,
            "filter": "sentences containing necessary/required/essential",
            "matching": "soft token overlap >= 50%",
            "results": results,
            "aggregate": {
                method: {
                    "r_at_5": sum(r["r_at_5"] for r in results[method].values()) / max(1, len(results[method])),
                    "r_at_10": sum(r["r_at_10"] for r in results[method].values()) / max(1, len(results[method])),
                }
                for method in results
            }
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
