#!/usr/bin/env python3
"""Canonical evaluator: recompute all headline metrics from raw artifacts.

This is the SINGLE SOURCE OF TRUTH for rank computation.
Rank = position in the CONFIDENCE-ORDERED assumption list (model's own ranking).
We do NOT re-sort by similarity.

Default behavior reproduces experiments/canonical_evaluation.json from
experiments/gpt4o_clean_prompt/. Use --input-dir to evaluate alternative
extraction outputs (e.g., OSS LLM runs at experiments/llama3_1_8b_clean/).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PARADIGMS = ["transformer", "diffusion", "icl", "vit", "gan", "batchnorm",
                     "resnet", "word2vec", "dropout", "bert"]
PRIMARY_PARADIGMS = ["transformer", "diffusion", "icl", "vit"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 0 else 0.0


def evaluate_paradigm(assumptions, gt_aliases, client, threshold=0.65, top_k=20):
    """Evaluate a single paradigm: compute confidence-based rank and recall.

    Args:
        assumptions: list of dicts with 'assumption' key, sorted by confidence (descending)
        gt_aliases: list of ground-truth assumption strings
        client: OpenAI client for embeddings
        threshold: similarity threshold
        top_k: max assumptions to evaluate

    Returns:
        dict with conf_rank, best_sim, r_at_5, r_at_10, top_matches
    """
    if not assumptions:
        return {"conf_rank": None, "best_sim": 0.0, "r_at_5": 0.0, "r_at_10": 0.0}

    # Take top-k by confidence (already in confidence order)
    eval_assumptions = assumptions[:top_k]
    texts = [a["assumption"] for a in eval_assumptions] + gt_aliases

    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embs = [e.embedding for e in resp.data]
    a_embs = embs[: len(eval_assumptions)]
    gt_embs = embs[len(eval_assumptions) :]

    # Compute similarity for each assumption IN CONFIDENCE ORDER
    sims = []
    for i, ae in enumerate(a_embs):
        sim = max(cosine_similarity(ae, ge) for ge in gt_embs)
        sims.append(sim)

    best_sim = max(sims) if sims else 0.0

    # Confidence-based rank: first position (1-indexed) where sim >= threshold
    conf_rank = None
    for i, sim in enumerate(sims):
        if sim >= threshold:
            conf_rank = i + 1
            break

    # Recall@K: any of top-K (by confidence) exceeds threshold?
    r_at_5 = 1.0 if any(s >= threshold for s in sims[:5]) else 0.0
    r_at_10 = 1.0 if any(s >= threshold for s in sims[:10]) else 0.0
    r_at_20 = 1.0 if any(s >= threshold for s in sims[:20]) else 0.0

    # Top matches for inspection
    top_matches = []
    for i, sim in enumerate(sims[:10]):
        top_matches.append({
            "conf_position": i + 1,
            "assumption": eval_assumptions[i]["assumption"],
            "similarity": round(sim, 4),
        })

    return {
        "conf_rank": conf_rank,
        "best_sim": round(best_sim, 4),
        "r_at_5": r_at_5,
        "r_at_10": r_at_10,
        "r_at_20": r_at_20,
        "top_matches": top_matches,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=str(PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt"),
                    help="Directory of {paradigm}.json extraction outputs")
    ap.add_argument("--output-path", default=str(PROJECT_ROOT / "experiments" / "canonical_evaluation.json"),
                    help="Where to write canonical_evaluation.json")
    ap.add_argument("--paradigms", nargs="+", default=DEFAULT_PARADIGMS,
                    help="Paradigms to evaluate (default: 10 historical breakthroughs)")
    ap.add_argument("--primary-only", action="store_true",
                    help="Evaluate only the 4 primary paradigms (transformer/diffusion/icl/vit)")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--label", default=None,
                    help="Optional label string saved into the output JSON (e.g. 'llama3_1_8b')")
    args = ap.parse_args()

    if args.primary_only:
        args.paradigms = PRIMARY_PARADIGMS

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["OPENAI_API_KEY"] = api_key

    client = OpenAI()

    # Load ground truths
    gt_path = PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"
    with open(gt_path) as f:
        gt_map = json.load(f)

    # Evaluate extraction outputs from input-dir
    results = []
    input_dir = Path(args.input_dir)
    paradigms = args.paradigms

    print(f"=== Canonical Evaluator: Confidence-Based Ranks ===")
    print(f"  input-dir: {input_dir}")
    print(f"  threshold: {args.threshold}, top_k: {args.top_k}\n")

    for paradigm in paradigms:
        result_file = input_dir / f"{paradigm}.json"
        if not result_file.exists():
            print(f"  {paradigm}: SKIPPED (no file)")
            continue

        with open(result_file) as f:
            data = json.load(f)

        assumptions = data.get("assumptions", [])
        if not assumptions:
            print(f"  {paradigm}: SKIPPED (no assumptions)")
            continue

        gt_info = gt_map.get(paradigm, {})
        aliases = [gt_info.get("broken_assumption", "")] + gt_info.get("aliases", [])
        aliases = [a for a in aliases if a]

        if not aliases:
            print(f"  {paradigm}: SKIPPED (no ground truth)")
            continue

        metrics = evaluate_paradigm(assumptions, aliases, client,
                                    threshold=args.threshold, top_k=args.top_k)
        metrics["paradigm"] = paradigm
        results.append(metrics)

        rank_str = str(metrics["conf_rank"]) if metrics["conf_rank"] else ">20"
        r5 = "✓" if metrics["r_at_5"] > 0 else "✗"
        r10 = "✓" if metrics["r_at_10"] > 0 else "✗"
        print(f"  {paradigm:15s}: rank={rank_str:>4s}  sim={metrics['best_sim']:.3f}  R@5={r5}  R@10={r10}")
        time.sleep(0.5)

    # Aggregate
    hits_5 = sum(1 for r in results if r["r_at_5"] > 0)
    hits_10 = sum(1 for r in results if r["r_at_10"] > 0)
    total = len(results)

    print(f"\n  Total: R@5={hits_5}/{total}, R@10={hits_10}/{total}")

    # Save canonical summary
    output = {
        "evaluator": "canonical_evaluator.py",
        "ranking_method": "confidence_order (NOT similarity-reranked)",
        "threshold": args.threshold,
        "top_k": args.top_k,
        "embedding_model": "text-embedding-3-small",
        "input_dir": str(input_dir),
        "label": args.label,
        "aggregate": {
            "total": total,
            "r_at_5": hits_5,
            "r_at_10": hits_10,
        },
        "results": results,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
