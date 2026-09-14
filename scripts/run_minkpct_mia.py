#!/usr/bin/env python3
"""Min-K% Prob (Shi et al. 2024, ICLR) membership-inference cross-check.

Behavioral controls in the paper (no-corpus 4/4, wrong-corpus 0/4) attribute
the headline 6/10 to pretraining memorization. This script runs the
*probabilistic* counterpart: Min-K%(20) Prob over the 4 primary pre-shift
corpora using 3 OSS models for which we can extract logprobs.

For each paper p:
    1. Feed (title + abstract) to the model as a prompt.
    2. Get per-prompt-token logprob via vLLM `prompt_logprobs=1`.
    3. Sort the token logprobs ascending; take the bottom K = 20% of them.
    4. Min-K% Prob(p) = mean logprob of those bottom-K tokens.

Higher (less negative) Min-K% Prob = even the model-rarest tokens are still
relatively probable -> stronger contamination signal.

Per paradigm we report:
    mean Min-K% Prob over 60 pre-shift papers, per model.

This is the probabilistic cross-check of the behavioral memorization claim.
The script writes both per-paper scores and per-paradigm aggregates so the
correlation with R@10 can be computed offline.

Usage (per GPU):
    CUDA_VISIBLE_DEVICES=1 python scripts/run_minkpct_mia.py \\
        --model llama-nous --output experiments/mia_minkpct/llama.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARADIGMS = ["transformer", "diffusion", "icl", "vit"]
K_PERCENT = 0.20  # Min-K% Prob standard K
PAPERS_PER = 60  # use full corpora (not 15-paper subset)

MODELS = {
    "llama-nous": {
        "hf_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "max_model_len": 8192,
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "max_model_len": 8192,
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 8192,
    },
}


def load_papers(paradigm: str, limit: int = PAPERS_PER) -> list[dict]:
    path = PROJECT_ROOT / "data" / paradigm / "papers.jsonl"
    papers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("abstract"):
                papers.append(d)
    return papers[:limit]


def minkpct_from_logprobs(token_logprobs: list[float], k: float = K_PERCENT) -> float:
    """Min-K% Prob: mean logprob of the bottom-k% rarest (lowest-logprob) tokens."""
    if not token_logprobs:
        return float("-inf")
    sorted_lp = sorted(token_logprobs)
    n_bottom = max(1, int(len(sorted_lp) * k))
    return sum(sorted_lp[:n_bottom]) / n_bottom


def run(model_key: str, output_path: Path) -> None:
    from vllm import LLM, SamplingParams  # noqa

    cfg = MODELS[model_key]
    print(f"[{model_key}] loading {cfg['hf_id']} ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=cfg["hf_id"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.35,
        trust_remote_code=False,
        dtype="bfloat16",
    )
    print(f"[{model_key}] loaded in {time.time()-t0:.1f}s", flush=True)

    # We only need the prompt logprobs (prefill). max_tokens=1 then ignore the
    # generation. prompt_logprobs=1 returns the top-1 PLUS the actual prompt
    # token at every position.
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)

    results = {"model": cfg["hf_id"], "k_percent": K_PERCENT,
               "papers_per_paradigm": PAPERS_PER, "per_paradigm": {}}

    for paradigm in PARADIGMS:
        papers = load_papers(paradigm)
        print(f"[{model_key}/{paradigm}] {len(papers)} papers ...", flush=True)
        prompts = [
            (p.get("title", "") or "") + "\n\n" + (p.get("abstract", "") or "")
            for p in papers
        ]
        outs = llm.generate(prompts, sp)

        per_paper = []
        for paper, out in zip(papers, outs):
            plp = out.prompt_logprobs or []
            tok_ids = out.prompt_token_ids or []
            token_lps: list[float] = []
            # First token has no logprob (None); skip it.
            for i, pos_dict in enumerate(plp):
                if pos_dict is None or i >= len(tok_ids):
                    continue
                actual = tok_ids[i]
                if actual in pos_dict:
                    lp_obj = pos_dict[actual]
                    lp = getattr(lp_obj, "logprob", lp_obj)
                    if isinstance(lp, (int, float)):
                        token_lps.append(float(lp))
            if not token_lps:
                continue
            mk = minkpct_from_logprobs(token_lps)
            per_paper.append({
                "paperId": paper.get("paperId"),
                "year": paper.get("year"),
                "n_tokens": len(token_lps),
                "min_kpct_prob": mk,
                "mean_logprob": sum(token_lps) / len(token_lps),
            })

        scores = [p["min_kpct_prob"] for p in per_paper]
        mean_logprobs = [p["mean_logprob"] for p in per_paper]
        results["per_paradigm"][paradigm] = {
            "n_papers": len(per_paper),
            "mean_min_kpct_prob": sum(scores) / len(scores) if scores else None,
            "mean_logprob": sum(mean_logprobs) / len(mean_logprobs) if mean_logprobs else None,
            "per_paper": per_paper,
        }
        # Best-effort incremental save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{model_key}/{paradigm}] mean Min-K%(20) Prob = "
              f"{results['per_paradigm'][paradigm]['mean_min_kpct_prob']:.4f}",
              flush=True)

    print(f"[{model_key}] DONE -> {output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
