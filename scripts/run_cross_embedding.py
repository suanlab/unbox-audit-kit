#!/usr/bin/env python3
"""Cross-embedding robustness check.

The canonical pipeline uses OpenAI text-embedding-3-small and threshold 0.65.
Reviewers will ask: does the 6/10 R@10 / 4/4 no-corpus / 0/4 wrong-corpus
pattern survive a change of embedding model? This script answers by
re-evaluating the SAME extracted assumptions (no new LLM calls) against the
target aliases using two alternative embedders, F1-calibrating the threshold
per embedding.

Sources of truth (extractions never re-run):
    experiments/canonical_evaluation.json        (main 10-case GPT-4o)
    experiments/no_corpus_control/per_paradigm/  (no-corpus)
    experiments/wrong_corpus_control/per_pairing/ (wrong-corpus)

Output: experiments/cross_embedding/summary.json

Embedders:
    1. OpenAI text-embedding-3-large   (1536/3072-dim; better recall)
    2. sentence-transformers all-MiniLM-L6-v2 (open, 384-dim; lower cost)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRIMARY = ["transformer", "diffusion", "icl", "vit"]
ALL_PARADIGMS = PRIMARY + ["batchnorm", "gan", "resnet", "word2vec", "dropout", "bert"]


def cosine(a, b):
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def load_extractions_canonical() -> dict:
    """Return {paradigm: [assumption_text, ...]} ordered by confidence desc."""
    out = {}
    # Use the per-paradigm extraction JSONs that feed canonical_evaluator
    src_dir = PROJECT_ROOT / "experiments" / "gpt4o_clean_prompt"
    for p in ALL_PARADIGMS:
        path = src_dir / f"{p}.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        ass = d.get("assumptions", [])
        # Sort by confidence desc
        ass_sorted = sorted(ass, key=lambda a: -float(a.get("confidence", 0)))
        # Dedup by text
        seen = set()
        uniq = []
        for a in ass_sorted:
            t = a.get("assumption", "").strip()
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        out[p] = uniq[:50]
    return out


def load_targets() -> dict:
    m = json.load(open(PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"))
    out = {}
    for p, v in m.items():
        out[p] = v.get("aliases") or [v.get("description", "")]
    return out


def evaluate_with_embedder(embed_fn, name: str, candidates: dict, targets: dict,
                           thresholds: list[float]) -> dict:
    """For each threshold, compute per-paradigm conf_rank/r_at_10."""
    results = {"embedding": name, "thresholds": {}, "per_paradigm_best_sim": {}}
    # Embed all targets and candidates once
    all_texts = []
    target_slices = {}
    cand_slices = {}
    for p in PRIMARY + [x for x in candidates if x not in PRIMARY]:
        tgts = targets.get(p, [])
        target_slices[p] = (len(all_texts), len(all_texts) + len(tgts))
        all_texts.extend(tgts)
        cands = candidates.get(p, [])
        cand_slices[p] = (len(all_texts), len(all_texts) + len(cands))
        all_texts.extend(cands)
    print(f"[{name}] embedding {len(all_texts)} texts ...", flush=True)
    embs = embed_fn(all_texts)
    # Compute best sim per paradigm
    for p in candidates:
        ts, te = target_slices.get(p, (0, 0))
        cs, ce = cand_slices.get(p, (0, 0))
        if cs == ce or ts == te:
            continue
        target_embs = embs[ts:te]
        cand_embs = embs[cs:ce]
        sims = []
        for ce_emb in cand_embs:
            best = max(cosine(ce_emb, t) for t in target_embs)
            sims.append(best)
        results["per_paradigm_best_sim"][p] = {
            "sims": [round(s, 4) for s in sims[:20]],
            "best_sim": round(max(sims), 4) if sims else 0,
        }
    # Threshold sweep
    for thr in thresholds:
        per_p = {}
        for p in candidates:
            if p not in results["per_paradigm_best_sim"]:
                continue
            sims = results["per_paradigm_best_sim"][p]["sims"]
            conf_rank = next((i + 1 for i, s in enumerate(sims) if s >= thr), None)
            per_p[p] = {
                "conf_rank": conf_rank,
                "best_sim": results["per_paradigm_best_sim"][p]["best_sim"],
                "r_at_5": 1 if conf_rank and conf_rank <= 5 else 0,
                "r_at_10": 1 if conf_rank and conf_rank <= 10 else 0,
            }
        primary_r10 = sum(per_p[p]["r_at_10"] for p in PRIMARY if p in per_p)
        all_r10 = sum(per_p[p]["r_at_10"] for p in per_p)
        results["thresholds"][f"{thr:.2f}"] = {
            "per_paradigm": per_p,
            "primary_r10": primary_r10,
            "all_r10": all_r10,
        }
    return results


def embed_openai_large(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    api_key = open("/tmp/unbox_oai_key.txt").read().strip()
    client = OpenAI(api_key=api_key)
    out = []
    BATCH = 100
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        r = client.embeddings.create(model="text-embedding-3-large", input=chunk)
        out.extend([d.embedding for d in r.data])
    return out


def embed_minilm(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = m.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs.tolist()


def main():
    out_dir = PROJECT_ROOT / "experiments" / "cross_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_extractions_canonical()
    targets = load_targets()
    print(f"Loaded {len(candidates)} paradigms; primary present: "
          f"{[p for p in PRIMARY if p in candidates]}", flush=True)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    full = {"thresholds_swept": thresholds}

    # 1) OpenAI text-embedding-3-large
    print("\n=== OpenAI text-embedding-3-large ===", flush=True)
    t0 = time.time()
    res_large = evaluate_with_embedder(embed_openai_large, "text-embedding-3-large",
                                       candidates, targets, thresholds)
    print(f"text-embedding-3-large done in {time.time()-t0:.1f}s", flush=True)
    full["text-embedding-3-large"] = res_large

    # 2) MiniLM
    print("\n=== sentence-transformers/all-MiniLM-L6-v2 ===", flush=True)
    t0 = time.time()
    res_minilm = evaluate_with_embedder(embed_minilm, "all-MiniLM-L6-v2",
                                        candidates, targets, thresholds)
    print(f"all-MiniLM-L6-v2 done in {time.time()-t0:.1f}s", flush=True)
    full["all-MiniLM-L6-v2"] = res_minilm

    # Aggregate summary
    summary = {
        "baseline_text-embedding-3-small_primary_r10": 4,
        "baseline_text-embedding-3-small_all_r10": 6,
        "baseline_threshold": 0.65,
        "per_embedding": {},
    }
    for name, res in [("text-embedding-3-large", res_large),
                      ("all-MiniLM-L6-v2", res_minilm)]:
        per_thr = res.get("thresholds", {})
        summary["per_embedding"][name] = {
            thr: {"primary_r10": v["primary_r10"], "all_r10": v["all_r10"]}
            for thr, v in per_thr.items()
        }

    full["aggregate"] = summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nDONE -> {out_dir/'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
