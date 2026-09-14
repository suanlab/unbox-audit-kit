#!/usr/bin/env python3
"""Wrong-Corpus Control, expanded to all 12 (4 x 3) cross pairings.

Mirrors scripts/run_wrong_corpus_control.py exactly (same prompt, same
threshold, same seed, same model, same canonical evaluation pipeline) but
enumerates every (target_paradigm, wrong_corpus) pair where target != corpus
across the 4 primary shifts. Confirms the 0/4 -> 0/12 wrong-corpus collapse.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from itertools import product

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from semantic_match import embed_texts, cosine_similarity  # noqa: E402

PRIMARY = ["transformer", "diffusion", "icl", "vit"]
PAPERS_PER_PAIR = 15
SOFT_THRESHOLD = 0.65
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL = "gpt-4o"
SEED = 42

CLEAN_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions that academic papers inherit from their broader research community.

Your task is NOT to find this specific paper's design choices. Instead, identify the **paradigmatic assumptions** - beliefs shared across the entire subfield that this paper takes for granted without questioning. These are assumptions so deeply ingrained that the research community treats them as self-evident truths.

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


def load_papers(paradigm: str, n: int = PAPERS_PER_PAIR) -> list[dict]:
    path = PROJECT_ROOT / "data" / paradigm / "papers.jsonl"
    papers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers[:n]


def load_target(paradigm: str) -> dict:
    m = json.load(open(PROJECT_ROOT / "data" / "paradigm_shift_mapping.json"))
    return m[paradigm]


def extract_paper(client: OpenAI, paper: dict) -> list[dict]:
    text = (paper.get("title", "") or "") + "\n\n" + (paper.get("abstract", "") or "")
    prompt = CLEAN_PROMPT.format(paper_text=text)
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=SEED,
    )
    raw = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("assumptions", [])
    except Exception:
        return []


def evaluate_pairing(client: OpenAI, target_p: str, corpus_p: str) -> dict:
    target = load_target(target_p)
    corpus = load_papers(corpus_p)
    print(f"  [{target_p} x {corpus_p}] {len(corpus)} papers", flush=True)
    all_assumptions = []
    for i, paper in enumerate(corpus):
        ass = extract_paper(client, paper)
        for a in ass:
            a["_paper_idx"] = i
            all_assumptions.append(a)
    # Sort by confidence (canonical pipeline)
    all_assumptions.sort(key=lambda a: -float(a.get("confidence", 0)))
    # Unique by assumption text
    seen = set()
    unique = []
    for a in all_assumptions:
        s = a.get("assumption", "").strip()
        if s and s not in seen:
            seen.add(s)
            unique.append(a)

    # Score against target aliases
    aliases = target.get("aliases", [target.get("description", target.get("target", ""))])
    if not aliases:
        aliases = [target.get("description", "")]
    target_embs = embed_texts(aliases)
    cand_texts = [a["assumption"] for a in unique[:50]]
    cand_embs = embed_texts(cand_texts) if cand_texts else []

    best_sim = 0.0
    conf_rank = None
    sims = []
    for rank, (a, e) in enumerate(zip(unique[:50], cand_embs), start=1):
        sim = max(cosine_similarity(e, t) for t in target_embs)
        sims.append(sim)
        if sim > best_sim:
            best_sim = sim
        if conf_rank is None and sim >= SOFT_THRESHOLD:
            conf_rank = rank
    r_at_5 = 1 if (conf_rank is not None and conf_rank <= 5) else 0
    r_at_10 = 1 if (conf_rank is not None and conf_rank <= 10) else 0
    r_at_20 = 1 if (conf_rank is not None and conf_rank <= 20) else 0
    return {
        "target": target_p,
        "corpus": corpus_p,
        "num_unique_assumptions": len(unique),
        "best_sim": round(best_sim, 4),
        "conf_rank": conf_rank,
        "r_at_5": r_at_5,
        "r_at_10": r_at_10,
        "r_at_20": r_at_20,
    }


def main():
    api_key = open("/tmp/unbox_oai_key.txt").read().strip()
    client = OpenAI(api_key=api_key)
    out_dir = PROJECT_ROOT / "experiments" / "wrong_corpus_12"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"

    pairings = [(t, c) for t, c in product(PRIMARY, PRIMARY) if t != c]
    print(f"Running {len(pairings)} pairings ...", flush=True)

    results = []
    # Resume support
    existing = {}
    if out_path.exists():
        try:
            existing_data = json.load(open(out_path))
            for r in existing_data.get("results", []):
                existing[(r["target"], r["corpus"])] = r
        except Exception:
            existing = {}

    t0 = time.time()
    for i, (t_p, c_p) in enumerate(pairings, 1):
        key = (t_p, c_p)
        if key in existing:
            print(f"[{i}/{len(pairings)}] {t_p} x {c_p} -- cached", flush=True)
            results.append(existing[key])
            continue
        print(f"[{i}/{len(pairings)}] {t_p} x {c_p} ...", flush=True)
        r = evaluate_pairing(client, t_p, c_p)
        results.append(r)
        # Incremental save
        summary = build_summary(results, pairings_total=len(pairings))
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary = build_summary(results, pairings_total=len(pairings))
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE -> {out_path}  ({time.time()-t0:.1f}s)", flush=True)


def build_summary(results, pairings_total: int) -> dict:
    hits = sum(r["r_at_10"] for r in results)
    avg_sim = sum(r["best_sim"] for r in results) / len(results) if results else 0
    return {
        "model": MODEL,
        "seed": SEED,
        "papers_per_pairing": PAPERS_PER_PAIR,
        "embedding_model": EMBEDDING_MODEL,
        "threshold": SOFT_THRESHOLD,
        "results": results,
        "aggregate": {
            "n_pairings_completed": len(results),
            "n_pairings_total": pairings_total,
            "wrong_corpus_r10": f"{hits}/{len(results)}",
            "avg_wrong_corpus_best_sim": round(avg_sim, 4),
        },
    }


if __name__ == "__main__":
    main()
