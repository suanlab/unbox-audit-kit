#!/usr/bin/env python3
"""G-1 + G-2 multi-seed variance for CRITICAL-A3.

Re-runs the 4 primary paradigms under {with-corpus, no-corpus} (G-1) and the
4 wrong-corpus pairings (G-2) across base_seeds {1, 2, 42}, using the SAME
prompts and matcher as the canonical pipeline (text-embedding-3-small,
cosine >= 0.65, confidence-ordered ranks). Per-cell = 15 GPT-4o calls;
API-level non-determinism produces the residual variation that aggregates
into the per-paradigm rank distribution.

Output:
    experiments/multiseed_variance/<condition>__<key>__seed<S>.json   (per-cell)
    experiments/multiseed_variance/summary.json                       (aggregate)

Aggregate summary structure:
    g1.with_corpus.<paradigm>.r_at_10_per_seed -> [int,int,int]
    g1.no_corpus.<paradigm>.r_at_10_per_seed   -> [int,int,int]
    g2.wrong_corpus.<pair>.r_at_10_per_seed    -> [int,int,int]
    aggregate.with_corpus.primary_r_at_10_per_seed  -> [4,4,4]  (or with variance)
    aggregate.no_corpus.primary_r_at_10_per_seed    -> [4,4,4]
    aggregate.wrong_corpus.primary_r_at_10_per_seed -> [0,0,0]

Cost: ~360 GPT-4o calls + ~36 embedding batches ~ $2-3, ~20-30 min.

Usage:
    OPENAI_API_KEY=... python scripts/run_multiseed_variance.py
    # or with the on-disk override key:
    OPENAI_API_KEY="$(cat /tmp/unbox_oai_key.txt)" python scripts/run_multiseed_variance.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "experiments" / "multiseed_variance"

# Verified identical to scripts/run_gpt4o_clean_all10.py CLEAN_PROMPT.
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

# Verified identical to scripts/run_no_corpus_control.py PROMPT_TEMPLATE.
NO_CORPUS_PROMPT = """You are an expert at identifying the foundational, field-wide assumptions
that researchers in a given field take for granted.

Identify the top 10 paradigmatic assumptions — beliefs shared across the
entire subfield that researchers take for granted without questioning.

Use the pattern: "[X] is necessary/required/essential for [Y]". State
what the field believes is NECESSARY, REQUIRED, or ESSENTIAL.

Return ONLY valid JSON:
{{"assumptions": [{{"assumption": "string", "confidence": 0.0-1.0, "category": "architectural|training|data|theoretical|evaluation"}}]}}

Field: {field_name}"""

FIELD_NAMES = {  # verified identical to FIELD_NAMES_NO_YEAR
    "transformer": "sequence modeling",
    "diffusion":   "generative modeling",
    "icl":         "transfer learning in NLP",
    "vit":         "visual recognition",
}

# Verified identical to scripts/run_wrong_corpus_control.py PAIRINGS.
WRONG_PAIRS = {
    "transformer_x_diffusion": {"target": "transformer", "corpus": "diffusion"},
    "diffusion_x_transformer": {"target": "diffusion",   "corpus": "transformer"},
    "icl_x_vit":               {"target": "icl",         "corpus": "vit"},
    "vit_x_icl":               {"target": "vit",         "corpus": "icl"},
}

PRIMARY = ["transformer", "diffusion", "icl", "vit"]
SEEDS = [1, 2, 42]
PAPERS_PER = 15
CALLS_PER_FIELD = 15  # no-corpus: same prompt 15 times (matches existing script)
THRESHOLD = 0.65
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o"
TOP_K = 20


def load_papers(paradigm: str, n: int = PAPERS_PER) -> list[dict]:
    path = DATA_DIR / paradigm / "papers.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def format_paper_text(paper: dict) -> str:
    parts = [f"Title: {paper.get('title', 'Unknown')}"]
    if paper.get("year"):
        parts.append(f"Year: {paper['year']}")
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    return "\n".join(parts)


def parse_assumptions(text: str) -> list[dict]:
    s = text.strip()
    try:
        d = json.loads(s)
        return d.get("assumptions", []) if isinstance(d, dict) else []
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        try:
            d = json.loads(m.group(1))
            return d.get("assumptions", []) if isinstance(d, dict) else []
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return d.get("assumptions", []) if isinstance(d, dict) else []
        except Exception:
            pass
    return []


def gpt4o(client, prompt: str, seed: int, max_retries: int = 3) -> list[dict]:
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
                seed=seed,
            )
            return parse_assumptions(r.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    [WARN] GPT-4o failed: {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
                return []
            time.sleep(2 * (attempt + 1))
    return []


def dedupe_rank(assumptions: list[dict]) -> list[dict]:
    seen, ranked = set(), []
    for a in sorted(assumptions, key=lambda x: -float(x.get("confidence", 0) or 0)):
        k = (a.get("assumption") or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            ranked.append(a)
    return ranked


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 0 else 0.0


def canonical_eval(client, assumptions: list[dict], target_aliases: list[str]) -> dict:
    """Match canonical_evaluator.py exactly: top_k=20, threshold 0.65, conf-ordered."""
    if not assumptions:
        return {"conf_rank": None, "best_sim": 0.0, "r_at_5": 0, "r_at_10": 0, "r_at_20": 0}
    cap = assumptions[:TOP_K]
    texts = [a["assumption"] for a in cap] + target_aliases
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = [e.embedding for e in resp.data]
    a_embs, gt_embs = embs[: len(cap)], embs[len(cap):]
    sims = [max(cosine(ae, ge) for ge in gt_embs) for ae in a_embs]
    best = max(sims) if sims else 0.0
    conf_rank = next((i + 1 for i, s in enumerate(sims) if s >= THRESHOLD), None)
    return {
        "conf_rank": conf_rank,
        "best_sim": round(best, 4),
        "r_at_5":  1 if any(s >= THRESHOLD for s in sims[:5])  else 0,
        "r_at_10": 1 if any(s >= THRESHOLD for s in sims[:10]) else 0,
        "r_at_20": 1 if any(s >= THRESHOLD for s in sims[:20]) else 0,
    }


def extract_with_corpus(client, paradigm: str, seed: int) -> list[dict]:
    assumptions = []
    for paper in load_papers(paradigm):
        prompt = CLEAN_PROMPT.format(paper_text=format_paper_text(paper))
        for a in gpt4o(client, prompt, seed):
            if isinstance(a, dict):
                a["source_paper"] = paper.get("title", "")
                assumptions.append(a)
    return dedupe_rank(assumptions)


def extract_no_corpus(client, paradigm: str, seed: int) -> list[dict]:
    prompt = NO_CORPUS_PROMPT.format(field_name=FIELD_NAMES[paradigm])
    assumptions = []
    for idx in range(CALLS_PER_FIELD):
        for a in gpt4o(client, prompt, seed):
            if isinstance(a, dict):
                a["call_idx"] = idx
                assumptions.append(a)
    return dedupe_rank(assumptions)


def extract_wrong_corpus(client, target_paradigm: str, corpus_paradigm: str, seed: int) -> list[dict]:
    assumptions = []
    for paper in load_papers(corpus_paradigm):  # wrong corpus
        prompt = CLEAN_PROMPT.format(paper_text=format_paper_text(paper))
        for a in gpt4o(client, prompt, seed):
            if isinstance(a, dict):
                a["source_paper"] = paper.get("title", "")
                assumptions.append(a)
    return dedupe_rank(assumptions)


def target_aliases(paradigm: str, gt: dict) -> list[str]:
    info = gt[paradigm]
    return [info["broken_assumption"]] + info.get("aliases", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-with", action="store_true", help="skip G-1 with-corpus cells")
    ap.add_argument("--skip-no",   action="store_true", help="skip G-1 no-corpus cells")
    ap.add_argument("--skip-wrong", action="store_true", help="skip G-2 wrong-corpus cells")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = json.loads((DATA_DIR / "paradigm_shift_mapping.json").read_text())

    results = {
        "g1": {"with_corpus": {}, "no_corpus": {}},
        "g2": {"wrong_corpus": {}},
        "seeds": args.seeds,
        "config": {
            "threshold": THRESHOLD, "embedding_model": EMBED_MODEL,
            "gpt_model": GPT_MODEL, "papers_per": PAPERS_PER,
            "calls_per_field_no_corpus": CALLS_PER_FIELD, "top_k": TOP_K,
        },
    }

    def cell_path(condition, key, seed):
        return OUT_DIR / f"{condition}__{key}__seed{seed}.json"

    def run_cell(condition, key, seed, extract_fn, aliases):
        path = cell_path(condition, key, seed)
        if path.exists():
            print(f"  [skip cached] {path.name}")
            return json.loads(path.read_text())
        t0 = time.time()
        print(f"  [{condition}] {key} seed={seed} ...", flush=True)
        assumptions = extract_fn()
        ev = canonical_eval(client, assumptions, aliases)
        cell = {"condition": condition, "key": key, "seed": seed,
                "num_unique": len(assumptions), **ev,
                "elapsed_s": round(time.time() - t0, 1)}
        path.write_text(json.dumps(cell, indent=2))
        print(f"    -> r@10={ev['r_at_10']} rank={ev['conf_rank']} sim={ev['best_sim']} "
              f"({cell['elapsed_s']}s)")
        return cell

    # G-1 with-corpus
    if not args.skip_with:
        for seed in args.seeds:
            for p in PRIMARY:
                aliases = target_aliases(p, gt)
                cell = run_cell("with_corpus", p, seed,
                                lambda p=p, s=seed: extract_with_corpus(client, p, s), aliases)
                results["g1"]["with_corpus"].setdefault(p, {})[str(seed)] = cell

    # G-1 no-corpus
    if not args.skip_no:
        for seed in args.seeds:
            for p in PRIMARY:
                aliases = target_aliases(p, gt)
                cell = run_cell("no_corpus", p, seed,
                                lambda p=p, s=seed: extract_no_corpus(client, p, s), aliases)
                results["g1"]["no_corpus"].setdefault(p, {})[str(seed)] = cell

    # G-2 wrong-corpus
    if not args.skip_wrong:
        for seed in args.seeds:
            for key, pair in WRONG_PAIRS.items():
                aliases = target_aliases(pair["target"], gt)
                cell = run_cell("wrong_corpus", key, seed,
                                lambda pair=pair, s=seed: extract_wrong_corpus(
                                    client, pair["target"], pair["corpus"], s), aliases)
                results["g2"]["wrong_corpus"].setdefault(key, {})[str(seed)] = cell

    # Aggregate: per-seed primary R@10 totals
    agg = {}
    if results["g1"]["with_corpus"]:
        agg["with_corpus_primary_r10_per_seed"] = [
            sum(results["g1"]["with_corpus"].get(p, {}).get(str(s), {}).get("r_at_10", 0)
                for p in PRIMARY)
            for s in args.seeds
        ]
    if results["g1"]["no_corpus"]:
        agg["no_corpus_primary_r10_per_seed"] = [
            sum(results["g1"]["no_corpus"].get(p, {}).get(str(s), {}).get("r_at_10", 0)
                for p in PRIMARY)
            for s in args.seeds
        ]
    if results["g2"]["wrong_corpus"]:
        agg["wrong_corpus_r10_per_seed"] = [
            sum(results["g2"]["wrong_corpus"].get(k, {}).get(str(s), {}).get("r_at_10", 0)
                for k in WRONG_PAIRS)
            for s in args.seeds
        ]
    results["aggregate"] = agg

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
