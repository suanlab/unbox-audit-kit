#!/usr/bin/env python3
"""Wrong-Corpus Control Experiment.

Tests whether providing a DIFFERENT field's corpus destroys the extraction of
a target paradigm's assumption. If wrong-corpus is comparable to with-corpus,
the corpus content does not actually steer extraction (memorization dominates).
If wrong-corpus destroys the signal, corpus content genuinely steers extraction.

Pairings (wrong corpus for each primary target):
  - transformer target  x  diffusion corpus
  - diffusion target    x  transformer corpus
  - icl target          x  vit corpus
  - vit target          x  icl corpus

Reports best similarity and confidence rank against the *right* target aliases,
using the canonical clean prompt and confidence-ordered ranking.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_match import embed_texts, cosine_similarity  # noqa: E402

# --- Configuration --------------------------------------------------------
PAIRINGS = {
    "transformer_x_diffusion": {"target": "transformer", "corpus": "diffusion"},
    "diffusion_x_transformer": {"target": "diffusion", "corpus": "transformer"},
    "icl_x_vit":               {"target": "icl",         "corpus": "vit"},
    "vit_x_icl":               {"target": "vit",         "corpus": "icl"},
}

PAPERS_PER_PAIR = 15
SOFT_THRESHOLD = 0.65
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL = "gpt-4o"

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


def load_papers(paradigm: str) -> list[dict]:
    path = PROJECT_ROOT / "data" / paradigm / "papers.jsonl"
    papers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers[:PAPERS_PER_PAIR]


def format_paper_text(paper: dict) -> str:
    parts = [f"Title: {paper.get('title', 'Unknown')}"]
    if paper.get("year"):
        parts.append(f"Year: {paper['year']}")
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    return "\n".join(parts)


def extract_assumptions(client: OpenAI, paper_text: str) -> list[dict]:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": CLEAN_PROMPT.format(paper_text=paper_text)}],
                temperature=0.0,
                seed=42,
            )
            content = resp.choices[0].message.content.strip()
            m = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if m:
                content = m.group(1).strip()
            parsed = json.loads(content)
            return parsed.get("assumptions", [])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"    parse retry {attempt+1}: {e}", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"    api retry {attempt+1}: {e}", flush=True)
            time.sleep(5)
    return []


def run_pair(client: OpenAI, pair_name: str, target_paradigm: str, corpus_paradigm: str,
             targets: dict) -> dict:
    print(f"\n=== {pair_name}: extracting from {corpus_paradigm} corpus ===", flush=True)
    papers = load_papers(corpus_paradigm)
    all_assumptions = []
    for i, paper in enumerate(papers):
        print(f"  [{i+1}/{len(papers)}] {paper.get('title', 'Unknown')[:60]}", flush=True)
        extractions = extract_assumptions(client, format_paper_text(paper))
        all_assumptions.extend(extractions)

    # Dedupe by normalized text (exact-string)
    seen = set()
    unique = []
    for a in all_assumptions:
        t = a.get("assumption", "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            unique.append(a)

    # Confidence-ordered (stable sort)
    unique.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

    # Embed against target aliases (the RIGHT target, not the corpus's target)
    target_info = targets[target_paradigm]
    aliases = [target_info["broken_assumption"]] + target_info.get("aliases", [])

    if not unique:
        return {
            "pair": pair_name,
            "target": target_paradigm,
            "corpus": corpus_paradigm,
            "num_assumptions": 0,
            "best_sim": None,
            "conf_rank": None,
            "r_at_5": 0,
            "r_at_10": 0,
        }

    assumption_texts = [a["assumption"] for a in unique]
    alias_embs = embed_texts(aliases)
    cand_embs = embed_texts(assumption_texts)

    best_sim = 0.0
    best_rank = None
    hit_rank = None
    for rank, (a, ce) in enumerate(zip(unique, cand_embs), start=1):
        sim = max(cosine_similarity(ce, ae) for ae in alias_embs)
        if sim > best_sim:
            best_sim = sim
        if sim >= SOFT_THRESHOLD and hit_rank is None:
            hit_rank = rank

    r_at_5 = 1 if (hit_rank is not None and hit_rank <= 5) else 0
    r_at_10 = 1 if (hit_rank is not None and hit_rank <= 10) else 0

    return {
        "pair": pair_name,
        "target": target_paradigm,
        "corpus": corpus_paradigm,
        "num_assumptions": len(unique),
        "best_sim": round(best_sim, 4),
        "conf_rank": hit_rank,
        "r_at_5": r_at_5,
        "r_at_10": r_at_10,
        "top_5": [{"rank": i+1, "assumption": a["assumption"][:150],
                   "confidence": a.get("confidence")} for i, a in enumerate(unique[:5])],
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    with open(PROJECT_ROOT / "data" / "paradigm_shift_mapping.json") as f:
        targets = json.load(f)

    out_dir = PROJECT_ROOT / "experiments" / "wrong_corpus_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pair_name, cfg in PAIRINGS.items():
        r = run_pair(client, pair_name, cfg["target"], cfg["corpus"], targets)
        results.append(r)
        print(f"  -> best_sim={r['best_sim']}, conf_rank={r['conf_rank']}, R@10={r['r_at_10']}", flush=True)

    with_corpus_reference = {
        "transformer": {"best_sim": 0.674, "conf_rank": 1, "r_at_10": 1},
        "diffusion":   {"best_sim": 0.759, "conf_rank": 2, "r_at_10": 1},
        "icl":         {"best_sim": 0.702, "conf_rank": 3, "r_at_10": 1},
        "vit":         {"best_sim": 0.676, "conf_rank": 5, "r_at_10": 1},
    }

    summary = {
        "experiment": "wrong_corpus_control",
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "soft_threshold": SOFT_THRESHOLD,
        "papers_per_pair": PAPERS_PER_PAIR,
        "description": "For each primary target paradigm, extract assumptions from a DIFFERENT field's corpus and evaluate against the right target aliases.",
        "pairings": PAIRINGS,
        "results": results,
        "with_corpus_reference": with_corpus_reference,
        "aggregate": {
            "avg_wrong_corpus_best_sim": round(
                sum((r["best_sim"] or 0.0) for r in results) / len(results), 4
            ),
            "avg_with_corpus_best_sim": round(
                sum(v["best_sim"] for v in with_corpus_reference.values()) / 4, 4
            ),
            "wrong_corpus_r10": f"{sum(r['r_at_10'] for r in results)}/{len(results)}",
            "with_corpus_r10": "4/4",
        },
    }
    summary["aggregate"]["delta_best_sim"] = round(
        summary["aggregate"]["avg_with_corpus_best_sim"] - summary["aggregate"]["avg_wrong_corpus_best_sim"], 4
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")
    print(f"Aggregate: wrong={summary['aggregate']['avg_wrong_corpus_best_sim']}, "
          f"right={summary['aggregate']['avg_with_corpus_best_sim']}, "
          f"delta={summary['aggregate']['delta_best_sim']}")


if __name__ == "__main__":
    main()
