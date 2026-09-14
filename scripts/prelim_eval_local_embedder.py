#!/usr/bin/env python3
"""Preliminary cross-model evaluation using LOCAL sentence-transformers.

WARNING: This is NOT the canonical evaluator. The headline metrics in the
paper use OpenAI text-embedding-3-small at threshold 0.65. This script uses
a local MiniLM model for an offline preliminary look while the OpenAI quota
is unavailable. Thresholds are NOT directly comparable across embedders.

What this provides:
  - apples-to-apples ranking across GPT-4o, Claude, Qwen, Mistral on the
    SAME local embedder, so relative ordering is informative even though
    absolute thresholds differ from canonical
  - best similarity and conf-rank per paradigm per model
  - markdown table for inspection

Usage:
    .venv/bin/python scripts/prelim_eval_local_embedder.py \\
        --output experiments/oss_llm_prelim_eval.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRIMARY = ["transformer", "diffusion", "icl", "vit"]
EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Where each model's per-paradigm extraction lives
SOURCES = [
    ("gpt-4o",          "experiments/gpt4o_clean_prompt/{p}.json"),
    ("claude-sonnet-4", "experiments/claude_clean_15paper/{p}.json"),
    ("llama-3.1-8b",    "experiments/llama3_1_8b_clean/{p}.json"),
    ("qwen-2.5-7b",     "experiments/qwen2_5_7b_clean/{p}.json"),
    ("mistral-7b",      "experiments/mistral_7b_clean/{p}.json"),
]


def cosine(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="experiments/oss_llm_prelim_eval.md")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="MiniLM-calibrated threshold (NOT canonical 0.65); chosen empirically for cross-model ordering")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"[prelim] loading {EMBEDDER_ID}", file=sys.stderr)
    embedder = SentenceTransformer(EMBEDDER_ID)

    gt_map = json.loads((PROJECT_ROOT / "data" / "paradigm_shift_mapping.json").read_text())

    results: dict[str, dict] = {label: {} for label, _ in SOURCES}

    for label, tmpl in SOURCES:
        for paradigm in PRIMARY:
            path = PROJECT_ROOT / tmpl.format(p=paradigm)
            if not path.exists():
                results[label][paradigm] = {"available": False}
                continue
            data = json.loads(path.read_text())
            assumptions = data.get("assumptions", [])[: args.top_k]
            if not assumptions:
                results[label][paradigm] = {"available": True, "empty": True}
                continue

            gt_info = gt_map.get(paradigm, {})
            aliases = [gt_info.get("broken_assumption", "")] + gt_info.get("aliases", [])
            aliases = [a for a in aliases if a]

            texts = [a["assumption"] for a in assumptions] + aliases
            embs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            a_embs = embs[: len(assumptions)]
            gt_embs = embs[len(assumptions):]
            sims = [max(cosine(ae, ge) for ge in gt_embs) for ae in a_embs]

            best_sim = max(sims) if sims else 0.0
            conf_rank = next((i + 1 for i, s in enumerate(sims) if s >= args.threshold), None)
            r5 = 1 if any(s >= args.threshold for s in sims[:5]) else 0
            r10 = 1 if any(s >= args.threshold for s in sims[:10]) else 0

            results[label][paradigm] = {
                "available": True,
                "conf_rank": conf_rank,
                "best_sim": round(best_sim, 4),
                "r_at_5": r5,
                "r_at_10": r10,
                "n_top_k": len(assumptions),
            }

    # Markdown
    lines = [
        "# Preliminary Cross-Model Evaluation (local embedder)",
        "",
        f"- **Embedder:** {EMBEDDER_ID}",
        f"- **Threshold:** {args.threshold} (MiniLM-calibrated; NOT canonical 0.65 which uses OpenAI text-embedding-3-small)",
        f"- **Top-K considered:** {args.top_k}",
        "- **Status:** PRELIMINARY — for relative ordering only. Final paper numbers will use canonical evaluator once OpenAI quota is restored.",
        "",
        "## Per-paradigm rank (1-indexed) and best similarity",
        "",
        "| Model | Transformer | Diffusion | ICL | ViT | R@5 | R@10 |",
        "|-------|-------------|-----------|-----|-----|-----|------|",
    ]
    for label, _ in SOURCES:
        cells = []
        r5_total = 0
        r10_total = 0
        n = 0
        for p in PRIMARY:
            r = results[label].get(p, {"available": False})
            if not r.get("available"):
                cells.append("missing")
                continue
            if r.get("empty"):
                cells.append("empty")
                continue
            n += 1
            r5_total += r["r_at_5"]
            r10_total += r["r_at_10"]
            rank = r["conf_rank"]
            sim = r["best_sim"]
            if rank is not None:
                cells.append(f"✓ r{rank} ({sim:.3f})")
            else:
                cells.append(f"✗ ({sim:.3f})")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {r5_total}/{n} | {r10_total}/{n} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- ✓ = best similarity at any of top-K reached threshold; rank = 1-indexed confidence-ordered position of first such hit")
    lines.append("- ✗ = no top-K extraction reached threshold (but best-sim shown)")
    lines.append("- This table is generated by `scripts/prelim_eval_local_embedder.py` and should be regenerated whenever extraction outputs change.")
    lines.append("- For paper Table 7, await canonical evaluation via `scripts/canonical_evaluator.py` once OpenAI quota is restored.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
