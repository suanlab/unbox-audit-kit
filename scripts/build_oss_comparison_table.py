#!/usr/bin/env python3
"""Build a cross-LLM comparison markdown table from canonical_evaluation.json files.

Reads multiple canonical_evaluation.json files (one per model) and emits a
markdown table showing R@5, R@10, and per-paradigm hit/miss for the 4 primary
paradigms. Useful for eyeballing OSS-vs-closed LLM results and for copy-paste
into the paper LaTeX after results land.

Usage:
    python scripts/build_oss_comparison_table.py \\
        --label gpt4o     --eval experiments/canonical_evaluation.json \\
        --label claude    --eval experiments/claude_clean_15paper/canonical_evaluation.json \\
        --label llama-8b  --eval experiments/llama3_1_8b_clean/canonical_evaluation.json \\
        --label qwen-7b   --eval experiments/qwen2_5_7b_clean/canonical_evaluation.json \\
        --output experiments/cross_llm_comparison.md

If a --eval path does not exist (e.g., OSS run not yet complete), the row is
emitted with "pending" cells so the table is well-formed during submission prep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY = ["transformer", "diffusion", "icl", "vit"]


def load_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def cell_for(paradigm: str, results: list[dict]) -> str:
    for r in results:
        if r.get("paradigm") == paradigm:
            rank = r.get("conf_rank")
            sim = r.get("best_sim")
            if rank is not None:
                return f"✓ (rank {rank}, sim {sim:.3f})"
            return f"✗ (sim {sim:.3f})"
    return "—"


def aggregate_primary(results: list[dict]) -> tuple[int, int, float | None]:
    sub = [r for r in results if r.get("paradigm") in PRIMARY]
    r5 = sum(1 for r in sub if r.get("r_at_5", 0) > 0)
    r10 = sum(1 for r in sub if r.get("r_at_10", 0) > 0)
    sims = [r.get("best_sim") for r in sub if isinstance(r.get("best_sim"), (int, float))]
    avg_sim = (sum(sims) / len(sims)) if sims else None
    return r5, r10, avg_sim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", action="append", required=True,
                    help="Model label; repeat for each --eval (paired in order)")
    ap.add_argument("--eval", action="append", required=True,
                    help="Path to canonical_evaluation.json; repeat per --label")
    ap.add_argument("--output", default="-",
                    help="Markdown output path; '-' for stdout (default)")
    args = ap.parse_args()

    if len(args.label) != len(args.eval):
        print("ERROR: --label and --eval must be paired (same count)", file=sys.stderr)
        return 1

    pairs = list(zip(args.label, args.eval))
    rows = []
    for label, eval_path in pairs:
        path = Path(eval_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        data = load_eval(path)
        if data is None:
            rows.append({
                "label": label,
                "available": False,
                "path": str(path),
            })
            continue
        results = data.get("results", [])
        r5, r10, avg_sim = aggregate_primary(results)
        rows.append({
            "label": label,
            "available": True,
            "results": results,
            "r5": r5,
            "r10": r10,
            "avg_sim": avg_sim,
            "ranking_method": data.get("ranking_method", ""),
        })

    # Markdown
    lines = []
    lines.append("# Cross-LLM Comparison (Primary 4 Paradigms)")
    lines.append("")
    lines.append("| Model | Transformer | Diffusion | ICL | ViT | R@5 | R@10 | Avg Sim |")
    lines.append("|-------|-------------|-----------|-----|-----|-----|------|---------|")
    for row in rows:
        if not row["available"]:
            lines.append(f"| {row['label']} | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |")
            continue
        results = row["results"]
        cells = [cell_for(p, results) for p in PRIMARY]
        avg_str = f"{row['avg_sim']:.3f}" if row["avg_sim"] is not None else "—"
        lines.append(f"| {row['label']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {row['r5']}/4 | {row['r10']}/4 | {avg_str} |")
    lines.append("")

    # LaTeX-ready summary (for paper paste)
    lines.append("## LaTeX-ready rows for tab:cross_llm")
    lines.append("```latex")
    for row in rows:
        if not row["available"]:
            lines.append(f"% {row['label']}: pending ({row['path']})")
            lines.append(f"{row['label']:25s} & \\multicolumn{{6}}{{c}}{{\\emph{{[PENDING — populate from OSS run]}}}} \\\\")
            continue
        results = row["results"]
        symbols = []
        for p in PRIMARY:
            for r in results:
                if r.get("paradigm") == p:
                    symbols.append("\\checkmark" if r.get("r_at_5", 0) > 0 else "\\texttimes")
                    break
            else:
                symbols.append("—")
        avg = f"{row['avg_sim']:.3f}" if row["avg_sim"] is not None else "—"
        lines.append(f"{row['label']:25s} & {symbols[0]} & {symbols[1]} & {symbols[2]} & {symbols[3]} & {row['r5']}/4 & {avg} \\\\")
    lines.append("```")
    lines.append("")

    # Sources note
    lines.append("## Sources")
    for row in rows:
        status = "OK" if row["available"] else "MISSING"
        path = row.get("path", "(loaded)") if not row["available"] else "(loaded)"
        lines.append(f"- **{row['label']}**: {status}  {path}")

    out = "\n".join(lines) + "\n"
    if args.output == "-":
        sys.stdout.write(out)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
