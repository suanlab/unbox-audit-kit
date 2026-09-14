#!/usr/bin/env python3
"""Run OSS LLM extraction via vLLM on a single A100.

Mirrors scripts/run_gpt4o_clean_all10.py: same CLEAN_PROMPT, same paradigms,
same 15 papers, same temperature=0.0, same JSON schema. Output structure
matches experiments/gpt4o_clean_prompt/ so canonical_evaluator.py can reuse
it with a path swap.

Models (safe option):
    llama  -> meta-llama/Llama-3.1-8B-Instruct  (gated, needs HF token)
    qwen   -> Qwen/Qwen2.5-7B-Instruct          (open)

Setup (one-time):
    pip install vllm
    huggingface-cli login   # for Llama-3.1 gated access

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_oss_llm_extraction.py \\
        --model llama --output-dir experiments/llama3_1_8b_clean

Parallel across A100 0/1 (recommended):
    bash scripts/run_oss_llm_parallel.sh
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- exact same prompt as GPT-4o run for fair cross-model comparison ---
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

MODEL_CONFIGS = {
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_model_len": 8192,
        "trust_remote_code": False,
        "note": "gated; request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "max_model_len": 8192,
        "trust_remote_code": False,
        "note": "open weights, no HF gate",
    },
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 8192,
        "trust_remote_code": False,
        "note": "open weights, no HF gate; second open-source comparator while Llama gate is pending",
    },
    "llama-nous": {
        "hf_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "max_model_len": 8192,
        "trust_remote_code": False,
        "note": "NousResearch un-gated mirror of meta-llama/Llama-3.1-8B-Instruct (identical weights); used while meta-llama gate is pending",
    },
}

PRIMARY_PARADIGMS = ["transformer", "diffusion", "icl", "vit"]
PAPERS_PER_PARADIGM = 15


def load_papers(paradigm: str) -> list[dict]:
    path = PROJECT_ROOT / "data" / paradigm / "papers.jsonl"
    papers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers[:PAPERS_PER_PARADIGM]


def format_paper_text(paper: dict) -> str:
    parts = [f"Title: {paper.get('title', 'Unknown')}"]
    if paper.get("year"):
        parts.append(f"Year: {paper['year']}")
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    return "\n".join(parts)


def parse_assumptions_json(text: str) -> list[dict]:
    """Robust JSON extraction from LLM output."""
    s = text.strip()
    # 1. direct parse
    try:
        d = json.loads(s)
        if isinstance(d, dict) and "assumptions" in d:
            return d["assumptions"]
    except Exception:
        pass
    # 2. strip markdown fence
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            if isinstance(d, dict) and "assumptions" in d:
                return d["assumptions"]
        except Exception:
            pass
    # 3. greedy first-{ to last-}
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and "assumptions" in d:
                return d["assumptions"]
        except Exception:
            pass
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--paradigms", nargs="+", default=PRIMARY_PARADIGMS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # lazy import so this script can be inspected without vLLM installed
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"ERROR: vllm not installed. pip install vllm. ({e})", file=sys.stderr)
        return 1

    print(f"[oss-extract] loading {cfg['hf_id']} ...")
    llm = LLM(
        model=cfg["hf_id"],
        max_model_len=cfg["max_model_len"],
        trust_remote_code=cfg["trust_remote_code"],
        seed=args.seed,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # build all prompts up-front for one-shot batched inference
    specs: list[tuple[str, dict, str]] = []
    for paradigm in args.paradigms:
        for paper in load_papers(paradigm):
            user_msg = CLEAN_PROMPT.format(paper_text=format_paper_text(paper))
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
            specs.append((paradigm, paper, chat))

    print(f"[oss-extract] generating {len(specs)} prompts in batch ...")
    outputs = llm.generate([s[2] for s in specs], sampling)

    grouped: dict[str, list[dict]] = {p: [] for p in args.paradigms}
    parse_failures: dict[str, int] = {p: 0 for p in args.paradigms}
    for (paradigm, paper, _), out in zip(specs, outputs):
        text = out.outputs[0].text
        assumptions = parse_assumptions_json(text)
        if not assumptions:
            parse_failures[paradigm] += 1
            continue
        for a in assumptions:
            if not isinstance(a, dict):
                continue
            a["source_paper"] = paper.get("title", "")
            grouped[paradigm].append(a)

    for paradigm in args.paradigms:
        # dedupe + confidence-rank (mirror gpt4o_clean_prompt format)
        seen = set()
        ranked: list[dict] = []
        items = sorted(
            grouped[paradigm],
            key=lambda x: -float(x.get("confidence", 0.0) or 0.0),
        )
        for a in items:
            key = (a.get("assumption") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ranked.append(a)

        record = {
            "paradigm": paradigm,
            "model": cfg["hf_id"],
            "num_papers": PAPERS_PER_PARADIGM,
            "num_raw_assumptions": len(grouped[paradigm]),
            "num_unique_assumptions": len(ranked),
            "num_parse_failures": parse_failures[paradigm],
            "seed": args.seed,
            "assumptions": ranked,
        }
        out_path = out_dir / f"{paradigm}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  {paradigm}: {len(ranked)} unique (raw {len(grouped[paradigm])}, "
              f"parse_fail {parse_failures[paradigm]}) -> {out_path}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": cfg["hf_id"],
            "seed": args.seed,
            "paradigms": args.paradigms,
            "papers_per_paradigm": PAPERS_PER_PARADIGM,
            "temperature": 0.0,
        }, f, indent=2)
    print(f"[oss-extract] done. summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
