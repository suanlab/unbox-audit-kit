#!/usr/bin/env python3
"""Min-K% Prob non-member baseline: same procedure on synthetic abstracts.

The 3 fictional fields in experiments/synthetic_benchmark/*/abstracts.json
are pure GPT-generated text on FICTIONAL field names; none of these
abstracts can be in any model's pretraining set. We compute Min-K%(20) Prob
on these and compare against the same statistic on the real pre-shift
corpora (per-model). A large gap (real >> synthetic) is a contamination
signal that survives the MIA modality; a small gap means MIA can't
distinguish, and the behavioral controls become the only sensitive signal.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
K_PERCENT = 0.20

MODELS = {
    "llama-nous": {"hf_id": "NousResearch/Meta-Llama-3.1-8B-Instruct", "max_model_len": 8192},
    "qwen":       {"hf_id": "Qwen/Qwen2.5-7B-Instruct",                "max_model_len": 8192},
    "mistral":    {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3",      "max_model_len": 8192},
}


def load_synthetic_abstracts() -> list[str]:
    fields = ["chromatic_signal_processing", "neuromorphic_memory_architecture",
              "adaptive_topology_networks"]
    texts = []
    for f in fields:
        path = PROJECT_ROOT / "experiments" / "synthetic_benchmark" / f / "abstracts.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        items = d if isinstance(d, list) else d.get("abstracts", d.get("papers", []))
        for item in items:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                txt = (item.get("title", "") or "") + "\n\n" + (item.get("abstract", "") or "")
                if txt.strip():
                    texts.append(txt)
    return texts


def minkpct(lps, k=K_PERCENT):
    s = sorted(lps)
    n = max(1, int(len(s) * k))
    return sum(s[:n]) / n


def run(model_key, output_path):
    from vllm import LLM, SamplingParams
    cfg = MODELS[model_key]
    texts = load_synthetic_abstracts()
    print(f"[{model_key}] {len(texts)} synthetic abstracts", flush=True)
    if not texts:
        print("No synthetic abstracts found"); return
    t0 = time.time()
    llm = LLM(model=cfg["hf_id"], max_model_len=cfg["max_model_len"],
              gpu_memory_utilization=0.35, dtype="bfloat16")
    print(f"[{model_key}] loaded in {time.time()-t0:.1f}s", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    outs = llm.generate(texts, sp)
    per_paper = []
    for txt, out in zip(texts, outs):
        plp = out.prompt_logprobs or []
        tok_ids = out.prompt_token_ids or []
        lps = []
        for i, pos_dict in enumerate(plp):
            if pos_dict is None or i >= len(tok_ids):
                continue
            tid = tok_ids[i]
            if tid in pos_dict:
                lp_obj = pos_dict[tid]
                lp = getattr(lp_obj, "logprob", lp_obj)
                if isinstance(lp, (int, float)):
                    lps.append(float(lp))
        if lps:
            per_paper.append({"n_tokens": len(lps), "min_kpct_prob": minkpct(lps),
                              "mean_logprob": sum(lps)/len(lps)})
    scores = [p["min_kpct_prob"] for p in per_paper]
    out_data = {
        "model": cfg["hf_id"],
        "kind": "synthetic_nonmember",
        "n_papers": len(per_paper),
        "mean_min_kpct_prob": sum(scores) / len(scores) if scores else None,
        "per_paper": per_paper,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[{model_key}] synth mean Min-K%(20) Prob = {out_data['mean_min_kpct_prob']:.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
