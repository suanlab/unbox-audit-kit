#!/usr/bin/env python3
"""Offline consistency check: assert the paper's headline numbers match the
frozen experiment artifacts.

No API calls. This does NOT recompute results (canonical_evaluator.py needs
the OpenAI embedding API and is single-run); it re-reads the stored artifacts
and asserts every headline number against them, printing any mismatch.

Usage:
    python scripts/reproduce_paper_tables.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASS = 0
FAIL = 0


def check(label: str, expected, actual, tol: float = 0.002):
    global PASS, FAIL
    if isinstance(expected, float) and isinstance(actual, float):
        ok = abs(expected - actual) < tol
    else:
        ok = expected == actual
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  [{status}] {label}: expected={expected}, got={actual}")
    else:
        PASS += 1


def table4_main_result():
    """Table 4: GPT-4o Clean Prompt main results."""
    print("\n=== Table 4: Main Results (GPT-4o Clean Prompt) ===")
    path = PROJECT_ROOT / "experiments" / "canonical_evaluation.json"
    if not path.exists():
        print("  SKIP: canonical_evaluation.json not found")
        return
    data = json.load(open(path))
    results = {r["paradigm"]: r for r in data["results"]}

    expected = {
        "transformer": {"conf_rank": 1, "best_sim": 0.674, "r_at_10": 1.0},
        "diffusion": {"conf_rank": 2, "best_sim": 0.759, "r_at_10": 1.0},
        "icl": {"conf_rank": 3, "best_sim": 0.702, "r_at_10": 1.0},
        "vit": {"conf_rank": 5, "best_sim": 0.676, "r_at_10": 1.0},
        "batchnorm": {"conf_rank": 7, "best_sim": 0.727, "r_at_10": 1.0},
        "gan": {"conf_rank": 10, "best_sim": 0.657, "r_at_10": 1.0},
        "resnet": {"conf_rank": None, "best_sim": 0.597, "r_at_10": 0.0},
        "word2vec": {"conf_rank": None, "best_sim": 0.633, "r_at_10": 0.0},
        "dropout": {"conf_rank": None, "best_sim": 0.604, "r_at_10": 0.0},
        "bert": {"conf_rank": None, "best_sim": 0.544, "r_at_10": 0.0},
    }

    for paradigm, exp in expected.items():
        r = results.get(paradigm, {})
        check(f"{paradigm} conf_rank", exp["conf_rank"], r.get("conf_rank"))
        check(f"{paradigm} best_sim", exp["best_sim"], r.get("best_sim"))
        check(f"{paradigm} R@10", exp["r_at_10"], r.get("r_at_10"))

    # Aggregate
    total_r10 = sum(1 for r in results.values() if r.get("r_at_10", 0) > 0)
    check("Total R@10", 6, total_r10)
    total_r5 = sum(1 for r in results.values() if r.get("r_at_5", 0) > 0)
    check("Total R@5", 4, total_r5)


def table_retrieval_baselines():
    """Section 5.13: TF-IDF/BM25 retrieval baselines."""
    print("\n=== Section 5.13: Retrieval Baselines ===")
    path = PROJECT_ROOT / "experiments" / "simple_baselines" / "summary.json"
    if not path.exists():
        print("  SKIP: simple_baselines/summary.json not found")
        return
    data = json.load(open(path))

    for method in ["tfidf", "bm25"]:
        agg = data["aggregate"][method]
        check(f"{method} R@10", 0.5, agg["r_at_10"])


def contamination_analysis():
    """Section 5.13: Popularity-performance correlations."""
    print("\n=== Section 5.13: Contamination Analysis ===")
    path = PROJECT_ROOT / "experiments" / "contamination" / "summary.json"
    if not path.exists():
        print("  SKIP: contamination/summary.json not found")
        return
    data = json.load(open(path))
    corr = data["correlations"]
    check("Spearman(citations, best_sim)", -0.6364, corr["spearman_citations_best_sim"], tol=0.01)
    check("Spearman(citations, R@10)", -0.3212, corr["spearman_citations_r10"], tol=0.01)


def calibration_analysis():
    """Section 5.13: Confidence calibration."""
    print("\n=== Section 5.13: Calibration Analysis ===")
    path = PROJECT_ROOT / "experiments" / "calibration" / "summary.json"
    if not path.exists():
        print("  SKIP: calibration/summary.json not found")
        return
    data = json.load(open(path))
    summary = data.get("summary", {})
    check("confidence_gap_top_to_hit", 0.025, summary.get("confidence_gap_top_to_hit"))
    mono = data.get("monotonicity", {})
    check("monotonic_paradigms", "10/10", mono.get("monotonic_paradigms"))


def error_analysis():
    """Section 5.13: Error analysis on 4 failures."""
    print("\n=== Section 5.13: Error Analysis ===")
    path = PROJECT_ROOT / "experiments" / "error_analysis" / "summary.json"
    if not path.exists():
        print("  SKIP: error_analysis/summary.json not found")
        return
    data = json.load(open(path))
    kw = data.get("keyword_analysis", {})
    check("resnet keywords in top-10", 0, kw.get("resnet", {}).get("count_in_top10", -1))
    check("bert keywords in top-10", 0, kw.get("bert", {}).get("count_in_top10", -1))


def synthetic_benchmark():
    """Section 5.8: Synthetic literature control."""
    print("\n=== Section 5.8: Synthetic Benchmark ===")
    path = PROJECT_ROOT / "experiments" / "synthetic_benchmark" / "summary.json"
    if not path.exists():
        print("  SKIP: synthetic_benchmark/summary.json not found")
        return
    data = json.load(open(path))
    scenarios = data.get("scenarios", [])
    check("num_synthetic_fields", 3, len(scenarios))


def wrong_corpus_control():
    """§5.3 abstract: wrong-corpus collapses to 0/4 vs with-corpus 4/4."""
    print("\n=== Section 5.3: Wrong-Corpus Control ===")
    path = PROJECT_ROOT / "experiments" / "wrong_corpus_control" / "summary.json"
    if not path.exists():
        print("  SKIP: wrong_corpus_control/summary.json not found")
        return
    d = json.load(open(path))
    ag = d.get("aggregate", {})
    check("wrong_corpus R@10", "0/4", ag.get("wrong_corpus_r10"))
    check("with_corpus R@10", "4/4", ag.get("with_corpus_r10"))
    hits = sum(1 for r in d.get("results", []) if r.get("r_at_10", 0))
    check("wrong-corpus pairings recovered", 0, hits)
    # 12-pairing expansion
    p12 = PROJECT_ROOT / "experiments" / "wrong_corpus_12" / "summary.json"
    if p12.exists():
        d12 = json.load(open(p12))
        ag12 = d12.get("aggregate", {})
        check("wrong_corpus_12 R@10", "0/12", ag12.get("wrong_corpus_r10"))
        check("wrong_corpus_12 pairings completed", 12, ag12.get("n_pairings_completed"))
        hits12 = sum(1 for r in d12.get("results", []) if r.get("r_at_10", 0))
        check("wrong_corpus_12 pairings recovered", 0, hits12)


def synthetic_scenario_ranks():
    """§5.3: synthetic Scenario 2 (Neuromorphic) rank 2 with corpus vs 27 without."""
    print("\n=== Section 5.3: Synthetic Scenario-2 Ranks ===")
    path = PROJECT_ROOT / "experiments" / "synthetic_benchmark" / "summary.json"
    if not path.exists():
        print("  SKIP: synthetic_benchmark/summary.json not found")
        return
    d = json.load(open(path))
    sc = {s.get("field"): s for s in d.get("scenarios", [])}
    neu = sc.get("Neuromorphic Memory Architecture", {})
    check("Scenario-2 with-corpus conf_rank", 2.0, neu.get("with_corpus", {}).get("conf_rank"))
    check("Scenario-2 no-corpus conf_rank", 27.0, neu.get("no_corpus", {}).get("conf_rank"))


def cross_llm_sweep():
    """§5.4 abstract: GPT-4o 4/4, Qwen 2/4, Claude 1/4, Llama 1/4, Mistral 0/4 (primary R@5)."""
    print("\n=== Section 5.4: Cross-LLM Sweep (primary R@5) ===")
    PRIM = {"transformer", "diffusion", "icl", "vit"}
    expected = {
        "canonical_evaluation": 4,
        "claude_clean_15paper/canonical_evaluation": 1,
        "llama3_1_8b_clean/canonical_evaluation": 1,
        "qwen2_5_7b_clean/canonical_evaluation": 2,
        "mistral_7b_clean/canonical_evaluation": 0,
    }
    for rel, exp in expected.items():
        p = PROJECT_ROOT / "experiments" / f"{rel}.json"
        if not p.exists():
            print(f"  SKIP: {rel}.json not found")
            continue
        res = json.load(open(p)).get("results", [])
        prim = [r for r in res if r.get("paradigm") in PRIM]
        got = sum(1 for r in prim if r.get("r_at_5", 0) > 0)
        check(f"{rel.split('/')[0]} primary R@5", exp, got)


def niche_benchmark():
    """§5.3: niche with-corpus 1/8, no-corpus 2/7 (KAN no-corpus incomplete)."""
    print("\n=== Section 5.3: Niche Subfield Control ===")
    path = PROJECT_ROOT / "experiments" / "niche_benchmark" / "summary.json"
    if not path.exists():
        print("  SKIP: niche_benchmark/summary.json not found")
        return
    res = json.load(open(path)).get("results", [])
    wc = sum(1 for r in res if r.get("with_corpus", {}).get("r_at_10") == 1)
    nc = sum(1 for r in res if r.get("no_corpus", {}).get("r_at_10") == 1)
    check("niche with-corpus hits", 1, wc)
    check("niche no-corpus hits", 2, nc)


def human_validation():
    """§5.4 abstract: human study alpha=0.63, human-vs-rule kappa=0.70, AC 79/80."""
    print("\n=== Section 5.4: Human Validation ===")
    path = PROJECT_ROOT / "annotation" / "g1_results.json"
    if not path.exists():
        print("  SKIP: annotation/g1_results.json not found")
        return
    g = json.load(open(path))
    check("Krippendorff alpha", 0.634, round(g.get("krippendorff_alpha_interval", -9), 3), tol=0.01)
    hv = g.get("human_vs_rule") or {}
    check("human-vs-rule kappa", 0.702, round(hv.get("kappa", -9), 3), tol=0.01)
    acp = g.get("attention_check_pass", {})
    passed = sum(v["passed"] for v in acp.values())
    total = sum(v["total"] for v in acp.values())
    check("attention-check passed", 79, passed)
    check("attention-check total", 80, total)


def main():
    print("=" * 60)
    print("Paper Table Reproduction (Offline, No API)")
    print("=" * 60)

    table4_main_result()
    table_retrieval_baselines()
    contamination_analysis()
    calibration_analysis()
    error_analysis()
    synthetic_benchmark()
    wrong_corpus_control()
    synthetic_scenario_ranks()
    cross_llm_sweep()
    niche_benchmark()
    human_validation()
    cross_embedding_robustness()
    mia_minkpct_check()
    multiseed_variance()


def cross_embedding_robustness():
    """App I: primary R@10 = 4/4 holds at calibrated threshold per embedder."""
    print("\n=== App I: Cross-Embedding Sensitivity ===")
    path = PROJECT_ROOT / "experiments" / "cross_embedding" / "summary.json"
    if not path.exists():
        print("  SKIP: cross_embedding/summary.json not found")
        return
    d = json.load(open(path))
    large = d.get("text-embedding-3-large", {}).get("thresholds", {}).get("0.50", {})
    minilm = d.get("all-MiniLM-L6-v2", {}).get("thresholds", {}).get("0.65", {})
    check("text-embedding-3-large @0.50: primary R@10", 4, large.get("primary_r10"))
    check("text-embedding-3-large @0.50: all R@10", 8, large.get("all_r10"))
    check("all-MiniLM-L6-v2 @0.65: primary R@10", 4, minilm.get("primary_r10"))
    check("all-MiniLM-L6-v2 @0.65: all R@10", 7, minilm.get("all_r10"))


def mia_minkpct_check():
    """App J: Min-K% Prob real-vs-synth gap is significantly negative in all 3 OSS models."""
    print("\n=== App J: Min-K%(20) Prob (real vs synthetic) ===")
    base = PROJECT_ROOT / "experiments" / "mia_minkpct"
    for m in ["llama", "qwen", "mistral"]:
        real_p = base / f"{m}.json"
        synth_p = base / f"{m}_synth.json"
        if not real_p.exists() or not synth_p.exists():
            print(f"  SKIP: mia_minkpct/{m}[_synth].json not found")
            continue
        real = json.load(open(real_p)).get("per_paradigm", {})
        synth = json.load(open(synth_p))
        if not real or synth.get("mean_min_kpct_prob") is None:
            continue
        real_mean = sum(v["mean_min_kpct_prob"] for v in real.values()) / len(real)
        synth_mean = synth["mean_min_kpct_prob"]
        gap = real_mean - synth_mean
        # Stored claim: all 3 models show negative gap (synth > real in member-likeness)
        check(f"{m}: real-synth gap is negative", True, gap < 0)
    gap_p = base / "gap_stats.json"
    if gap_p.exists():
        gs = json.load(open(gap_p))
        for m in ["llama", "qwen", "mistral"]:
            entry = gs.get(m, {})
            check(f"{m}: 95% CI upper bound < 0 (significant)", True,
                  entry.get("gap_significantly_negative", False))


def multiseed_variance():
    """§Limitations: Multi-seed run-variance assertions (3 seeds, 4 primary)."""
    print("\n=== Limitations: Multi-seed Run Variance ===")
    path = PROJECT_ROOT / "experiments" / "multiseed_variance" / "summary.json"
    if not path.exists():
        print("  SKIP: multiseed_variance/summary.json not found")
        return
    ag = json.load(open(path)).get("aggregate", {})
    check("no_corpus per-seed R@10 (3 seeds, all 4 primary stable)",
          [4, 4, 4], ag.get("no_corpus_primary_r10_per_seed"))
    check("wrong_corpus per-seed R@10 (3 seeds, 0/4 stable)",
          [0, 0, 0], ag.get("wrong_corpus_r10_per_seed"))
    # with-corpus mildly unstable: seed 1=4, 2=4, 42=3 in current re-run
    check("with_corpus per-seed R@10",
          [4, 4, 3], ag.get("with_corpus_primary_r10_per_seed"))

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print("\nWARNING: Some paper numbers do not match stored artifacts!")
        sys.exit(1)
    else:
        print("\nAll paper numbers match stored artifacts.")
        sys.exit(0)


if __name__ == "__main__":
    main()
