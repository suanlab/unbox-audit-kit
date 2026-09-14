#!/usr/bin/env python3
"""T1.3: Contamination analysis.

Quantify how "famous" each breakthrough is, then correlate with
R@10 performance. If popularity correlates with success,
the benchmark measures popularity more than extraction.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_wikipedia_pageviews(title: str, days: int = 365) -> int:
    """Fetch Wikipedia pageviews (approximate popularity proxy).

    Uses Wikipedia REST API. Returns total pageviews over last N days.
    """
    # URL-encode title
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{encoded}/monthly/20240101/20241231"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UnboxResearch/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            items = data.get("items", [])
            return sum(item.get("views", 0) for item in items)
    except Exception:
        return 0


# Manual popularity scores based on common knowledge
# (since we can't easily crawl Google Scholar)
POPULARITY_DATA = {
    # Primary 4 (very famous)
    "transformer": {
        "wiki_title": "Transformer_(deep_learning_architecture)",
        "citation_count_2024": 100000,  # approx for Vaswani et al.
        "first_paper_year": 2017,
        "fame_tier": "extreme",
    },
    "diffusion": {
        "wiki_title": "Diffusion_model",
        "citation_count_2024": 15000,  # Ho et al. DDPM
        "first_paper_year": 2020,
        "fame_tier": "very_high",
    },
    "icl": {
        "wiki_title": "In-context_learning_(natural_language_processing)",
        "citation_count_2024": 30000,  # GPT-3 paper
        "first_paper_year": 2020,
        "fame_tier": "very_high",
    },
    "vit": {
        "wiki_title": "Vision_transformer",
        "citation_count_2024": 25000,  # Dosovitskiy et al.
        "first_paper_year": 2020,
        "fame_tier": "very_high",
    },
    # Additional 6 (varying fame)
    "gan": {
        "wiki_title": "Generative_adversarial_network",
        "citation_count_2024": 60000,
        "first_paper_year": 2014,
        "fame_tier": "extreme",
    },
    "batchnorm": {
        "wiki_title": "Batch_normalization",
        "citation_count_2024": 50000,
        "first_paper_year": 2015,
        "fame_tier": "very_high",
    },
    "resnet": {
        "wiki_title": "Residual_neural_network",
        "citation_count_2024": 200000,
        "first_paper_year": 2015,
        "fame_tier": "extreme",
    },
    "word2vec": {
        "wiki_title": "Word2vec",
        "citation_count_2024": 40000,
        "first_paper_year": 2013,
        "fame_tier": "extreme",
    },
    "dropout": {
        "wiki_title": "Dropout_(neural_networks)",
        "citation_count_2024": 45000,
        "first_paper_year": 2014,
        "fame_tier": "extreme",
    },
    "bert": {
        "wiki_title": "BERT_(language_model)",
        "citation_count_2024": 80000,
        "first_paper_year": 2018,
        "fame_tier": "extreme",
    },
}


# Our paper's GPT-4o clean results (from canonical_evaluation.json)
RESULTS = {
    "transformer": {"conf_rank": 1, "r_at_5": 1, "r_at_10": 1, "best_sim": 0.674},
    "diffusion": {"conf_rank": 2, "r_at_5": 1, "r_at_10": 1, "best_sim": 0.759},
    "icl": {"conf_rank": 3, "r_at_5": 1, "r_at_10": 1, "best_sim": 0.702},
    "vit": {"conf_rank": 5, "r_at_5": 1, "r_at_10": 1, "best_sim": 0.676},
    "batchnorm": {"conf_rank": 7, "r_at_5": 0, "r_at_10": 1, "best_sim": 0.727},
    "gan": {"conf_rank": 10, "r_at_5": 0, "r_at_10": 1, "best_sim": 0.657},
    "resnet": {"conf_rank": None, "r_at_5": 0, "r_at_10": 0, "best_sim": 0.597},
    "word2vec": {"conf_rank": None, "r_at_5": 0, "r_at_10": 0, "best_sim": 0.633},
    "dropout": {"conf_rank": None, "r_at_5": 0, "r_at_10": 0, "best_sim": 0.604},
    "bert": {"conf_rank": None, "r_at_5": 0, "r_at_10": 0, "best_sim": 0.544},
}


def spearman_correlation(x: list, y: list) -> float:
    """Simple Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return 0.0

    def ranks(values):
        sorted_vals = sorted(enumerate(values), key=lambda p: p[1])
        r = [0] * len(values)
        for rank, (i, _) in enumerate(sorted_vals, 1):
            r[i] = rank
        return r

    rx = ranks(x)
    ry = ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
    den_y = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5
    return num / (den_x * den_y) if den_x * den_y > 0 else 0.0


def main():
    # Fetch wikipedia pageviews
    print("Fetching Wikipedia pageviews...")
    pageviews = {}
    for paradigm, info in POPULARITY_DATA.items():
        pv = get_wikipedia_pageviews(info["wiki_title"])
        pageviews[paradigm] = pv
        print(f"  {paradigm:15s}: {info['wiki_title']:45s} → {pv:>10,} views")

    # Compute correlations
    paradigms = list(POPULARITY_DATA.keys())

    # Contamination proxies
    citations = [POPULARITY_DATA[p]["citation_count_2024"] for p in paradigms]
    views = [pageviews.get(p, 0) for p in paradigms]

    # Performance
    r_at_10_vals = [RESULTS[p]["r_at_10"] for p in paradigms]
    r_at_5_vals = [RESULTS[p]["r_at_5"] for p in paradigms]
    best_sims = [RESULTS[p]["best_sim"] for p in paradigms]

    # Correlations
    spearman_cite_r10 = spearman_correlation(citations, r_at_10_vals)
    spearman_views_r10 = spearman_correlation(views, r_at_10_vals)
    spearman_cite_sim = spearman_correlation(citations, best_sims)
    spearman_views_sim = spearman_correlation(views, best_sims)

    print("\n=== Contamination vs Performance Correlations (N=10) ===")
    print(f"Spearman(citations, R@10):      ρ = {spearman_cite_r10:+.3f}")
    print(f"Spearman(wiki_views, R@10):     ρ = {spearman_views_r10:+.3f}")
    print(f"Spearman(citations, best_sim):  ρ = {spearman_cite_sim:+.3f}")
    print(f"Spearman(wiki_views, best_sim): ρ = {spearman_views_sim:+.3f}")

    # Save
    out = {
        "experiment": "contamination_analysis",
        "n": 10,
        "popularity_proxies": {
            p: {
                "wiki_title": POPULARITY_DATA[p]["wiki_title"],
                "pageviews_2024": pageviews.get(p, 0),
                "citations_approx": POPULARITY_DATA[p]["citation_count_2024"],
                "fame_tier": POPULARITY_DATA[p]["fame_tier"],
                "first_paper_year": POPULARITY_DATA[p]["first_paper_year"],
            }
            for p in paradigms
        },
        "performance": {
            p: RESULTS[p] for p in paradigms
        },
        "correlations": {
            "spearman_citations_r10": round(spearman_cite_r10, 4),
            "spearman_wiki_views_r10": round(spearman_views_r10, 4),
            "spearman_citations_best_sim": round(spearman_cite_sim, 4),
            "spearman_wiki_views_best_sim": round(spearman_views_sim, 4),
        },
        "interpretation": (
            "Higher popularity correlates with higher performance if Spearman ρ > 0. "
            "Small N=10 means correlation stability is limited. "
            "A positive correlation would support the memorization hypothesis — "
            "benchmark success is partly a function of breakthrough fame."
        ),
    }
    out_path = PROJECT_ROOT / "experiments" / "contamination" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
