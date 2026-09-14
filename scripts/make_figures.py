#!/usr/bin/env python3
"""Generate the 5 paper figures as vector PDFs into paper/figs/.

Figures (all vector PDF, matplotlib):
  fig_mia_box.pdf       : Min-K% Prob real vs synth per OSS model (box plot)
  fig_cross_llm_bar.pdf : 5-LLM R@5 per primary paradigm (grouped bar)
  fig_multiseed.pdf     : best_sim across 3 seeds per (paradigm, condition)
  fig_threshold.pdf     : threshold sweep R@10 per embedder
  fig_rank_heatmap.pdf  : per-paradigm conf_rank heatmap (10 paradigms x conditions)

All output is vector (Type 1 fonts, no rasterization).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Vector PDF output -- ensure no rasterized text
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    p = FIGS / name
    fig.savefig(p, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")


# ---------------------------------------------------------------------------
# Figure: MIA real vs synthetic box plot
# ---------------------------------------------------------------------------
def fig_mia_box():
    models = ["llama", "qwen", "mistral"]
    labels = ["Llama-3.1-8B", "Qwen-2.5-7B", "Mistral-7B"]
    data = []
    for m in models:
        real = json.load(open(ROOT / "experiments" / "mia_minkpct" / f"{m}.json"))
        synth = json.load(open(ROOT / "experiments" / "mia_minkpct" / f"{m}_synth.json"))
        r_scores = [p["min_kpct_prob"]
                    for v in real["per_paradigm"].values()
                    for p in v.get("per_paper", [])]
        s_scores = [p["min_kpct_prob"] for p in synth.get("per_paper", [])]
        data.append((r_scores, s_scores))

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    positions = []
    colors_real = "#3a76b4"
    colors_synth = "#d97557"
    for i, (real, synth) in enumerate(data):
        x_real = i * 3
        x_synth = i * 3 + 1
        bp_real = ax.boxplot([real], positions=[x_real], widths=0.7,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor=colors_real, alpha=0.7, edgecolor="black"),
                             medianprops=dict(color="black"))
        bp_synth = ax.boxplot([synth], positions=[x_synth], widths=0.7,
                              patch_artist=True, showfliers=False,
                              boxprops=dict(facecolor=colors_synth, alpha=0.7, edgecolor="black"),
                              medianprops=dict(color="black"))
        positions.append(x_real + 0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Min-K%(20) Prob\n(nats / token; less negative = more probable)")
    ax.axhline(y=-6.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors_real, alpha=0.7, edgecolor="black",
                     label="Real pre-shift papers ($n{=}60{\\times}4$)"),
               Patch(facecolor=colors_synth, alpha=0.7, edgecolor="black",
                     label="Synthetic non-member abstracts ($n{=}45$)")]
    # Place legend below the axes so it never overlaps the whiskers.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=8, frameon=False)
    ax.set_title("Min-K\\% Prob: synthetic non-members appear $\\it{more}$ member-like\n"
                 "than real pre-shift papers (paired-bootstrap 95\\% CI $<0$ for all 3 models)",
                 fontsize=9)
    save(fig, "fig_mia_box.pdf")


# ---------------------------------------------------------------------------
# Figure: 5-LLM R@5 grouped bar chart per primary paradigm
# ---------------------------------------------------------------------------
def fig_cross_llm_bar():
    """Categorical dot-matrix replacement for the original grouped bar chart.

    The underlying R@5 signal is binary per (model, paradigm) cell, so a
    bar chart wastes most of its vertical space on whitespace; a 5x4 dot
    matrix conveys the same information at a fraction of the page area
    and makes each model's total recovery count read off at a glance.
    Each row shows one LLM across the 4 primary paradigms; filled circle
    = R@5 match, open circle = miss. A trailing count column reports the
    per-model total. Kept distinct from Figure 6 (which encodes
    continuous confidence rank by colour gradient over 10 paradigms).
    """
    PRIMARY = ["transformer", "diffusion", "icl", "vit"]
    LABELS = {"transformer": "Transformer", "diffusion": "Diffusion",
              "icl": "ICL", "vit": "ViT"}
    sources = [
        ("GPT-4o", "experiments/canonical_evaluation.json", "#1f77b4"),
        ("Claude Sonnet 4", "experiments/claude_clean_15paper/canonical_evaluation.json", "#ff7f0e"),
        ("Llama-3.1-8B", "experiments/llama3_1_8b_clean/canonical_evaluation.json", "#2ca02c"),
        ("Qwen-2.5-7B", "experiments/qwen2_5_7b_clean/canonical_evaluation.json", "#d62728"),
        ("Mistral-7B", "experiments/mistral_7b_clean/canonical_evaluation.json", "#9467bd"),
    ]
    matrix = []  # (name, row, color)
    for name, path, color in sources:
        d = json.load(open(ROOT / path))
        per = {r["paradigm"]: r for r in d["results"] if r["paradigm"] in PRIMARY}
        row = [per.get(p, {}).get("r_at_5", 0) for p in PRIMARY]
        matrix.append((name, row, color))

    n_rows = len(matrix)
    n_cols = len(PRIMARY)
    fig, ax = plt.subplots(figsize=(5.3, 2.2))

    # Faint horizontal guide for each row, faint vertical guide for each column.
    for r in range(n_rows):
        ax.axhline(y=r, color="lightgray", linewidth=0.4, zorder=1)
    for c in range(n_cols):
        ax.axvline(x=c, color="lightgray", linewidth=0.4, zorder=1)

    for r, (name, row, color) in enumerate(matrix):
        for c, val in enumerate(row):
            if val:
                ax.scatter(c, r, s=160, color=color, edgecolors="black",
                           linewidths=0.6, zorder=3)
            else:
                ax.scatter(c, r, s=160, facecolors="none", edgecolors="lightgray",
                           linewidths=0.8, zorder=3)
        # Per-model total at the right
        total = int(round(sum(row)))
        ax.text(n_cols - 0.35, r, f"{total}/{n_cols}", va="center", ha="left",
                fontsize=8, color="black", fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([LABELS[p] for p in PRIMARY])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([m[0] for m in matrix])
    ax.invert_yaxis()  # GPT-4o on top
    ax.set_xlim(-0.5, n_cols + 0.2)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.tick_params(left=False, bottom=False)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Cross-LLM Soft-R@5 on identical prompts and 15-paper corpora\n"
                 "(canonical pipeline: text-embedding-3-small, threshold 0.65)",
                 fontsize=9)
    # Inline legend at the bottom margin
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", color="black",
                      markerfacecolor="#777777", markeredgecolor="black",
                      markersize=8, label="match (R@5 = 1)"),
               Line2D([0], [0], marker="o", linestyle="", color="black",
                      markerfacecolor="none", markeredgecolor="lightgray",
                      markersize=8, label="miss (R@5 = 0)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=7.5, frameon=False)
    save(fig, "fig_cross_llm_bar.pdf")


# ---------------------------------------------------------------------------
# Figure: Multi-seed strip plot
# ---------------------------------------------------------------------------
def fig_multiseed():
    d = json.load(open(ROOT / "experiments" / "multiseed_variance" / "summary.json"))
    PRIMARY = ["transformer", "diffusion", "icl", "vit"]
    LABELS = {"transformer": "Trans.", "diffusion": "Diff.", "icl": "ICL", "vit": "ViT"}
    SEEDS = ["1", "2", "42"]
    g1 = d["g1"]
    g2 = d.get("g2", {})

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    cond_meta = [
        ("with_corpus", 0, "#1f77b4", "with-corpus"),
        ("no_corpus", 1, "#2ca02c", "no-corpus"),
        # wrong_corpus is per-pairing under g2 -- expand below
    ]
    # x-axis: paradigm groups
    x_positions = {}
    pos = 0
    for p in PRIMARY:
        x_positions[p] = pos
        pos += 1
    threshold = 0.65

    for cond_name, cond_offset, color, label in cond_meta:
        for p in PRIMARY:
            seeds = g1[cond_name][p]
            xs = []
            ys = []
            for s in SEEDS:
                xs.append(x_positions[p] + cond_offset * 0.18 - 0.09)
                ys.append(seeds[s]["best_sim"])
            ax.scatter(xs, ys, s=22, color=color, alpha=0.85, edgecolors="black",
                       linewidth=0.4, label=label if p == PRIMARY[0] else None,
                       zorder=3)
    # wrong-corpus (g2) entries
    wc_pairs = list(g2.keys())  # e.g. "transformer_x_diffusion"
    for pair_key in wc_pairs:
        tgt = pair_key.split("_x_")[0]
        if tgt not in PRIMARY:
            continue
        xs, ys = [], []
        for s in SEEDS:
            if s not in g2[pair_key]:
                continue
            xs.append(x_positions[tgt] + 0.27)
            ys.append(g2[pair_key][s]["best_sim"])
        if xs:
            ax.scatter(xs, ys, s=22, color="#d62728", alpha=0.85, marker="x",
                       label="wrong-corpus" if tgt == PRIMARY[0] else None, zorder=3)

    ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=0.5,
               label="threshold = 0.65")
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([LABELS[p] for p in PRIMARY])
    ax.set_ylabel("best cosine similarity")
    ax.set_title("Multi-seed variance (3 seeds per cell): with-corpus 4/4/3,\n"
                 "no-corpus 4/4/4, wrong-corpus 0/0/0 across seeds",
                 fontsize=9)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_ylim(0.30, 0.90)
    save(fig, "fig_multiseed.pdf")


# ---------------------------------------------------------------------------
# Figure: threshold sweep curve per embedder
# ---------------------------------------------------------------------------
def fig_threshold():
    d = json.load(open(ROOT / "experiments" / "cross_embedding" / "summary.json"))
    embedders = [
        ("text-embedding-3-small (baseline)", "baseline", "#1f77b4"),
        ("text-embedding-3-large", "text-embedding-3-large", "#ff7f0e"),
        ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "#2ca02c"),
    ]
    thr_axis = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.5))

    # Baseline only has one point at threshold 0.65 from canonical_evaluation
    # Approximate the baseline curve by reading the canonical extractions and
    # re-thresholding the SAME sims at different thresholds.
    canon = json.load(open(ROOT / "experiments" / "canonical_evaluation.json"))
    base_sims = {r["paradigm"]: r["best_sim"] for r in canon["results"]}
    PRIM = {"transformer", "diffusion", "icl", "vit"}

    def baseline_curve():
        # primary R@10: how many of the 4 primary have best_sim >= thr
        p, a = [], []
        for thr in thr_axis:
            n_p = sum(1 for k, v in base_sims.items() if k in PRIM and v >= thr)
            n_a = sum(1 for v in base_sims.values() if v >= thr)
            p.append(n_p)
            a.append(n_a)
        return p, a

    for name, key, color in embedders:
        if key == "baseline":
            p_r, a_r = baseline_curve()
        else:
            entry = d.get(key, {}).get("thresholds", {})
            p_r = [entry.get(f"{thr:.2f}", {}).get("primary_r10", 0) for thr in thr_axis]
            a_r = [entry.get(f"{thr:.2f}", {}).get("all_r10", 0) for thr in thr_axis]
        ax1.plot(thr_axis, p_r, marker="o", label=name.split(" (")[0],
                 color=color, linewidth=1.4)
        ax2.plot(thr_axis, a_r, marker="o", label=name.split(" (")[0],
                 color=color, linewidth=1.4)

    for ax, title, ylab, ymax in [(ax1, "Primary R@10 (of 4)", "matches", 4.4),
                                  (ax2, "All-paradigm R@10 (of 10)", "matches", 10.5)]:
        ax.axvline(x=0.65, color="gray", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_xlabel("threshold")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.2, ymax)
        ax.grid(linestyle=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=7, frameon=False)
    save(fig, "fig_threshold.pdf")


# ---------------------------------------------------------------------------
# Figure: per-paradigm rank heatmap
# ---------------------------------------------------------------------------
def fig_rank_heatmap():
    PARADIGMS = ["transformer", "diffusion", "icl", "vit",
                 "batchnorm", "gan", "resnet", "word2vec", "dropout", "bert"]
    LABELS = {"transformer": "Transformer", "diffusion": "Diffusion", "icl": "ICL",
              "vit": "ViT", "batchnorm": "BatchNorm", "gan": "GAN", "resnet": "ResNet",
              "word2vec": "Word2Vec", "dropout": "Dropout", "bert": "BERT"}
    conds = [
        ("GPT-4o w/ corpus", "experiments/canonical_evaluation.json"),
        ("Claude w/ corpus", "experiments/claude_clean_15paper/canonical_evaluation.json"),
        ("Llama-3.1 w/ corpus", "experiments/llama3_1_8b_clean/canonical_evaluation.json"),
        ("Qwen-2.5 w/ corpus", "experiments/qwen2_5_7b_clean/canonical_evaluation.json"),
        ("Mistral w/ corpus", "experiments/mistral_7b_clean/canonical_evaluation.json"),
    ]
    matrix = np.full((len(PARADIGMS), len(conds)), np.nan)
    for j, (name, path) in enumerate(conds):
        try:
            d = json.load(open(ROOT / path))
        except FileNotFoundError:
            continue
        by_p = {r["paradigm"]: r for r in d["results"]}
        for i, p in enumerate(PARADIGMS):
            r = by_p.get(p, {})
            rank = r.get("conf_rank")
            if rank is None:
                matrix[i, j] = np.nan
            else:
                # cap rank at 20 for visualization
                matrix[i, j] = min(rank, 20)

    from matplotlib.colors import BoundaryNorm, ListedColormap
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    nrows, ncols = matrix.shape
    # Discrete bins so both heatmap AND colorbar render as vector paths
    bounds = [1, 3, 6, 10, 15, 21]  # bin edges
    bin_labels = ["1–2", "3–5", "6–9", "10–14", "15+"]
    base = plt.cm.viridis_r
    discrete_colors = [base(0.10), base(0.30), base(0.50), base(0.72), base(0.92)]
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(bounds, cmap.N)
    X = np.arange(ncols + 1) - 0.5
    Y = np.arange(nrows + 1) - 0.5
    masked = np.ma.masked_invalid(matrix)
    pc = ax.pcolormesh(X, Y, masked, cmap=cmap, norm=norm,
                       edgecolors="white", linewidth=0.5, rasterized=False)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([c[0] for c in conds], rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([LABELS[p] for p in PARADIGMS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    # Overlay text
    for i in range(nrows):
        for j in range(ncols):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "—"; color = "darkred"
            else:
                txt = f"{int(v)}" + ("+" if v == 20 else "")
                color = "white" if v >= 10 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7.5)
    cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02,
                      boundaries=bounds, ticks=[(bounds[i]+bounds[i+1])/2
                                                for i in range(len(bounds)-1)])
    cb.ax.set_yticklabels(bin_labels, fontsize=7)
    cb.set_label("conf-rank (lower better; — = no match)", fontsize=7)
    ax.set_title("Per-paradigm conf-rank across 5 LLMs (15-paper corpora)", fontsize=9)
    save(fig, "fig_rank_heatmap.pdf")


def main():
    print("Generating 5 vector PDF figures into paper/figs/ ...")
    fig_mia_box()
    fig_cross_llm_bar()
    fig_multiseed()
    fig_threshold()
    fig_rank_heatmap()
    print("Done.")


if __name__ == "__main__":
    main()
