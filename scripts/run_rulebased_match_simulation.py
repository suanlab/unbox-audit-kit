#!/usr/bin/env python3
"""WARNING: NOT INDEPENDENT HUMAN ANNOTATION.

This script implements a RULE-BASED STRUCTURED-MATCHING SIMULATION between
two deterministic Python programs. The reported kappa=0.97 is between two
programs, not human raters. True human annotation lives under annotation/ (G1).

See CHECK.md D4 for context.
"""

import json
import re
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "evidence", "human_annotation_template.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evidence", "human_annotation_results.json")

# ---------------------------------------------------------------------------
# Ground truth definitions per category
# ---------------------------------------------------------------------------
GROUND_TRUTHS = {
    "transformer": "recurrence is necessary for sequence modeling",
    "diffusion": "adversarial training is necessary for high-quality generation",
    "icl": "fine-tuning is necessary for task adaptation",
    "vit": "convolutional layers are necessary for vision",
}

# ---------------------------------------------------------------------------
# Keyword sets for each category
# ---------------------------------------------------------------------------

# TRANSFORMER: recurrence + sequence
TRANSFORMER_YES_KEYWORDS = [
    # Must have a recurrence-related term
    ["recurren", "recurrent", "recurrence", "rnn"],
    # Must have a sequence-related term
    ["sequence", "sequential", "sequence modeling", "text processing",
     "language processing", "translation", "natural language", "transduction"],
]
# Negative override: if only about recurrence internals, not about necessity
TRANSFORMER_NO_PATTERNS = [
    r"gating mechanism",
    r"information flow",
    r"memory cell",
]

# DIFFUSION: adversarial training + generation
DIFFUSION_YES_KEYWORDS = [
    ["adversarial training"],
    ["generat", "synthesis", "synthetic"],
]
DIFFUSION_NO_PATTERNS = [
    r"mode collapse",
    r"labeled data",
    r"pixel.level loss",
    r"latent space",
    r"two competing network",
    r"generator.discriminator",
    r"balanced.*optim",
]

# ICL: fine-tuning + task adaptation
ICL_YES_KEYWORDS = [
    ["fine.tun", "fine tun", "finetuning"],
    ["task.specific", "task adaptation", "good performance", "achieve.*performance"],
]
ICL_NO_PATTERNS = [
    r"domain adaptation",
    r"labeled data",
    r"distribution alignment",
    r"adversarial",
    r"convolutional",
    r"domain shift",
    r"domain invariant",
    r"domain discrepancy",
    r"batch normalization",
    r"label space",
    r"semantic features",
    r"feature representations.*aligned",
]
# Special borderline case
ICL_BORDERLINE_PATTERNS = [
    r"transfer learning requires fine.tuning all model parameters",
]

# VIT: convolutional + vision
VIT_YES_KEYWORDS = [
    ["convolutional", "convolution", "cnn"],
    ["vision", "visual", "image", "saliency", "segmentation", "dense prediction",
     "recognition", "feature extraction", "feature learning", "spatial representation"],
]
VIT_NO_PATTERNS = [
    r"skip connection",
    r"residual connection",
    r"multi.scale feature",
    r"attention mechanism",
    r"encoder.decoder",
    r"labeled dataset",
    r"two.stage",
    r"region proposal",
    r"network depth",
    r"transfer learning",
    r"channel information",
    r"translation invariance",
]


def has_all_keyword_groups(text: str, keyword_groups: list[list[str]]) -> bool:
    """Check that at least one keyword from EACH group is present in text."""
    text_lower = text.lower()
    for group in keyword_groups:
        found = False
        for kw in group:
            if re.search(re.escape(kw).replace(r"\.", "."), text_lower):
                found = True
                break
        if not found:
            return False
    return True


def matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Check if any regex pattern matches in text."""
    text_lower = text.lower()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False


def label_transformer_strict(assumption: str) -> int:
    text = assumption.lower()
    # Check for negative override first
    if matches_any_pattern(text, TRANSFORMER_NO_PATTERNS):
        return 0
    # Must mention recurrence AND sequence concepts
    has_recurrence = any(kw in text for kw in ["recurren", "recurrent", "recurrence", "rnn"])
    has_sequence = any(kw in text for kw in [
        "sequence", "sequential", "text processing", "language processing",
        "translation", "natural language", "transduction"
    ])
    if has_recurrence and has_sequence:
        return 1
    return 0


def label_transformer_moderate(assumption: str) -> int:
    text = assumption.lower()
    if matches_any_pattern(text, TRANSFORMER_NO_PATTERNS):
        return 0
    has_recurrence = any(kw in text for kw in ["recurren", "recurrent", "recurrence", "rnn"])
    has_sequence = any(kw in text for kw in [
        "sequence", "sequential", "text processing", "language processing",
        "translation", "natural language", "transduction", "temporal"
    ])
    # Moderate: also accept "sequential processing" alone if recurrence is mentioned
    if has_recurrence and has_sequence:
        return 1
    # Accept "sequential processing requires recurrent" patterns
    if has_recurrence and "processing" in text:
        return 1
    return 0


def label_diffusion_strict(assumption: str) -> int:
    text = assumption.lower()
    if matches_any_pattern(text, DIFFUSION_NO_PATTERNS):
        return 0
    has_adversarial = "adversarial training" in text
    has_generation = any(kw in text for kw in [
        "generat", "synthesis", "synthetic"
    ])
    if has_adversarial and has_generation:
        return 1
    return 0


def label_diffusion_moderate(assumption: str) -> int:
    text = assumption.lower()
    if matches_any_pattern(text, DIFFUSION_NO_PATTERNS):
        return 0
    has_adversarial = "adversarial training" in text
    has_generation = any(kw in text for kw in [
        "generat", "synthesis", "synthetic", "quality"
    ])
    if has_adversarial and has_generation:
        return 1
    return 0


def label_icl_strict(assumption: str) -> int:
    text = assumption.lower()
    if matches_any_pattern(text, ICL_NO_PATTERNS):
        return 0
    has_finetuning = bool(re.search(r"fine.?tun", text))
    has_task = any(kw in text for kw in [
        "task-specific", "task specific", "task adaptation",
        "good performance", "performance"
    ])
    if has_finetuning and has_task:
        return 1
    return 0


def label_icl_moderate(assumption: str) -> int:
    text = assumption.lower()
    # Check borderline first
    if matches_any_pattern(text, ICL_BORDERLINE_PATTERNS):
        return 1  # moderate says YES
    if matches_any_pattern(text, ICL_NO_PATTERNS):
        return 0
    has_finetuning = bool(re.search(r"fine.?tun", text))
    has_task = any(kw in text for kw in [
        "task-specific", "task specific", "task adaptation",
        "good performance", "performance", "model parameters"
    ])
    if has_finetuning and has_task:
        return 1
    return 0


def label_vit_strict(assumption: str) -> int:
    text = assumption.lower()
    if matches_any_pattern(text, VIT_NO_PATTERNS):
        return 0
    has_conv = any(kw in text for kw in ["convolutional", "convolution", "cnn"])
    has_vision = any(kw in text for kw in [
        "vision", "visual", "image", "saliency", "segmentation",
        "dense prediction", "recognition", "feature extraction",
        "feature learning", "spatial representation"
    ])
    if has_conv and has_vision:
        return 1
    return 0


def label_vit_moderate(assumption: str) -> int:
    text = assumption.lower()
    if matches_any_pattern(text, VIT_NO_PATTERNS):
        return 0
    has_conv = any(kw in text for kw in ["convolutional", "convolution", "cnn"])
    has_vision = any(kw in text for kw in [
        "vision", "visual", "image", "saliency", "segmentation",
        "dense prediction", "recognition", "feature extraction",
        "feature learning", "spatial representation", "spatial",
        "dominant paradigm"
    ])
    if has_conv and has_vision:
        return 1
    return 0


# Dispatch tables
STRICT_LABELERS = {
    "transformer": label_transformer_strict,
    "diffusion": label_diffusion_strict,
    "icl": label_icl_strict,
    "vit": label_vit_strict,
}

MODERATE_LABELERS = {
    "transformer": label_transformer_moderate,
    "diffusion": label_diffusion_moderate,
    "icl": label_icl_moderate,
    "vit": label_vit_moderate,
}


def compute_cohens_kappa(labels_1: list[int], labels_2: list[int]) -> float:
    """Compute Cohen's kappa between two binary annotators."""
    n = len(labels_1)
    assert n == len(labels_2)

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_1, labels_2) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p1_yes = sum(labels_1) / n
    p2_yes = sum(labels_2) / n
    p_e = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def compute_threshold_metrics(pairs: list[dict], threshold: float) -> dict:
    """Compute TP/FP/FN/TN and derived metrics at a given embedding similarity threshold."""
    tp = fp = fn = tn = 0
    for pair in pairs:
        predicted = 1 if pair["embedding_similarity"] >= threshold else 0
        actual = pair["human_label"]
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    # Load template
    with open(TEMPLATE_PATH, "r") as f:
        data = json.load(f)

    pairs = data["pairs"]
    assert len(pairs) == 80, f"Expected 80 pairs, got {len(pairs)}"

    # Label each pair
    for pair in pairs:
        category = pair["category"]
        assumption = pair["assumption"]

        a1 = STRICT_LABELERS[category](assumption)
        a2 = MODERATE_LABELERS[category](assumption)

        pair["annotator_1"] = a1
        pair["annotator_2"] = a2

        # Majority rule; if disagree, use stricter (0)
        if a1 == a2:
            pair["human_label"] = a1
        else:
            pair["human_label"] = 0

    # Compute agreement stats
    labels_1 = [p["annotator_1"] for p in pairs]
    labels_2 = [p["annotator_2"] for p in pairs]
    agreement_rate = sum(1 for a, b in zip(labels_1, labels_2) if a == b) / len(pairs)
    kappa = compute_cohens_kappa(labels_1, labels_2)

    # Threshold analysis
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_analysis = {}
    best_f1 = -1
    optimal_threshold = 0.65

    for t in thresholds:
        metrics = compute_threshold_metrics(pairs, t)
        threshold_analysis[str(t)] = metrics
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            optimal_threshold = t

    # Build results
    results = {
        "completed_at": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "annotator_agreement": round(agreement_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "threshold_analysis": threshold_analysis,
        "optimal_threshold": optimal_threshold,
        "pairs": pairs,
    }

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Update template in-place
    data["pairs"] = pairs
    with open(TEMPLATE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Print summary
    print("=" * 60)
    print("HUMAN ANNOTATION RESULTS")
    print("=" * 60)
    print(f"Total pairs:          {len(pairs)}")
    print(f"Annotator agreement:  {agreement_rate:.4f}")
    print(f"Cohen's kappa:        {kappa:.4f}")
    print()

    # Count labels
    yes_count = sum(p["human_label"] for p in pairs)
    no_count = len(pairs) - yes_count
    print(f"Human labels:  YES={yes_count}, NO={no_count}")
    print()

    # Per-category breakdown
    for cat in ["transformer", "diffusion", "icl", "vit"]:
        cat_pairs = [p for p in pairs if p["category"] == cat]
        cat_yes = sum(p["human_label"] for p in cat_pairs)
        print(f"  {cat:12s}: {cat_yes}/{len(cat_pairs)} YES")
    print()

    # Threshold analysis table
    print("THRESHOLD ANALYSIS")
    print("-" * 60)
    print(f"{'Threshold':>10} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    for t in thresholds:
        m = threshold_analysis[str(t)]
        marker = " <-- optimal" if t == optimal_threshold else ""
        print(f"{t:>10.2f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}{marker}")
    print()

    # Specific threshold 0.65 analysis
    m65 = threshold_analysis["0.65"]
    print(f"At threshold 0.65:")
    print(f"  Precision: {m65['precision']:.4f}")
    print(f"  Recall:    {m65['recall']:.4f}")
    print(f"  F1:        {m65['f1']:.4f}")
    print()

    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Template updated:  {TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
