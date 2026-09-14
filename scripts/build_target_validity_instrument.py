#!/usr/bin/env python3
"""Build the expert target-validation instrument.

Independent-expert study that validates the *targets themselves* (distinct from the
prior 10-rater study, which validated the similarity *matcher*). For each of the 10
author-defined targets we present the real assumption plus one plausible-but-wrong
decoy; experts rate accuracy (Q1) and whether the breakthrough overturned it (Q2) on
a 1-5 Likert, and may propose an alternative phrasing (Q3). Decoys give a discriminant
control; a real-vs-decoy gap with 95% bootstrap CI > 0 shows experts discriminate.

Emits (into --out-dir):
  instrument_master.csv  -- includes item type (real/decoy/attention); for analysis only
  instrument_raters.csv  -- blind, seeded-shuffled, no type column; hand to raters
  freeze.json            -- canonical payload + SHA-256 for pre-registration

Usage:
    python scripts/build_target_validity_instrument.py \
        --mapping data/paradigm_shift_mapping.json \
        --out-dir annotation/expert_targets --seed 42
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

# Human-readable field label + breakthrough display name per paradigm key.
FIELD = {
    "transformer": ("sequence modeling / NLP", "the Transformer (2017)"),
    "diffusion": ("generative image modeling", "denoising diffusion models (2020)"),
    "icl": ("task adaptation in LLMs", "in-context learning / GPT-3 (2020)"),
    "vit": ("computer vision", "the Vision Transformer (2020)"),
    "resnet": ("deep network training", "ResNet / residual learning (2015)"),
    "gan": ("generative modeling", "GANs (2014)"),
    "word2vec": ("word representation", "Word2Vec (2013)"),
    "dropout": ("neural-network regularization", "Dropout (2014)"),
    "batchnorm": ("deep-network optimization", "Batch Normalization (2015)"),
    "bert": ("language representation", "BERT (2018)"),
}

# One plausible-but-wrong decoy per paradigm. Experts should rate these LOW.
DECOY = {
    "transformer": "attention requires convolutional feature maps",
    "diffusion": "generation requires discrete latent codebooks",
    "icl": "task adaptation requires reinforcement learning",
    "vit": "vision requires hand-crafted SIFT/HOG features",
    "resnet": "network depth requires recurrent weight sharing",
    "gan": "generation requires paired supervised data",
    "word2vec": "word vectors require sentence-level labels",
    "dropout": "regularization requires ensembling separate models",
    "batchnorm": "deep training requires second-order optimizers",
    "bert": "language representations require supervised parse trees",
}

# Blind attention checks, disguised with the same field scaffolding as real items so
# raters cannot tell them apart. Each has an unambiguous Q1 answer.
ATTENTION_CHECKS = [
    ("attention-true", "computer vision", "before 2020", "the Vision Transformer (2020)",
     "Labeled image data is useful for training supervised visual recognition models",
     "expect high (>=4)"),
    ("attention-false", "sequence modeling / NLP", "before 2017", "the Transformer (2017)",
     "Sequence models were expected to be trained without using any data",
     "expect low (<=2)"),
]


def build_rows(mapping: dict, seed: int) -> list[dict]:
    rows: list[dict] = []
    for key, rec in mapping.items():
        if key not in FIELD:
            continue
        field_label, breakthrough = FIELD[key]
        year = rec["year"]
        common = {
            "paradigm": key,
            "field_label": field_label,
            "year_guard": f"before {year}",
            "breakthrough": breakthrough,
        }
        rows.append({**common, "type": "real",
                     "candidate_assumption": rec["broken_assumption"]})
        rows.append({**common, "type": "decoy",
                     "candidate_assumption": DECOY[key]})
    for ac_id, field_label, year_guard, breakthrough, text, note in ATTENTION_CHECKS:
        rows.append({"paradigm": ac_id, "field_label": field_label,
                     "year_guard": year_guard, "breakthrough": breakthrough,
                     "type": "attention", "candidate_assumption": text,
                     "expected": note})

    rng = random.Random(seed)
    rng.shuffle(rows)
    for i, row in enumerate(rows, start=1):
        row["item_id"] = f"T{i:02d}"
    return rows


def sha256_of(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", default="data/paradigm_shift_mapping.json")
    ap.add_argument("--out-dir", default="annotation/expert_targets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    rows = build_rows(mapping, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    master_cols = ["item_id", "paradigm", "type", "field_label", "year_guard",
                   "breakthrough", "candidate_assumption", "expected"]
    with (out / "instrument_master.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=master_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Blind rater view: no 'type'/'expected'; Likert + free-text columns appended.
    rater_cols = ["item_id", "field_label", "year_guard", "breakthrough",
                  "candidate_assumption",
                  "q1_accuracy_1to5", "q2_overturned_1to5",
                  "q3_alternative_phrasing", "q4_scope_broad_ok_narrow"]
    with (out / "instrument_raters.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rater_cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({**{k: "" for k in rater_cols[5:]}, **row})

    n_real = sum(r["type"] == "real" for r in rows)
    n_decoy = sum(r["type"] == "decoy" for r in rows)
    n_att = sum(r["type"] == "attention" for r in rows)
    freeze = {
        "instrument": "expert_target_validation",
        "seed": args.seed,
        "n_items": len(rows),
        "n_real": n_real, "n_decoy": n_decoy, "n_attention": n_att,
        "likert": "1-5; binary validated = q1>=4 and q2>=4 (pre-registered)",
        "primary_analysis": "per-target median q1,q2; target validated if both medians>=4",
        "reliability": "Krippendorff alpha (interval) on q1,q2; mean pairwise Cohen kappa on validated-flag",
        "discriminant": "real vs decoy q1,q2: Mann-Whitney U + 95% bootstrap gap CI (expect >0)",
        "items": [{"item_id": r["item_id"], "paradigm": r["paradigm"],
                   "type": r["type"], "candidate_assumption": r["candidate_assumption"]}
                  for r in rows],
    }
    freeze["sha256"] = sha256_of({k: v for k, v in freeze.items() if k != "sha256"})
    (out / "freeze.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote instrument to {out}/ : {len(rows)} items "
          f"({n_real} real, {n_decoy} decoy, {n_att} attention).")
    print(f"Pre-registration SHA-256: {freeze['sha256']}")
    print("Next: freeze the hash (commit), then hand instrument_raters.csv to blind experts.")


if __name__ == "__main__":
    main()
