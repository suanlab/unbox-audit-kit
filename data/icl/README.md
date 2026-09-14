# data/icl — Canonical ICL Corpus

This directory contains the canonical in-context learning (ICL) corpus used for the headline **R@10 = 6/10** result reported in the paper.

**File:** `papers.jsonl` — 60 papers, year range 2017–2019.

## Alternative variants (§5.6 ablation)

- `data/icl_curated` — manually curated subset used in the curation ablation
- `data/icl_temporal` — temporally stratified variant used in the temporal-split ablation

Both alternative variants are provided for reproducibility of §5.6 ablation results only. The canonical result uses this directory (`data/icl/`).
