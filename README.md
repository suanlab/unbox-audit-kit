# Unbox: An Audit Kit for LLM Retrospective Assumption Recovery

> Accepted to **AACL-IJCNLP 2026** (main conference).
> Artifacts for *Memorization or Extraction? Auditing LLM-Based Recovery of
> Scientific Assumptions* — Suan Lee and JaeSeong Kim, Semyung University.

We audit whether retrospective assumption-recovery benchmarks measure corpus-grounded extraction or pretraining memorization, and release the audit kit (five factorable controls, ~$15 of API credits) as a reusable benchmark-validity probe.

## Headline result (GPT-4o, clean prompt, 10 breakthroughs)

- **R@10 = 6/10** breakthroughs recovered within the top-10 confidence-ranked extractions (~0.19 pair-level precision under independent 10-rater study)
- **No-corpus 4/4 on primary shifts** — solvable without any paper input (memorization confound)
- **Wrong-corpus 0/4 → 0/12 across all cross-pairings** — corpus content IS causally read (Sec.~5.3)
- **5-LLM canonical sweep**: GPT-4o 4/4, Qwen 2/4, Claude 1/4, Llama-3.1-8B 1/4, Mistral 0/4 (format-compliance confound)
- **Min-K%(20) Prob cross-check**: synthetic non-members appear MORE member-like than real pre-shift papers in all 3 OSS models (App.~J) — token-level MIA is confounded by stylistic similarity; behavioral controls are the sensitive signal
- **Cross-embedding**: primary R@10 = 4/4 holds for `text-embedding-3-{small,large}` and `all-MiniLM-L6-v2` at calibrated threshold (App.~I)

## Audit-kit recommendations (Sec.~5.6, App.~H)

Five factorable controls applicable to any LLM-based scientific-text-mining benchmark:
1. No-corpus baseline (field name only)
2. Wrong-corpus pairing (target × other-field)
3. Pair-level precision alongside any-of-top-K recall
4. ≥2 cross-LLM evaluation
5. Multi-seed or sensitivity-range reporting for closed-model APIs

## Quick reproduction (no API required)

The paper's headline numbers can be re-verified offline against the frozen
artifacts (a consistency check, not recomputation — see
`experiments/SOURCE_OF_TRUTH.md`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # use the pinned lockfile
make reproduce-tables
# Expected: 91/91 passed
```

The optional open-source 5-LLM sweep (Llama/Qwen/Mistral) has its own pinned,
CUDA-only stack in `requirements-oss.txt` (not needed for the core pipeline,
`make reproduce-tables`, or `make test`).

Unit tests:
```bash
make test
```

Compile the paper:
```bash
make paper
# -> paper/unbox_arr.pdf
```

Build the anonymized supplement zip (respects `.releaseignore`):
```bash
make supplement
# -> unbox_supplement_anonymized.zip
```

## Source of truth

All headline numbers in `paper/unbox_arr.tex` come from `experiments/canonical_evaluation.json`, produced by `scripts/canonical_evaluator.py`. The full table-to-artifact-to-script manifest is in `experiments/SOURCE_OF_TRUTH.md` (and Appendix A of the paper).

Stale and failed artifacts are quarantined:
- `experiments/gpt4o_clean_prompt/summary_STALE_similarity_reranked.json` — retained for audit trail; not used for any paper claim.
- `experiments/failed_runs/` — runs that terminated due to API quota exhaustion. Notably includes a quota-failed wrong-corpus pilot; the script `scripts/run_wrong_corpus_control.py` and pairing design are released but the full run is left as future work.

## Provenance

This repository is the artifact release for the AACL-IJCNLP 2026 paper, published
as a clean snapshot. The work was first drafted for NeurIPS and then migrated to
the ACL Rolling Review template (`paper/unbox_arr.tex`, `acl.sty`); it was
reviewed through ARR and committed to AACL-IJCNLP 2026.

Reproducibility does not depend on commit history: every headline number is
checked against the frozen artifacts by `make reproduce-tables`, and the
table-to-artifact-to-script manifest is in `experiments/SOURCE_OF_TRUTH.md`.

## Full API-based pipeline (for those who want to re-extract)

Re-extraction requires OpenAI + Anthropic API keys and is subject to model drift:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# Collect corpus (≤2016 transformer, ≤2019 diffusion/ICL/ViT, strict year guards)
python3 scripts/collect_curated_corpus.py --output-dir data --target-per-category 60
python3 scripts/verify_corpus.py --data-dir data --min-count 50

# Main 10-case GPT-4o clean-prompt run (~$3 per paradigm, 3-10 min each)
.venv/bin/python scripts/run_gpt4o_clean_all10.py

# Canonical evaluator (confidence-ordered ranking)
.venv/bin/python scripts/canonical_evaluator.py

# Negative controls
.venv/bin/python scripts/run_no_corpus_control.py
.venv/bin/python scripts/run_synthetic_benchmark.py
.venv/bin/python scripts/run_niche_benchmark.py

# Diagnostic analyses (offline)
.venv/bin/python scripts/run_simple_baselines.py
.venv/bin/python scripts/run_contamination_analysis.py
.venv/bin/python scripts/run_error_analysis.py
.venv/bin/python scripts/run_calibration_analysis.py

# Verify paper numbers
make reproduce-tables
```

## Project structure

```
paper/
├── unbox_arr.tex              # ACL Rolling Review (EMNLP) submission (primary)
├── custom.bib                 # BibTeX bibliography (43 entries)
├── acl.sty / acl_natbib.bst   # Official ACL style files (vendored)
└── unbox_arr.pdf              # Compiled submission PDF

src/
├── extraction.py              # Field-wide assumption extraction (Claude / GPT-4o)
├── extraction_prompts.py      # Clean + few-shot + paper-specific prompt variants
├── semantic_match.py          # OpenAI embedding similarity (threshold 0.65)
├── metrics.py                 # Recall@K, Mean Rank, Precision@K
├── taxonomy.py                # 5 categories × 21 subcategories
├── multi_agent_evaluator.py   # Advocate/Critic/Judge debate (optional)
├── constraint_breaker.py      # TRIZ transformations (not validated; Appendix F of paper)
└── pipeline.py                # End-to-end CLI

scripts/
├── canonical_evaluator.py     # SSoT for headline numbers
├── reproduce_paper_tables.py  # 91 offline checks
├── run_gpt4o_clean_all10.py   # Main 10-case extraction
├── run_no_corpus_control.py   # Memorization control
├── run_synthetic_benchmark.py
├── run_niche_benchmark.py
├── run_wrong_corpus_control.py   # 4 wrong-corpus pairings (0/4 R@10, GPT-4o)
├── run_oss_llm_extraction.py     # vLLM extraction (Llama / Qwen / Mistral)
├── build_oss_comparison_table.py # 5-model canonical comparison
├── prelim_eval_local_embedder.py # MiniLM prelim eval (non-canonical, directional only)
├── post_quota_pipeline.sh        # End-to-end G2 + canonical-eval automation
├── run_simple_baselines.py    # TF-IDF / BM25
├── run_contamination_analysis.py
├── run_error_analysis.py
├── run_calibration_analysis.py
├── run_ablation_study.py
└── run_rulebased_match_simulation.py    # Rule-based program-vs-program SIMULATION (not human annotation; superseded by annotation/g1_results.json)

experiments/
├── canonical_evaluation.json  # Source of truth
├── SOURCE_OF_TRUTH.md         # Artifact manifest
├── ablation/                  # Prompt ablation (N=3, corpus files identical)
├── no_corpus_control/         # 4/4 primary shifts
├── synthetic_benchmark/       # N=3 fictional fields
├── niche_benchmark/           # 8 less-famous breakthroughs
├── simple_baselines/          # TF-IDF / BM25
├── contamination/             # Popularity correlations
├── error_analysis/            # 4 failure cases
├── calibration/               # Confidence gap 0.025
├── claude_clean_15paper/      # Claude cross-model comparison
├── prospective/               # LLM-judge exploratory pilot
├── wrong_corpus_control/      # 4 wrong-corpus pairings (0/4 R@10)
├── llama3_1_8b_clean/         # Llama-3.1-8B vLLM extraction (open mirror)
├── qwen2_5_7b_clean/          # Qwen-2.5-7B vLLM extraction
├── mistral_7b_clean/          # Mistral-7B vLLM extraction
├── cross_llm_comparison.md    # 5-model canonical summary table
├── oss_llm_prelim_eval.md     # MiniLM-based directional check
├── failed_runs/               # Quota-failed runs (quarantined)
└── archive/                   # Superseded intermediate runs

data/
├── {transformer,diffusion,icl,icl_curated,icl_temporal,vit}/papers.jsonl
├── paradigm_shift_mapping.json   # Author-defined targets + aliases
└── corpus_collection_report.json
```

## Citation

```bibtex
@inproceedings{lee2026unbox,
  title     = {Memorization or Extraction? Auditing {LLM}-Based Recovery of Scientific Assumptions},
  author    = {Lee, Suan and Kim, JaeSeong},
  booktitle = {Proceedings of AACL-IJCNLP 2026},
  year      = {2026}
}
```

## License

MIT for code. Corpus metadata is sourced from OpenAlex (CC0). LLM extractions are derived works; please consult OpenAI/Anthropic terms of service.

## Limitations (read the paper first)

- Author-defined targets. Each is the belief its originating paper is framed against (paper App. B), but no expert panel has validated them; the pre-registered expert study was not completed (App. C).
- N = 10 breakthroughs, 4 pre-registered (Fisher p = 0.07 on the primary comparison; underpowered)
- The matcher is validated against 10 human raters (human-vs-rule kappa = 0.70); the earlier kappa = 0.97 was between two rule-based programs, not humans
- Cross-LLM table includes only 7-8B open-source comparators; larger open models (70B+) remain future work
- Multi-seed variance covers the 4 primary shifts only (3 seeds); all other extractions are single runs
- Prospective pilot is LLM-judge only; human expert evaluation remains future work
