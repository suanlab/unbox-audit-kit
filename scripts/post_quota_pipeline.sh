#!/usr/bin/env bash
# Post-quota recovery pipeline: runs every remaining API-dependent task end-to-end.
# Triggers ONLY after OpenAI quota has been restored (top-up or fresh key).
#
# What this does (in order):
#   1. G2 wrong-corpus full run (4 pairings x 15 papers, GPT-4o)
#   2. Claude canonical evaluation (uses existing experiments/claude_clean_15paper/)
#   3. OSS LLM evaluation: Llama + Qwen extractions (if directories exist)
#   4. Refresh cross-LLM comparison markdown
#   5. Regenerate make reproduce-tables + make test
#
# Prereq:
#   - .env present with valid OPENAI_API_KEY (quota replenished)
#   - (optional) experiments/llama3_1_8b_clean/ from local vLLM run
#   - (optional) experiments/qwen2_5_7b_clean/ from local vLLM run

set -uo pipefail   # NOTE: not -e; we want to continue past per-step failures
cd "$(dirname "$0")/.."

mkdir -p logs experiments/wrong_corpus_control
PY=".venv/bin/python"
[ -x "$PY" ] || PY=python3
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/post_quota_${TS}.log"

note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# load .env first, then allow /tmp/unbox_oai_key.txt to override (latest key)
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi
if [ -f /tmp/unbox_oai_key.txt ]; then
    OPENAI_API_KEY="$(tr -d '\n\r ' < /tmp/unbox_oai_key.txt)"
    export OPENAI_API_KEY
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    note "ERROR: OPENAI_API_KEY not set; abort."
    exit 2
fi
# Quick ping before committing to the long pipeline
PING_RC=$("$PY" -c "
from openai import OpenAI
try:
    OpenAI().embeddings.create(model='text-embedding-3-small', input=['ping'])
    print(0)
except Exception as e:
    print(1)
    import sys; print(f'PING FAIL: {type(e).__name__}: {str(e)[:200]}', file=sys.stderr)
" 2>&1)
if [ "${PING_RC%%$'\n'*}" != "0" ]; then
    note "ERROR: OpenAI ping failed: ${PING_RC}"
    exit 3
fi
note "OpenAI ping OK"

note "=== Post-quota pipeline start ==="

# ---------- Step 1: G2 wrong-corpus full ----------
note "[1/5] G2 wrong-corpus full (~30 min)"
if "$PY" scripts/run_wrong_corpus_control.py >> "$LOG" 2>&1; then
    note "[1/5] G2 OK"
else
    note "[1/5] G2 FAILED (check $LOG)"
fi

# ---------- Step 2: Claude canonical eval ----------
note "[2/5] Claude canonical evaluation (4 primary)"
if "$PY" scripts/canonical_evaluator.py \
        --input-dir experiments/claude_clean_15paper \
        --output-path experiments/claude_clean_15paper/canonical_evaluation.json \
        --primary-only --label claude_sonnet_4 >> "$LOG" 2>&1; then
    note "[2/5] Claude eval OK"
else
    note "[2/5] Claude eval FAILED"
fi

# ---------- Step 3: OSS LLM eval (if extractions present) ----------
for model_dir in llama3_1_8b_clean qwen2_5_7b_clean mistral_7b_clean; do
    label="${model_dir%_clean}"
    if [ -d "experiments/${model_dir}" ] && \
       ls experiments/${model_dir}/transformer.json >/dev/null 2>&1; then
        note "[3/5] Eval ${label}"
        if "$PY" scripts/canonical_evaluator.py \
                --input-dir "experiments/${model_dir}" \
                --output-path "experiments/${model_dir}/canonical_evaluation.json" \
                --primary-only --label "${label}" >> "$LOG" 2>&1; then
            note "[3/5] ${label} eval OK"
        else
            note "[3/5] ${label} eval FAILED"
        fi
    else
        note "[3/5] SKIP ${label} (no extraction at experiments/${model_dir}/)"
    fi
done

# ---------- Step 4: Refresh cross-LLM comparison ----------
note "[4/5] Refresh cross-LLM comparison table"
"$PY" scripts/build_oss_comparison_table.py \
    --label gpt-4o     --eval experiments/canonical_evaluation.json \
    --label claude     --eval experiments/claude_clean_15paper/canonical_evaluation.json \
    --label llama-8b   --eval experiments/llama3_1_8b_clean/canonical_evaluation.json \
    --label qwen-7b    --eval experiments/qwen2_5_7b_clean/canonical_evaluation.json \
    --label mistral-7b --eval experiments/mistral_7b_clean/canonical_evaluation.json \
    --output experiments/cross_llm_comparison.md >> "$LOG" 2>&1 \
    && note "[4/5] cross_llm_comparison.md regenerated" \
    || note "[4/5] cross-LLM table generation FAILED"

# ---------- Step 5: Regression gates ----------
note "[5/5] make test"
make test >> "$LOG" 2>&1 && note "[5/5] make test OK" || note "[5/5] make test FAILED"
note "[5/5] make reproduce-tables"
make reproduce-tables >> "$LOG" 2>&1 && note "[5/5] reproduce OK" || note "[5/5] reproduce FAILED"

note "=== Done. Summary log: $LOG ==="
echo
echo "Next manual steps:"
echo "  - inspect experiments/cross_llm_comparison.md"
echo "  - if G2 succeeded, edit paper §5.3 (wrong-corpus paragraph) and §5.4 table"
echo "  - if OSS LLM evals succeeded, replace [PENDING] rows in tab:cross_llm"
echo "  - commit and recompile paper"
