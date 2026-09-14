#!/usr/bin/env bash
# Parallel OSS LLM extraction across A100 GPUs.
# Assumes A100 80GB on GPU 0 (Llama) and GPU 1 (Qwen). Adjust CUDA_VISIBLE_DEVICES if needed.
#
# Prereq:
#   pip install vllm
#   huggingface-cli login    # for gated meta-llama/Llama-3.1-8B-Instruct
#
# Outputs:
#   experiments/llama3_1_8b_clean/{transformer,diffusion,icl,vit}.json
#   experiments/qwen2_5_7b_clean/{transformer,diffusion,icl,vit}.json

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

LLAMA_GPU="${LLAMA_GPU:-0}"
QWEN_GPU="${QWEN_GPU:-1}"

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

echo "[parallel] Llama-3.1-8B  -> GPU $LLAMA_GPU  (log: logs/llama_extract.log)"
echo "  NOTE: meta-llama/Llama-3.1-8B-Instruct is gated. If access is not"
echo "  yet granted, this run will fail with GatedRepoError. Either request"
echo "  access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
echo "  or set USE_MISTRAL=1 to substitute mistralai/Mistral-7B-Instruct-v0.3."
SECOND_MODEL="${SECOND_MODEL:-llama}"
if [ "${USE_MISTRAL:-0}" = "1" ]; then SECOND_MODEL="mistral"; fi
CUDA_VISIBLE_DEVICES="$LLAMA_GPU" "$PY" scripts/run_oss_llm_extraction.py \
    --model "$SECOND_MODEL" \
    --output-dir "experiments/${SECOND_MODEL}_clean" \
    --seed 42 \
    > "logs/${SECOND_MODEL}_extract.log" 2>&1 &
LLAMA_PID=$!

echo "[parallel] Qwen-2.5-7B   -> GPU $QWEN_GPU  (log: logs/qwen_extract.log)"
CUDA_VISIBLE_DEVICES="$QWEN_GPU" "$PY" scripts/run_oss_llm_extraction.py \
    --model qwen \
    --output-dir experiments/qwen2_5_7b_clean \
    --seed 42 \
    > logs/qwen_extract.log 2>&1 &
QWEN_PID=$!

echo "[parallel] PIDs: llama=$LLAMA_PID qwen=$QWEN_PID -- waiting ..."
wait $LLAMA_PID; LLAMA_RC=$?
wait $QWEN_PID;  QWEN_RC=$?

echo
echo "[parallel] llama exit=$LLAMA_RC, qwen exit=$QWEN_RC"
echo "  logs/llama_extract.log"
echo "  logs/qwen_extract.log"

if [ "$LLAMA_RC" -ne 0 ] || [ "$QWEN_RC" -ne 0 ]; then
    echo "[parallel] one or more runs failed; inspect logs above" >&2
    exit 1
fi
echo "[parallel] both runs completed."
