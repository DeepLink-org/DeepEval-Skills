#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-sglang}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
MODEL_PATH="${MODEL_PATH:-/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-2048}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
SEED="${SEED:-42}"
CURRENT_DATE="$(date +%Y%m%d)"

mkdir -p "$CURRENT_DATE"

TRANSFORMERS_OFFLINE=1 \
python3 -m sglang.bench_serving \
  --model "${MODEL_PATH}" \
  --random-range-ratio 1 \
  --backend "${BACKEND}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --output-file "${CURRENT_DATE}/speed_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}_n${NUM_PROMPTS}_dsr1.csv" \
  --seed "${SEED}" 2>&1 | tee "${CURRENT_DATE}/speed_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}_n${NUM_PROMPTS}_dsr1.log"
