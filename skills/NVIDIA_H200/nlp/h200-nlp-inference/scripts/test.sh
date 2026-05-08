#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-deepseek-r1-0528}"
CONFIG_FILE="${SCRIPT_DIR}/configs/${MODEL_NAME}.sh"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[test.sh] Unknown MODEL_NAME=${MODEL_NAME}. Available configs:" >&2
  ls "${SCRIPT_DIR}/configs" | sed 's/\.sh$//' >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${MODEL_PATH:?MODEL_PATH not set in ${CONFIG_FILE}}"
: "${PORT:?PORT not set in ${CONFIG_FILE}}"

BACKEND="${BACKEND:-sglang}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
HOST="${HOST:-127.0.0.1}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-2048}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
SEED="${SEED:-42}"
CURRENT_DATE="$(date +%Y%m%d)"

mkdir -p "$CURRENT_DATE"

OUT_BASE="${CURRENT_DATE}/speed_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}_n${NUM_PROMPTS}_${MODEL_NAME}"

echo "[test.sh] MODEL_NAME=${MODEL_NAME}  HOST=${HOST}  PORT=${PORT}"
echo "[test.sh] output=${OUT_BASE}.csv"

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
  --output-file "${OUT_BASE}.csv" \
  --seed "${SEED}" 2>&1 | tee "${OUT_BASE}.log"
