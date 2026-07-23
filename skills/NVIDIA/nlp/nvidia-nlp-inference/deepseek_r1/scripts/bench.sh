#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"
DATASET_PATH="$(resolve_dataset_path "${DATASET_PATH:-}")"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"

test -f "$MODEL_PATH/config.json"
test -f "$DATASET_PATH"
mkdir -p "$LOG_ROOT"

set -o pipefail
python3 -m sglang.bench_serving \
  --model "$MODEL_PATH" \
  --random-range-ratio 1 \
  --backend sglang \
  --dataset-name random \
  --dataset-path "$DATASET_PATH" \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --host "$HOST" \
  --port "$PORT" \
  --output-file "$LOG_ROOT/bench.csv" \
  --seed 42 2>&1 | tee "$LOG_ROOT/bench.log"
