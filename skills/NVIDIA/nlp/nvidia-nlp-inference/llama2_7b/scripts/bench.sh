#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"
# SGLang's ``random`` workload downloads ShareGPT when its JSON input is
# absent. Resolve a mounted local JSON before starting the offline benchmark.
DATASET_PATH="$(resolve_dataset_path "${DATASET_PATH:-}")"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"

test -f "$MODEL_PATH/config.json"
mkdir -p "$LOG_ROOT"

args=(
  --model "$MODEL_PATH"
  --random-range-ratio 1
  --backend sglang
  --dataset-name random
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  --num-prompts "$NUM_PROMPTS"
  --host "$HOST"
  --port "$PORT"
  --output-file "$LOG_ROOT/bench.csv"
  --seed 42
)

case "$DATASET_PATH" in
  *.json) args+=(--dataset-path "$DATASET_PATH") ;;
  *) echo "DATASET_PATH must be a JSON file: $DATASET_PATH" >&2; exit 2 ;;
esac

set -o pipefail
python3 -u -m sglang.bench_serving "${args[@]}" 2>&1 | tee "$LOG_ROOT/bench.log"
