#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"; DATASET_PATH="$(resolve_dataset_path "${DATASET_PATH:-}")"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"; HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-1024}"; OUTPUT_LEN="${OUTPUT_LEN:-1024}"; NUM_PROMPTS="${NUM_PROMPTS:-1000}"
[[ "$DATASET_PATH" == *.json ]] || { echo "DATASET_PATH must be JSON" >&2; exit 2; }; mkdir -p "$LOG_ROOT"
python3 -u -m sglang.bench_serving --model "$MODEL_PATH" --random-range-ratio 1 --backend sglang --dataset-name random --dataset-path "$DATASET_PATH" --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" --host "$HOST" --port "$PORT" --output-file "$LOG_ROOT/bench.csv" --seed 42 2>&1 | tee "$LOG_ROOT/bench.log"
