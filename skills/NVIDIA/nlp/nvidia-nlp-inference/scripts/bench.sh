#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"; DATASET_PATH="$(resolve_dataset_path "${DATASET_PATH:-}")"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"; HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-30000}"
INPUT_LEN="${INPUT_LEN:-1024}"; OUTPUT_LEN="${OUTPUT_LEN:-1024}"; NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"; BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
[[ "$DATASET_PATH" == *.json ]] || { echo "DATASET_PATH must be JSON" >&2; exit 2; }; mkdir -p "$LOG_ROOT"
# SGLang may append its JSON summary when the output file already exists.  The
# same container is intentionally reused across Agent retries, so a stale file
# would become multi-line JSON and make calc.sh reject it as invalid evidence.
rm -f -- "$LOG_ROOT/bench.csv"
require_positive_integer BENCH_TIMEOUT "$BENCH_TIMEOUT"
if [[ -n "$MAX_CONCURRENCY" ]]; then require_positive_integer MAX_CONCURRENCY "$MAX_CONCURRENCY"; fi
args=(python3 -u -m sglang.bench_serving --model "$MODEL_PATH" --random-range-ratio 1 --backend sglang --dataset-name random --dataset-path "$DATASET_PATH" --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" --host "$HOST" --port "$PORT" --output-file "$LOG_ROOT/bench.csv" --seed 42)
[[ -n "$MAX_CONCURRENCY" ]] && args+=(--max-concurrency "$MAX_CONCURRENCY")
echo "benchmark max_concurrency=${MAX_CONCURRENCY:-unlimited} timeout=${BENCH_TIMEOUT}s" | tee "$LOG_ROOT/bench.log"
timeout --foreground "$BENCH_TIMEOUT" "${args[@]}" 2>&1 | tee -a "$LOG_ROOT/bench.log"
