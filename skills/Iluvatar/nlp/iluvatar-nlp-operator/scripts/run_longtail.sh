#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Usage: run_longtail.sh [fp32|fp16]
PRECISION="${1:-fp32}"
require_choice PRECISION "$PRECISION" fp32 fp16
prepare_operator_dirs

if [[ "$PRECISION" == fp16 ]]; then
  BENCH_ROOT="$OPERATOR_PROJECT_ROOT/LongTail-Bench-fp16"
  CASES="$OPERATOR_PROJECT_ROOT/longtail_perf_gpu_fp16.csv"
else
  BENCH_ROOT="$OPERATOR_PROJECT_ROOT/LongTail-Bench"
  CASES="$OPERATOR_PROJECT_ROOT/longtail_perf_gpu.csv"
fi

OUTPUT="$OPERATOR_RESULTS_DIR/longtail_${PRECISION}.csv"
LOG="$OPERATOR_LOGS_DIR/longtail_${PRECISION}.log"
test -f "$BENCH_ROOT/long_tail_bench/api/api.py"
test -f "$CASES"

cd "$BENCH_ROOT"
export PYTHONPATH="$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 ./long_tail_bench/api/api.py -f "$CASES" --outcsv "$OUTPUT" \
  2>&1 | tee "$LOG"
test -s "$OUTPUT"
