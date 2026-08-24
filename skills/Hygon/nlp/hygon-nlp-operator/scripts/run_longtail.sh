#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
PRECISION="${1:-fp32}"; require_choice PRECISION "$PRECISION" fp32 fp16
prepare_operator_dirs; require_dcu
SPEED_DIR="$OPERATOR_PROJECT_ROOT/speed_test"
if [[ "$PRECISION" == fp16 ]]; then BENCH_ROOT="$SPEED_DIR/LongTail-Bench-fp16"; CASES="$SPEED_DIR/longtail_perf_gpu_fp16.csv"; else BENCH_ROOT="$SPEED_DIR/LongTail-Bench"; CASES="$SPEED_DIR/longtail_perf_gpu.csv"; fi
OUTPUT="$OPERATOR_RESULTS_DIR/longtail_${PRECISION}.csv"; LOG="$OPERATOR_LOGS_DIR/longtail_${PRECISION}.log"
test -f "$BENCH_ROOT/long_tail_bench/api/api.py"; test -f "$CASES"
cd "$BENCH_ROOT"; export PYTHONPATH="$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 ./long_tail_bench/api/api.py -f "$CASES" --outcsv "$OUTPUT" 2>&1 | tee "$LOG"
test -s "$OUTPUT"
