#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
source /opt/dtk/env.sh; source /opt/dtk/cuda/env.sh
OPERATOR="${1:-all}"; PRECISION="${2:-all}"
require_choice OPERATOR "$OPERATOR" gemm conv all
require_choice PRECISION "$PRECISION" fp16 fp32 all
prepare_operator_dirs; require_dcu
SPEED_TEST_DIR="$OPERATOR_PROJECT_ROOT/speed_test"
for operator in gemm conv; do
  [[ "$OPERATOR" == all || "$OPERATOR" == "$operator" ]] || continue
  for bits in 16 32; do
    [[ "$PRECISION" == all || "$PRECISION" == "fp$bits" ]] || continue
    input="$SPEED_TEST_DIR/${operator}_f${bits}.csv"
    output="$OPERATOR_RESULTS_DIR/${operator}_fp${bits}.csv"
    temp="/tmp/${operator}_f${bits}.csv"
    test -f "$input"; cp "$input" "$temp"
    if [[ "$operator" == gemm ]]; then
      test -x "$SPEED_TEST_DIR/cuda_ops/build/gemm"
      python3 "$SPEED_TEST_DIR/run_native_gemm.py" "$temp" "$bits" "$temp" 2>&1 | tee "$OPERATOR_LOGS_DIR/gemm_fp${bits}.log"
    else
      test -f "$SPEED_TEST_DIR/test_conv_dcu.py"
      BENCH_WARMUP="${BENCH_WARMUP:-10}" BENCH_ITERATIONS="${BENCH_ITERATIONS:-1000}" \
        python3 "$SPEED_TEST_DIR/test_conv_dcu.py" "$temp" "$bits" 2>&1 | tee "$OPERATOR_LOGS_DIR/conv_fp${bits}.log"
    fi
    cp "$temp" "$output"
  done
done