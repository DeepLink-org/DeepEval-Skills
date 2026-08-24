#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"

# Usage: run_gemm_conv.sh [gemm|conv|all] [baseline|test] [fp16|fp32|all]
OPERATOR="${1:-all}"; MODE="${2:-baseline}"; PRECISION="${3:-all}"
require_choice OPERATOR "$OPERATOR" gemm conv all
require_choice MODE "$MODE" baseline test
require_choice PRECISION "$PRECISION" fp16 fp32 all
prepare_operator_dirs
test -x "$OPERATOR_PROJECT_ROOT/cuda_ops/build/gemm"
test -x "$OPERATOR_PROJECT_ROOT/cuda_ops/build/conv"

validate=0; [[ "$MODE" == test ]] && validate=1
run_one() {
  local operator="$1" dtype="$2" suffix script csv log
  suffix="f${dtype}"; script="$OPERATOR_PROJECT_ROOT/test_${operator}.py"
  csv="$OPERATOR_PROJECT_ROOT/${operator}_${suffix}.csv"
  log="$OPERATOR_LOGS_DIR/${operator}_${suffix}_${MODE}.log"
  test -f "$script"; test -f "$csv"
  python3 "$script" "$csv" "$dtype" "$validate" 2>&1 | tee "$log"
}
for operator in gemm conv; do
  [[ "$OPERATOR" == all || "$OPERATOR" == "$operator" ]] || continue
  [[ "$PRECISION" == all || "$PRECISION" == fp16 ]] && run_one "$operator" 16
  [[ "$PRECISION" == all || "$PRECISION" == fp32 ]] && run_one "$operator" 32
done
