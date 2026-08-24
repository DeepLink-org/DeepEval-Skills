#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Usage: run_gemm_conv.sh [gemm|conv|all] [fp16|fp32|all]
OPERATOR="${1:-all}"
PRECISION="${2:-all}"
require_choice OPERATOR "$OPERATOR" gemm conv all
require_choice PRECISION "$PRECISION" fp16 fp32 all
prepare_operator_dirs

run_one() {
  local operator="$1" dtype="$2"
  local source_csv output_csv temp_csv script log

  source_csv="$OPERATOR_PROJECT_ROOT/${operator}_f${dtype}.csv"
  output_csv="$OPERATOR_RESULTS_DIR/${operator}_fp${dtype}.csv"
  temp_csv="/tmp/${operator}_fp${dtype}.csv"
  script="$OPERATOR_PROJECT_ROOT/test_${operator}.py"
  log="$OPERATOR_LOGS_DIR/${operator}_fp${dtype}.log"

  test -f "$script"
  test -f "$source_csv"
  if [[ "$operator" == gemm ]]; then
    test -x "$OPERATOR_PROJECT_ROOT/cuda_ops/build/gemm"
  fi

  cp "$source_csv" "$temp_csv"
  cd "$OPERATOR_PROJECT_ROOT"
  if [[ "$operator" == conv ]]; then
    python3 "$script" "$temp_csv" "$dtype" 0 torch 2>&1 | tee "$log"
  else
    python3 "$script" "$temp_csv" "$dtype" 0 2>&1 | tee "$log"
  fi
  cp "$temp_csv" "$output_csv"
}

for operator in gemm conv; do
  [[ "$OPERATOR" == all || "$OPERATOR" == "$operator" ]] || continue
  [[ "$PRECISION" == all || "$PRECISION" == fp16 ]] && run_one "$operator" 16
  [[ "$PRECISION" == all || "$PRECISION" == fp32 ]] && run_one "$operator" 32
done
