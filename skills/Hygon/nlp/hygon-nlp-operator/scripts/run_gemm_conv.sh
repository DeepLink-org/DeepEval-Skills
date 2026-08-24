#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"

# Usage: run_gemm_conv.sh [gemm|conv|all] [fp16|fp32|all]
OPERATOR="${1:-all}"; PRECISION="${2:-all}"
require_choice OPERATOR "$OPERATOR" gemm conv all
require_choice PRECISION "$PRECISION" fp16 fp32 all
prepare_operator_dirs

SPEED_TEST_DIR="$OPERATOR_PROJECT_ROOT/speed_test"
require_dcu

run_one() {
  local operator="$1" dtype="$2" suffix source_csv output_csv temp_csv script log
  suffix="f${dtype}"
  source_csv="$SPEED_TEST_DIR/${operator}_${suffix}.csv"
  output_csv="$OPERATOR_RESULTS_DIR/${operator}_fp${dtype}.csv"
  temp_csv="/tmp/${operator}_fp${dtype}.csv"
  script="$SPEED_TEST_DIR/test_${operator}.py"
  log="$OPERATOR_LOGS_DIR/${operator}_fp${dtype}.log"

  test -f "$script"
  test -f "$source_csv"
  {
    echo "=== Hygon DCU ${operator} FP${dtype} performance ==="
    echo "started_at=$(date -Is)"
    echo "source_csv=$source_csv"
    echo "output_csv=$output_csv"
    echo "warmup=${BENCH_WARMUP:-10} iterations=${BENCH_ITERATIONS:-1000}"
    python3 -c 'import torch; print(f"device={torch.cuda.get_device_name(0)} visible_dcus={torch.cuda.device_count()}")'
    cp "$source_csv" "$temp_csv"
    cd "$SPEED_TEST_DIR"
    python3 "$script" "$temp_csv" "$dtype" 1 torch
    python3 - "$temp_csv" <<'PY'
import math, sys
import pandas as pd
frame = pd.read_csv(sys.argv[1])
values = pd.to_numeric(frame['baseline'], errors='coerce')
if values.isna().any() or not values.map(math.isfinite).all():
    raise SystemExit('invalid baseline values')
print(f'completed_cases={len(frame)} min_ms={values.min():.6f} median_ms={values.median():.6f} max_ms={values.max():.6f}')
PY
    echo "finished_at=$(date -Is)"
  } 2>&1 | tee "$log"
  cp "$temp_csv" "$output_csv"
}

for operator in gemm conv; do
  [[ "$OPERATOR" == all || "$OPERATOR" == "$operator" ]] || continue
  if [[ "$PRECISION" == all || "$PRECISION" == fp16 ]]; then
    run_one "$operator" 16
  fi
  if [[ "$PRECISION" == all || "$PRECISION" == fp32 ]]; then
    run_one "$operator" 32
  fi
done
