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

validate_csv() {
  local csv_path="$1" expected_rows="$2"
  python3 - "$csv_path" "$expected_rows" <<'PY'
import csv, math, sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path, newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
if len(rows) != expected:
    raise SystemExit(f"CSV row count mismatch: expected {expected}, got {len(rows)} ({path})")
for i, row in enumerate(rows):
    try:
        value = float(row['baseline'])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid baseline at row {i} ({path}): {exc}")
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"non-positive/non-finite baseline at row {i}: {value} ({path})")
PY
}

run_one() {
  local operator="$1" dtype="$2"
  local source_csv output_csv temp_csv publish_csv script log expected_rows

  source_csv="$OPERATOR_PROJECT_ROOT/${operator}_f${dtype}.csv"
  output_csv="$OPERATOR_RESULTS_DIR/${operator}_fp${dtype}.csv"
  temp_csv="/tmp/${operator}_fp${dtype}.csv"
  publish_csv="${output_csv}.tmp.$$"
  script="$OPERATOR_PROJECT_ROOT/test_${operator}.py"
  log="$OPERATOR_LOGS_DIR/${operator}_fp${dtype}.log"
  expected_rows=63
  [[ "$operator" == gemm ]] && expected_rows=224

  test -f "$script"
  test -f "$source_csv"
  if [[ "$operator" == gemm ]]; then
    test -x "$OPERATOR_PROJECT_ROOT/cuda_ops/build/gemm"
  fi

  rm -f -- "$temp_csv" "$publish_csv"
  cp -- "$source_csv" "$temp_csv"
  cd "$OPERATOR_PROJECT_ROOT"
  if [[ "$operator" == conv ]]; then
    python3 "$script" "$temp_csv" "$dtype" 0 torch 2>&1 | tee "$log"
  else
    python3 "$script" "$temp_csv" "$dtype" 0 2>&1 | tee "$log"
  fi

  validate_csv "$temp_csv" "$expected_rows"
  cp -- "$temp_csv" "$publish_csv"
  mv -f -- "$publish_csv" "$output_csv"
  echo "published $output_csv ($(wc -l < "$output_csv") lines)" | tee -a "$log"
}

for operator in gemm conv; do
  [[ "$OPERATOR" == all || "$OPERATOR" == "$operator" ]] || continue
  [[ "$PRECISION" == all || "$PRECISION" == fp16 ]] && run_one "$operator" 16
  [[ "$PRECISION" == all || "$PRECISION" == fp32 ]] && run_one "$operator" 32
done