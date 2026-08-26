#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
prepare_operator_dirs
RUNNER="$OPERATOR_PROJECT_ROOT/accuracy_test/dcu_accuracy_test.py"
OUTPUT="$OPERATOR_RESULTS_DIR/dcu_accuracy_result.json"
LOG="$OPERATOR_LOGS_DIR/accuracy.log"
test -f "$RUNNER"
{
  echo "=== Hygon DCU operator execution report ==="
  echo "started_at=$(date -Is)"
  echo "mode=dcu_execution_only_no_a100_comparison"
  echo "output=$OUTPUT"
  require_dcu
  python3 "$RUNNER" --output "$OUTPUT"
  echo "finished_at=$(date -Is)"
} 2>&1 | tee "$LOG"