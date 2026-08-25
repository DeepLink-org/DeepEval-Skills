#!/usr/bin/env bash
# Deterministic operator-task orchestration for generated AIBenchAgent scripts.
set -euo pipefail

TARGET="${1:?usage: run_operator_task.sh <gemm|gemm-conv|longtail|all> <task-id> <workload-fingerprint>}"
TASK_ID="${2:?missing task id}"
WORKLOAD_FINGERPRINT="${3:?missing workload fingerprint}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_TIME=$(date +%s)

case "$TARGET" in
  gemm)
    "$SCRIPT_DIR/run_gemm_conv.sh" gemm all
    ;;
  gemm-conv)
    "$SCRIPT_DIR/run_gemm_conv.sh" all all
    ;;
  longtail)
    "$SCRIPT_DIR/run_longtail.sh" fp32
    "$SCRIPT_DIR/run_longtail.sh" fp16
    ;;
  all)
    "$SCRIPT_DIR/run_gemm_conv.sh" all all
    "$SCRIPT_DIR/run_longtail.sh" fp32
    "$SCRIPT_DIR/run_longtail.sh" fp16
    "$SCRIPT_DIR/run_transformer.sh"
    ;;
  *)
    echo "unsupported target: $TARGET" >&2
    exit 2
    ;;
esac

"$SCRIPT_DIR/collect_results.sh" "$TARGET"
DURATION_SECONDS=$(( $(date +%s) - START_TIME ))
(( DURATION_SECONDS > 0 )) || DURATION_SECONDS=1
"$SCRIPT_DIR/write_result_contract.py" \
  --task-id "$TASK_ID" \
  --workload-fingerprint "$WORKLOAD_FINGERPRINT" \
  --duration-seconds "$DURATION_SECONDS" \
  --target "$TARGET"