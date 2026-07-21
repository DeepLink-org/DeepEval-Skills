#!/usr/bin/env bash
set -euo pipefail

TRAIN_LOG="${1:?Usage: calc.sh TRAIN_LOG [WORLD_SIZE]}"
WORLD_SIZE="${2:-8}"
LOG_ROOT="${QWEN3_LOG_ROOT:-${LOGS_DIR:-/workspace/logs}}"
RESULT_ROOT="${QWEN3_RESULT_ROOT:-${RESULTS_DIR:-/workspace/results}}"

test -s "$TRAIN_LOG"
case "$TRAIN_LOG" in
  "$LOG_ROOT"/train_Qwen3_8B_8_node0_*.log) ;;
  *) echo "Unexpected training log path: $TRAIN_LOG" >&2; exit 1 ;;
esac
case "$WORLD_SIZE" in ''|*[!0-9]*) echo "WORLD_SIZE must be a positive integer" >&2; exit 1 ;; esac
if [ "$WORLD_SIZE" -lt 1 ]; then
  echo "WORLD_SIZE must be a positive integer" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"

python3 - "$TRAIN_LOG" "$WORLD_SIZE" "$RESULT_ROOT/result.json" <<'PY'
import json
import math
import re
import sys

log_path, world_size_text, result_path = sys.argv[1:]
values = [float(v) for v in re.findall(
    r"tokens_per_sec_per_gpu:\s*([0-9.+\-eE]+)",
    open(log_path, encoding="utf-8", errors="replace").read(),
)]
if len(values) <= 20:
    raise SystemExit(f"insufficient tokens_per_sec_per_gpu records: {len(values)}")
if not all(math.isfinite(value) for value in values):
    raise SystemExit("tokens_per_sec_per_gpu contains a non-finite value")

trimmed = values[10:-10]
average = sum(trimmed) / len(trimmed)
world_size = int(world_size_text)
metrics = {
    "tokens_per_sec_per_gpu_avg": round(average, 2),
    "tokens_per_sec_total": round(average * world_size, 2),
    "step_count_used": len(trimmed),
}
result = {"status": "success", "metrics": metrics}
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("result.json:", json.dumps(result, ensure_ascii=False))
PY