#!/usr/bin/env bash
set -euo pipefail

# Run the fixed Llama-2-7B precision matrix. This script intentionally owns
# the server lifecycle so FP16 and INT8 cannot share a server, log, result, or
# process. It writes per-precision metric files plus a combined JSON payload.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"
TP="${TP:-1}"
PRECISIONS="${PRECISIONS:-fp16 int8}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

stop_server() {
  local pid_file="$1/serve.pid"
  if [[ -s "$pid_file" ]]; then
    local server_pid
    server_pid="$(cat "$pid_file")"
    kill -TERM "$server_pid" 2>/dev/null || true
    # launch_server creates scheduler/detokenizer children which can outlive
    # the launcher.  Leaving them alive makes the next precision's readiness
    # probe see the old server on the same port.
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  pkill -TERM -f 'sglang::scheduler|sglang::detokenizer' 2>/dev/null || true
  # Do not start the next precision until the old HTTP listener is gone.
  for _ in $(seq 1 60); do
    if ! curl -fs -m 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      rm -f "$pid_file"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: SGLang listener on ${HOST}:${PORT} did not stop" >&2
  return 1
}

for precision in $PRECISIONS; do
  case "$precision" in fp16|int8) ;; *) echo "unsupported precision: $precision" >&2; exit 2 ;; esac
  precision_log_root="$LOG_ROOT/$precision"
  precision_result_root="$RESULT_ROOT/$precision"
  mkdir -p "$precision_log_root" "$precision_result_root"

  cleanup_precision() { stop_server "$precision_log_root"; }
  trap cleanup_precision EXIT

  PRECISION="$precision" MODEL_PATH="$MODEL_PATH" TP="$TP" PORT="$PORT" \
    LOG_ROOT="$precision_log_root" bash /workspace/scripts/serve.sh
  MODEL_PATH="$MODEL_PATH" HOST="$HOST" PORT="$PORT" LOG_ROOT="$precision_log_root" \
    bash /workspace/scripts/bench.sh
  stop_server "$precision_log_root"
  trap - EXIT
  RESULT_ROOT="$precision_result_root" bash /workspace/scripts/calc.sh \
    "$precision_log_root/bench.log" "$TP"
done

python3 - "$RESULT_ROOT" $PRECISIONS <<'PY'
import json
import math
import sys

result_root, *precisions = sys.argv[1:]
metrics = {}
for precision in precisions:
    path = f"{result_root}/{precision}/result.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("status") != "success" or not isinstance(payload.get("metrics"), dict):
        raise SystemExit(f"invalid precision result: {path}")
    metrics[f"{precision}_precision_bits"] = 16 if precision == "fp16" else 8
    for name, value in payload["metrics"].items():
        value = float(value)
        if not math.isfinite(value):
            raise SystemExit(f"non-finite {precision} metric: {name}")
        metrics[f"{precision}_{name}"] = value

with open(f"{result_root}/precision_metrics.json", "w", encoding="utf-8") as f:
    # The evaluation collector consumes a flat numeric JSON mapping. Per-run
    # result.json files retain the richer status/metrics envelope.
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
PY
