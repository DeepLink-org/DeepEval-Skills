#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"; LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
TP="${TP:-1}"; SERVER_HOST="${SERVER_HOST:-0.0.0.0}"; PORT="${PORT:-30000}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"; TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"; EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
require_positive_integer TP "$TP"; require_positive_integer READY_TIMEOUT "$READY_TIMEOUT"; mkdir -p "$LOG_ROOT"; rm -f "$LOG_ROOT/serve.pid"
if [[ -n "${GPU_IDS:-}" ]]; then
  validate_gpu_ids "$GPU_IDS" "$TP"
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
args=(--model-path "$MODEL_PATH" --tp "$TP" --host "$SERVER_HOST" --port "$PORT")
[[ "$TRUST_REMOTE_CODE" == 1 ]] && args+=(--trust-remote-code)
if [[ -n "$EXTRA_SERVER_ARGS" ]]; then read -r -a extra_args <<< "$EXTRA_SERVER_ARGS"; args+=("${extra_args[@]}"); fi
nohup python3 -m sglang.launch_server "${args[@]}" >"$LOG_ROOT/serve.log" 2>&1 &
server_pid=$!; printf '%s\n' "$server_pid" >"$LOG_ROOT/serve.pid"
elapsed=0
while (( elapsed < READY_TIMEOUT )); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo "ERROR: server pid $server_pid died" >&2; tail -n 200 "$LOG_ROOT/serve.log" >&2 || true; exit 1; fi
  if curl -fs -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "server ready after ${elapsed}s (pid=$server_pid)"; exit 0; fi
  sleep 10; elapsed=$((elapsed + 10))
done
echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2; tail -n 200 "$LOG_ROOT/serve.log" >&2 || true; exit 1
