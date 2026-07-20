#!/usr/bin/env bash
set -euo pipefail

# Executor 将 Skill 的 scripts/ 预置到 /workspace/scripts；模型和日志目录通过卷挂载提供。
MODEL_PATH="${MODEL_PATH:-/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
TP="${TP:-8}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"

test -d "$MODEL_PATH"
case "$TP" in ''|*[!0-9]*) echo "TP must be a positive integer" >&2; exit 1 ;; esac
if [ "$TP" -lt 1 ]; then
  echo "TP must be a positive integer" >&2
  exit 1
fi
case "$READY_TIMEOUT" in ''|*[!0-9]*) echo "READY_TIMEOUT must be a positive integer" >&2; exit 1 ;; esac

mkdir -p "$LOG_ROOT"
rm -f "$LOG_ROOT/serve.pid"

nohup python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp "$TP" \
  --host "$SERVER_HOST" \
  --port "$PORT" \
  --trust-remote-code \
  > "$LOG_ROOT/serve.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$LOG_ROOT/serve.pid"

elapsed=0
while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server pid $SERVER_PID died" >&2
    tail -n 200 "$LOG_ROOT/serve.log" >&2 || true
    exit 1
  fi
  if curl -fs -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "server ready after ${elapsed}s (pid=$SERVER_PID)"
    exit 0
  fi
  sleep 10
  elapsed=$((elapsed + 10))
done

echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2
tail -n 200 "$LOG_ROOT/serve.log" >&2 || true
exit 1
