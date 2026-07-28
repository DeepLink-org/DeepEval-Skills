#!/usr/bin/env bash
set -euo pipefail

# Llama-2 precision workflow. Profiles select supported values through
# PRECISIONS (for example, 7B uses fp16/int8 and 70B uses fp16/int8).
# Generic serving, benching and metric parsing remain in the parent scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRECISIONS="${PRECISIONS:-fp16 int8}"; TP="${TP:-1}"; RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"; HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-30000}"; MODEL_PATH="${MODEL_PATH:-}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"; INPUT_LEN="${INPUT_LEN:-1024}"; OUTPUT_LEN="${OUTPUT_LEN:-1024}"; NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"; BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
stop_server() { local pid_file="$1/serve.pid" pid; [[ -s "$pid_file" ]] && { pid="$(<"$pid_file")"; kill -TERM "$pid" 2>/dev/null || true; for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -KILL "$pid" 2>/dev/null || true; }; for _ in $(seq 1 60); do curl -fs -m 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 || { rm -f "$pid_file"; return 0; }; sleep 1; done; echo "ERROR: SGLang listener on ${HOST}:${PORT} did not stop" >&2; return 1; }
cleanup_log_root=''; cleanup() { [[ -n "$cleanup_log_root" ]] && stop_server "$cleanup_log_root" || true; }; trap cleanup EXIT
for precision in $PRECISIONS; do
  case "$precision" in fp16) extra='--dtype float16';; int8) extra='--dtype float16 --torchao-config int8wo';; *) echo "unsupported Llama2 precision: $precision" >&2; exit 2;; esac
  precision_log_root="$LOG_ROOT/$precision"; precision_result_root="$RESULT_ROOT/$precision"; mkdir -p "$precision_log_root" "$precision_result_root"; cleanup_log_root="$precision_log_root"
  MODEL_PATH="$MODEL_PATH" TP="$TP" GPU_IDS="${GPU_IDS:-}" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" LOG_ROOT="$precision_log_root" EXTRA_SERVER_ARGS="$extra" bash "$SCRIPT_DIR/serve.sh"
  MODEL_PATH="$MODEL_PATH" HOST="$HOST" PORT="$PORT" INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" MAX_CONCURRENCY="$MAX_CONCURRENCY" BENCH_TIMEOUT="$BENCH_TIMEOUT" LOG_ROOT="$precision_log_root" DATASET_PREFER="${DATASET_PREFER:-llama2_7b_sharegpt.json}" bash "$SCRIPT_DIR/bench.sh"
  stop_server "$precision_log_root"; cleanup_log_root=''; RESULT_ROOT="$precision_result_root" bash "$SCRIPT_DIR/calc.sh" "$precision_log_root/bench.log" "$TP"
done
# 每个精度的 calc.sh 已写出符合 Agent result contract 的 result.json。
# Agent 只回收 $RESULT_ROOT/result.json，因此以 fp16 作为该 profile 的主结果；
# int8 的完整结果仍保留在 $RESULT_ROOT/int8/result.json 供后续分析。
for precision in $PRECISIONS; do
  test -s "$RESULT_ROOT/$precision/result.json"
done
test -s "$RESULT_ROOT/fp16/result.json"
cp "$RESULT_ROOT/fp16/result.json" "$RESULT_ROOT/result.json.tmp"
mv "$RESULT_ROOT/result.json.tmp" "$RESULT_ROOT/result.json"
echo "result.json: $(cat "$RESULT_ROOT/result.json")"
