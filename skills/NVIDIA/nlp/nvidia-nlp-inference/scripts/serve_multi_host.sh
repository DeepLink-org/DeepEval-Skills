#!/usr/bin/env bash
set -euo pipefail

# Generic rank-aware SGLang launch. Network variables (NCCL_* / NVSHMEM_*)
# are inherited from the selected profile; this script does not guess hardware.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
: "${MASTER_ADDR:?Executor must set MASTER_ADDR}"; : "${MASTER_PORT:?Executor must set MASTER_PORT}"
: "${NNODES:?Executor must set NNODES}"; : "${NODE_RANK:?Executor must set NODE_RANK}"
: "${GPUS_PER_NODE:?Executor must set GPUS_PER_NODE}"
require_positive_integer NNODES "$NNODES"; require_positive_integer GPUS_PER_NODE "$GPUS_PER_NODE"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE)); MODEL_PATH="$(resolve_model_path "${MODEL_PATH:-}")"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"; PORT="${PORT:-30000}"; READY_TIMEOUT="${READY_TIMEOUT:-2400}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"; EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
require_positive_integer READY_TIMEOUT "$READY_TIMEOUT"

# A multi-host profile may reserve only part of every node. GPU_IDS must then
# name exactly the local devices and GPUS_PER_NODE must equal their count.
if [[ -n "${GPU_IDS:-}" ]]; then
  validate_gpu_ids "$GPU_IDS" "$GPUS_PER_NODE"
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
pkill -9 -f 'sglang.srt' 2>/dev/null || true
sleep 3
mkdir -p "$LOG_ROOT"; log_path="$LOG_ROOT/serve.rank${NODE_RANK}.log"; pid_path="$LOG_ROOT/serve.rank${NODE_RANK}.pid"
args=(--model-path "$MODEL_PATH" --dist-init-addr "${MASTER_ADDR}:${MASTER_PORT}" --nnodes "$NNODES" --node-rank "$NODE_RANK" --tp "$WORLD_SIZE" --host 0.0.0.0 --port "$PORT")
[[ "$TRUST_REMOTE_CODE" == 1 ]] && args+=(--trust-remote-code)
if [[ -n "$EXTRA_SERVER_ARGS" ]]; then read -r -a extra_args <<< "$EXTRA_SERVER_ARGS"; args+=("${extra_args[@]}"); fi
nohup python3 -m sglang.launch_server "${args[@]}" >"$log_path" 2>&1 &
server_pid=$!; printf '%s\n' "$server_pid" >"$pid_path"
elapsed=0
while (( elapsed < READY_TIMEOUT )); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo "ERROR: rank${NODE_RANK} server pid ${server_pid} died" >&2; tail -n 200 "$log_path" >&2 || true; exit 1; fi
  if [[ "$NODE_RANK" == 0 ]]; then
    if curl -fs -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "rank0 server ready after ${elapsed}s"; exit 0; fi
  elif grep -q 'Capture cuda graph end\|The server is fired up' "$log_path" 2>/dev/null; then
    echo "rank${NODE_RANK} worker ready after ${elapsed}s"; exit 0
  fi
  sleep 10; elapsed=$((elapsed + 10))
done
echo "ERROR: rank${NODE_RANK} not ready after ${READY_TIMEOUT}s" >&2; tail -n 200 "$log_path" >&2 || true; exit 1
