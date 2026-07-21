#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${QWEN3_CODE_ROOT:-${CODE_DIR:-/workspace/code/qwen_pretrain}}"
LOG_ROOT="${QWEN3_LOG_ROOT:-${LOGS_DIR:-/workspace/logs}}"
TMP_ROOT="${TMP_DIR:-/workspace/tmp}"
MODEL_ROOT="${MODEL_DIR:-/data/models/qwen3_8b}"
RESOURCE_SCRIPT="$CODE_ROOT/scripts/pretraining_qwen.sh"
PROCESSED_PREFIX="${PROCESSED_DATA_PREFIX:-$TMP_ROOT/datasets_processed/qwen3_8b/arxiv_sample_text_document}"
NODE_COUNT="${NODE_COUNT:-${NNODES:-1}}"
PROC_PER_NODE="${PROC_PER_NODE:-${GPUS_PER_NODE:-8}}"

case "$NODE_COUNT" in ''|*[!0-9]*) echo "NODE_COUNT/NNODES must be a positive integer" >&2; exit 1 ;; esac
case "$PROC_PER_NODE" in ''|*[!0-9]*) echo "PROC_PER_NODE/GPUS_PER_NODE must be a positive integer" >&2; exit 1 ;; esac
if [ "$NODE_COUNT" -ne 1 ] || [ "$PROC_PER_NODE" -ne 8 ]; then
  echo "Qwen3-8B resource currently supports exactly one node with 8 GPUs" >&2
  exit 1
fi

test -f "$RESOURCE_SCRIPT"
test -f "$CODE_ROOT/nemotron_pretraining_qwen3_8b.py"
test -f "$MODEL_ROOT/tokenizer.json"
test -s "${PROCESSED_PREFIX}.bin"
test -s "${PROCESSED_PREFIX}.idx"
mkdir -p "$LOG_ROOT" "$TMP_ROOT"

export MODEL_DIR="$MODEL_ROOT"
export TMP_DIR="$TMP_ROOT"
export PROCESSED_DATA_PREFIX="$PROCESSED_PREFIX"
export NNODES=1
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE=8

LOG_FILE="$LOG_ROOT/train_Qwen3_8B_8_node0_$(date +%Y%m%d_%H%M%S).log"
PATH_FILE="$LOG_ROOT/train_Qwen3_8B_node0.path"
printf '%s\n' "$LOG_FILE" > "$PATH_FILE"

mkdir -p "$TMP_ROOT/nemo_run"
cd "$TMP_ROOT/nemo_run"
echo "Training log: $LOG_FILE"
bash "$RESOURCE_SCRIPT" > "$LOG_FILE" 2>&1
test -s "$LOG_FILE"
echo "Training completed; log saved to $LOG_FILE"