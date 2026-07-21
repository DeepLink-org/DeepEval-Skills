#!/usr/bin/env bash
set -euo pipefail

# Executor 将 Skill 的 scripts/ 预置到 /workspace/scripts；实际 resource 代码由
# CODE_DIR 挂载到 /workspace/code/qwen_pretrain。
CODE_ROOT="${QWEN3_CODE_ROOT:-${CODE_DIR:-/workspace/code/qwen_pretrain}}"
LOG_ROOT="${QWEN3_LOG_ROOT:-${LOGS_DIR:-/workspace/logs}}"
TMP_ROOT="${TMP_DIR:-/workspace/tmp}"
MODEL_ROOT="${MODEL_DIR:-/data/models/qwen3_8b}"
DATA_ROOT="${DATASET_DIR:-/data/datasets}"
RESOURCE_SCRIPT="$CODE_ROOT/scripts/preprocess_data.sh"
OUTPUT_PREFIX="$TMP_ROOT/datasets_processed/qwen3_8b/arxiv_sample_text_document"

test -f "$RESOURCE_SCRIPT"
test -f "$CODE_ROOT/scripts/nlp_language_modeling/preprocess_data_for_megatron.py"
test -f "$MODEL_ROOT/tokenizer.json"
test -f "$DATA_ROOT/arxiv_sample.jsonl"
mkdir -p "$LOG_ROOT" "$TMP_ROOT"

export MODEL_DIR="$MODEL_ROOT"
export DATASET_DIR="$DATA_ROOT"
export TMP_DIR="$TMP_ROOT"

bash "$RESOURCE_SCRIPT" > "$LOG_ROOT/preprocess.log" 2>&1
test -s "${OUTPUT_PREFIX}.bin"
test -s "${OUTPUT_PREFIX}.idx"
printf '%s\n' "$OUTPUT_PREFIX" > "$LOG_ROOT/preprocess.prefix"
echo "Preprocessing completed; output prefix: $OUTPUT_PREFIX"