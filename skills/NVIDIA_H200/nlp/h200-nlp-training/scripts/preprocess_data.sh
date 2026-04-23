#!/bin/bash

# Qwen LLM 数据预处理脚本
# 该脚本用于预处理训练数据集

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH="/data/datasets/arxiv_sample.jsonl"

# 检查数据集是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH!"
    exit 1
fi

MODEL_PATH="/data/models/qwen3_8b/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Qwen3-8B model directory not found at $MODEL_PATH!"
    exit 1
fi

# 创建预处理输出目录
mkdir -p "/workspace/tmp/datasets_processed/qwen3_8b"

# 执行数据预处理
python "${SKILL_ROOT}/tools/preprocess_data_for_megatron.py" \
    --input="${DATA_PATH}" \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type="${MODEL_PATH}" \
    --output-prefix="/workspace/tmp/datasets_processed/qwen3_8b/arxiv_sample" \
    --workers=48

echo "Data preprocessing completed!"