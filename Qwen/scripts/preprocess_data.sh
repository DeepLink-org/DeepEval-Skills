#!/bin/bash

# Qwen LLM 数据预处理脚本
# 该脚本用于预处理训练数据集

set -ex

# 获取基础目录，支持环境变量或默认值
BASE_DIR=${BASE_DIR:-./workspace}

# 切换到工作目录
cd "${BASE_DIR}"

DATA_PATH="${BASE_DIR}/arxiv_sample.jsonl"

# 检查数据集是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH!"
    exit 1
fi

MODEL_PATH="${BASE_DIR}/model/qwen3_8b/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Qwen3-8B model directory not found at $MODEL_PATH!"
    exit 1
fi

# 创建预处理输出目录
mkdir -p "${BASE_DIR}/datasets_processed/qwen3_8b"

# 执行数据预处理
python "${BASE_DIR}/preprocess_data_for_megatron.py" \
    --input="${DATA_PATH}" \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type="${MODEL_PATH}" \
    --output-prefix="${BASE_DIR}/datasets_processed/qwen3_8b/arxiv_sample" \
    --workers=48

echo "Data preprocessing completed!"