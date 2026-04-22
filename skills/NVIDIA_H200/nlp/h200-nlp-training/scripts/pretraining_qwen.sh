#!/bin/bash

# Qwen LLM 预训练启动脚本

set -ex

# 获取基础目录，支持环境变量或默认值
BASE_DIR=${BASE_DIR:-./workspace}

# 环境变量：未设置时使用默认值，并自动导出
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

# 切换目录
cd "${BASE_DIR}"

echo "Starting Qwen-3B pretraining..."
python "${BASE_DIR}/tools/nemotron_pretraining_qwen3_8b.py"