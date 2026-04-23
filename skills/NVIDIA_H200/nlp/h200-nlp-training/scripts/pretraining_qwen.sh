#!/bin/bash

# Qwen LLM 预训练启动脚本

set -ex

# 环境变量：未设置时使用默认值，并自动导出
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Starting Qwen-3B pretraining..."
python "${SKILL_ROOT}/tools/nemotron_pretraining_qwen3_8b.py"