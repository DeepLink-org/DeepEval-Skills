#!/usr/bin/env bash
set -euo pipefail

# Executor 将 Skill 的 scripts/ 预置到 /workspace/scripts；训练代码仍挂载在
# /workspace/code。不要再用本脚本的 SCRIPT_DIR 推导 xtuner、配置或日志路径。
CODE_ROOT="${MIXTRAL_CODE_ROOT:-/workspace/code}"
LOG_ROOT="${MIXTRAL_LOG_ROOT:-/workspace/logs}"
CONFIG_NAME="${MIXTRAL_CONFIG:-mixtral_8x7b_instruct_full_oasst1_e3_copy2.py}"


export PYTHONPATH="$CODE_ROOT/xtuner:${PYTHONPATH:-}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_3,mlx5_4,mlx5_5,mlx5_6}"
echo "CHECK NCCL_IB_HCA=$NCCL_IB_HCA"
echo "CHECK NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST:-unset}"

REQUESTED_NNODES="${NODE_COUNT:-${NNODES:-1}}"
REQUESTED_NPROC="${PROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
case "$REQUESTED_NNODES" in
    ''|*[!0-9]*) echo "NODE_COUNT/NNODES must be a positive integer" >&2; exit 1 ;;
esac
case "$REQUESTED_NPROC" in
    ''|*[!0-9]*) echo "PROC_PER_NODE/GPUS_PER_NODE must be a positive integer" >&2; exit 1 ;;
esac
if [ "$REQUESTED_NNODES" -lt 1 ] || [ "$REQUESTED_NPROC" -lt 1 ]; then
    echo "Node and process counts must be greater than zero" >&2
    exit 1
fi

export NNODES="$REQUESTED_NNODES"
if [ "$NNODES" -gt 1 ]; then
    : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node hostname or IP for multi-node training}"
    : "${NODE_RANK:?Set NODE_RANK to the unique rank of this node (0 to NNODES-1). Rank 0 is the master node}"
fi
export NODE_RANK="${NODE_RANK:-0}"
export ADDR="${MASTER_ADDR:-127.0.0.1}"
export PORT="${MASTER_PORT:-29600}"
export NPROC_PER_NODE="$REQUESTED_NPROC"
case "$NODE_RANK" in
    ''|*[!0-9]*) echo "NODE_RANK must be a non-negative integer" >&2; exit 1 ;;
esac
if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "NODE_RANK must be in [0, NNODES)" >&2
    exit 1
fi
EXPECTED_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [ -n "${WORLD_SIZE:-}" ]; then
    case "$WORLD_SIZE" in
        *[!0-9]*) echo "WORLD_SIZE must be a positive integer" >&2; exit 1 ;;
    esac
    if [ "$WORLD_SIZE" -ne "$EXPECTED_WORLD_SIZE" ]; then
        echo "WORLD_SIZE=$WORLD_SIZE does not match NNODES*NPROC_PER_NODE=$EXPECTED_WORLD_SIZE" >&2
        exit 1
    fi
fi

test -d "$CODE_ROOT/xtuner"
test -f "$CODE_ROOT/$CONFIG_NAME"
mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/train_Full_${EXPECTED_WORLD_SIZE}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
LOG_PATH_FILE="$LOG_ROOT/train_Full_node${NODE_RANK}.path"
printf '%s\n' "$LOG_FILE" >"$LOG_PATH_FILE"

export HF_HOME="${HF_HOME:-$CODE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cd "$CODE_ROOT"

if [ "$NODE_RANK" -eq 0 ]; then
    echo "This is MASTER node (rank=0), starting first..."
else
    echo "This is WORKER node (rank=$NODE_RANK), waiting for master to be ready..."
fi

echo "Training config: $CODE_ROOT/$CONFIG_NAME"
echo "Training log: $LOG_FILE"
xtuner train "$CONFIG_NAME" --deepspeed deepspeed_zero3 >"$LOG_FILE" 2>&1
echo "Training completed; log saved to $LOG_FILE"
