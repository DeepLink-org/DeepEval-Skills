# `deepseek_r1`：DeepSeek-R1 Profile

本 Profile 定义 DeepSeek-R1 的已验证推理基线。通用输入输出、脚本变量和 `generic` 流程见
`../model_profiles.md`；多机编排见`../multi_host.md`。

## 拓扑选择

| 任务资源 / `NNODES` | 使用流程 | 并行度 | 说明 |
|---|---|---:|---|
| 1 节点、8 张 NVIDIA GPU；`NNODES=1` | 单机 | `TP=8` | 默认 8 卡基线。 |
| 2 节点、每节点 8 张 H200；`NNODES=2` | 多机 | `WORLD_SIZE=16`、`TP=16` | 16 卡跨机 TP 基线。 |
| 其他规模 | 不直接复用基线 | — | 先验证显存、网络与 SGLang 拓扑，再新增或更新 Profile。 |

不要以模型加载失败、吞吐较低或节点空闲为理由自动切换拓扑。

## 单机 8 卡配置与流程

```bash
TP=8
GPU_IDS='0,1,2,3,4,5,6,7'
PORT=30000
READY_TIMEOUT=2400
TRUST_REMOTE_CODE=1
INPUT_LEN=2048
OUTPUT_LEN=2048
NUM_PROMPTS=1000
DATASET_PREFER='ShareGPT_V3_unfiltered_cleaned_split.json'
EXTRA_SERVER_ARGS=''

GPU_IDS="$GPU_IDS" TP="$TP" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS" bash /workspace/scripts/serve.sh
INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" \
  DATASET_PREFER="$DATASET_PREFER" HOST=127.0.0.1 PORT="$PORT" bash /workspace/scripts/bench.sh
bash /workspace/scripts/calc.sh /workspace/logs/bench.log "$TP"
```

`TRUST_REMOTE_CODE=1` 是该 Profile 的显式基线选择，不是通用默认值。仅对可信、固定版本的模型代码启用。

结果位于`/workspace/results/result.json`，日志和 CSV 位于 `/workspace/logs/`。

## 双机 16 卡配置与流程

以下网络参数仅适用于已验证的 2×8 H200 IB 集群。`MASTER_ADDR`、`MASTER_PORT`、
`NNODES`、`NODE_RANK`、`GPUS_PER_NODE` 由执行环境提供，不在 Profile 中写死。

```bash
PORT=30000
READY_TIMEOUT=2400
TRUST_REMOTE_CODE=1
INPUT_LEN=2048
OUTPUT_LEN=2048
NUM_PROMPTS=1000
DATASET_PREFER='ShareGPT_V3_unfiltered_cleaned_split.json'
EXTRA_SERVER_ARGS=''
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IBGDA_NUM_RC_PER_PE=8
export NVSHMEM_IB_TRAFFIC_CLASS=186
export NVSHMEM_DISABLE_NVLs=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA='=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7'
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=186
export NCCL_NVLS_ENABLE=0
```

按 `../multi_host.md` 的固定步骤执行：所有节点启动 `serve_multi_host.sh`；rank0 执行
`bench.sh`；最后由 rank0 执行 `calc.sh`，其 TP 使用 `NNODES * GPUS_PER_NODE`。

结果位于`/workspace/results/result.json`，日志和 CSV 位于 `/workspace/logs/`。