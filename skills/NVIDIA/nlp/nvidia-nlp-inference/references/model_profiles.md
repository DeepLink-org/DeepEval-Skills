# 模型 Profile 与执行清单

本文件是通用推理 skill 的**唯一模型配置来源**。`SKILL.md` 只定义通用评测契约，
`multi_host.md` 只定义通用多机编排，`scripts/` 只执行通用或模型专用动作；三者均不应
自行猜测模型参数。新增或调整评测时，先修改本文件的 profile，再调用既有脚本。

## 选择规则

1. 模型名已在本文件列出时，使用对应 profile。
2. 未列出的 HuggingFace 文本模型使用下方 `generic` profile；这不是“无配置”，而是一套
   可执行的单机默认配置。首次成功后，应将实际模型名、显存/拓扑、参数和结果约束新增为
   独立 profile，避免以后再次猜测。
3. **单机或多机由任务分配的 `NNODES` 决定，不由脚本自动切换。** `NNODES=1` 使用
   profile 的单机流程；`NNODES>1` 仅在该 profile 标明支持多机时使用多机流程。不同拓扑
   的结果不能互相替代，也不能在失败时静默从双机降级为单机。

## 通用脚本变量

每个 profile 必须只通过以下变量配置脚本，避免硬编码模型仓名、snapshot、IP 或宿主路径。

| 脚本 | 可由 profile 设置的变量 |
|---|---|
| `scripts/serve.sh` | `MODEL_PATH`、`GPU_IDS`、`TP`、`SERVER_HOST`、`PORT`、`READY_TIMEOUT`、`TRUST_REMOTE_CODE`、`EXTRA_SERVER_ARGS`、`LOG_ROOT` |
| `scripts/serve_multi_host.sh` | 上述服务变量（除 `TP`）以及 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK`、`GPUS_PER_NODE`、`NCCL_*`、`NVSHMEM_*` |
| `scripts/bench.sh` | `MODEL_PATH`、`DATASET_PATH`、`DATASET_PREFER`、`INPUT_LEN`、`OUTPUT_LEN`、`NUM_PROMPTS`、`HOST`、`PORT`、`LOG_ROOT` |
| `scripts/calc.sh` | 第一个位置参数 `LOG_PATH`、第二个位置参数 `TP`、`RESULT_ROOT` |

所有 profile 都使用本地 JSON 数据集，`bench.sh` 不会联网下载 ShareGPT。产物固定为
`/workspace/logs/{serve,bench}.log`、`bench.csv` 和 `/workspace/results/result.json`；多机
服务日志另带 rank 后缀。

`GPU_IDS` 是单节点参与评测的逗号分隔物理 GPU 编号，例如 `GPU_IDS=0`。服务脚本会将其
导出为 `CUDA_VISIBLE_DEVICES`；因此它是**显卡选择**，`TP` 是**可见卡之间的张量并行度**，
两者必须一致。未设置 `GPU_IDS` 时，脚本保留容器原有的可见卡集合；不能用 `TP=1` 代替
GPU 隔离。

---

## `generic`：未列出的 HuggingFace 文本模型

### 适用范围与限制

- 适用于 SGLang 支持、权重目录中可找到 `config.json` 的普通文本生成模型。
- 默认只保证 **单机** 基础评测；需要 `trust_remote_code`、量化、特殊 chat template、MoE/EP
  或多机通信的模型，先新增独立 profile 后再作为正式基线运行。
- 由任务提供实际 GPU 数；`TP` 应等于该单机参与评测的 GPU 数。

### 固定配置

```bash
GPU_IDS=<allocated_gpu_ids>
TP=<number_of_allocated_gpu_ids>
PORT=30000
READY_TIMEOUT=1200
TRUST_REMOTE_CODE=0
INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_PROMPTS=1000
DATASET_PREFER=''
EXTRA_SERVER_ARGS=''
```

### 单机流程

```bash
GPU_IDS="$GPU_IDS" TP="$TP" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS" \
  bash /workspace/scripts/serve.sh
INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" \
  DATASET_PREFER="$DATASET_PREFER" HOST=127.0.0.1 PORT="$PORT" \
  bash /workspace/scripts/bench.sh
bash /workspace/scripts/calc.sh /workspace/logs/bench.log "$TP"
```

---

## `deepseek_r1`：DeepSeek-R1

### 拓扑选择

| 任务资源 / `NNODES` | 使用流程 | 并行度 | 说明 |
|---|---|---:|---|
| 1 节点、8 张 NVIDIA GPU；`NNODES=1` | 单机 | `TP=8` | 默认 8 卡基线。 |
| 2 节点、每节点 8 张 H200；`NNODES=2` | 多机 | `WORLD_SIZE=16`、`TP=16` | 16 卡跨机 TP 基线。 |
| 其他规模 | 不直接复用基线 | — | 先验证显存、网络与 SGLang 拓扑，再新增或更新 profile。 |

因此，任务明确分配一节点时跑单机 8 卡；明确分配两个节点/16 卡，且 H200 IB 拓扑符合
下列网络配置时跑双机 16 卡。不要以模型加载失败、吞吐较低或节点空闲为理由自动切换拓扑。

### 单机配置与流程

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

### 双机 16 卡配置与流程

以下网络参数仅适用于已验证的 2×8 H200 IB 集群；`MASTER_ADDR`、`MASTER_PORT`、
`NNODES`、`NODE_RANK`、`GPUS_PER_NODE` 由 Executor 注入，不能在 profile 中写死。

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

按 `multi_host.md` 的三步 CommandGroup 执行：全部节点以以上环境运行
`bash /workspace/scripts/serve_multi_host.sh`；rank0 运行
`HOST=127.0.0.1 INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" DATASET_PREFER="$DATASET_PREFER" bash /workspace/scripts/bench.sh`；
最后 rank0 运行
`bash /workspace/scripts/calc.sh /workspace/logs/bench.log "$((NNODES * GPUS_PER_NODE))"`。

---

## `llama2_7b`：Llama-2-7B-Chat

### 适用拓扑与配置

当前基线为 `NNODES=1`、单张高显存 NVIDIA GPU、`TP=1`。该模型的评测**必须**执行
FP16 和 INT8 weight-only 两个精度，不能只跑一个精度，也不支持用该 profile 做多机基线。

```bash
TP=1
GPU_IDS='0'
PORT=30000
READY_TIMEOUT=1200
INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_PROMPTS=1000
DATASET_PREFER='llama2_7b_sharegpt.json'
PRECISIONS='fp16 int8'
```

### 单机精度矩阵流程

`scripts/llama2/run_precision_matrix.sh` 固定为 FP16 的 `--dtype float16` 和 INT8 的
`--dtype float16 --torchao-config int8wo`；脚本为每个精度隔离服务、日志和结果，最后写入
`/workspace/results/precision_metrics.json`。运行：

```bash
GPU_IDS="$GPU_IDS" TP="$TP" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" INPUT_LEN="$INPUT_LEN" \
  OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" DATASET_PREFER="$DATASET_PREFER" \
  PRECISIONS="$PRECISIONS" bash /workspace/scripts/llama2/run_precision_matrix.sh
```

最终任务结果必须纳入 `precision_metrics.json` 的全部数值，包括
`fp16_precision_bits=16` 和 `int8_precision_bits=8`。
