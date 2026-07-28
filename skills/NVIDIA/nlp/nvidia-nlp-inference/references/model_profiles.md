# 模型 Profile 与执行清单

本文件是通用推理 skill 的**模型 Profile 索引和通用配置来源**。`SKILL.md` 只定义通用评测契约，
`multi_host.md` 只定义通用多机编排，`scripts/` 只执行通用或模型专用动作；三者均不应
自行猜测模型参数。模型专有 Profile 位于 `references/models/`；新增或调整评测时，先增加或修改
对应模型文档，再调用既有脚本。

## 选择规则

1. 模型名已在下方索引列出时，读取对应 Profile。
2. 未列出的 HuggingFace 文本模型使用下方 `generic` profile；这不是“无配置”，而是一套
   可执行的单机默认配置。首次成功后，应将实际模型名、显存/拓扑、参数和结果约束新增为
   独立 profile，避免以后再次猜测。
3. **单机或多机由任务分配的 `NNODES` 决定，不由脚本自动切换。** `NNODES=1` 使用
   profile 的单机流程；`NNODES>1` 仅在该 profile 标明支持多机时使用多机流程。不同拓扑
   的结果不能互相替代，也不能在失败时静默从双机降级为单机。

## 模型 Profile 索引

| 模型键 | Profile | 支持拓扑 | 说明 |
|---|---|---|---|
| `generic` | 本文件下方 | 单机 | 未列出的、SGLang 支持的普通文本模型默认流程。 |
| `deepseek_r1` | models/deepseek_r1.md | 单机 8 卡；双机 16 卡 | DeepSeek-R1 已验证基线。 |
| `llama2_7b` | models/llama2_7b.md | 单机 1 卡 | Llama-2-7B FP16/INT8 精度矩阵基线。 |
| `llama2_70b` | models/llama2_70b.md | 单机 8 卡 | Llama-2-70B FP16 SGLang 已验证基线。 |

新增模型时，在此表增加稳定模型键和链接；不要把模型专有参数重新写回本文件。

## 通用脚本变量

若`references/models/`中无特殊规定，每个 profile 必须只通过以下变量配置脚本，避免硬编码模型仓名、snapshot、IP 或宿主路径。

| 脚本 | 可由 profile 设置的变量 |
|---|---|
| `scripts/serve.sh` | `MODEL_PATH`、`GPU_IDS`、`TP`、`SERVER_HOST`、`PORT`、`READY_TIMEOUT`、`TRUST_REMOTE_CODE`、`EXTRA_SERVER_ARGS`、`LOG_ROOT` |
| `scripts/serve_multi_host.sh` | 上述服务变量（除 `TP`）以及 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK`、`GPUS_PER_NODE`、`NCCL_*`、`NVSHMEM_*` |
| `scripts/bench.sh` | `MODEL_PATH`、`DATASET_PATH`、`DATASET_PREFER`、`INPUT_LEN`、`OUTPUT_LEN`、`NUM_PROMPTS`、`MAX_CONCURRENCY`、`BENCH_TIMEOUT`、`HOST`、`PORT`、`LOG_ROOT` |
| `scripts/calc.sh` | 第一个位置参数 `LOG_PATH`、第二个位置参数 `TP`、`RESULT_ROOT`、`TASK_ID`（可选，默认 `NVIDIA_nlp_inference`）、`SCHEMA_VERSION`（可选，默认 `1.0`） |

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
