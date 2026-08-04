# `llama2_70b`：Llama-2-70B-Chat Profile

本 Profile 定义 Llama-2-70B-Chat 的单机 8 卡 FP16/INT8 weight-only 精度矩阵基线。通用输入
输出和脚本变量见 `../model_profiles.md`。

## 拓扑与配置

当前基线为单节点 8 张高显存 NVIDIA GPU、`TP=8`。该模型必须执行 FP16 和 INT8 weight-only
两个精度；不支持以此 Profile 进行多机基线评测。

```bash
TP=8
GPU_IDS='0,1,2,3,4,5,6,7'
PORT=30000
READY_TIMEOUT=2400
INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_PROMPTS=1000
MAX_CONCURRENCY=64
BENCH_TIMEOUT=1800
DATASET_PREFER='llama2_7b_sharegpt.json'
PRECISIONS='fp16 int8'
```

## 单机精度流程

Llama-2 使用 `scripts/llama/run_precision_matrix.sh` 统一编排服务、压测、清理和结果采集。
脚本为两个精度隔离服务、日志和结果；通用结果写入 `/workspace/results/result.json`，日志和 CSV
位于 `/workspace/logs/`。

顶层评测脚本的前三个有效命令必须从当前 Generator 结果契约把本次任务的动态字段以**字面量**导出；
不得先检查变量、不得使用默认值、`unknown`、固定 hash、变量展开或猜测值。矩阵脚本会生成并校验最终
结果、导出 `DURATION_SECONDS`；调用后顶层脚本必须直接结束，不得用 Python、`python -c`、heredoc 或
shell 逻辑重建、校验或覆盖 `result.json`：

```bash
export TASK_ID="<当前任务 ID>"
export WORKLOAD_FINGERPRINT="<当前结果契约给出的 workload_fingerprint>"
export SCHEMA_VERSION="1.2"
```

```bash
GPU_IDS="$GPU_IDS" TP="$TP" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" INPUT_LEN="$INPUT_LEN" \
  OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  BENCH_TIMEOUT="$BENCH_TIMEOUT" DATASET_PREFER="$DATASET_PREFER" PRECISIONS="$PRECISIONS" \
  bash /workspace/scripts/llama/run_precision_matrix.sh
```
