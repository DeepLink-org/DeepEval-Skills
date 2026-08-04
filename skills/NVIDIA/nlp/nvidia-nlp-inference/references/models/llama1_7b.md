# `llama1_7b`：Llama-1-7B Profile

本 Profile 定义 Llama-1-7B 的已验证精度矩阵基线。通用输入输出和脚本变量见`../model_profiles.md`。

## 拓扑与配置

当前基线为单张高显存 NVIDIA GPU、`TP=1`。该模型必须执行 FP16 和 INT8 weight-only 两个精度；
不能只跑其中一个精度，也不支持以此 Profile 进行多机基线评测。

```bash
TP=1
GPU_IDS='0'
PORT=30000
READY_TIMEOUT=1200
INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_PROMPTS=1000
MAX_CONCURRENCY=64
BENCH_TIMEOUT=1800
DATASET_PREFER='llama1_7b_sharegpt.json'
PRECISIONS='fp16 int8'
```

`MAX_CONCURRENCY=64` 限制同时在飞请求数，不改变 `NUM_PROMPTS=1000` 的样本规模；这是避免
INT8 weight-only 场景把 KV cache 填满、持续 request retraction 后导致 SGLang 调度无进展的
基线保护参数。`BENCH_TIMEOUT` 到期时必须视为该精度失败，脚本会清理服务，不能无限等待。

## 单机精度矩阵流程

`scripts/llama/run_precision_matrix.sh` 固定为 FP16 的 `--dtype float16` 和 INT8 的
`--dtype float16 --torchao-config int8wo`；脚本为每个精度隔离服务、日志和结果，最后写入
`/workspace/results/result.json`。

顶层评测脚本的前三个有效命令必须把本次 Generator 结果契约给出的动态身份字段以**字面量**导出；
不得先检查变量、不得使用默认值、`unknown`、固定 hash、变量展开或任何猜测值。这样不同任务、版本和
workload 的结果不会误复用：

```bash
export TASK_ID="<当前任务 ID>"
export WORKLOAD_FINGERPRINT="<当前结果契约给出的 workload_fingerprint>"
export SCHEMA_VERSION="1.2"
```

矩阵脚本负责生成并校验最终 `/workspace/results/result.json`，并导出 `DURATION_SECONDS`。外层评测脚本
在调用后必须直接结束：禁止再通过 Python、`python -c`、heredoc 或 shell 逻辑读取、重建、校验或覆盖
`result.json`。

```bash
GPU_IDS="$GPU_IDS" TP="$TP" PORT="$PORT" READY_TIMEOUT="$READY_TIMEOUT" INPUT_LEN="$INPUT_LEN" \
  OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  BENCH_TIMEOUT="$BENCH_TIMEOUT" DATASET_PREFER="$DATASET_PREFER" PRECISIONS="$PRECISIONS" \
  bash /workspace/scripts/llama/run_precision_matrix.sh
```

最终结果必须纳入 `/workspace/results/result.json` 的全部数值，包括
`fp16_precision_bits=16` 和 `int8_precision_bits=8`，日志和 CSV 位于 `/workspace/logs/`。
