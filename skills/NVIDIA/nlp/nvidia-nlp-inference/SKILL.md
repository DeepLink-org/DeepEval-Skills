---
name: nvidia-nlp-inference
description: NVIDIA GPU 上基于 SGLang 的通用文本推理评测技能。用于不同 HuggingFace 模型的容器启动、服务启动、离线压测、日志采集和吞吐/延迟指标分析；模型特有参数由 profile 指定。
---

# nvidia-nlp-inference

本技能提供统一的单机 SGLang 推理评测骨架。必须通过本技能的
`scripts/serve.sh`、`scripts/bench.sh`、`scripts/calc.sh` 执行，不要在生成的任务
脚本中重复实现 `sglang.launch_server`、`sglang.bench_serving` 或指标正则。

`scripts/` 根目录仅放跨模型通用脚本（含 `serve_multi_host.sh`）；仅模型使用的流程
放入 `scripts/<model>/`。当前 Llama-2 精度矩阵位于 `scripts/llama2/`，不能用于其他模型。

先从 `references/model_profiles.md` 选择模型 profile；其中已定义 `generic`（未列出
模型的默认入口）、`deepseek_r1`、`llama2_7b`。**profile 是模型参数、拓扑选择、执行
顺序和定制脚本的唯一来源**：未列出的模型先使用 `generic` 的完整流程，再在该文件中新增
独立 profile 后才能固化为基线。多机评测再读取 `references/multi_host.md`。模型特有流程
（例如精度矩阵）由 profile 明确启用，不应混入通用流程。

## 触发条件

- NVIDIA GPU 上用 SGLang 启动 HuggingFace 文本模型并压测
- 收集生成吞吐、TTFT、TPOT、ITL 或端到端延迟
- 为新 NLP 模型增加与现有模型一致的推理评测流程

## 统一输入、输出与容器

| 宿主环境变量 | 容器路径 | 权限 | 用途 |
|---|---|---|---|
| `MODEL_DIR` | `/data/models` | `ro` | HuggingFace 模型目录或 Hub cache |
| `DATASET_DIR` | `/data/datasets` | `ro` | 本地 ShareGPT 格式 JSON |
| `RESULTS_DIR` | `/workspace/results` | `rw` | `result.json` 及可选矩阵结果 |
| `LOGS_DIR` | `/workspace/logs` | `rw` | 服务、压测日志与 CSV |
| `CODE_DIR` | `/workspace/code` | `rw` | 可选的用户代码 |

```bash
docker run -it --name sglang_inference --gpus all --shm-size=128g \
  -v "$MODEL_DIR:/data/models:ro" \
  -v "$DATASET_DIR:/data/datasets:ro" \
  -v "$RESULTS_DIR:/workspace/results:rw" \
  -v "$LOGS_DIR:/workspace/logs:rw" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-inference:latest bash
```

`MODEL_DIR` 下必须能找到 `config.json`；可以是直接模型目录，也可以是
HuggingFace cache 的 `snapshots/<revision>`。`DATASET_DIR` 下必须有 JSON；脚本
优先选择 profile 指定的文件名，其次选择标准 ShareGPT 文件，最后选择首个 JSON。
所有默认值均可由环境变量覆盖，但变更 `TP` 时必须同步传给 `calc.sh`。

## 单机通用流程

以下示例使用 `generic` profile。执行前必须以所选 profile 的完整命令替换 `TP`、长度、启动
参数和超时；不要在本文件自行推断模型参数。Executor 将 `scripts/` 预置到容器内
`/workspace/scripts/`。

```bash
# 1. 验证输入和脚本
nvidia-smi
find /data/models -name config.json -print
find /data/datasets -name '*.json' -print
test -f /workspace/scripts/serve.sh
test -f /workspace/scripts/bench.sh
test -f /workspace/scripts/calc.sh

# 2. 启动并等待 HTTP 就绪。EXTRA_SERVER_ARGS 是以空格分隔的额外 SGLang 参数。
TP=<profile_tp> READY_TIMEOUT=1200 bash /workspace/scripts/serve.sh

# 3. 使用本地 JSON 压测；不要让 SGLang 下载数据集。
HOST=127.0.0.1 PORT=30000 INPUT_LEN=1024 OUTPUT_LEN=1024 NUM_PROMPTS=1000 \
  bash /workspace/scripts/bench.sh

# 4. 解析最后一次 benchmark 汇总，生成唯一的结构化结果。
bash /workspace/scripts/calc.sh /workspace/logs/bench.log "$TP"
```

产物固定为：`/workspace/logs/serve.log`、`serve.pid`、`bench.log`、`bench.csv` 和
`/workspace/results/result.json`。`calc.sh` 只接受完整压测日志，并校验所有指标为有限数值。
压测后可执行 `kill "$(cat /workspace/logs/serve.pid)"` 停止服务。

## 指标契约

`result.json` 的格式固定如下：

```json
{
  "status": "success",
  "metrics": {
    "output_token_throughput": 0,
    "output_tokens_per_sec_per_gpu": 0,
    "total_token_throughput": 0,
    "concurrency": 0,
    "mean_e2e_latency_ms": 0,
    "mean_ttft_ms": 0,
    "mean_tpot_ms": 0,
    "mean_itl_ms": 0
  }
}
```

`output_tokens_per_sec_per_gpu = output_token_throughput / TP`。不得用单卡日志或
其他路径替代该全局结果；精度矩阵的汇总由专用脚本产生后，再按 profile 的要求写入最终结果。

## 新模型接入规则

1. 先复用三个通用脚本，并在 `model_profiles.md` 新增 profile：模型标识、支持拓扑与选择条件、TP、请求长度、超时、数据集优先级、所有传入脚本的变量和完整执行命令。
2. 只有服务生命周期或结果契约不同（如多精度矩阵、预处理、跨机拓扑）时，才在 `scripts/<model>/` 新增专用脚本；在 profile 中说明适用条件、输入变量和调用顺序，避免改写通用脚本。
3. 不要硬编码模型仓名、snapshot hash、宿主路径、IP 或网卡名。需要固定模型时传 `MODEL_PATH`，需要新启动参数时传 `EXTRA_SERVER_ARGS`。
4. 多机时使用 `references/multi_host.md`，其中 `TP=WORLD_SIZE`，仅 rank0 跑 bench 和收集指标。
