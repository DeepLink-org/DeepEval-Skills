---
name: nvidia-nlp-inference
description: NVIDIA GPU 上基于 SGLang 的通用文本推理评测技能。用于不同 HuggingFace 模型的容器启动、服务启动、离线压测、日志采集和吞吐/延迟指标分析；模型特有参数由 profile 指定。
---

# nvidia-nlp-inference

本技能提供统一的单机 SGLang 推理评测骨架。必须通过本技能的`scripts/serve.sh`、`scripts/bench.sh`、`scripts/calc.sh` 执行，不要绕过脚本直接调用 `sglang.launch_server`、`sglang.bench_serving` 或内嵌指标采集代码。

`scripts/` 根目录仅放跨模型通用脚本（含 `serve_multi_host.sh`）；仅特定模型使用的流程放入 `scripts/<model>/`。当前 Llama 精度矩阵位于 `scripts/llama/`，不能用于未声明该流程的模型。

先从 `references/model_profiles.md` 的索引选择模型。Profile：`generic` 保留在索引文件中，`deepseek_r1`、`llama2_7b` 等模型专有 Profile 位于 `references/models/`。多机评测还需要读取`references/multi_host.md`。模型特有流程（例如精度矩阵）由 Profile 明确启用，不应混入通用流程。

**Profile 是模型参数、拓扑选择、执行顺序和定制脚本的唯一来源**：未列出的模型先使用 `generic` 的完整流程，再在`references/models/` 增加独立 Profile、并登记到索引后才能固化为基线。

## 触发条件

- NVIDIA GPU 上用 SGLang 启动 HuggingFace 文本模型并压测
- 收集生成吞吐、TTFT、TPOT、ITL 或端到端延迟
- 为新 NLP 模型增加与现有模型一致的推理评测流程

## 统一输入、输出与容器

| 宿主环境变量 | 容器路径 | 权限 | 用途 | 是否必需 |
|---|---|---|---|
| `MODEL_DIR` | `/data/models` | `ro` | HuggingFace 模型目录或 Hub cache | 是 |
| `DATASET_DIR` | `/data/datasets` | `ro` | 本地 ShareGPT 格式 JSON | 是 |
| `RESULTS_DIR` | `/workspace/results` | `rw` | `result.json` 及可选矩阵结果 | 是 |
| `LOGS_DIR` | `/workspace/logs` | `rw` | 服务、压测日志与 CSV | 是 |
| `CODE_DIR` | `/workspace/code` | `rw` | 可选的用户代码 | 否 |

**说明**
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径
- **MODEL_DIR** 需要外部提供，挂载预训练模型权重根目录（HuggingFace 格式）
- **DATASET_DIR** 需要外部提供，挂载压测数据集目录
- **CODE_DIR** 可选，若用户有自定义 serve / bench 脚本可通过此目录挂载；本 skill 默认直接调用容器内 `python3 -m sglang.*` 命令，无需挂载代码目录
- **RESULTS_DIR** 需要外部提供，挂载评测结果目录。所有结构化产物（metrics、状态汇总）以 `result.json` 形式写入此目录
- **LOGS_DIR** 需要外部提供，挂载日志目录。`sglang.launch_server` 与 `sglang.bench_serving` 的 `stdout`/`stderr` 重定向、压测 csv、容器内异常堆栈等运行期文本均写入此目录，便于事后排查

**Docker 镜像**
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-inference:latest
```

**目录结构说明**

- `MODEL_DIR`: 目录下必须能找到 `config.json`；可以是直接模型目录，也可以是HuggingFace cache 的 `snapshots/<revision>`。
- `DATASET_DIR` 下必须有 JSON；脚本优先选择 profile 指定的文件名，其次选择标准 ShareGPT 文件，最后选择首个 JSON。所有默认值均可由环境变量覆盖，但变更 `TP` 时必须同步传给 `calc.sh`。

## 单机通用流程

以下示例使用 `generic` profile。执行前必须以所选 profile 的完整命令替换 `TP`、长度、启动
参数和超时；不要在本文件自行推断模型参数。Executor 将 `scripts/` 预置到容器内
`/workspace/scripts/`。

### 容器启动

**完整启动命令**
```bash
docker run -it --name sglang_inference --gpus all --shm-size=128g \
  -v "$MODEL_DIR:/data/models:ro" \
  -v "$DATASET_DIR:/data/datasets:ro" \
  -v "$RESULTS_DIR:/workspace/results:rw" \
  -v "$LOGS_DIR:/workspace/logs:rw" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-inference:latest bash
```

**说明**
- 所有大文件路径通过 `MODEL_DIR`、`DATASET_DIR` 环境变量提供，避免命令中硬编码
- 若已存在同名容器，先执行 `docker rm -f sglang_inference`
- `--shm-size=128g`：避免大吞吐推理时共享内存不足；若仍报错，可适当增大
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行后续步骤；如需后台常驻可改为 `-d` 并配合 `docker exec`
- 如有自定义脚本目录，可追加 `-v $CODE_DIR:/workspace/code:rw`

#### 容器管理命令

**进入容器**
```bash
# 如果容器已在运行
docker exec -it sglang_inference /bin/bash

# 如果容器已停止，先启动再进入
docker start sglang_inference
docker exec -it sglang_inference /bin/bash
```

**验证容器环境**
```bash
# 检查 GPU 设备
nvidia-smi

# 检查挂载的目录
find /data/models -name config.json -print
find /data/datasets -name '*.json' -print
ls -lh /data/datasets/

# 检查 sglang 是否可用
python3 -m sglang.launch_server --help | head -5
python3 -m sglang.bench_serving --help | head -5

# 检查 Skill 预置脚本
test -f /workspace/scripts/serve.sh
test -f /workspace/scripts/bench.sh
test -f /workspace/scripts/calc.sh
```

#### 启动模型服务

```bash
# 1. 启动并等待 HTTP 就绪。EXTRA_SERVER_ARGS 是以空格分隔的额外 SGLang 参数。
TP=<profile_tp> READY_TIMEOUT=1200 bash /workspace/scripts/serve.sh

# 2. 使用本地 JSON 压测；不要让 SGLang 下载数据集。
HOST=127.0.0.1 PORT=30000 INPUT_LEN=1024 OUTPUT_LEN=1024 NUM_PROMPTS=1000 \
  bash /workspace/scripts/bench.sh

# 3. calc.sh 直接生成最终 v1.2 结果；不得再转换或覆盖 result.json。
bash /workspace/scripts/calc.sh /workspace/logs/bench.log "$TP" /workspace/logs/bench.csv
```

**注意**
- 若 GPU 数量改变，`TP` 必须同步调整，并与 `calc.sh` 保持一致
- 默认 `HOST=127.0.0.1`、`PORT=30000`，若服务运行在其他节点上，按实际 IP 调整 `HOST`
- **不要修改** `INPUT_LEN`、`OUTPUT_LEN`、`NUM_PROMPTS` 默认值，否则与基线指标不可比
- 顶层评测脚本必须用 `trap` 在 `EXIT` 时停止本次 `serve.pid` 记录的服务；即使 `bench.sh` 或
  `calc.sh` 失败也必须清理，避免重试因遗留服务占满 GPU 而被拒绝。

**输出产物**

产物固定为：`/workspace/logs/serve.log`、`serve.pid`、`bench.log`、`bench.csv` 和
`/workspace/results/result.json`。当前 SGLang 版本虽将结构化结果命名为 `bench.csv`，但其内容是
**单行 JSON summary**，其中 `completed` 是样本数、`duration` 是测量时长；禁止用 `tail`、`head`
或 `wc` 按 CSV 行数解析。`calc.sh` 同时读取完整 `bench.log` 与该 JSON summary，并直接生成最终结果。
流程结束后可执行 `kill "$(cat /workspace/logs/serve.pid)"` 停止服务并释放资源。

## 指标契约

`calc.sh` 输出 Agent 结果契约的 `schema_version="1.2"`。评测脚本必须从当前 Generator 结果契约
导出本次任务的 `TASK_ID`、`WORKLOAD_FINGERPRINT` 与 `SCHEMA_VERSION`；不得猜测、伪造、复用旧值或
写死其它任务的值。每次调用 `bench.sh` 会先清除该日志目录下旧的 `bench.csv`，确保汇总为单个 JSON 对象。
`result.json` 的格式固定如下：

```json
{
  "schema_version": "1.2",
  "task_id": "NVIDIA_nlp_inference",
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
  },
  "metadata": {
    "workload_fingerprint": "<Agent injected fingerprint>",
    "measurement_count": 1,
    "duration_seconds": 1.0,
    "source": "/workspace/logs/bench.log, /workspace/logs/bench.csv"
  }
}
```

`output_tokens_per_sec_per_gpu = output_token_throughput / TP`。不得用单卡日志或
其他路径替代该全局结果；禁止在 `calc.sh` 成功后重新生成、转换或覆盖 `result.json`。精度矩阵的
汇总由专用脚本产生后，再按 profile 的要求写入最终结果。

如果 profile 指定专用编排脚本已生成并校验 `result.json`，顶层任务脚本只做三件事：在开头以
当前 Generator 结果契约中的**字面量**导出 `TASK_ID`、`WORKLOAD_FINGERPRINT`、`SCHEMA_VERSION`，设置
profile 参数，然后调用专用脚本。不得在顶层任务脚本内嵌 `python -c`、heredoc、JSON 重建、JSON 校验或
结果覆盖；这些操作必须由预置的 Skill 脚本完成。不要先用 `${VAR:?}` 检查再期望外部环境提供这些值。

## 新模型接入规则

1. 先复用三个通用脚本，并在 `model_profiles.md` 新增 profile：模型标识、支持拓扑与选择条件、TP、请求长度、超时、数据集优先级、所有传入脚本的变量和完整执行命令。
2. 只有服务生命周期或结果契约不同（如多精度矩阵、预处理、跨机拓扑）时，才在 `scripts/<model>/` 新增专用脚本；在 profile 中说明适用条件、输入变量和调用顺序，避免改写通用脚本。
3. 不要硬编码模型仓名、snapshot hash、宿主路径、IP 或网卡名。需要固定模型时传 `MODEL_PATH`，需要新启动参数时传 `EXTRA_SERVER_ARGS`。
4. 多机时使用 `references/multi_host.md`，其中 `TP=WORLD_SIZE`，仅 rank0 跑 bench 和收集指标。
