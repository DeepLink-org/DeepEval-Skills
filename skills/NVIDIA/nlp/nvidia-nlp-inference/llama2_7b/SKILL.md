---
name: nvidia-nlp-inference
description: NVIDIA GPU 上基于 sglang 的 Llama-2-7B-Chat 文本推理评测技能。用于指导 executor 完成容器启动、模型服务启动、压测脚本执行、推理日志采集与吞吐/延迟指标分析。
metadata:
  test_case: llama2_7b
---

# nvidia-nlp-inference

本 SKILL.md 描述**单机** Llama-2-7B-Chat 推理评测流程。Llama-2-7B-Chat 在单张高显存 NVIDIA GPU 上即可完成基线评测。该文件会被 Generator 在 `nnodes > 1` 时自动拼入LLM prompt，单机用户无需关注。

推理启动、压测和指标采集脚本分别是本 Skill 自带的`scripts/serve.sh`、`scripts/bench.sh` 和 `scripts/calc.sh`。Executor 会将它们预置到容器内 `/workspace/scripts/`；评测必须通过这些脚本执行，不要绕过脚本直接调用`sglang.launch_server`、`sglang.bench_serving` 或内嵌指标采集代码。

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Llama2 推理"
- "帮我测试 Llama-2-7B 推理性能"
- "在 nvidia 上压测 Llama2-7B-Chat"
- "帮我启动 sglang 服务并跑 bench_serving"
- "采集 Llama2 推理吞吐"

## 硬件要求

- 单机：1 节点，1 张 NVIDIA GPU（对齐 `sglang.launch_server --tp 1`）
- 足够显存支撑 Llama-2-7B-Chat 服务化推理与压测

## 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-inference:latest
```

容器内已预装 sglang 及相关依赖，可直接调用：
```bash
python3 -m sglang.launch_server
python3 -m sglang.bench_serving
```

## 环境变量

### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `MODEL_DIR` | `/data/models` | 是 | 模型权重根目录，存放 Llama-2-7B-Chat HuggingFace 格式权重 |
| `DATASET_DIR` | `/data/datasets` | 是 | 存放本地 ShareGPT 格式 JSON；默认读取 `llama2_7b_sharegpt.json`，不联网下载 |
| `CODE_DIR` | `/workspace/code` | 否 | 推理相关脚本/代码目录（如有自定义脚本可挂载；默认可不挂载，直接使用容器内命令） |
| `RESULTS_DIR` | `/workspace/results` | 是 | 评测结果目录，存放 metrics 汇总文件 `result.json`（由步骤 4 的指标采集脚本生成） |
| `LOGS_DIR` | `/workspace/logs` | 是 | 日志目录，存放服务日志（`serve.log`）、压测日志（`bench.log`）与压测结果 csv（`bench.csv`） |

## 精度矩阵（必须全部执行）

每次 Llama-2-7B 推理任务必须依次执行下列两个精度；它们使用相同的模型、请求长度、GPU数量和 benchmark 参数，但必须使用独立的服务进程、`results/<precision>`、`logs/<precision>`目录，禁止覆盖或混合结果。

| 精度 | 配置字段 | 启动方式 | 说明 |
|---|---|---|---|
| FP16 | `precision: fp16` | `PRECISION=fp16` | 以 `--dtype float16` 加载模型，是基线结果。 |
| INT8 | `precision: int8` | `PRECISION=int8` | 以 `--dtype float16 --torchao-config int8wo` 做 TorchAO weight-only INT8 服务。 |

生成的评测脚本不得自行拼接 serve / bench 命令，也不得只跑其中一种精度。必须只通过下列
固定命令执行矩阵：

```bash
TP=1 PRECISIONS="fp16 int8" bash /workspace/scripts/run_precision_matrix.sh
```

`int8wo` 是 weight-only INT8 量化：激活与 KV cache 不等同于 INT8。首次 INT8 启动会在容器内
量化权重，启动时间可能比 FP16 更长。矩阵脚本会写入
`/workspace/results/precision_metrics.json`，其中指标以 `fp16_` / `int8_` 前缀区分，并包含
`fp16_precision_bits: 16` 与 `int8_precision_bits: 8`。生成脚本必须将该文件的全部数值 metrics
原样写入最终 `/workspace/results/result.json`，补齐当前任务的 `schema_version`、`task_id` 与
`status: "success"` 后再做结果契约校验。

**说明**：
- **MODEL_DIR** 需要外部提供，挂载预训练模型权重根目录（HuggingFace 格式）
- **DATASET_DIR** 需要外部提供，挂载压测数据集目录
- **CODE_DIR** 可选，若用户有自定义 serve / bench 脚本可通过此目录挂载；本 skill 默认直接调用容器内 `python3 -m sglang.*` 命令，无需挂载代码目录
- **RESULTS_DIR** 需要外部提供，挂载评测结果目录。所有结构化产物（metrics、状态汇总）以 `result.json` 形式写入此目录
- **LOGS_DIR** 需要外部提供，挂载日志目录。`sglang.launch_server` 与 `sglang.bench_serving` 的 `stdout`/`stderr` 重定向、压测 csv、容器内异常堆栈等运行期文本均写入此目录，便于事后排查
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

**目录结构说明**：

- `$MODEL_DIR`: 模型权重根目录。不要在生成的评测脚本中硬编码模型目录名；脚本会在容器挂载的 `/data/models` 下自动定位 `config.json`，并将其父目录作为 `--model-path`。典型结构如下：
  ```
  $MODEL_DIR/
  └── llama-2-7b-chat-hf/
      ├── config.json
      ├── generation_config.json
      ├── tokenizer.json
      ├── tokenizer.model
      ├── tokenizer_config.json
      ├── model-00001-of-00002.safetensors
      ├── model-00002-of-00002.safetensors
      └── model.safetensors.index.json
  ```

  **注意**：需要指定模型版本时才传入 `MODEL_PATH=/data/models/<实际模型目录>`；默认使用自动发现的模型目录。

- `$DATASET_DIR`: 应包含 ShareGPT 格式 JSON。脚本优先选择 `llama2_7b_sharegpt.json`，其次选择标准 ShareGPT 文件，最后选择目录中的首个 JSON；压测不会联网下载 ShareGPT。

- `$RESULTS_DIR`: 评测结果目录，典型结构如下：
  ```
  $RESULTS_DIR/
  └── result.json
  ```

- `$LOGS_DIR`: 日志目录，典型结构如下：
  ```
  $LOGS_DIR/
  ├── serve.log
  ├── serve.pid
  ├── bench.log
  └── bench.csv
  ```

**注意**：
- 必需的参数为 `MODEL_DIR`、`RESULTS_DIR`、`LOGS_DIR`；`DATASET_DIR` 仅在使用 JSON 数据集时需要
- 容器内路径已通过卷挂载固定，对应 `docker run` 命令中的 `-v` 参数
- 宿主机路径建议存放在大容量磁盘上，避免占用系统盘空间

## 执行流程

### 步骤 1：容器启动

**挂载权限约定**：
- `:ro` — 只读，用于输入数据（模型权重、数据集等），防止误修改
- `:rw` — 读写，用于输出目录（日志、压测结果、metrics 汇总等）

**完整启动命令**：

```bash
docker run -it \
  --name sglang_inference \
  --gpus all \
  --shm-size=32g \
  -v $MODEL_DIR:/data/models:ro \
  -v $DATASET_DIR:/data/datasets:ro \
  -v $RESULTS_DIR:/workspace/results:rw \
  -v $LOGS_DIR:/workspace/logs:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-inference:latest \
  bash
```

**注意**：
- 所有大文件路径通过 `MODEL_DIR`、`DATASET_DIR` 环境变量提供，避免命令中硬编码
- 若已存在同名容器，先执行 `docker rm -f sglang_inference`
- `--shm-size=32g` 适用于 7B 基线；高并发或长上下文可适当增大
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行后续步骤；如需后台常驻可改为 `-d` 并配合 `docker exec`
- 如有自定义脚本目录，可追加 `-v $CODE_DIR:/workspace/code:rw`

#### 容器管理命令

**进入已创建的容器**：
```bash
docker exec -it sglang_inference /bin/bash
```

**验证容器环境**：
```bash
nvidia-smi
find /data/models -name config.json -print
ls -lh /data/datasets/
python3 -m sglang.launch_server --help | head -5
python3 -m sglang.bench_serving --help | head -5
test -f /workspace/scripts/serve.sh
test -f /workspace/scripts/bench.sh
test -f /workspace/scripts/calc.sh
```

### 步骤 2：启动模型服务

在容器内启动 `sglang.launch_server`，对 Llama-2-7B-Chat 进行单卡服务化推理：

```bash
TP=1 PORT=30000 bash /workspace/scripts/serve.sh
```

脚本会启动服务、写入 `serve.log` 和 `serve.pid`，并通过 `/v1/models` 等待服务就绪；只有服务进程存活且 HTTP 检查成功时才返回成功。

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `serve.log` | `/workspace/logs/serve.log` | 服务启动日志，含模型加载、KV-cache 分配、监听端口等信息 |
| `serve.pid` | `/workspace/logs/serve.pid` | 服务进程 PID，便于步骤 3 完成后停止服务 |

**注意**：
- 模型目录变更时可通过 `MODEL_PATH` 指定实际目录；默认自动发现挂载目录中的模型
- 若 GPU 数量改变，`TP` 必须同步调整，并与步骤 4 `calc.sh` 的第二个参数保持一致
- 默认监听 `0.0.0.0:30000`，与步骤 3 `bench.sh` 的 `HOST` / `PORT` 默认值一致

### 步骤 3：执行压测

服务就绪后，使用 `sglang.bench_serving` 对其发起压测：

```bash
HOST=127.0.0.1 PORT=30000 bash /workspace/scripts/bench.sh
```

脚本自动从 `/data/datasets` 选择本地 JSON，使用固定的 1024 输入 token、1024 输出 token 和 1000 个请求，并写入 `/workspace/logs/bench.log` 和 `/workspace/logs/bench.csv`。该本地 JSON 使 SGLang 不会访问 Hugging Face。`DATASET_PATH` 可覆盖 JSON 路径，`LOG_ROOT` 可覆盖日志目录。

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `bench.log` | `/workspace/logs/bench.log` | 压测日志，末尾含性能汇总（吞吐 / 延迟） |
| `bench.csv` | `/workspace/logs/bench.csv` | 压测结构化结果，bench_serving 自身写入 |

**验证压测结果**：
```bash
tail -50 /workspace/logs/bench.log
head -2 /workspace/logs/bench.csv
```

**注意**：
- 请保持默认的 `INPUT_LEN=1024`、`OUTPUT_LEN=1024`、`NUM_PROMPTS=1000`，确保结果可与同一基线比较
- 默认 `HOST=127.0.0.1`、`PORT=30000`，与步骤 2 服务监听地址保持一致
- 不要把 token-id 文本（如 `overall_input_token_ids.txt`）传给 `DATASET_PATH`；该参数只接受 JSON 文件
- 压测完成后建议停止服务：`kill $(cat /workspace/logs/serve.pid)`

### 步骤 4：指标采集

压测完成后，`bench.log` 末尾会出现性能汇总（由 `sglang.bench_serving` 在压测结束时打印），所有性能指标均从该段提取。

#### 关键性能指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `Output token throughput (tok/s)` | 输出 token 总吞吐，核心吞吐指标（全局聚合） |
| 性能（必采） | `output_tokens_per_sec_per_gpu` | 单卡输出吞吐 = `Output token throughput / tp`（默认 1） |
| 性能（辅助） | `Total token throughput (tok/s)` | 输入 + 输出 token 总吞吐 |
| 性能（辅助） | `Mean TTFT (ms)` | 首 token 平均延迟 |
| 性能（辅助） | `Mean TPOT (ms)` | 每输出 token 平均延迟（不含首 token） |
| 性能（辅助） | `Mean ITL (ms)` | 平均 token 间延迟 |
| 性能（辅助） | `Mean E2E Latency (ms)` | 端到端平均延迟 |
| 性能（辅助） | `Concurrency` | 实际并发数 |

#### 指标采集方法

`calc.sh` 校验所有必需指标均存在且为有限数值，随后写入并回显 `/workspace/results/result.json`：

```bash
bash /workspace/scripts/calc.sh /workspace/logs/bench.log 1
```

**结果文件**：
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

**注意**：
- 必须等待压测完成（`bench.log` 末尾出现 `Output token throughput` 汇总行）才能执行 `calc.sh`
- 切换 `TP` 后，需将 `calc.sh` 的第二个参数同步调整为实际值
- 如通过 `LOG_ROOT` 改动日志目录，需将实际 `bench.log` 路径作为 `calc.sh` 的第一个参数
