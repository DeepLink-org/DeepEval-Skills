---
name: nvidia-nlp-inference
description: NVIDIA GPU 上基于 sglang 的 DeepSeek 文本推理评测技能。用于指导 executor 完成容器启动、模型服务启动、压测脚本执行、推理日志采集与吞吐/延迟指标分析。
metadata:
  test_case: deepseek_r1
  multi_host_hint: references/multi_host.md
---

# nvidia-nlp-inference

本 SKILL.md 描述**单机** 8 卡推理评测流程。**多机评测**（2 节点 16 卡跨机 TP
等）请参见 `references/multi_host.md`——该文件会被 Generator 在
`nnodes > 1` 时自动拼入 LLM prompt，单机用户无需关注。

推理启动、压测和指标采集脚本分别是本 Skill 自带的
`scripts/serve.sh`、`scripts/bench.sh` 和 `scripts/calc.sh`。Executor 会将它们预置到
容器内 `/workspace/scripts/`；评测必须通过这些脚本执行，不要绕过脚本直接调用
`sglang.launch_server`、`sglang.bench_serving` 或内嵌指标采集代码。

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 DeepSeek 推理"
- "帮我测试 sglang 推理性能"
- "在 nvidia 上压测 DeepSeek-R1"
- "帮我启动 sglang 服务并跑 bench_serving"
- "采集 DeepSeek-R1 推理吞吐"

## 硬件要求

- 1 节点，8 张 NVIDIA GPU（对齐 `sglang.launch_server --tp 8`）
- 足够显存支撑 DeepSeek-R1 服务化推理与压测

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
| `MODEL_DIR` | `/data/models` | 是 | 模型权重根目录，存放 DeepSeek-R1 权重（HuggingFace Hub 缓存布局） |
| `DATASET_DIR` | `/data/datasets` | 是 | 压测数据集目录，存放 `ShareGPT_V3_unfiltered_cleaned_split.json` |
| `CODE_DIR` | `/workspace/code` | 否 | 推理相关脚本/代码目录（如有自定义脚本可挂载；默认可不挂载，直接使用容器内命令） |
| `RESULTS_DIR` | `/workspace/results` | 是 | 评测结果目录，存放 metrics 汇总文件 `result.json`（由步骤 4 的指标采集脚本生成） |
| `LOGS_DIR` | `/workspace/logs` | 是 | 日志目录，存放服务日志（`serve.log`）、压测日志（`bench.log`）与压测结果 csv（`bench.csv`） |

**说明**：
- **MODEL_DIR** 需要外部提供，挂载预训练模型权重根目录（HuggingFace 格式）
- **DATASET_DIR** 需要外部提供，挂载压测数据集目录
- **CODE_DIR** 可选，若用户有自定义 serve / bench 脚本可通过此目录挂载；本 skill 默认直接调用容器内 `python3 -m sglang.*` 命令，无需挂载代码目录
- **RESULTS_DIR** 需要外部提供，挂载评测结果目录。所有结构化产物（metrics、状态汇总）以 `result.json` 形式写入此目录
- **LOGS_DIR** 需要外部提供，挂载日志目录。`sglang.launch_server` 与 `sglang.bench_serving` 的 `stdout`/`stderr` 重定向、压测 csv、容器内异常堆栈等运行期文本均写入此目录，便于事后排查
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

**目录结构说明**：

- `$MODEL_DIR`: 模型权重目录，采用 HuggingFace Hub 缓存布局，典型结构如下：
  ```
  $MODEL_DIR/                                              # 例如 /data/models
    ├── blobs/                                           # 实际权重文件（哈希命名）
    ├── refs/                                            # 分支/标签引用
    └── snapshots/                                       # 各 commit 快照（软链至 blobs/）
        └── 4236a6af538feda4548eca9ab308586007567f52/    # 当前使用的 commit 快照
            ├── config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── model-00001-of-000xx.safetensors
            ├── ...SS
            └── model.safetensors.index.json
  ```

  **注意**：不要在生成的评测脚本中硬编码模型仓名或 snapshot hash。`serve.sh` 与 `bench.sh` 会从容器挂载的 `/data/models` 下自动定位首个 `config.json`，并将其父目录作为 `--model-path`；需要指定版本时才传入 `MODEL_PATH=/data/models/.../snapshots/<commit_hash>`。

- `$DATASET_DIR`: 数据集目录，典型结构如下：
  ```
  $DATASET_DIR/
  └── ShareGPT_V3_unfiltered_cleaned_split.json   # bench_serving 默认 sharegpt 数据集
  ```

- `$RESULTS_DIR`: 评测结果目录，典型结构如下：
  ```
  $RESULTS_DIR/
  └── result.json   # 指标采集脚本生成的结构化结果（{"status": "success", "metrics": {...}}）
  ```

  **注意**：内容由步骤 4 的指标采集脚本写入；上层 agent 会从该路径（容器内 `/workspace/results/result.json`）读取或从脚本 stdout 解析 metrics。

- `$LOGS_DIR`: 日志目录，典型结构如下：
  ```
  $LOGS_DIR/
  ├── serve.log    # sglang.launch_server 的 stdout/stderr（步骤 2 通过 tee 写入）
  ├── bench.log    # sglang.bench_serving 的 stdout/stderr（步骤 3 通过 tee 写入）
  └── bench.csv    # sglang.bench_serving 的结构化结果输出（--output-file 指定）
  ```

  **注意**：步骤 4 的指标采集脚本默认从 `/workspace/logs/bench.log` 中提取性能汇总行；若改了 `tee` 路径，需同步更新脚本中的 `log_path`。

**注意**：
- 必需的参数（`MODEL_DIR`、`DATASET_DIR`、`RESULTS_DIR`、`LOGS_DIR`）必须提供
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
  --shm-size=128g \
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
- `--shm-size=128g`：避免大吞吐推理时共享内存不足；若仍报错，可适当增大
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行后续步骤；如需后台常驻可改为 `-d` 并配合 `docker exec`
- 如有自定义脚本目录，可追加 `-v $CODE_DIR:/workspace/code:rw`

#### 容器管理命令

**进入已创建的容器**：
```bash
# 如果容器已在运行
docker exec -it sglang_inference /bin/bash

# 如果容器已停止，先启动再进入
docker start sglang_inference
docker exec -it sglang_inference /bin/bash
```

**验证容器环境**：
```bash
# 检查 GPU 设备
nvidia-smi

# 检查挂载的目录
find /data/models -name config.json -print
ls -lh /data/datasets/

# 检查 sglang 是否可用
python3 -m sglang.launch_server --help | head -5
python3 -m sglang.bench_serving --help | head -5

# 检查 Skill 预置脚本
test -f /workspace/scripts/serve.sh
test -f /workspace/scripts/bench.sh
test -f /workspace/scripts/calc.sh
```

### 步骤 2：启动模型服务

在容器内启动 `sglang.launch_server`，对 DeepSeek-R1 进行 8 卡张量并行推理服务化：

脚本会启动服务、写入 `serve.log` 和 `serve.pid`，并通过
`/v1/models` 等待服务就绪；只有服务进程存活且 HTTP 检查成功时才返回成功。

```bash
# 脚本从 /data/models 自动发现模型；仅在需要固定版本时传 MODEL_PATH。
TP=8 PORT=30000 bash /workspace/scripts/serve.sh
```

> 多机评测的环境变量（NVSHMEM / NCCL）、额外启动参数（`--dist-init-addr` /
> `--nnodes` / `--node-rank` / 跨机 `--tp`）以及 rank-aware 脚本模板，
> 统一在 `references/multi_host.md` 内描述，本文不重复。

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `serve.log` | `/workspace/logs/serve.log` | 服务启动日志，含模型加载、KV-cache 分配、监听端口等信息 |
| `serve.pid` | `/workspace/logs/serve.pid` | 服务进程 PID，便于步骤 3 完成后停止服务 |

**注意**：
- 模型版本切换时可通过 `MODEL_PATH` 指定实际的 `snapshots/<commit_hash>`；默认自动发现挂载目录中的模型
- 若 GPU 数量改变，`TP` 必须同步调整，并与步骤 4 `calc.sh` 的第二个参数保持一致
- 默认监听 `0.0.0.0:30000`，与步骤 3 `bench.sh` 的 `HOST` / `PORT` 默认值一致

### 步骤 3：执行压测

服务就绪后，使用 `sglang.bench_serving` 对其发起压测：

```bash
# 请保持默认的 INPUT_LEN、OUTPUT_LEN、NUM_PROMPTS，确保结果可与基线比较。
HOST=127.0.0.1 PORT=30000 bash /workspace/scripts/bench.sh
```

脚本优先读取 `/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`，否则自动选择挂载目录下的首个 JSON 文件，并写入 `/workspace/logs/bench.log` 和 `/workspace/logs/bench.csv`。可用 `DATASET_PATH`、`LOG_ROOT` 覆盖相应路径。

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `bench.log` | `/workspace/logs/bench.log` | 压测日志，末尾含性能汇总（吞吐 / 延迟） |
| `bench.csv` | `/workspace/logs/bench.csv` | 压测结构化结果，bench_serving 自身写入 |

**验证压测结果**：
```bash
# 查看压测日志末尾汇总
tail -50 /workspace/logs/bench.log

# 检查 csv 结果
head -2 /workspace/logs/bench.csv
```

**注意**：
- **不要修改** `INPUT_LEN`、`OUTPUT_LEN`、`NUM_PROMPTS` 默认值，否则与基线指标不可比
- 默认 `HOST=127.0.0.1`、`PORT=30000`，与步骤 2 服务监听地址保持一致；若服务运行在其他节点上，按实际 IP 调整 `HOST`
- 压测完成后建议停止服务：`kill $(cat /workspace/logs/serve.pid)`

### 步骤 4：指标采集

压测完成后，`bench.log` 末尾会出现性能汇总（由 `sglang.bench_serving` 在压测结束时打印），所有性能指标均从该段提取：

```
============ Serving Benchmark Result ============
...
Output token throughput (tok/s):         0
Total token throughput (tok/s):          0
Concurrency:                             0
Mean E2E Latency (ms):                   0
Mean TTFT (ms):                          0
Mean TPOT (ms):                          0
Mean ITL (ms):                           0
==================================================
```

#### 关键性能指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `Output token throughput (tok/s)` | 输出 token 总吞吐，核心吞吐指标（全局聚合） |
| 性能（必采） | `output_tokens_per_sec_per_gpu` | 单卡输出吞吐 = `Output token throughput / tp`（默认 8） |
| 性能（辅助） | `Total token throughput (tok/s)` | 输入 + 输出 token 总吞吐 |
| 性能（辅助） | `Mean TTFT (ms)` | 首 token 平均延迟 |
| 性能（辅助） | `Mean TPOT (ms)` | 每输出 token 平均延迟（不含首 token） |
| 性能（辅助） | `Mean ITL (ms)` | 平均 token 间延迟 |
| 性能（辅助） | `Mean E2E Latency (ms)` | 端到端平均延迟 |
| 性能（辅助） | `Concurrency` | 实际并发数 |

#### 指标采集方法

`calc.sh` 校验所有必需指标均存在且为有限数值，随后写入并回显
`/workspace/results/result.json`：

```bash
# 用实际的 TP 值替换 8；第一个参数可替换为其他 bench.log 路径。
bash /workspace/scripts/calc.sh /workspace/logs/bench.log 8
```

**结果文件**（`/workspace/results/result.json`）：
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
