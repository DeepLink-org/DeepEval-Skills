---
name: nvidia-nlp-inference
description: NVIDIA GPU 上基于 sglang 的 DeepSeek 文本推理评测技能。用于指导 executor 完成容器启动、模型服务启动、压测脚本执行、推理日志采集与吞吐/延迟指标分析。
multi_host_hint: references/multi_host.md
---

# nvidia-nlp-inference

本 SKILL.md 描述**单机** 8 卡推理评测流程。**多机评测**（2 节点 16 卡跨机 TP
等）请参见 `references/multi_host.md`——该文件会被 Generator 在
`nnodes > 1` 时自动拼入 LLM prompt，单机用户无需关注。

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

  **注意**：`sglang.launch_server --model-path` 必须指向具体的 snapshot 子目录（即 `$MODEL_DIR/models--deepseek-ai--DeepSeek-R1-0528/snapshots/<commit_hash>`），而非 `$MODEL_DIR` 本身。`snapshots/` 下可能存在多个 commit 版本，按需选择。

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
ls -lh /data/models/snapshots/
ls -lh /data/datasets/

# 检查 sglang 是否可用
python3 -m sglang.launch_server --help | head -5
python3 -m sglang.bench_serving --help | head -5
```

### 步骤 2：启动模型服务

在容器内启动 `sglang.launch_server`，对 DeepSeek-R1 进行 8 卡张量并行推理服务化：

```bash
mkdir -p /workspace/logs

# 后台启动 sglang 服务（推荐，便于在同一 shell 内继续执行压测）
nohup python3 -m sglang.launch_server \
  --model-path /data/models/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  > /workspace/logs/serve.log 2>&1 &
SERVER_PID=$!

echo ${SERVER_PID} > /workspace/logs/serve.pid
```

> 多机评测的环境变量（NVSHMEM / NCCL）、额外启动参数（`--dist-init-addr` /
> `--nnodes` / `--node-rank` / 跨机 `--tp`）以及 rank-aware 脚本模板，
> 统一在 `references/multi_host.md` 内描述，本文不重复。

**等待服务就绪**（关键，必须等到 HTTP `/v1/models` 真就绪才能进入步骤 3）：

```bash
# ready check：进程存活 + HTTP /v1/models 双重确认。
# 端口 listen 早于模型加载完成（大模型可能要 20+ 分钟），仅检端口或仅 grep
# 日志关键字会让步骤 3 在服务尚未就绪时打过来，命中 connection refused 或
# NCCL 卡死。严禁仅靠 sleep N 或 `tail -F | grep`。
TIMEOUT=2400   # 40 分钟，足够大模型加载 + capture cuda graph
ELAPSED=0
while [ ${ELAPSED} -lt ${TIMEOUT} ]; do
  # 服务进程死了立刻失败，不要傻等
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "ERROR: server pid ${SERVER_PID} died" >&2
    tail -n 200 /workspace/logs/serve.log >&2
    exit 1
  fi
  # 必须命中真正的 model-ready 端点
  if curl -fs -m 5 http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
    echo "server ready after ${ELAPSED}s"
    break
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
done
if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
  echo "ERROR: server not ready after ${TIMEOUT}s" >&2
  tail -n 200 /workspace/logs/serve.log >&2
  exit 1
fi
```

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `serve.log` | `/workspace/logs/serve.log` | 服务启动日志，含模型加载、KV-cache 分配、监听端口等信息 |
| `serve.pid` | `/workspace/logs/serve.pid` | 服务进程 PID，便于步骤 3 完成后停止服务 |

**注意**：
- 若模型版本切换，需同步修改 `--model-path` 中的 `snapshots/<commit_hash>`
- 若 GPU 数量改变，`--tp` 必须同步调整，并与步骤 4 指标采集脚本中的 `nproc` 对齐
- 默认监听 `0.0.0.0:30000`，与步骤 3 压测脚本的 `HOST` / `PORT` 默认值一致

### 步骤 3：执行压测

服务就绪后，使用 `sglang.bench_serving` 对其发起压测：

```bash
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-2048}
NUM_PROMPTS=${NUM_PROMPTS:-1000}

python3 -m sglang.bench_serving \
  --model /data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 \
  --random-range-ratio 1 \
  --backend sglang \
  --dataset-name random \
  --dataset-path /data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --output-file /workspace/logs/bench.csv \
  --seed 42 2>&1 | tee /workspace/logs/bench.log
```

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

**Python 脚本提取**

脚本职责：
1. 从 `bench.log` 中提取性能汇总段（含 `Output token throughput`、`Mean TTFT` 等字段）
2. 计算 `output_tokens_per_sec_per_gpu`（输出吞吐 / tp）
3. 把 metrics 写入 `/workspace/results/result.json`（`{"status": "success", "metrics": {...}}` 格式）
4. 同时把 `result.json` 的内容回显到 stdout（前缀 `result.json: `），供 mcp__agent 从标准输出解析

```bash
python3 - <<'EOF'
import json
import os
import re

log_path    = '/workspace/logs/bench.log'
result_path = '/workspace/results/result.json'
tp          = 8  # 与步骤 2 中 sglang.launch_server --tp 保持一致

with open(log_path) as f:
    text = f.read()

def grab(pattern, cast=float):
    """抓取最后一次匹配（避免被中间日志干扰），返回 cast 后的值或 None。"""
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return cast(matches[-1])

metrics_raw = {
    'output_token_throughput': grab(r'Output token throughput \(tok/s\):\s+([\d.]+)'),
    'total_token_throughput':  grab(r'Total token throughput \(tok/s\):\s+([\d.]+)'),
    'concurrency':             grab(r'Concurrency:\s+([\d.]+)'),
    'mean_e2e_latency_ms':     grab(r'Mean E2E Latency \(ms\):\s+([\d.]+)'),
    'mean_ttft_ms':            grab(r'Mean TTFT \(ms\):\s+([\d.]+)'),
    'mean_tpot_ms':            grab(r'Mean TPOT \(ms\):\s+([\d.]+)'),
    'mean_itl_ms':             grab(r'Mean ITL \(ms\):\s+([\d.]+)'),
}

if metrics_raw['output_token_throughput'] is None:
    raise SystemExit("未找到压测汇总行（Output token throughput），压测可能未结束或日志被截断")

ottp = metrics_raw['output_token_throughput']

metrics = {
    'output_token_throughput':       round(ottp, 2),
    'output_tokens_per_sec_per_gpu': round(ottp / tp, 2),
    'total_token_throughput':        round(metrics_raw['total_token_throughput'], 2),
    'concurrency':                   round(metrics_raw['concurrency'], 2),
    'mean_e2e_latency_ms':           round(metrics_raw['mean_e2e_latency_ms'], 2),
    'mean_ttft_ms':                  round(metrics_raw['mean_ttft_ms'], 2),
    'mean_tpot_ms':                  round(metrics_raw['mean_tpot_ms'], 2),
    'mean_itl_ms':                   round(metrics_raw['mean_itl_ms'], 2),
}

# 1) 控制台人类可读打印
print(f"Output token throughput (tok/s)       : {metrics['output_token_throughput']:.2f}")
print(f"output_tokens_per_sec_per_gpu ({tp} GPUs) : {metrics['output_tokens_per_sec_per_gpu']:.2f}")
print(f"Total token throughput (tok/s)        : {metrics['total_token_throughput']:.2f}")
print(f"Concurrency                           : {metrics['concurrency']:.2f}")
print(f"Mean E2E Latency (ms)                 : {metrics['mean_e2e_latency_ms']:.2f}")
print(f"Mean TTFT (ms)                        : {metrics['mean_ttft_ms']:.2f}")
print(f"Mean TPOT (ms)                        : {metrics['mean_tpot_ms']:.2f}")
print(f"Mean ITL (ms)                         : {metrics['mean_itl_ms']:.2f}")

# 2) 写入 /workspace/results/result.json
os.makedirs(os.path.dirname(result_path), exist_ok=True)
result = {'status': 'success', 'metrics': metrics}
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 3) 把 result.json 内容回显到 stdout（必须与文件路径在同一行，
#    便于上层 mcp__agent 通过 "result.json" 关键字 + {...} 正则提取）
print(f"result.json: {json.dumps(result, ensure_ascii=False)}")
EOF
```

**输出示例**（基于上方样例日志）：
```
Output token throughput (tok/s)       : 0
output_tokens_per_sec_per_gpu (8 GPUs) : 0
Total token throughput (tok/s)        : 0
Concurrency                           : 0
Mean E2E Latency (ms)                 : 0
Mean TTFT (ms)                        : 0
Mean TPOT (ms)                        : 0
Mean ITL (ms)                         : 0
result.json: {"status": "success", "metrics": {"output_token_throughput": 0, "output_tokens_per_sec_per_gpu": 0, "total_token_throughput": 0, "concurrency": 0, "mean_e2e_latency_ms": 0, "mean_ttft_ms": 0, "mean_tpot_ms": 0, "mean_itl_ms": 0}}
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
- 必须等待压测完成（`bench.log` 末尾出现 `Output token throughput` 汇总行）才能采集，否则正则会无输出
- 切换 `--tp` 后，需将脚本中的 `tp` 同步调整为步骤 2 `sglang.launch_server` 的实际值
- 如改动了 `tee` 输出路径，需同步更新 `log_path`