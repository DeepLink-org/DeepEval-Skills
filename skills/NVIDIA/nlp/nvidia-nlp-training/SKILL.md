---
name: nvidia-nlp-training
description: NVIDIA GPU 上 Qwen3-8B 模型预训练性能评测技能（基于 NeMo）。用于指导 executor 完成容器启动、数据预处理、预训练脚本执行、训练日志采集与性能指标分析。
---

# nvidia-nlp-training

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上训练 Qwen3 8B 模型"
- "Qwen3 8B 模型预训练"
- "在 nvidia 上跑 nemo 预训练 benchmark"
- "采集 Qwen3-8B 训练 tokens_per_sec_per_gpu"

## 硬件要求

- 1 节点，8 张 NVIDIA GPU
- 至少 1TB NVMe SSD 数据盘（用于挂载模型权重、数据集、预处理产物）

## 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-training:latest
```

容器内已预装 NeMo、PyTorch、Megatron-LM 等，无需在宿主机额外安装。

## 环境变量

### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `MODEL_DIR` | `/data/models/qwen3_8b` | 是 | Qwen3-8B 权重目录（HuggingFace 标准模型仓库扁平布局，根目录直接包含 `config.json` / `tokenizer.json` / `model-*.safetensors` 等） |
| `DATASET_DIR` | `/data/datasets` | 是 | 预训练数据集目录，存放 `arxiv_sample.jsonl`（RedPajama-Data-1T-Sample arxiv 子集） |
| `CODE_DIR` | `/workspace/code/qwen_pretrain` | 是 | 预训练代码目录，包含 `scripts/` 与 `tools/` 子目录（启动脚本与训练入口 Python） |
| `RESULTS_DIR` | `/workspace/results` | 是 | 评测结果目录，存放 metrics 汇总文件 `result.json`（由步骤 4 的指标采集脚本生成） |
| `LOGS_DIR` | `/workspace/logs` | 是 | 日志目录，存放预处理日志（`preprocess.log`）、训练日志（`training.log`）以及 `stdout`/`stderr` 重定向输出 |

> **不需要外部提供的容器内路径**：`/workspace/tmp/`（临时缓存目录，存放预处理产物，例如 `datasets_processed/qwen3_8b/arxiv_sample_text_document.{bin,idx}`）。该目录仅在容器生命周期内有效，由步骤 2 的预处理脚本按需 `mkdir -p` 创建，**不需要从宿主机挂载**。如希望预处理产物跨容器复用，可自行追加一条 `-v <host_tmp>:/workspace/tmp:rw`。

**说明**：
- **MODEL_DIR** 需要外部提供，挂载预训练模型权重目录（HuggingFace 标准模型仓库扁平布局，根目录直接含 `config.json` / `tokenizer.json` / `model-*.safetensors`）
- **DATASET_DIR** 需要外部提供，挂载原始数据集目录（包含 `arxiv_sample.jsonl`）
- **CODE_DIR** 需要外部提供，挂载预训练代码目录。`scripts/preprocess_data.sh` 与 `scripts/pretraining_qwen.sh` 通过相对路径解析 `SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"`，因此 `tools/` 必须与 `scripts/` 同级
- **RESULTS_DIR** 需要外部提供，挂载评测结果目录。所有结构化产物（metrics、状态汇总）以 `result.json` 形式写入此目录
- **LOGS_DIR** 需要外部提供，挂载日志目录。预处理 / 训练日志、`stdout`/`stderr` 重定向、容器内异常堆栈等运行期文本均写入此目录，便于事后排查
- 容器内的 `/workspace/tmp/` 用作预处理产物缓存（`$TMP_DIR/datasets_processed/qwen3_8b/`），**默认不挂载到宿主机**：预处理脚本会在容器启动后 `mkdir -p` 创建该目录，训练入口 Python (`nemotron_pretraining_qwen3_8b.py`) 从同一路径读取；容器销毁后产物丢失，如需跨容器复用，自行挂载即可
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

**目录结构说明**：

- `$MODEL_DIR`: 模型权重目录，采用 HuggingFace 标准模型仓库扁平布局（与 `git clone https://huggingface.co/Qwen/Qwen3-8B` 后的目录一致），典型结构如下：
  ```
  $MODEL_DIR/                              # 例如 qwen3_8b 或 Qwen3-8B
  ├── config.json
  ├── generation_config.json
  ├── tokenizer.json
  ├── tokenizer_config.json
  ├── merges.txt
  ├── model.safetensors.index.json
  ├── model-00001-of-00005.safetensors
  ├── model-00002-of-00005.safetensors
  ├── model-00003-of-00005.safetensors
  ├── model-00004-of-00005.safetensors
  ├── model-00005-of-00005.safetensors
  ├── LICENSE
  └── README.md
  ```

  **注意**：预处理脚本与训练入口直接以 `$MODEL_DIR` 根目录作为模型路径（即 `/data/models/qwen3_8b/`），无需再下钻到 `snapshots/<commit_hash>/` 子目录。如使用的是 HuggingFace Hub 缓存布局（含 `blobs/` / `refs/` / `snapshots/`），请改为指向具体的 `snapshots/<commit_hash>/` 作为 `MODEL_DIR`，或直接调整为本扁平布局。

- `$DATASET_DIR`: 数据集目录，典型结构如下：
  ```
  $DATASET_DIR/
  └── arxiv_sample.jsonl   # RedPajama-Data-1T-Sample arxiv 子集（jsonl 文本）
  ```

- `$CODE_DIR`: 预训练代码目录，典型结构如下：
  ```
  $CODE_DIR/                                  # qwen_pretrain
  ├── scripts/
  │   ├── preprocess_data.sh                  # 数据预处理启动脚本
  │   └── pretraining_qwen.sh                 # 预训练启动脚本（torchrun 入口）
  └── tools/
      ├── preprocess_data_for_megatron.py     # Megatron 预处理工具
      └── nemotron_pretraining_qwen3_8b.py    # 训练主入口（基于 NeMo）
  ```

  **注意**：
  - `scripts/*.sh` 通过 `SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"` 自动解析 `tools/` 的位置，因此必须保持 `scripts/` 与 `tools/` 同级
  - `tools/nemotron_pretraining_qwen3_8b.py` 中硬编码了 tokenizer 与预处理产物路径（`/data/models/qwen3_8b`、`/workspace/tmp/datasets_processed/qwen3_8b/`），如调整 `MODEL_DIR`/`TMP_DIR` 的容器内挂载点，需同步修改该 Python 文件

- `$RESULTS_DIR`: 评测结果目录，典型结构如下：
  ```
  $RESULTS_DIR/
  └── result.json   # 指标采集脚本生成的结构化结果（{"status": "success", "metrics": {...}}）
  ```

  **注意**：内容由步骤 4 的指标采集脚本写入；上层 agent 会从该路径（容器内 `/workspace/results/result.json`）读取或从脚本 stdout 解析 metrics。

- `$LOGS_DIR`: 日志目录，典型结构如下：
  ```
  $LOGS_DIR/
  ├── preprocess.log   # 数据预处理输出（步骤 2，`tee` 重定向）
  └── training.log     # 训练日志（步骤 3，`tee` 重定向；指标采集源）
  ```

- `$TMP_DIR`: 临时缓存目录（容器内 `/workspace/tmp/`，**默认不从宿主机挂载**），典型结构如下：
  ```
  /workspace/tmp/
  └── datasets_processed/
      └── qwen3_8b/
          ├── arxiv_sample_text_document.bin
          └── arxiv_sample_text_document.idx
  ```

  **注意**：
  - 该目录由步骤 2 的预处理脚本在容器内 `mkdir -p` 创建，容器销毁后产物随之丢失
  - 如希望跨容器复用预处理产物（避免每次重做），可在 `docker run` 中追加 `-v <host_tmp>:/workspace/tmp:rw`

**注意**：
- 必需的参数（`MODEL_DIR`、`DATASET_DIR`、`CODE_DIR`、`RESULTS_DIR`、`LOGS_DIR`）必须提供
- 容器内路径已通过卷挂载固定，对应 `docker run` 命令中的 `-v` 参数
- 宿主机路径建议存放在大容量 NVMe 磁盘上，避免占用系统盘空间

## 执行流程

### 步骤 1：容器启动

**挂载权限约定**：
- `:ro` — 只读，用于输入数据（模型权重、数据集等），防止误修改
- `:rw` — 读写，用于输出目录（代码目录下的临时文件、预处理产物、训练日志等）

**完整启动命令**：

```bash
docker run -it \
  --name nemo_pretrain \
  --gpus all \
  --shm-size=128g \
  -v $MODEL_DIR:/data/models/qwen3_8b:ro \
  -v $DATASET_DIR:/data/datasets:ro \
  -v $CODE_DIR:/workspace/code/qwen_pretrain:rw \
  -v $RESULTS_DIR:/workspace/results:rw \
  -v $LOGS_DIR:/workspace/logs:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-training:latest \
  bash
```

**注意**：
- 所有大文件路径通过 `MODEL_DIR`、`DATASET_DIR`、`CODE_DIR` 环境变量提供，避免命令中硬编码
- `--shm-size=128g`：避免大数据加载 / NCCL 时共享内存不足
- 若已存在同名容器，先执行 `docker rm -f nemo_pretrain`
- `CODE_DIR` 必须挂载为 `:rw`，因为可能写入临时缓存
- 容器内 `/workspace/tmp/` 不挂载，由预处理脚本在容器内创建；如需跨容器复用预处理产物，自行追加 `-v <host_tmp>:/workspace/tmp:rw` 即可

#### 容器管理命令

**进入已创建的容器**：
```bash
# 如果容器已在运行
docker exec -it nemo_pretrain /bin/bash

# 如果容器已停止，先启动再进入
docker start nemo_pretrain
docker exec -it nemo_pretrain /bin/bash
```

**验证容器环境**：
```bash
# 检查 GPU 设备
nvidia-smi

# 检查挂载的目录
ls -lh /data/models/qwen3_8b/
ls -lh /data/datasets/arxiv_sample.jsonl
ls -lh /workspace/code/qwen_pretrain/scripts/
ls -lh /workspace/code/qwen_pretrain/tools/
```

### 步骤 2：数据预处理

```bash
cd /workspace/code/qwen_pretrain

# 执行预处理（产物写入 /workspace/tmp/datasets_processed/qwen3_8b/）
bash scripts/preprocess_data.sh 2>&1 | tee /workspace/logs/preprocess.log
```

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `arxiv_sample_text_document.{bin,idx}` | `/workspace/tmp/datasets_processed/qwen3_8b/` | Megatron 格式的二进制 token 与索引文件 |
| `preprocess.log` | `/workspace/logs/preprocess.log` | 预处理 stdout/stderr 重定向产物 |

**验证预处理结果**：
```bash
ls -lh /workspace/tmp/datasets_processed/qwen3_8b/
tail -20 /workspace/logs/preprocess.log
```

**注意**：
- 脚本会自动校验 `/data/datasets/arxiv_sample.jsonl` 与 `/data/models/qwen3_8b/` 是否存在
- 若 `/workspace/tmp/datasets_processed/qwen3_8b/arxiv_sample_text_document.bin` 已存在，可跳过本步骤直接执行步骤 3
- 如调整 `MODEL_DIR` 的容器内路径，需同步修改 `scripts/preprocess_data.sh` 中的 `MODEL_PATH`

### 步骤 3：执行训练

```bash
cd /workspace/code/qwen_pretrain

# 执行预训练（默认 1 节点 8 卡，global_batch_size=128，seq_length=8192）
bash scripts/pretraining_qwen.sh 2>&1 | tee /workspace/logs/training.log
```

**默认行为**：
- 通过环境变量 `MASTER_PORT`、`GPUS_PER_NODE`、`NNODES`、`NODE_RANK`、`MASTER_ADDR` 配置分布式参数（未设置时自带默认值）
- 训练入口 `tools/nemotron_pretraining_qwen3_8b.py` 中固定 `global_batch_size=128`、`micro_batch_size=2`、`seq_length=8192`、`max_steps=100`、`warmup_steps=10`
- **不要修改** `global_batch_size`、`seq_length` 等核心超参，否则与基线指标不可比

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `training.log` | `/workspace/logs/training.log` | 训练日志（指标采集源） |

**验证训练结果**：
```bash
# 查看训练日志末尾
tail -50 /workspace/logs/training.log

# 检查是否包含 tokens_per_sec_per_gpu 行
grep -c "tokens_per_sec_per_gpu" /workspace/logs/training.log
```

### 步骤 4：指标采集

训练过程中 NeMo `TimingCallback`（`log_tokens_per_sec=True`）会在每个有效 step 打印一行包含 `tokens_per_sec_per_gpu` 的指标，例如：

```text
tokens_per_sec_per_gpu: 0
```

平均吞吐计算时需丢弃前后若干步以排除热身与尾部噪声（前 10 步 warmup、末尾 10 步收尾）。

#### 关键性能指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `tokens_per_sec_per_gpu_avg` | 单卡平均吞吐（剔除前 10 步与末尾 10 步后的算术平均） |
| 性能（辅助） | `tokens_per_sec_total` | 全局吞吐 = `tokens_per_sec_per_gpu_avg * world_size` |
| 性能（辅助） | `step_count_used` | 参与平均的 step 数（用于核对样本量） |
| 资源（辅助） | `GPU 利用率` | 训练阶段 GPU 使用率（外部 `nvidia-smi` / dcgm 采集） |
| 质量（辅助） | `Loss` | 训练收敛曲线（NeMo 默认输出，留作离线分析） |

#### 指标采集方法

**Python 脚本提取**

脚本职责：
1. 从训练日志中提取所有 `tokens_per_sec_per_gpu` 行
2. 丢弃前 10 步与末尾 10 步，对剩余取算术平均
3. 计算总吞吐 = 单卡均值 × `world_size`（默认 8）
4. 把 metrics 写入 `/workspace/results/result.json`（`{"status": "success", "metrics": {...}}` 格式）
5. 同时把 `result.json` 的内容回显到 stdout（前缀 `result.json: `），供 agent 从标准输出解析

```bash
python - <<'EOF'
import json
import os
import re

log_path    = '/workspace/logs/training.log'
result_path = '/workspace/results/result.json'
world_size  = 8       # 与 pretraining_qwen.sh 中 GPUS_PER_NODE * NNODES 保持一致
warmup_skip = 10      # 丢弃前 10 步
tail_skip   = 10      # 丢弃末尾 10 步

with open(log_path) as f:
    text = f.read()

# 抓取所有 "tokens_per_sec_per_gpu: <number>" 行
values = [float(x) for x in re.findall(r"tokens_per_sec_per_gpu:\s*([0-9.+\-eE]+)", text)]
if len(values) <= warmup_skip + tail_skip:
    raise SystemExit(
        f"训练日志中 tokens_per_sec_per_gpu 行数不足（{len(values)}），训练可能未完成或被截断"
    )

trimmed = values[warmup_skip : len(values) - tail_skip]
avg_per_gpu = sum(trimmed) / len(trimmed)

metrics = {
    'tokens_per_sec_per_gpu_avg': round(avg_per_gpu, 2),
    'tokens_per_sec_total':       round(avg_per_gpu * world_size, 2),
    'step_count_used':            len(trimmed),
}

# 1) 控制台人类可读打印
print(f"tokens_per_sec_per_gpu_avg ({world_size} GPUs) : {metrics['tokens_per_sec_per_gpu_avg']:.2f}")
print(f"tokens_per_sec_total                : {metrics['tokens_per_sec_total']:.2f}")
print(f"step_count_used                     : {metrics['step_count_used']}")

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

**输出示例**（基于一次正常训练）：
```
tokens_per_sec_per_gpu_avg (8 GPUs) : 0
tokens_per_sec_total                : 0
step_count_used                     : 0
result.json: {"status": "success", "metrics": {"tokens_per_sec_per_gpu_avg": 0, "tokens_per_sec_total": 0, "step_count_used": 0}}
```

**结果文件**（`/workspace/results/result.json`）：
```json
{
  "status": "success",
  "metrics": {
    "tokens_per_sec_per_gpu_avg": 0,
    "tokens_per_sec_total": 0,
    "step_count_used": 0
  }
}
```

**注意**：
- 必须等待训练正常结束（`max_steps=100` 全部跑完）才能采集，否则可用 step 数不足
- 切换 GPU 数后，需将脚本中的 `world_size` 同步调整为 `GPUS_PER_NODE * NNODES` 的实际值
- 切换 `max_steps` 后，若有效 step 数 ≤ `warmup_skip + tail_skip` 会直接报错，需相应调小 `tail_skip`
