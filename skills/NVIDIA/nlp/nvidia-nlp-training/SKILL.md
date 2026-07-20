---
name: nvidia-nlp-training-qwen3
description: NVIDIA GPU 上 Qwen3-8B 模型预训练性能评测技能（基于 NeMo）。用于指导 executor 完成容器启动、数据预处理、预训练脚本执行、训练日志采集与性能指标分析。
metadata:
  test_case: qwen3
---

# nvidia-nlp-training: qwen3

本 Skill 的预处理、训练和指标采集分别使用自带的
`scripts/preprocess.sh`、`scripts/pretrain.sh` 和 `scripts/calc.sh`。Executor 会将它们预置到
容器内 `/workspace/scripts/`；训练代码仍由 `CODE_DIR` 挂载，评测时不要绕过 Skill 脚本直接调用
resource 中的 `preprocess_data.sh`、`pretraining_qwen.sh` 或 Python 训练入口。

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
| `CODE_DIR` | `/workspace/code/qwen_pretrain` | 是 | 预训练代码目录，包含 resource 的 `scripts/` 子目录与 `nemotron_pretraining_qwen3_8b.py` 训练入口 |
| `RESULTS_DIR` | `/workspace/results` | 是 | 评测结果目录，存放 metrics 汇总文件 `result.json`（由步骤 4 的指标采集脚本生成） |
| `LOGS_DIR` | `/workspace/logs` | 是 | 日志目录，存放预处理日志（`preprocess.log`）、训练日志（`training.log`）以及 `stdout`/`stderr` 重定向输出 |

> **不需要外部提供的容器内路径**：`/workspace/tmp/`（临时缓存目录，存放预处理产物，例如 `datasets_processed/qwen3_8b/arxiv_sample_text_document.{bin,idx}`）。该目录仅在容器生命周期内有效，由步骤 2 的预处理脚本按需 `mkdir -p` 创建，**不需要从宿主机挂载**。如希望预处理产物跨容器复用，可自行追加一条 `-v <host_tmp>:/workspace/tmp:rw`。

**说明**：
- **MODEL_DIR** 需要外部提供，挂载预训练模型权重目录（HuggingFace 标准模型仓库扁平布局，根目录直接含 `config.json` / `tokenizer.json` / `model-*.safetensors`）
- **DATASET_DIR** 需要外部提供，挂载原始数据集目录（包含 `arxiv_sample.jsonl`）
- **CODE_DIR** 需要外部提供，挂载 resource 的 `code/` 目录。`scripts/preprocess_data.sh` 与 `scripts/pretraining_qwen.sh` 通过相对路径解析代码根目录，因此 `nemotron_pretraining_qwen3_8b.py` 必须与 `scripts/` 同级
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
  ├── nemotron_pretraining_qwen3_8b.py        # 训练主入口（基于 NeMo）
  └── scripts/
      ├── preprocess_data.sh                  # 数据预处理启动脚本
      ├── pretraining_qwen.sh                 # 预训练启动脚本
      └── nlp_language_modeling/
          └── preprocess_data_for_megatron.py # Megatron 预处理工具
  ```

  **注意**：
  - resource 的两个启动脚本均从其所在位置自动解析代码根目录，因此必须保持 `nemotron_pretraining_qwen3_8b.py` 与 `scripts/` 同级
  - Skill 脚本将 `MODEL_DIR` 与 `TMP_DIR` 传给 resource 代码；如修改容器内挂载点，调用脚本时显式设置相应环境变量

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
ls -lh /workspace/code/qwen_pretrain/nemotron_pretraining_qwen3_8b.py

# 检查 Skill 预置脚本
test -f /workspace/scripts/preprocess.sh
test -f /workspace/scripts/pretrain.sh
test -f /workspace/scripts/calc.sh
```

### 步骤 2：数据预处理

```bash
# 调用 resource 中的 scripts/preprocess_data.sh；日志由 Skill 脚本写入。
bash /workspace/scripts/preprocess.sh
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
- 如调整模型、数据或临时目录的容器内路径，调用 `preprocess.sh` 时显式设置 `MODEL_DIR`、`DATASET_DIR` 或 `TMP_DIR`

### 步骤 3：执行训练

```bash
# 调用 resource 中的 scripts/pretraining_qwen.sh；仅支持单机 8 卡。
NODE_COUNT=1 PROC_PER_NODE=8 bash /workspace/scripts/pretrain.sh
```

**默认行为**：
- `pretrain.sh` 校验 `NODE_COUNT=1`、`PROC_PER_NODE=8`，并向 resource 脚本设置 `MASTER_PORT`、`GPUS_PER_NODE`、`NNODES`、`NODE_RANK`、`MASTER_ADDR`
- 训练入口 `nemotron_pretraining_qwen3_8b.py` 中固定 `global_batch_size=128`、`micro_batch_size=2`、`seq_length=8192`、`max_steps=100`、`warmup_steps=10`
- **不要修改** `global_batch_size`、`seq_length` 等核心超参，否则与基线指标不可比

**输出产物**：

| 文件 / 目录 | 容器内路径 | 描述 |
| :--- | :--- | :--- |
| `train_Qwen3_8B_8_node0_<timestamp>.log` | `/workspace/logs/` | 训练日志（指标采集源） |
| `train_Qwen3_8B_node0.path` | `/workspace/logs/train_Qwen3_8B_node0.path` | 本次实际训练日志的完整路径 |

**验证训练结果**：
```bash
TRAIN_LOG="$(cat /workspace/logs/train_Qwen3_8B_node0.path)"
test -s "$TRAIN_LOG"

# 检查是否包含 tokens_per_sec_per_gpu 行
grep -c "tokens_per_sec_per_gpu" "$TRAIN_LOG"
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

**Skill 脚本提取**

`calc.sh` 负责：
1. 从本次训练日志中提取所有 `tokens_per_sec_per_gpu` 行；
2. 丢弃前 10 步与末尾 10 步，对剩余取算术平均；
3. 计算总吞吐 = 单卡均值 × `world_size`（固定为 8）；
4. 把 metrics 写入 `/workspace/results/result.json`，并将 `result.json` 回显到 stdout。

```bash
TRAIN_LOG="$(cat /workspace/logs/train_Qwen3_8B_node0.path)"
bash /workspace/scripts/calc.sh "$TRAIN_LOG" 8
```

**输出示例**（基于一次正常训练）：
```
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
- resource 的 Qwen3-8B 训练入口固定为单机 8 卡；`calc.sh` 的第二个参数必须保持为 `8`
- 切换 `max_steps` 后，若有效 step 数 ≤ 20，`calc.sh` 会直接报错
