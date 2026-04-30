---
name: nvidia-nlp-training
description: NVIDIA GPU 上 Qwen3-8B 模型预训练性能评测技能。基于 NeMo，用于指导 executor 完成容器启动、数据预处理、预训练脚本执行、日志采集与性能指标分析。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上训练 Qwen3 8B 模型"
- "Qwen3 8B 模型预训练"
- "在 nvidia 上跑 nemo 预训练 benchmark"
- "采集 Qwen3-8B 训练 tokens_per_sec_per_gpu"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 数据集目录：`/data/datasets`
- 代码挂载目录：`/workspace/code`
- 训练结果输出目录：`/workspace/results`
- 临时缓存目录：`/workspace/tmp`
- 训练日志输出目录：`/workspace/logs`

---

### 支持的模型配置

**当前支持模型**：
- **Qwen3-8B**：8 卡训练，global_batch_size=128，seq_length=8192

**当前支持任务**：
- 基于 NeMo 的 Qwen3-8B 预训练性能测试
- 使用 RedPajama-Data-1T-Sample arxiv_sample.jsonl 数据集

**硬件要求**：
- 1 节点，8 张 NVIDIA GPU
- 至少 1TB NVMe SSD 数据盘

---

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
nvcr.io/nvidia/nemo:25.09.00
```

容器内已预装 NeMo、PyTorch 等。

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/qwen3_8b/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5/   # Qwen3-8B tokenizer
```

**数据集路径**：
```bash
/data/datasets/arxiv_sample.jsonl   # RedPajama-Data-1T-Sample arxiv 子集
```

**预处理输出**：
```bash
/workspace/tmp/datasets_processed/qwen3_8b/arxiv_sample_text_document
```

如模型或数据集路径发生变化，应同步修改 `scripts/preprocess_data.sh` 中的相关路径。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name nemo_pretrain \
  --gpus all \
  --shm-size=128g \
  -v /data/models:/data/models \
  -v /data/datasets:/data/datasets \
  -v /workspace/code:/workspace/code \
  -v /workspace/results:/workspace/results \
  -v /workspace/tmp:/workspace/tmp \
  -v /workspace/logs:/workspace/logs \
  nvcr.io/nvidia/nemo:25.09.00 \
  bash
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行脚本；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=128g`**：避免大数据加载时共享内存不足。
- 若已存在同名容器，需先执行 `docker rm -f nemo_pretrain` 或更换 `--name`。

---

### 数据预处理

预处理脚本位于 skill 目录下的 `scripts/preprocess_data.sh`，部署时拷贝到 `/workspace/code/`：

```bash
cp scripts/preprocess_data.sh /workspace/code/
cp tools/preprocess_data_for_megatron.py /workspace/code/
```

执行方式：

```bash
cd /workspace/code
bash preprocess_data.sh 2>&1 | tee /workspace/logs/preprocess.log
```

说明：
- 脚本会自动检查 `/data/datasets/arxiv_sample.jsonl` 是否存在
- 如果 `/workspace/tmp/datasets_processed/qwen3_8b/arxiv_sample_text_document` 已存在，可跳过预处理

---

### 训练脚本

训练脚本位于 skill 目录下的 `scripts/pretraining_qwen.sh`，部署时拷贝到 `/workspace/code/`：

```bash
cp scripts/pretraining_qwen.sh /workspace/code/
cp tools/nemotron_pretraining_qwen3_8b.py /workspace/code/
```

执行方式：

```bash
cd /workspace/code
bash pretraining_qwen.sh 2>&1 | tee /workspace/logs/training.log
```

当前脚本默认行为：
- 通过环境变量 `MASTER_PORT`、`GPUS_PER_NODE`、`NNODES`、`NODE_RANK`、`MASTER_ADDR` 配置分布式参数
- 默认 1 节点 8 卡，global_batch_size=128，seq_length=8192
- **不要修改** `global_batch_size`、`seq_length` 等核心超参，否则与基线指标不可比

---

### 关键性能指标

训练日志中包含每 GPU 每秒 token 处理数，例如：

```text
tokens_per_sec_per_gpu: 1234.5
```

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `tokens_per_sec_per_gpu` | 每 GPU 每秒处理的 token 数，核心吞吐指标 |
| 性能（辅助） | `GPU 利用率` | 训练阶段 GPU 使用率 |
| 资源（辅助） | `内存使用` | 训练阶段内存占用 |
| 质量 | `Loss` | 训练收敛曲线 |

**采集命令**（将 `LOG` 替换为实际日志路径，如 `/workspace/logs/training.log`）：

```bash
# 核心：tokens_per_sec_per_gpu 平均值
grep "tokens_per_sec_per_gpu" "$LOG" | tail -n +11 | head -n -10 | awk '{sum+=$2} END {print "Average:", sum/NR}'
```

---

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f nemo_pretrain` 后重试，或改用新容器名。

2. **找不到预处理/训练脚本**
   - 检查 `/workspace/code` 下是否已拷贝 `preprocess_data.sh`、`pretraining_qwen.sh` 及 `tools/` 下的 Python 文件。

3. **数据集找不到**
   - 检查 `/data/datasets/arxiv_sample.jsonl` 是否存在。

4. **模型/tokenizer 找不到**
   - 检查 `/data/models/qwen3_8b/snapshots/` 下对应路径是否存在。

5. **共享内存不足**
   - 已使用 `--shm-size=128g`；若仍报错，可适当增大。

6. **预处理输出已存在**
   - 若 `/workspace/tmp/datasets_processed/qwen3_8b/` 已有预处理结果，跳过预处理步骤直接训练。
