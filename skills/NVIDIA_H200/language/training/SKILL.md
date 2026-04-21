---
name: LLM-pretrain-Qwen
description: "NVIDIA H200 GPU AI训练专家，支持Qwen大模型预训练，提供分布式训练配置、性能优化和问题诊断 | NVIDIA H200 GPU AI training expert, supports Qwen LLM pretraining with distributed training config, performance optimization and issue diagnosis"
user-invocable: true
allowed-tools: Read, Bash
---

# NVIDIA H200 AI 训练专家

## 触发条件

当用户说以下任意内容时启动：
- "我要在 H200 上训练 Qwen3 8B 模型"
- "Qwen3 8B 模型预训练"

---

## 工具使用规则

使用以下工具：

| 任务 | 使用工具 |
|------|---------|
| 读取配置文件 | `Read` 工具 |
| 读取训练脚本 | `Read` 工具 |
| 执行训练命令 | `Bash` 工具 |
| 性能监控日志 | `Bash` 工具 |

**基础目录配置**：训练相关文件在 `BASE_DIR` 目录下。默认值为 `${BASE_DIR:-./workspace}`，可以通过环境变量或命令行参数自定义。

**配置方法**：
1. 通过环境变量设置：`export BASE_DIR=/your/custom/path`
2. 通过命令行参数设置：`--base-dir /your/custom/path`
3. 默认使用：`./workspace`（如果未设置）

---

## PART A：训练配置专家

### 支持的模型配置

**模型规模选择**：
- **8B**: Qwen3-8B，8卡训练，global_batch_size=128，seq_length=8192

**硬件要求**：
- 8张 NVIDIA H200 GPU 
- 100GB 共享内存
- 数据盘：至少 1TB NVMe SSD

**环境变量配置**：
```bash
export MASTER_PORT=29500
export MASTER_ADDR=${MASTER_ADDR}
export NODE_RANK=${NODE_RANK}
export WORLD_SIZE=${WORLD_SIZE}
export NNODES=${NODE_COUNT}
export GPUS_PER_NODE=${GPUS_PER_NODE}
```

### 数据预处理

**数据集准备**：
1. 先检查是否存在处理完成的数据集，如果有就不用再进行预处理：`${BASE_DIR}/datasets_processed/qwen3_8b/arxiv_sample_text_document`
2. 使用 RedPajama-Data-1T-Sample 中的 arxiv_sample.jsonl 数据集并进行预处理，数据集位置：`${BASE_DIR}/arxiv_sample.jsonl`

**预处理命令**：
```bash
sh scripts/preprocess_data.sh
```

### 启动配置

**Docker 运行命令**：
```bash
docker run -d --gpus all --shm-size=100g \
  -v ${BASE_DIR}:${BASE_DIR} \
  nvcr.io/nvidia/nemo:25.09.00
```
**训练命令**：
```bash
sh scripts/pretraining_qwen.sh
```

**训练脚本选择**：
- 自动检测模型规模并选择对应脚本
- 支持 `tools/nemotron_pretraining_qwen3_8b.py`
---

### 性能监控

**关键指标**：
- `tokens_per_sec_per_gpu` - 每 GPU 每秒处理的 token 数
- Loss 收敛曲线
- GPU 利用率
- 内存使用情况

**分析命令**：
```bash
# 提取性能指标
grep "tokens_per_sec_per_gpu" training.log | tail -n +11 | head -n -10 | awk '{sum+=$2} END {print "Average:", sum/NR}'

# 检查 GPU 利用率
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1
```

