---
name: nvidia-nlp-training
description: "nvidia 芯片-语言场景-训练任务的评测流程。用于指导executor完成docker容器启动、脚本生成、上传和执行的完整评测链路。"
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上训练 Qwen3 8B 模型"
- "Qwen3 8B 模型预训练"

---

### 支持的模型配置

**模型规模选择**：
- **8B**: Qwen3-8B，8卡训练，global_batch_size=128，seq_length=8192

**硬件要求**：
- 8张 NVIDIA GPU 
- 100GB 共享内存
- 数据盘：至少 1TB NVMe SSD

### 数据预处理

**数据集准备**：
1. 先检查是否存在处理完成的数据集，如果有就不用再进行预处理：`/workspace/tmp/datasets_processed/qwen3_8b/arxiv_sample_text_document`
2. 使用 RedPajama-Data-1T-Sample 中的 arxiv_sample.jsonl 数据集并进行预处理，数据集位置：`/data/datasets/arxiv_sample.jsonl`

**预处理命令**：
```bash
sh scripts/preprocess_data.sh
```

### 启动配置

**Docker 运行命令**：
```bash
docker run -d --gpus all --shm-size=100g \
  -v /data/models:/data/models \
  -v /data/datasets:/data/datasets \
  -v /workspace/results:/workspace/results \
  -v /workspace/tmp:/workspace/tmp \
  -v /workspace/logs:/workspace/logs \
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
```

