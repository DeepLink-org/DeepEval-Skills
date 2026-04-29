---
name: h200-nlp-finetune
description: NVIDIA H200 GPU 上语言模型微调任务的评测技能。用于指导 executor 完成容器启动、微调脚本执行、训练日志采集与性能/质量指标分析。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 H200 上跑语言模型微调"
- "帮我测试 LoRA 微调性能"
- "我要跑 Alpaca-LoRA 微调"
- "用 LLaMA-7B 跑 alpaca-lora finetune"
- "采集 alpaca-lora 训练 train_tokens_per_second(tgs)"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 数据集目录：`/data/datasets`
- 代码挂载目录：`/workspace/code`
- 训练结果输出目录：`/workspace/results`
- 训练日志输出目录：`/workspace/logs`

---

### 支持的模型配置

**当前支持模型**：
- **LLaMA-7B**

**当前支持任务**：
- **Alpaca-LoRA 微调**：使用挂载代码仓库中的 `finetune.sh`

**硬件要求**：
- NVIDIA H200 GPU
- 足够显存支撑 LLaMA-7B + LoRA 训练 (`batch_size=128`, `cutoff_len=512`)

---

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:xtuner
```

容器内需可直接执行：

```bash
bash finetune.sh
```

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/   # LLaMA-7B 权重所在目录，由 finetune.sh 内部指向
```

**数据集路径**：
```bash
/data/datasets/   # Alpaca 数据集所在目录，由 finetune.sh 内部指向
```

如模型或数据集发生变化，应同步修改 `finetune.sh` 中的相关路径。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name alpaca_finetune \
  --gpus all \
  --shm-size=128g \
  -v /data/models:/data/models \
  -v /data/datasets:/data/datasets \
  -v /workspace/results:/workspace/results \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:xtuner \
  bash
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行 `finetune.sh`；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=128g`**：避免大 batch 数据加载时 `/dev/shm` 不足。
- 若已存在同名容器，需先执行 `docker rm -f alpaca_finetune` 或更换 `--name`。

---

### 微调脚本

微调脚本位置（由挂载的代码仓库提供）：

```bash
/workspace/code/finetune.sh
```

执行方式：

```bash
cd /workspace/code
bash finetune.sh 2>&1 | tee /workspace/logs/train.log
```

当前默认行为：
- `finetune.sh` 内部指向已挂载的 `/data/models`、`/data/datasets` 与输出目录 `/workspace/results`
- 完整训练日志写入 `/workspace/logs/train.log`，便于宿主机侧 `grep` / 脚本采集
- **不要修改** `batch_size` (=128) / `num_epochs` (=3) / `cutoff_len` (=512) / `lora_r` (=8)，否则与基线指标不可比

---

### 关键性能指标

训练日志中包含每步 Loss 与最终训练汇总行，例如：

```text
{'loss': 0.8408, 'learning_rate': 7.518796992481203e-07, 'epoch': 2.98}
{'train_runtime': 3581.7097, 'train_samples_per_second': 41.678, 'train_steps_per_second': 0.325, 'train_tokens_per_second(tgs)': 1228.267, 'train_loss': 0.9499486045739085, 'epoch': 2.99}
```

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `train_tokens_per_second(tgs)` | 唯一核心吞吐指标 |
| 性能（辅助） | `train_samples_per_second`、`train_steps_per_second`、`train_runtime` | 排查与对照用 |
| 质量 | `train_loss` | 目标 **`< 0.95`** |

**采集命令**（将 `LOG` 替换为实际日志路径，如 `/workspace/logs/train.log`）：

```bash
# 核心：tokens_per_second(tgs)
grep -oP "'train_tokens_per_second\(tgs\)':\s*\K[0-9.eE+-]+" "$LOG"

# 质量：train_loss
grep -oP "'train_loss':\s*\K[0-9.eE+-]+" "$LOG"
```

---

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f alpaca_finetune` 后重试，或改用新容器名

2. **找不到 `finetune.sh`**
   - 检查 `/workspace/code` 挂载是否包含该脚本
   - 检查当前工作目录是否为仓库根目录

3. **找不到模型或数据集**
   - 检查 `/data/models/...` 和 `/data/datasets/...` 映射是否正确
   - 检查 `finetune.sh` 内部路径是否与挂载一致

4. **`grep -P` 不可用**
   - 换用支持 Perl 正则的环境执行命令，或将日志行复制到本地用 `python -c` 解析

5. **共享内存不足**
   - 已使用 `--shm-size=128g`；若仍报错，检查数据加载 `num_workers` 与 Docker `--shm-size`

6. **训练日志或结果文件未生成**
   - 检查 `/workspace/logs` 与 `/workspace/results` 写权限
   - 检查 `tee` 重定向是否生效
