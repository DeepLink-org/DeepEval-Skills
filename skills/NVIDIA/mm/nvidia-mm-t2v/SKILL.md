---
name: nvidia-mm-t2v
description: "nvidia芯片-多模态场景-Open-Sora文生视频推理任务的评测流程。基于PyTorch进行推理，用于指导executor完成容器启动、模型构建、推理执行和性能采集的完整评测链路。"
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Open-Sora 文生视频推理"
- "Open-Sora 推理"
- "nvidia 文生视频推理评测"

---

### 支持的模型配置

**模型**：
- **Open-Sora v2**: `Open_Sora_v2.safetensors`，权重路径：`/data/models/Open-Sora-v2`

**模型组件**：
- ae: `hunyuan_vae.safetensors`
- t5: `google/t5-v1_1-xxl`
- clip: `openai/clip-vit-large-patch14`

**推理参数**：
- resolution: 256px / 768px
- aspect_ratio: 16:9 / 9:16 / 1:1
- num_frames: 129（默认）
- num_steps: 50（默认）
- guidance: 7.5（默认）
- FP16 精度推理

**硬件要求**：
- 1 张 NVIDIA H200 GPU

---

### 环境准备

**代码位置**：
- `/workspace/code/Open-Sora`

**推理脚本**：
- `scripts/diffusion/inference.py` - 文生视频推理入口
- 配置文件：`configs/diffusion/inference/256px.py`

**环境依赖**：
- 已打包在镜像 `registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3` 中

---

### 启动配置

**Docker 运行命令**：
```bash
docker run -d --gpus all --shm-size=16g \
  -v /data/models:/data/models \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3
```

**测试命令**：
```bash
cd /workspace/code/Open-Sora

# 256px 分辨率
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
  configs/diffusion/inference/256px.py \
  --prompt "raining, sea" \
  --offload True \
  > /workspace/logs/opensora_256.log 2>&1

# 768px 分辨率
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
  configs/diffusion/inference/768px.py \
  --prompt "raining, sea" \
  --offload True \
  > /workspace/logs/opensora_768.log 2>&1
```

**参数说明**：
- `--nproc_per_node`: GPU 数量
- `configs/diffusion/inference/256px.py`: 推理配置文件，指定模型路径、分辨率、帧数等
- `--prompt`: 文本提示词
- `--offload`: 是否启用模型 offload 以节省显存

---

### 性能监控

**关键指标**：
- `s/it` - 每步推理耗时

**日志示例**（256px）：
```
Inference progress: 100%|██████████| 1/1 [00:48<00:00, 48.56s/it]
CUDA max memory max memory allocated at inference: 52.5 GB
CUDA max memory max memory reserved at inference: 70.1 GB
```

**日志示例**（768px）：
```
Inference progress: 100%|██████████| 1/1 [19:34<00:00, 1174.06s/it]
CUDA max memory max memory allocated at inference: 58.3 GB
CUDA max memory max memory reserved at inference: 85.6 GB
```

**分析命令**：
```bash
# 提取每步推理耗时
grep -oP '\d+\.\d+s/it' /workspace/logs/opensora_256.log | tail -1

# 提取推理阶段显存占用
grep "CUDA max memory" /workspace/logs/opensora_256.log | grep "inference"
```
