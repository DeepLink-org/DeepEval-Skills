---
name: nvidia-mm-t2v
description: NVIDIA GPU 上 Open-Sora 文生视频推理任务的评测技能。用于指导 executor 完成容器启动、推理脚本执行、日志采集与性能指标分析。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Open-Sora 文生视频推理"
- "Open-Sora 推理评测"
- "nvidia 文生视频推理评测"
- "采集 Open-Sora 推理性能"
- "跑 Open-Sora v2 t2v"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 代码挂载目录：`/workspace/code`
- 推理日志输出目录：`/workspace/logs`

---

### 支持的模型配置

**当前支持模型**：
- **Open-Sora v2**（文生视频）

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
- 1 张 NVIDIA GPU

---

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3
```

容器内已预装 torch==2.4、torchvision==0.19.0、colossalai、mmengine、ftfy、liger-kernel、flash-attn 等。

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/Open-Sora-v2/   # Open_Sora_v2.safetensors, hunyuan_vae.safetensors, google/t5-v1_1-xxl, openai/clip-vit-large-patch14
```

配置文件 `configs/diffusion/inference/256px.py` 内部已指向上述路径。如模型路径发生变化，应同步修改配置文件中的 `from_pretrained`。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name opensora_inference \
  --gpus all \
  --shm-size=16g \
  -v /data/models:/data/models \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3 \
  bash
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行推理命令；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=16g`**：避免大分辨率推理时共享内存不足。
- 若已存在同名容器，需先执行 `docker rm -f opensora_inference` 或更换 `--name`。

---

### 推理脚本

推理脚本位置（由挂载的代码仓库提供）：

```bash
/workspace/code/Open-Sora/scripts/diffusion/inference.py
```

配置文件位置：

```bash
/workspace/code/Open-Sora/configs/diffusion/inference/256px.py
/workspace/code/Open-Sora/configs/diffusion/inference/768px.py
```

执行方式：

```bash
cd /workspace/code/Open-Sora

# 256px 分辨率
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
  configs/diffusion/inference/256px.py \
  --prompt "raining, sea" \
  --offload True \
  2>&1 | tee /workspace/logs/opensora_256.log

# 768px 分辨率
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
  configs/diffusion/inference/768px.py \
  --prompt "raining, sea" \
  --offload True \
  2>&1 | tee /workspace/logs/opensora_768.log
```

说明：
- 配置文件内部已指向 `/data/models/Open-Sora-v2/` 下的模型权重。
- 推理日志通过 `tee` 写入 `/workspace/logs/`，便于宿主机侧采集。
- `--offload True` 启用模型 offload 可降低显存占用。

---

### 关键性能指标

推理日志中包含进度条与显存汇总行，例如（256px）：

```text
Inference progress: 100%|██████████| 1/1 [00:48<00:00, 48.56s/it]
CUDA max memory max memory allocated at inference: 52.5 GB
CUDA max memory max memory reserved at inference: 70.1 GB
```

768px：

```text
Inference progress: 100%|██████████| 1/1 [19:34<00:00, 1174.06s/it]
CUDA max memory max memory allocated at inference: 58.3 GB
CUDA max memory max memory reserved at inference: 85.6 GB
```

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `s/it` | 每步推理耗时，核心吞吐指标 |
| 资源（辅助） | `CUDA max memory allocated` | 推理阶段峰值显存分配 |
| 资源（辅助） | `CUDA max memory reserved` | 推理阶段峰值显存预留 |

**采集命令**（将 `LOG` 替换为实际日志路径，如 `/workspace/logs/opensora_256.log`）：

```bash
# 核心：每步推理耗时 (s/it)
grep -oP '\d+\.\d+s/it' "$LOG" | tail -1

# 辅助：推理阶段显存占用
grep "CUDA max memory" "$LOG" | grep "inference"
```

---

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f opensora_inference` 后重试，或改用新容器名。

2. **共享内存不足**
   - 已使用 `--shm-size=16g`；若大分辨率（768px）仍报错，可适当增大。

3. **找不到模型权重**
   - 检查 `/data/models/Open-Sora-v2/` 是否包含 `Open_Sora_v2.safetensors`、`hunyuan_vae.safetensors`、`google/t5-v1_1-xxl`、`openai/clip-vit-large-patch14`。
   - 检查配置文件 `from_pretrained` 路径是否与挂载一致。

4. **`grep -P` 不可用**
   - 换用支持 Perl 正则的环境执行命令，或将日志行复制到本地用 `python -c` 解析。

5. **768px 推理 OOM**
   - 确认已开启 `--offload True`。
   - 768px 推理显存峰值约 85 GB，确保单卡显存足够。

6. **推理日志或输出视频未生成**
   - 检查 `/workspace/logs` 写权限。
   - 检查 `tee` 重定向是否生效。
   - 输出视频默认保存在 `samples/` 目录下。
