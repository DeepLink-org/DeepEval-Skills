---
name: nvidia-mm-t2v
description: NVIDIA GPU 上 Open-Sora v2 文生视频单卡推理性能评测技能。基于 /workspace/code/Open-Sora 和 t2v.sh，用于指导 executor 完成容器启动、脚本执行、日志采集与 frames_per_second 性能指标分析。默认运行 256px，可切换 768px，固定单卡。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 Open-Sora 文生视频推理"
- "Open-Sora 推理评测"
- "nvidia 文生视频推理评测"
- "采集 Open-Sora 推理性能"
- "跑 Open-Sora v2 t2v"
- "帮我测试 t2v / 文生视频 / text-to-video"

---

## 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `MM_T2V_PROJECT_ROOT` | `/workspace/code` | 是 | 项目根目录，需外部提供，包含 `Open-Sora/`；`t2v.sh` 可由 skill 资源复制到该目录 |
| `MM_T2V_WEIGHT_DIR` | `/workspace/weight` | 是 | 权重根目录，需外部提供，包含 `Open-Sora-v2/` |
| `MM_T2V_LOGS_DIR` | `/workspace/logs` | 否 | 推理日志和汇总结果 `eval_result.json` 输出目录 |

**说明**：
- **MM_T2V_PROJECT_ROOT** 需要外部提供，挂载到 `/workspace/code`，至少应包含 `Open-Sora/`；`t2v.sh` 可由 agent 从 `/workspace/scripts/t2v.sh` 复制到 `/workspace/code/t2v.sh`
- 容器启动后、执行评测前，如果 `/workspace/code/t2v.sh` 不存在，应将预置的 `/workspace/scripts/t2v.sh` 复制到 `/workspace/code/t2v.sh` 并添加执行权限
- **MM_T2V_WEIGHT_DIR** 需要外部提供，目录下应包含 `Open-Sora-v2/`
- **MM_T2V_LOGS_DIR** 用于保存推理日志和汇总结果 `eval_result.json`
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

---

## 目录结构约定

- `$MM_T2V_PROJECT_ROOT`: 项目根目录，默认结构如下：
  ```text
  $MM_T2V_PROJECT_ROOT/                 # = /workspace/code
  ├── t2v.sh                            # 运行前需要存在；可由本 skill 的 scripts/t2v.sh 复制到此处
  └── Open-Sora/
      ├── scripts/diffusion/inference.py
      └── configs/diffusion/inference/
          ├── 256px.py
          └── 768px.py
  ```

- `$MM_T2V_WEIGHT_DIR`: 权重根目录，默认结构如下：
  ```text
  $MM_T2V_WEIGHT_DIR/                   # = /workspace/weight
  └── Open-Sora-v2/
      ├── Open_Sora_v2.safetensors
      ├── hunyuan_vae.safetensors
      ├── google/t5-v1_1-xxl/
      └── openai/clip-vit-large-patch14/
  ```

- `$MM_T2V_LOGS_DIR`: 日志和结果目录，默认映射到 `/workspace/logs`。

---

## 支持的模型配置

当前实现默认运行：
- 模型：`Open-Sora v2`
- 任务：text-to-video inference
- GPU 数：固定 `1`
- 分辨率：默认 `256px`，可切换 `768px`
- 帧数：`129`
- 权重目录：`/workspace/weight/Open-Sora-v2`
- Prompt：`raining, sea`
- Offload：`True`

---

## 依赖要求

Docker 镜像：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-mm-t2v:latest
```

容器内需要具备 Python 3.10、PyTorch 2.4 + CUDA、torchrun、colossalai、mmengine、flash-attn、ftfy、liger-kernel 等 Open-Sora 推理依赖。

**注意**：Docker 启动时必须设置 `NVIDIA_DRIVER_CAPABILITIES=compute,utility`，确保 PyTorch CUDA runtime 可用；仅 `nvidia-smi` 可用不代表 compute 能力可用。

---

## 交互参数映射规则

| 用户表达 | 必须设置的变量 | 结果 |
|---------|----------------|------|
| `1卡`、`单卡`、未指定卡数 | `export MM_T2V_NGPU=1` | 固定 `torchrun --nproc_per_node 1`。 |
| `256px`、`256` | `export MM_T2V_RESOLUTION=256px` | 使用 `configs/diffusion/inference/256px.py`。 |
| `768px`、`768` | `export MM_T2V_RESOLUTION=768px` | 使用 `configs/diffusion/inference/768px.py`。 |
| 指定 prompt | `export MM_T2V_PROMPT='<prompt>'` | 覆盖默认 `raining, sea`。 |
| `offload`、`开启 offload` | `export MM_T2V_OFFLOAD=True` | 降低显存占用。 |
| `关闭 offload` | `export MM_T2V_OFFLOAD=False` | 仅在用户明确要求时关闭。 |

生成评测脚本时，如果 `/workspace/code/t2v.sh` 不存在，应先将 agent 预置的 `/workspace/scripts/t2v.sh` 复制到 `/workspace/code/t2v.sh`，再执行：

```bash
if [ ! -f /workspace/code/t2v.sh ] && [ -f /workspace/scripts/t2v.sh ]; then
  cp /workspace/scripts/t2v.sh /workspace/code/t2v.sh
  chmod +x /workspace/code/t2v.sh
fi
test -f /workspace/code/t2v.sh
bash /workspace/code/t2v.sh
```

---

## 第一阶段：容器启动

```bash
docker run --gpus all \
  --network host --ipc host --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -it --name mm_t2v_bench \
  -v $MM_T2V_PROJECT_ROOT:/workspace/code:rw \
  -v $MM_T2V_WEIGHT_DIR:/workspace/weight:ro \
  -v $MM_T2V_LOGS_DIR:/workspace/logs:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-mm-t2v:latest \
  bash
```

如果容器名已存在：`docker rm -f mm_t2v_bench`。

**注意**：
- `MM_T2V_PROJECT_ROOT`、`MM_T2V_WEIGHT_DIR` 必须在宿主机提供
- 容器启动后、执行评测前，如果 `/workspace/code/t2v.sh` 不存在，应将预置的 `/workspace/scripts/t2v.sh` 复制到 `/workspace/code/t2v.sh` 并添加执行权限
- 输出目录 `MM_T2V_LOGS_DIR` 不存在时需先在宿主机创建，或由 executor 创建后再挂载

---

## 第二阶段：容器中执行评测

### 步骤 1：选择推理参数

容器内代码、权重和日志路径已通过卷挂载固定（详见[环境变量定义](#环境变量定义)）。

```bash
RESOLUTION="256px"            # 可选: 256px, 768px
PROMPT="raining, sea"
OFFLOAD="True"                # 可选: True, False
FRAME_COUNT="129"
```

### 配置文件说明

**关键路径**：
```text
/workspace/code/Open-Sora                     # Open-Sora 项目目录
/workspace/code/t2v.sh                        # 推理评测脚本；代码/镜像没有时由 skill scripts/t2v.sh 提供
/workspace/weight/Open-Sora-v2                # Open-Sora v2 权重目录
/workspace/logs                               # 推理日志和 eval_result.json 输出目录
```

### 步骤 2：执行文生视频推理评测

运行推理脚本，日志和结果文件将保存至 `/workspace/logs`：

```bash
mkdir -p /workspace/logs
if [ ! -f /workspace/code/t2v.sh ] && [ -f /workspace/scripts/t2v.sh ]; then
  cp /workspace/scripts/t2v.sh /workspace/code/t2v.sh
  chmod +x /workspace/code/t2v.sh
fi
test -f /workspace/code/t2v.sh

export MM_T2V_RESOLUTION="$RESOLUTION"
export MM_T2V_PROMPT="$PROMPT"
export MM_T2V_OFFLOAD="$OFFLOAD"
export MM_T2V_FRAME_COUNT="$FRAME_COUNT"
export MM_T2V_NGPU=1
bash /workspace/code/t2v.sh
```

上述指令的默认行为：
- 运行 Open-Sora v2 文生视频推理
- 固定使用 1 张 NVIDIA GPU
- 默认运行 `256px` 分辨率
- 推理日志写入 `/workspace/logs/opensora_${RESOLUTION}_gpus1.log`
- 结构化结果统一写入 `/workspace/logs/eval_result.json`

**验证执行结果**：
```bash
LOG=/workspace/logs/opensora_${RESOLUTION}_gpus1.log
test -f "$LOG"
tail -50 "$LOG"
cat /workspace/logs/eval_result.json
```

---

## 评测输出产物

推理日志和汇总结果均输出到 `/workspace/logs`。

| 文件路径 | 描述 |
| :--- | :--- |
| `/workspace/logs/opensora_${RESOLUTION}_gpus1.log` | Open-Sora 推理日志，包含 `s/it`、输出视频路径和显存信息 |
| `/workspace/logs/eval_result.json` | 本次运行所有分辨率的汇总结果 |

**注意**：如果通过 `MM_T2V_RESOLUTIONS=256px,768px` 一次运行多个分辨率，所有结果都写入同一个 `eval_result.json`，不再生成额外的 `*_result.json` 文件。

**汇总结果示例**：

```json
{
  "status": "success",
  "task": "mm_t2v",
  "model": "opensora_v2",
  "gpu_count": 1,
  "resolutions": ["256px"],
  "frame_count": 129,
  "metric": {
    "name": "frames_per_second",
    "formula": "frame_count / seconds_per_iter",
    "expression": "129/48.56s/it",
    "value": 2.656507,
    "unit": "frames/s"
  },
  "results": {
    "opensora_256px_gpus1": {
      "status": "success",
      "model": "opensora_v2",
      "gpu_count": 1,
      "resolution": "256px",
      "frame_count": 129,
      "seconds_per_iter": 48.56,
      "frame_count_over_seconds_per_iter": "129/48.56s/it",
      "frames_per_second": 2.656507,
      "log": "/workspace/logs/opensora_256px_gpus1.log",
      "output_video": "samples/video_256px/prompt_0000.mp4",
      "cuda_memory_allocated_gb": 52.5,
      "cuda_memory_reserved_gb": 70.1
    }
  }
}
```

---

## 关键性能指标

推理日志中应包含进度条、输出视频路径与显存汇总行，例如：

```text
Saved to samples/video_256px/prompt_0000.mp4
Inference progress: 100%|██████████| 1/1 [00:48<00:00, 48.56s/it]
CUDA max memory max memory allocated at inference: 52.5 GB
CUDA max memory max memory reserved at inference: 70.1 GB
```

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `frames_per_second` | `frame_count / seconds_per_iter`，即 `129 / s/it` |
| 性能（辅助） | `seconds_per_iter` | 日志中的 `s/it` |
| 产物（必采） | `output_video` | 输出视频路径，例如 `samples/video_768px/prompt_0000.mp4` |
| 资源（辅助） | `cuda_memory_allocated_gb` / `cuda_memory_reserved_gb` | 推理阶段峰值显存 |

#### 指标采集

优先读取汇总结果文件：

```bash
python3 -c "
import json
path='/workspace/logs/eval_result.json'
print(json.dumps(json.load(open(path, 'r', encoding='utf-8')), indent=2, ensure_ascii=False))
"
```

---

## 常见问题

1. **找不到 `t2v.sh`**：检查 `/workspace/code/t2v.sh` 是否存在；如果不存在，应将 agent 预置的 `/workspace/scripts/t2v.sh` 复制到 `/workspace/code/t2v.sh` 并添加执行权限。
2. **找不到 Open-Sora 项目**：检查 `/workspace/code/Open-Sora` 是否存在，或设置 `MM_T2V_OPENSORA_DIR` 指向实际路径。
3. **找不到模型权重**：检查 `/workspace/weight/Open-Sora-v2` 是否存在，且包含 `Open_Sora_v2.safetensors`、`hunyuan_vae.safetensors`、`google/t5-v1_1-xxl`、`openai/clip-vit-large-patch14`。
4. **没有 `frames_per_second`**：检查日志中是否有 `s/it`；只要 `Inference finished.`、`Saved to` 和 `s/it` 已出现，不应把 NCCL warning 判定为失败。
5. **768px 推理 OOM 或运行时间过长**：保持 `MM_T2V_OFFLOAD=True`；768px 推理参考耗时约 `1174.06s/it`，agent 的命令执行超时时间需要大于实际推理时间。
6. **GPU 数不匹配**：当前只支持单卡，脚本固定 `MM_T2V_NGPU=1` 和 `--nproc_per_node 1`。
