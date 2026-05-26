---
name: nvidia-mm-t2v
description: NVIDIA GPU 上 Open-Sora v2 文生视频单卡推理性能评测技能。使用 skill 自带 scripts/t2v.sh 运行 /workspace/code/Open-Sora，权重位于 /workspace/weight/Open-Sora-v2，最终指标为 129 / s/it 的 frames/s。
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

| 环境变量 | 默认值/映射目录 | 是否必需 | 说明 |
|---------|----------------|----------|------|
| `MM_T2V_HOST_CODE_DIR` | `/path/to/code` | 是 | 宿主机代码目录，Docker 启动时挂载到容器 `/workspace/code`；该目录下应包含 `Open-Sora/`。 |
| `MM_T2V_HOST_LOG_DIR` | `/path/to/logs` | 是 | 宿主机日志目录，Docker 启动时挂载到容器 `/workspace/logs`。 |
| `MM_T2V_HOST_WEIGHT_DIR` | `/path/to/weight` | 是 | 宿主机权重目录，Docker 启动时只读挂载到容器 `/workspace/weight`；该目录下应包含 `Open-Sora-v2/`。 |
| `MM_T2V_PROJECT_ROOT` | `/workspace/code` | 是 | 容器内代码根目录。 |
| `MM_T2V_OPENSORA_DIR` | `/workspace/code/Open-Sora` | 是 | Open-Sora 项目目录。 |
| `MM_T2V_SCRIPT` | `/workspace/scripts/t2v.sh` | 是 | skill 自带测试脚本，agent 上传后从 `/workspace/scripts` 执行。 |
| `MM_T2V_WEIGHT_PATH` | `/workspace/weight/Open-Sora-v2` | 是 | Open-Sora v2 权重目录。 |
| `MM_T2V_LOG_DIR` | `/workspace/logs` | 是 | 推理日志、单项结果和汇总结果输出目录。 |
| `MM_T2V_DOCKER_IMAGE` | `registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3` | 是 | 运行镜像。 |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | 是 | Docker 启动时必须设置，确保 PyTorch CUDA runtime 可用；仅 `nvidia-smi` 可用不代表 compute 能力可用。 |
| `MM_T2V_CONTAINER_NAME` | `mm_t2v_bench` | 否 | 容器名。 |
| `MM_T2V_NGPU` | `1` | 否 | 固定单卡。生成脚本必须显式 `export MM_T2V_NGPU=1`。 |
| `MM_T2V_RESOLUTION` | `256px` | 否 | 单次运行分辨率，支持 `256px` 或 `768px`。 |
| `MM_T2V_RESOLUTIONS` | 空 | 否 | 兼容旧变量；如果未设置 `MM_T2V_RESOLUTION`，脚本取第一个分辨率。 |
| `MM_T2V_FRAME_COUNT` | `129` | 否 | Open-Sora 默认帧数。最终指标按 `129 / seconds_per_iter` 计算，由生成的评测脚本解析日志后写入 JSON。 |
| `MM_T2V_PROMPT` | `raining, sea` | 否 | 推理 prompt。 |
| `MM_T2V_OFFLOAD` | `True` | 否 | 是否启用 offload。 |

**说明**：
- 容器启动后必须同时满足 `nvidia-smi` 可用和 `python3 -c "import torch; torch.cuda.init()"` 可用；如果前者可用但后者报 `Found no NVIDIA driver`，通常是 Docker 没有启用 `compute` driver capability。
- 当前 t2v 只跑 1 卡，不根据 task config 的其他 `card_count` 扩展到多卡。
- 用户历史手动命令来自 `/workspace/code/Open-Sora/test.sh`，核心是 `torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --prompt "raining, sea" --offload True`。
- `scripts/t2v.sh` 贴近用户手动验证脚本，只增加 `MM_T2V_RESOLUTION` 分辨率切换，并把日志写到 `/workspace/logs/opensora_${MM_T2V_RESOLUTION}_gpus1.log`。
- `scripts/t2v.sh` 不写结果 JSON；JSON 提取逻辑应由 agent 按本 skill 的「结果文件命名与汇总规则」生成。

---

## 目录结构约定

```text
/workspace/code/
└── Open-Sora/
    ├── test.sh
    ├── scripts/diffusion/inference.py
    └── configs/diffusion/inference/
        ├── 256px.py
        └── 768px.py

/workspace/weight/
└── Open-Sora-v2/
    ├── Open_Sora_v2.safetensors
    ├── hunyuan_vae.safetensors
    ├── google/t5-v1_1-xxl/
    └── openai/clip-vit-large-patch14/

/workspace/scripts/
└── t2v.sh

/workspace/logs/
```

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
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3
```

容器内需要具备 Python 3.10、PyTorch 2.4 + CUDA、torchrun、colossalai、mmengine、flash-attn、ftfy、liger-kernel 等 Open-Sora 推理依赖。

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

生成的评测脚本必须执行 skill 自带入口：

```bash
chmod +x /workspace/scripts/t2v.sh
MM_T2V_RESOLUTION=256px bash /workspace/scripts/t2v.sh
```

切换 768px 时只改分辨率：

```bash
MM_T2V_RESOLUTION=768px bash /workspace/scripts/t2v.sh
# 或
bash /workspace/scripts/t2v.sh --resolution 768px
```

---

## 第一阶段：容器启动

推荐启动命令：

```bash
docker run --gpus all \
  --network host --ipc host --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility} \
  -it --name ${MM_T2V_CONTAINER_NAME:-mm_t2v_bench} \
  -v ${MM_T2V_HOST_CODE_DIR:-/path/to/code}:/workspace/code:rw \
  -v ${MM_T2V_HOST_LOG_DIR:-/path/to/logs}:/workspace/logs:rw \
  -v ${MM_T2V_HOST_WEIGHT_DIR:-/path/to/weight}:/workspace/weight:ro \
  ${MM_T2V_DOCKER_IMAGE:-registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:torch2.4-cu12-fla3} \
  bash
```

如果容器名已存在：

```bash
docker rm -f ${MM_T2V_CONTAINER_NAME:-mm_t2v_bench}
```

---

## 第二阶段：运行评测

进入容器后执行：

```bash
export MM_T2V_OPENSORA_DIR=${MM_T2V_OPENSORA_DIR:-/workspace/code/Open-Sora}
export MM_T2V_WEIGHT_PATH=${MM_T2V_WEIGHT_PATH:-/workspace/weight/Open-Sora-v2}
export MM_T2V_LOG_DIR=${MM_T2V_LOG_DIR:-/workspace/logs}
export MM_T2V_NGPU=1
export MM_T2V_FRAME_COUNT=${MM_T2V_FRAME_COUNT:-129}
export MM_T2V_RESOLUTION=${MM_T2V_RESOLUTION:-256px}
export MM_T2V_PROMPT="${MM_T2V_PROMPT:-raining, sea}"
export MM_T2V_OFFLOAD=${MM_T2V_OFFLOAD:-True}

chmod +x /workspace/scripts/t2v.sh
bash /workspace/scripts/t2v.sh --resolution "$MM_T2V_RESOLUTION"
export MM_T2V_LOG="${MM_T2V_LOG_DIR}/opensora_${MM_T2V_RESOLUTION}_gpus1.log"
test -f "$MM_T2V_LOG"
```

运行前检查：

```bash
test -d "$MM_T2V_OPENSORA_DIR"
test -f "/workspace/scripts/t2v.sh"
test -d "$MM_T2V_WEIGHT_PATH"
test -f "$MM_T2V_WEIGHT_PATH/Open_Sora_v2.safetensors"
test -f "$MM_T2V_WEIGHT_PATH/hunyuan_vae.safetensors"
test -d "$MM_T2V_WEIGHT_PATH/google/t5-v1_1-xxl"
test -d "$MM_T2V_WEIGHT_PATH/openai/clip-vit-large-patch14"
```

---

## 结果文件命名与汇总规则

每个分辨率必须写入独立结果文件，文件名固定包含 `gpus1`：

```bash
RESULT_JSON="${MM_T2V_LOG_DIR}/opensora_${MM_T2V_RESOLUTION}_gpus1_result.json"
```

示例：

```text
/workspace/logs/opensora_256px_gpus1_result.json
/workspace/logs/opensora_768px_gpus1_result.json
```

同时必须写入汇总文件：

```text
/workspace/logs/eval_result.json
```

`eval_result.json` 的核心输出必须包含最终指标 `frame_count / seconds_per_iter`：

```json
{
  "status": "success",
  "task": "mm_t2v",
  "model": "opensora_v2",
  "gpu_count": 1,
  "resolution": "256px",
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
      "result_json": "/workspace/logs/opensora_256px_gpus1_result.json",
      "output_video": "samples/video_256px/prompt_0000.mp4",
      "cuda_memory_allocated_gb": 52.5,
      "cuda_memory_reserved_gb": 70.1
    }
  }
}
```

768px 参考输出的最终指标应为：

```json
{
  "resolution": "768px",
  "frame_count": 129,
  "seconds_per_iter": 1174.06,
  "frame_count_over_seconds_per_iter": "129/1174.06s/it",
  "frames_per_second": 0.109875
}
```

生成评测脚本必须在 `t2v.sh` 运行完成后解析日志并写入 JSON。下面是参考实现，应放在生成的评测脚本中，而不是放进 `scripts/t2v.sh`：

```bash
python3 - <<'PY'
import json, os, pathlib, re

resolution = os.environ.get("MM_T2V_RESOLUTION", "256px")
frame_count = int(os.environ.get("MM_T2V_FRAME_COUNT", "129"))
log_dir = pathlib.Path(os.environ.get("MM_T2V_LOG_DIR", "/workspace/logs"))
opensora_dir = pathlib.Path(os.environ.get("MM_T2V_OPENSORA_DIR", "/workspace/code/Open-Sora"))
prompt = os.environ.get("MM_T2V_PROMPT", "raining, sea")
offload = os.environ.get("MM_T2V_OFFLOAD", "True")

log = pathlib.Path(os.environ.get("MM_T2V_LOG", str(log_dir / f"opensora_{resolution}_gpus1.log")))
result_path = log_dir / f"opensora_{resolution}_gpus1_result.json"
eval_path = log_dir / "eval_result.json"
text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""

s_iter_rows = re.findall(r"([0-9]+(?:\.[0-9]+)?)s/it", text)
video_rows = re.findall(r"Saved to\s+([^\r\n]+)", text)
allocated_rows = re.findall(r"CUDA max memory.*?allocated at inference:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", text)
reserved_rows = re.findall(r"CUDA max memory.*?reserved at inference:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", text)

seconds_text = s_iter_rows[-1] if s_iter_rows else None
seconds = float(seconds_text) if seconds_text else None
fps = frame_count / seconds if seconds else None
output_video = video_rows[-1].strip() if video_rows else None
output_video_abs = None
if output_video:
    p = pathlib.Path(output_video)
    output_video_abs = str(p if p.is_absolute() else opensora_dir / p)

status = "success" if "Inference finished." in text and seconds is not None else "partial"
key = f"opensora_{resolution}_gpus1"
item = {
    "status": status,
    "model": "opensora_v2",
    "gpu_count": 1,
    "resolution": resolution,
    "prompt": prompt,
    "offload": offload,
    "frame_count": frame_count,
    "seconds_per_iter": seconds,
    "frame_count_over_seconds_per_iter": f"{frame_count}/{seconds_text}s/it" if seconds_text else None,
    "frames_per_second": round(fps, 6) if fps is not None else None,
    "log": str(log),
    "result_json": str(result_path),
    "output_video": output_video,
    "output_video_abs": output_video_abs,
    "cuda_memory_allocated_gb": float(allocated_rows[-1]) if allocated_rows else None,
    "cuda_memory_reserved_gb": float(reserved_rows[-1]) if reserved_rows else None,
}
result_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
aggregate = {
    "status": status,
    "task": "mm_t2v",
    "model": "opensora_v2",
    "gpu_count": 1,
    "resolution": resolution,
    "frame_count": frame_count,
    "metric": {
        "name": "frames_per_second",
        "formula": "frame_count / seconds_per_iter",
        "expression": item["frame_count_over_seconds_per_iter"],
        "value": item["frames_per_second"],
        "unit": "frames/s",
    },
    "results": {key: item},
}
eval_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"result json written: {result_path}")
print(f"eval result json written: {eval_path}")
PY
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

用户历史参考日志：
- `256px` 单卡：`129/48.56s/it = 2.656507 frames/s`，inference 显存 `52.5 GB allocated / 70.1 GB reserved`。
- `768px` 单卡：`129/1174.06s/it = 0.109875 frames/s`，inference 显存 `58.3 GB allocated / 85.6 GB reserved`。
- 两个参考日志的 build model 显存均约为 `33.6 GB allocated / 34.4 GB reserved`。

关注指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `frames_per_second` | `frame_count / seconds_per_iter`，即 `129 / s/it`。 |
| 性能（辅助） | `seconds_per_iter` | 日志中的 `s/it`。 |
| 产物（必采） | `Saved to` | 输出视频路径，例如 `samples/video_768px/prompt_0000.mp4`。 |
| 资源（辅助） | `CUDA max memory allocated/reserved at inference` | 推理阶段峰值显存。 |

---

## 常见问题

1. **找不到 `/workspace/scripts/t2v.sh`**
   - 检查 agent 是否上传了 skill 的 `scripts/t2v.sh` 到容器 `/workspace/scripts`。

2. **找不到 Open-Sora 项目**
   - 检查 Docker 是否挂载了宿主机代码目录到容器 `/workspace/code`。
   - 或设置 `MM_T2V_OPENSORA_DIR` 指向实际路径。

3. **找不到模型权重**
   - 检查 `MM_T2V_WEIGHT_PATH=/workspace/weight/Open-Sora-v2` 是否存在。
   - 必须包含 `Open_Sora_v2.safetensors`、`hunyuan_vae.safetensors`、`google/t5-v1_1-xxl`、`openai/clip-vit-large-patch14`。

4. **没有 `frames_per_second`**
   - 检查日志中是否有 `s/it`。
   - 参考日志里会出现 PyTorch 2.4 的 `ProcessGroupNCCL` 未销毁 warning；只要 `Inference finished.`、`Saved to` 和 `s/it` 已出现，不应把该 warning 判定为失败。

5. **768px 推理 OOM 或运行时间过长**
   - 保持 `MM_T2V_OFFLOAD=True`。
   - 768px 推理参考耗时约 `1174.06s/it`，agent 的命令执行超时时间需要大于实际推理时间。

6. **GPU 数不匹配**
   - 当前只支持单卡，脚本固定 `MM_T2V_NGPU=1` 和 `--nproc_per_node 1`。
