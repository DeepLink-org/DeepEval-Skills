---
name: nvidia-cv-detection
description: NVIDIA GPU 上 CV 目标检测训练性能评测技能。基于 /workspace/code 中的 onedl-mmdetection 实现和 batch_detection.sh，用于指导 executor 完成容器启动、脚本执行、日志采集与 AVG_ITER_TIME 性能指标分析。默认适配 faster_rcnn + COCO + ResNet50 backbone 权重，可通过环境变量扩展到 mask_rcnn、cascade_rcnn、retinanet、yolov3、fcos、ssd300、centernet、solo、swin_mask_rcnn 等模型。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 CV detection 模型测试"
- "帮我测试 faster rcnn / mask rcnn / yolov3 检测训练性能"
- "在 nvidia 上跑 mmdetection benchmark"
- "帮我批量测试 CV detection 模型 FP32/FP16 性能"
- "采集 detection 模型 AVG_ITER_TIME"

---

## 环境变量定义

| 环境变量 | 默认值/映射目录 | 是否必需 | 说明 |
|---------|----------------|----------|------|
| `CV_DET_HOST_CODE_DIR` | `/path/to/cv/code` | 是 | 宿主机代码目录，Docker 启动时挂载到容器 `/workspace/code`。 |
| `CV_DET_HOST_LOG_DIR` | `/path/to/cv/logs` | 是 | 宿主机日志目录，Docker 启动时挂载到容器 `/workspace/logs`。 |
| `CV_DET_HOST_DATASET_DIR` | `/path/to/cv/datasets` | 是 | 宿主机数据集目录，Docker 启动时只读挂载到容器 `/workspace/datasets`。 |
| `CV_DET_HOST_WEIGHT_DIR` | `/path/to/cv/weight` | 是 | 宿主机权重目录，Docker 启动时只读挂载到容器 `/workspace/weight`。 |
| `CV_DET_PROJECT_ROOT` | `/workspace/code` | 是 | 容器内代码根目录。 |
| `CV_DET_MMDET_DIR` | `/workspace/code/onedl-mmdetection` | 是 | onedl-mmdetection 项目目录。 |
| `CV_DET_MMCV_DIR` | `/workspace/code/onedl-mmcv` | 是 | onedl-mmcv 项目目录，会加入 `PYTHONPATH`。 |
| `CV_DET_SCRIPT` | `/workspace/code/batch_detection.sh` 或 `/workspace/scripts/batch_detection.sh` | 是 | 实际批量运行脚本。代码目录已有脚本时优先使用，否则用 skill 预置脚本。 |
| `CV_DET_WEIGHT_DIR` | `/workspace/weight` | 是 | 预训练权重目录。 |
| `CV_DET_DATASET_DIR` | `/workspace/datasets/coco` | 是 | COCO 数据集目录；脚本会建立 `onedl-mmdetection/data/coco` 软链接。 |
| `CV_DET_LOG_DIR` | `/workspace/logs` | 是 | 训练日志和 work-dir 输出根目录。 |
| `CV_DET_WORK_DIR` | `/workspace/logs/${MODEL_NAME}_gpus${CV_DET_NGPU}_${PRECISION}` | 是 | mmdet `--work-dir` 输出目录，实际日志会在其时间戳子目录下自动生成。 |
| `CV_DET_LOG_GLOB` | `/workspace/logs/${MODEL_NAME}_gpus${CV_DET_NGPU}_${PRECISION}/*/*.log` | 否 | 用于采集指标的日志匹配模式；为空时按模型、卡数、精度自动推导。 |
| `CV_DET_DOCKER_IMAGE` | `registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:cv` | 是 | 运行镜像。 |
| `CV_DET_CONTAINER_NAME` | `cv_det_bench` | 否 | 容器名。 |
| `CV_DET_NGPU` | 来自 task config 的 `card_count`，默认 `1` | 否 | 训练使用的 GPU 数；生成脚本必须显式设置。 |
| `CV_DET_MODELS` | `faster_rcnn` | 否 | 逗号分隔模型列表。 |
| `CV_DET_PRECISIONS` | `fp16,fp32` | 否 | 逗号分隔精度列表。 |

说明：
- 本 skill 以用户当前 `/workspace/code` 实现为准，不再假设代码位于 `./models/onedl-mmdetection`。
- Docker 启动阶段使用 `CV_DET_HOST_*`，运行阶段只使用容器内 `/workspace/*` 路径。
- executor 会把 skill assets 上传到 `/workspace/scripts` 和 `/workspace/tools`。如果 `/workspace/code/batch_detection.sh` 不存在，生成脚本应使用 `/workspace/scripts/batch_detection.sh`。

---

## 目录结构约定

```text
/workspace/code/
├── batch_detection.sh              # 可选；存在时优先执行
├── onedl-mmdetection/
│   ├── configs/
│   ├── tools/train.py
│   └── ...
└── onedl-mmcv/

/workspace/weight/
├── resnet50-0676ba61.pth
├── resnet50_msra-5891d200.pth
├── darknet53-a628ea1b.pth
├── vgg16_caffe-292e1171.pth
├── resnet18-f37072fd.pth
└── swin_tiny_patch4_window7_224.pth

/workspace/datasets/coco/
├── annotations/
├── train2017/
└── val2017/

/workspace/logs/
```

---

## 支持的模型配置

默认运行：`faster_rcnn`，默认精度 `fp16,fp32`，默认 GPU 数来自 task config。

支持模型：`faster_rcnn`、`mask_rcnn`、`cascade_rcnn`、`retinanet`、`yolov3`、`fcos`、`ssd300`、`centernet`、`solo`、`swin_mask_rcnn`。

默认权重位于 `/workspace/weight`，脚本会按模型自动选择对应 backbone 权重。

---

## 依赖要求

Docker 镜像：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:cv
```

容器内需要具备 Python 3.10、PyTorch + CUDA、MMEngine、onedl-mmcv、onedl-mmdetection。

---

## 交互参数映射规则

| 用户表达 | 必须设置的变量 | 结果 |
|---------|----------------|------|
| `1卡`、`单卡`、`card_count=1` | `export CV_DET_NGPU=1` | `--nproc_per_node=1`，日志目录包含 `_gpus1_`。 |
| `8卡`、`八卡`、`card_count=8` | `export CV_DET_NGPU=8` | `--nproc_per_node=8`，日志目录包含 `_gpus8_`。 |
| `fp16`、`半精度` | `export CV_DET_PRECISIONS=fp16` | 使用 `AmpOptimWrapper`。 |
| `fp32`、`单精度` | `export CV_DET_PRECISIONS=fp32` | 使用 `OptimWrapper`。 |
| `faster_rcnn` / `mask_rcnn` / `retinanet` 等 | `export CV_DET_MODELS=<model>` | 运行对应模型配置。 |

如果用户没有指定，默认值为：`CV_DET_NGPU=1`、`CV_DET_MODELS=faster_rcnn`、`CV_DET_PRECISIONS=fp16,fp32`。

生成的评测脚本应执行：

```bash
if [ -f /workspace/code/batch_detection.sh ]; then
  export CV_DET_SCRIPT=/workspace/code/batch_detection.sh
else
  export CV_DET_SCRIPT=/workspace/scripts/batch_detection.sh
fi
bash "$CV_DET_SCRIPT"
```

---

## 第一阶段：容器启动

```bash
docker run --gpus all \
  --network host --ipc host --shm-size=128g \
  -it --name ${CV_DET_CONTAINER_NAME:-cv_det_bench} \
  -v ${CV_DET_HOST_CODE_DIR:-/path/to/cv/code}:/workspace/code:rw \
  -v ${CV_DET_HOST_LOG_DIR:-/path/to/cv/logs}:/workspace/logs:rw \
  -v ${CV_DET_HOST_DATASET_DIR:-/path/to/cv/datasets}:/workspace/datasets:ro \
  -v ${CV_DET_HOST_WEIGHT_DIR:-/path/to/cv/weight}:/workspace/weight:ro \
  ${CV_DET_DOCKER_IMAGE:-registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:cv} \
  bash
```

如果容器名已存在：`docker rm -f ${CV_DET_CONTAINER_NAME:-cv_det_bench}`。

---

## 第二阶段：运行评测

```bash
export CV_DET_PROJECT_ROOT=${CV_DET_PROJECT_ROOT:-/workspace/code}
export CV_DET_MMDET_DIR=${CV_DET_MMDET_DIR:-/workspace/code/onedl-mmdetection}
export CV_DET_MMCV_DIR=${CV_DET_MMCV_DIR:-/workspace/code/onedl-mmcv}
if [ -f /workspace/code/batch_detection.sh ]; then
  export CV_DET_SCRIPT=${CV_DET_SCRIPT:-/workspace/code/batch_detection.sh}
else
  export CV_DET_SCRIPT=${CV_DET_SCRIPT:-/workspace/scripts/batch_detection.sh}
fi
export CV_DET_WEIGHT_DIR=${CV_DET_WEIGHT_DIR:-/workspace/weight}
export CV_DET_DATASET_DIR=${CV_DET_DATASET_DIR:-/workspace/datasets/coco}
export CV_DET_LOG_DIR=${CV_DET_LOG_DIR:-/workspace/logs}
export CV_DET_NGPU=${CV_DET_NGPU:-${CARD_COUNT:-1}}
export CV_DET_MODELS=${CV_DET_MODELS:-faster_rcnn}
export CV_DET_PRECISIONS=${CV_DET_PRECISIONS:-fp16,fp32}
chmod +x "$CV_DET_SCRIPT"

CV_DET_RUN_MARKER=${CV_DET_RUN_MARKER:-/tmp/cv_det_run_marker.$(date +%s).$$}
touch "$CV_DET_RUN_MARKER"
cd "$CV_DET_MMDET_DIR"
bash "$CV_DET_SCRIPT"

MODEL_NAME=${CV_DET_MODELS%%,*}
PRECISION=${CV_DET_PRECISIONS%%,*}
CV_DET_WORK_DIR=${CV_DET_WORK_DIR:-${CV_DET_LOG_DIR}/${MODEL_NAME}_gpus${CV_DET_NGPU}_${PRECISION}}
CV_DET_LATEST_LOG=$(find "$CV_DET_WORK_DIR" -type f -name "*.log" -newer "$CV_DET_RUN_MARKER" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
test -n "$CV_DET_LATEST_LOG"
export CV_DET_LATEST_LOG
```

运行前检查：

```bash
test -d "$CV_DET_MMDET_DIR"
test -d "$CV_DET_MMCV_DIR"
test -f "$CV_DET_SCRIPT"
test -d "$CV_DET_WEIGHT_DIR"
test -d "$CV_DET_DATASET_DIR"
```

---

## 结果文件命名与汇总规则

生成评测脚本时禁止把所有单项结果固定写到 `/workspace/logs/result.json`，否则不同模型、卡数或精度会互相覆盖。

每个模型、卡数、精度组合必须写入独立结果文件：

```bash
RESULT_JSON="${CV_DET_LOG_DIR}/${MODEL_NAME}_gpus${CV_DET_NGPU}_${PRECISION}_result.json"
```

示例：

```text
/workspace/logs/faster_rcnn_gpus1_fp16_result.json
/workspace/logs/faster_rcnn_gpus1_fp32_result.json
/workspace/logs/mask_rcnn_gpus8_fp16_result.json
/workspace/logs/mask_rcnn_gpus8_fp32_result.json
```

同时，生成评测脚本必须把本次运行产生的所有单项结果汇总写入：

```text
/workspace/logs/eval_result.json
```

`eval_result.json` 格式建议如下：

```json
{
  "status": "success",
  "task": "cv_detection",
  "model": "faster_rcnn",
  "gpu_count": 1,
  "precisions": ["fp16", "fp32"],
  "results": {
    "faster_rcnn_gpus1_fp16": {
      "model": "faster_rcnn",
      "gpu_count": 1,
      "precision": "fp16",
      "log": "/workspace/logs/faster_rcnn_gpus1_fp16/<timestamp>/<timestamp>.log",
      "result_json": "/workspace/logs/faster_rcnn_gpus1_fp16_result.json",
      "avg_iter_time": 0.0,
      "data_time": 0.0,
      "op_time": 0.0
    }
  }
}
```

如果 `CV_DET_PRECISIONS=fp16,fp32`，生成脚本必须分别解析 fp16 和 fp32 的 mmengine 日志，写两个单项结果文件，并把两个结果都写入同一个 `/workspace/logs/eval_result.json`。如果用户只指定 `fp32`，则 `eval_result.json` 只包含 `fp32` 的结果。

生成脚本必须使用 `CV_DET_RUN_MARKER` 只解析 marker 之后新生成/更新的日志，避免读取历史日志。

下面是结果汇总逻辑的参考实现，生成脚本应采用等价逻辑：

```bash
python3 - <<'CV_DET_PARSE'
import glob, json, os, re

models = [m.strip() for m in os.environ.get("CV_DET_MODELS", "faster_rcnn").split(",") if m.strip()]
gpu = os.environ.get("CV_DET_NGPU", "1")
log_dir = os.environ.get("CV_DET_LOG_DIR", "/workspace/logs")
marker = os.environ.get("CV_DET_RUN_MARKER")
precisions = [p.strip() for p in os.environ.get("CV_DET_PRECISIONS", "fp16,fp32").split(",") if p.strip()]
marker_mtime = os.path.getmtime(marker) if marker and os.path.exists(marker) else 0

results = {}
for model in models:
    for precision in precisions:
        work_dir = f"{log_dir}/{model}_gpus{gpu}_{precision}"
        logs = [p for p in glob.glob(f"{work_dir}/*/*.log") if os.path.getmtime(p) > marker_mtime]
        logs.sort(key=os.path.getmtime, reverse=True)
        log = logs[0] if logs else ""
        text = open(log, "r", encoding="utf-8", errors="ignore").read() if log else ""
        rows = re.findall(r"AVG_ITER_TIME:\s*([0-9.]+)s\s*\|\s*DATA:\s*([0-9.]+)s\s*\|\s*OP:\s*([0-9.]+)s", text)
        last = rows[-1] if rows else None
        key = f"{model}_gpus{gpu}_{precision}"
        item = {
            "model": model,
            "gpu_count": int(gpu),
            "precision": precision,
            "log": log,
            "avg_iter_time": float(last[0]) if last else None,
            "data_time": float(last[1]) if last else None,
            "op_time": float(last[2]) if last else None,
        }
        item_path = os.path.join(log_dir, f"{key}_result.json")
        item["result_json"] = item_path
        with open(item_path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
        results[key] = item

aggregate = {
    "status": "success" if results and all(v.get("avg_iter_time") is not None for v in results.values()) else "partial",
    "task": "cv_detection",
    "model": ",".join(models),
    "gpu_count": int(gpu),
    "precisions": precisions,
    "results": results,
}
with open(os.path.join(log_dir, "eval_result.json"), "w", encoding="utf-8") as f:
    json.dump(aggregate, f, ensure_ascii=False, indent=2)
print("eval result json written: /workspace/logs/eval_result.json")
CV_DET_PARSE
```

---

## 关键性能指标

训练日志中应包含 `CustomIterTimerHook` 输出，例如：

```text
=== AVG_ITER_TIME: 0.0951s | DATA: 0.0036s | OP: 0.0915s ===
```

关注指标：`AVG_ITER_TIME`、`DATA`、`OP`。

采集命令：

```bash
if [ -n "${CV_DET_LATEST_LOG:-}" ]; then
  LOG="$CV_DET_LATEST_LOG"
else
  MODEL_NAME=${CV_DET_MODELS%%,*}
  PRECISION=${CV_DET_PRECISIONS%%,*}
  CV_DET_WORK_DIR=${CV_DET_WORK_DIR:-${CV_DET_LOG_DIR:-/workspace/logs}/${MODEL_NAME:-faster_rcnn}_gpus${CV_DET_NGPU:-1}_${PRECISION:-fp16}}
  test -n "${CV_DET_RUN_MARKER:-}"
  LOG=$(find "$CV_DET_WORK_DIR" -type f -name "*.log" -newer "$CV_DET_RUN_MARKER" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
fi
test -n "$LOG"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "AVG_ITER_TIME: \K[0-9.]+"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "DATA: \K[0-9.]+"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "OP: \K[0-9.]+"
```

---

## 常见问题

1. **找不到 `batch_detection.sh`**：检查 `/workspace/code/batch_detection.sh` 是否存在；不存在时使用 agent 上传的 `/workspace/scripts/batch_detection.sh`。
2. **找不到 `onedl-mmdetection` 或 `onedl-mmcv`**：检查 `CV_DET_MMDET_DIR` 和 `CV_DET_MMCV_DIR`。
3. **预训练权重加载失败**：检查 `CV_DET_WEIGHT_DIR=/workspace/weight` 下对应权重文件是否存在。
4. **数据集路径错误**：检查 `CV_DET_DATASET_DIR=/workspace/datasets/coco` 是否包含 COCO 数据集；脚本会建立 `onedl-mmdetection/data/coco` 软链接。
5. **没有 `AVG_ITER_TIME`**：确认 `custom_iter_timer_hook.py` 已被项目 config 引入，且不要直接读取历史最新日志，应使用 `CV_DET_RUN_MARKER`。
6. **GPU 数不匹配**：导出 `CV_DET_NGPU=<card_count>`，并确认容器可见 GPU 数。
