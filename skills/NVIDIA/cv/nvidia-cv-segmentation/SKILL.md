---
name: nvidia-cv-segmentation
description: NVIDIA GPU 上 CV 语义分割训练性能评测技能。基于 /workspace/code 中的 onedl-mmsegmentation 实现和 batch_segmentation.sh，用于指导 executor 完成容器启动、脚本执行、日志采集与 AVG_ITER_TIME 性能指标分析。默认适配 fcn + Cityscapes + ResNet50 backbone 权重，可通过环境变量扩展到 帮我测试deeplabv3分割训练性能，1卡，nvidia上、pspnet、apcnet 等模型。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 CV 分割模型测试"
- "帮我测试 fcn / deeplabv3 / pspnet / apcnet 分割训练性能"
- "在 nvidia 上跑 mmsegmentation segmentation benchmark"
- "帮我采集 segmentation 模型 AVG_ITER_TIME"

---

## 环境变量定义

| 环境变量 | 默认值/映射目录 | 是否必需 | 说明 |
|---------|----------------|----------|------|
| `CV_SEG_HOST_CODE_DIR` | `/path/to/cv/code` | 是 | 宿主机代码目录，Docker 启动时挂载到容器 `/workspace/code`。 |
| `CV_SEG_HOST_LOG_DIR` | `/path/to/cv/logs` | 是 | 宿主机日志目录，Docker 启动时挂载到容器 `/workspace/logs`。 |
| `CV_SEG_HOST_DATASET_DIR` | `/path/to/cv/datasets` | 是 | 宿主机数据集目录，Docker 启动时只读挂载到容器 `/workspace/datasets`。 |
| `CV_SEG_HOST_WEIGHT_DIR` | `/path/to/cv/weight` | 是 | 宿主机权重目录，Docker 启动时只读挂载到容器 `/workspace/weight`。 |
| `CV_SEG_PROJECT_ROOT` | `/workspace/code` | 是 | 容器内代码根目录，包含 `onedl-mmsegmentation`、`onedl-mmcv` 和 `batch_segmentation.sh`。 |
| `CV_SEG_MMSEG_DIR` | `/workspace/code/onedl-mmsegmentation` | 是 | onedl-mmsegmentation 项目目录，训练命令从这里执行。 |
| `CV_SEG_MMCV_DIR` | `/workspace/code/onedl-mmcv` | 是 | onedl-mmcv 项目目录，会加入 `PYTHONPATH`。 |
| `CV_SEG_SCRIPT` | `/workspace/code/batch_segmentation.sh` | 是 | 实际批量运行脚本。当前用户实现位于 `/workspace/code/batch_segmentation.sh`。 |
| `CV_SEG_WEIGHT_DIR` | `/workspace/weight` | 是 | 预训练权重目录，Docker 启动时挂载这个目录。 |
| `CV_SEG_WEIGHT_PATH` | `/workspace/weight/resnet50_v1c-2cccc1ad.pth` | 是 | ResNetV1c-50 backbone 预训练权重文件，训练命令使用这个文件路径。 |
| `CV_SEG_DATASET_DIR` | `/workspace/datasets/cityscapes` 或配置内路径 | 是 | Cityscapes 数据集目录；若项目 config 已固定数据路径，按实际 config 为准。 |
| `CV_SEG_LOG_DIR` | `/workspace/logs` | 是 | 训练日志和 work-dir 输出根目录。 |
| `CV_SEG_WORK_DIR` | `/workspace/logs/${MODEL_NAME}_gpus${CV_SEG_NGPU}_${PRECISION}` | 是 | mmseg `--work-dir` 输出目录，实际日志会在其时间戳子目录下自动生成；1卡为 `fcn_gpus1_fp16`，8卡为 `fcn_gpus8_fp16`。 |
| `CV_SEG_LOG_GLOB` | `/workspace/logs/${MODEL_NAME}_gpus${CV_SEG_NGPU}_${PRECISION}/*/*.log` | 否 | 用于采集指标的日志匹配模式；为空时按模型、卡数、精度自动推导。 |
| `CV_SEG_DOCKER_IMAGE` | `registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:mm_segmentation` | 是 | 运行镜像。 |
| `CV_SEG_CONTAINER_NAME` | `cv_seg_bench` | 否 | 容器名，默认使用用户现有命名规范。 |
| `CV_SEG_NGPU` | 来自 task config 的 `card_count`，默认 `1` | 否 | 训练使用的 GPU 数。1卡任务设为 `1`，8卡任务设为 `8`；生成评测脚本时必须显式 `export CV_SEG_NGPU=<card_count>`。 |
| `CV_SEG_MODELS` | `fcn` | 否 | 当前实现默认只跑 `fcn`；可扩展为 `deeplabv3,fcn,pspnet,apcnet`。 |
| `CV_SEG_PRECISIONS` | `fp16,fp32` | 否 | 默认按顺序运行 `fp16` 再运行 `fp32`；用户可指定单个精度，如 `fp16` 或 `fp32`。 |

**说明**：
- 本 skill 以用户当前 `/workspace/code` 实现为准，不再假设代码位于 `./models/onedl-mmsegmentation`。
- Docker 启动阶段使用 `CV_SEG_HOST_*` 作为宿主机源路径，挂载到容器内固定路径 `/workspace/code`、`/workspace/logs`、`/workspace/datasets`、`/workspace/weight`。
- 运行评测阶段只使用容器内路径，包括 `CV_SEG_PROJECT_ROOT`、`CV_SEG_MMSEG_DIR`、`CV_SEG_MMCV_DIR`、`CV_SEG_SCRIPT`、`CV_SEG_WEIGHT_PATH`、`CV_SEG_LOG_DIR`。
- 即使 executor 会把 skill assets 上传到 `/workspace/scripts` 和 `/workspace/tools`，生成脚本也必须优先执行 `CV_SEG_SCRIPT=/workspace/code/batch_segmentation.sh`，因为这是用户当前验证过的运行入口。

---

## 目录结构约定

容器内推荐结构：

```text
/workspace/code/
├── batch_segmentation.sh
├── onedl-mmsegmentation/
│   ├── configs/
│   ├── tools/train.py
│   └── ...
└── onedl-mmcv/

/workspace/weight/
└── resnet50_v1c-2cccc1ad.pth

/workspace/logs/
```

当前宿主机目录按用户实际环境挂载：`/path/to/cv/code` -> `/workspace/code`，`/path/to/cv/logs` -> `/workspace/logs`，`/path/to/cv/datasets` -> `/workspace/datasets`，`/path/to/cv/weight` -> `/workspace/weight`。

---

## 支持的模型配置

当前用户实现默认运行：
- 模型：`fcn`
- 精度：默认按顺序运行 `fp16`、`fp32`
- GPU 数：由 task config 的 `card_count` 决定，支持 `1` 和 `8`
- 配置文件：`configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py`
- 权重：`/workspace/weight/resnet50_v1c-2cccc1ad.pth`

可扩展模型：
- `deeplabv3`
- `fcn`
- `pspnet`
- `apcnet`

扩展时应同步修改 `batch_segmentation.sh` 或使用 skill 自带脚本中的 `CV_SEG_MODELS` / `CV_SEG_PRECISIONS` 环境变量。

---

## 依赖要求

Docker 镜像：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:mm_segmentation
```

容器内需要具备：
- Python 3.10
- PyTorch + CUDA
- MMEngine
- onedl-mmcv
- onedl-mmsegmentation

---


## 交互参数映射规则

当用户不使用 `--config`，而是在对话中指定模型、卡数、精度时，生成评测脚本必须按以下规则设置环境变量：

| 用户表达 | 必须设置的变量 | 结果 |
|---------|----------------|------|
| `1卡`、`单卡`、`card_count=1` | `export CV_SEG_NGPU=1` | `--nproc_per_node=1`，日志目录包含 `_gpus1_`。 |
| `8卡`、`八卡`、`card_count=8` | `export CV_SEG_NGPU=8` | `--nproc_per_node=8`，日志目录包含 `_gpus8_`。 |
| `fp16`、`半精度` | `export CV_SEG_PRECISIONS=fp16` | 使用 `AmpOptimWrapper`。 |
| `fp32`、`单精度` | `export CV_SEG_PRECISIONS=fp32` | 使用 `OptimWrapper`。 |
| `fcn` / `deeplabv3` / `pspnet` / `apcnet` | `export CV_SEG_MODELS=<model>` | 运行对应模型配置。 |

如果用户没有指定，默认值为：`CV_SEG_NGPU=1`、`CV_SEG_MODELS=fcn`、`CV_SEG_PRECISIONS=fp16,fp32`，即先跑 fp16，再跑 fp32。

生成的评测脚本必须执行用户验证过的入口：

```bash
bash /workspace/code/batch_segmentation.sh
```

不要把 `/workspace/scripts/batch_segmentation.sh` 作为首选入口。`/workspace/scripts` 只是 agent 上传 skill assets 的位置，当前 CV segmentation 以 `/workspace/code/batch_segmentation.sh` 为准。

---

## 第一阶段：容器启动

推荐启动命令：

```bash
docker run --gpus all \
  --network host --ipc host --shm-size=16g \
  -it --name ${CV_SEG_CONTAINER_NAME:-cv_seg_bench} \
  -v ${CV_SEG_HOST_CODE_DIR:-/path/to/cv/code}:/workspace/code:rw \
  -v ${CV_SEG_HOST_LOG_DIR:-/path/to/cv/logs}:/workspace/logs:rw \
  -v ${CV_SEG_HOST_DATASET_DIR:-/path/to/cv/datasets}:/workspace/datasets:ro \
  -v ${CV_SEG_HOST_WEIGHT_DIR:-/path/to/cv/weight}:/workspace/weight:ro \
  ${CV_SEG_DOCKER_IMAGE:-registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:mm_segmentation} \
  bash
```

如果容器名已存在：

```bash
docker rm -f ${CV_SEG_CONTAINER_NAME:-cv_seg_bench}
```

---

## 第二阶段：运行评测

进入容器后执行：

```bash
export CV_SEG_PROJECT_ROOT=${CV_SEG_PROJECT_ROOT:-/workspace/code}
export CV_SEG_MMSEG_DIR=${CV_SEG_MMSEG_DIR:-/workspace/code/onedl-mmsegmentation}
export CV_SEG_MMCV_DIR=${CV_SEG_MMCV_DIR:-/workspace/code/onedl-mmcv}
export CV_SEG_SCRIPT=${CV_SEG_SCRIPT:-/workspace/code/batch_segmentation.sh}
export CV_SEG_WEIGHT_DIR=${CV_SEG_WEIGHT_DIR:-/workspace/weight}
export CV_SEG_WEIGHT_PATH=${CV_SEG_WEIGHT_PATH:-${CV_SEG_WEIGHT_DIR}/resnet50_v1c-2cccc1ad.pth}
export CV_SEG_LOG_DIR=${CV_SEG_LOG_DIR:-/workspace/logs}
# 必须使用 task config 中的 card_count 设置 CV_SEG_NGPU：1卡任务设为1，8卡任务设为8。
export CV_SEG_NGPU=${CV_SEG_NGPU:-${CARD_COUNT:-1}}
export CV_SEG_MODELS=${CV_SEG_MODELS:-fcn}
export CV_SEG_PRECISIONS=${CV_SEG_PRECISIONS:-fp16,fp32}

# 如果用户在对话中指定了 1卡/8卡或单个精度 fp16/fp32，必须在运行前覆盖上述变量。
# 默认不指定精度时会按顺序运行 fp16,fp32。
# 示例：export CV_SEG_NGPU=8; export CV_SEG_PRECISIONS=fp32
chmod +x "$CV_SEG_SCRIPT"

# 创建本次运行 marker，后续只读取 marker 之后新生成/更新的日志，避免误读历史日志。
CV_SEG_RUN_MARKER=${CV_SEG_RUN_MARKER:-/tmp/cv_seg_run_marker.$(date +%s).$$}
touch "$CV_SEG_RUN_MARKER"

cd "$CV_SEG_MMSEG_DIR"
bash "$CV_SEG_SCRIPT"

# mmengine 会在 --work-dir 下生成时间戳目录和 .log，例如：
# /workspace/logs/fcn_gpus1_fp16/20260520_032625/20260520_032625.log
MODEL_NAME=${CV_SEG_MODELS%%,*}
PRECISION=${CV_SEG_PRECISIONS%%,*}
CV_SEG_WORK_DIR=${CV_SEG_WORK_DIR:-${CV_SEG_LOG_DIR}/${MODEL_NAME}_gpus${CV_SEG_NGPU}_${PRECISION}}
CV_SEG_LATEST_LOG=$(find "$CV_SEG_WORK_DIR" -type f -name "*.log" -newer "$CV_SEG_RUN_MARKER" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
test -n "$CV_SEG_LATEST_LOG"
export CV_SEG_LATEST_LOG
echo "Latest segmentation log from current run: $CV_SEG_LATEST_LOG"
```


### 1卡与8卡选择

- 使用 `config/task/h200_cv_segmentation_1card.json` 时，`card_count=1`，评测脚本应导出 `CV_SEG_NGPU=1`。
- 使用 `config/task/h200_cv_segmentation_8card.json` 时，`card_count=8`，评测脚本应导出 `CV_SEG_NGPU=8`。
- `batch_segmentation.sh` 会读取 `CV_SEG_NGPU` 并传给 `torch.distributed.launch --nproc_per_node`。
- 日志目录会随卡数和精度变化，例如 `/workspace/logs/fcn_gpus1_fp16/...`、`/workspace/logs/fcn_gpus1_fp32/...`、`/workspace/logs/fcn_gpus8_fp16/...`、`/workspace/logs/fcn_gpus8_fp32/...`。

运行前检查：

```bash
test -d "$CV_SEG_MMSEG_DIR"
test -d "$CV_SEG_MMCV_DIR"
test -f "$CV_SEG_SCRIPT"
test -d "$CV_SEG_WEIGHT_DIR"
test -f "$CV_SEG_WEIGHT_PATH"
```

---


## 结果文件命名与汇总规则

生成评测脚本时禁止把所有单项结果固定写到 `/workspace/logs/result.json`，否则不同模型、卡数或精度会互相覆盖。

每个模型、卡数、精度组合必须写入独立结果文件：

```bash
RESULT_JSON="${CV_SEG_LOG_DIR}/${MODEL_NAME}_gpus${CV_SEG_NGPU}_${PRECISION}_result.json"
```

示例：

```text
/workspace/logs/fcn_gpus1_fp16_result.json
/workspace/logs/fcn_gpus1_fp32_result.json
/workspace/logs/fcn_gpus8_fp16_result.json
/workspace/logs/fcn_gpus8_fp32_result.json
```

同时，生成评测脚本必须把本次运行产生的所有单项结果汇总写入：

```text
/workspace/logs/eval_result.json
```

`eval_result.json` 格式建议如下：

```json
{
  "status": "success",
  "task": "cv_segmentation",
  "model": "fcn",
  "gpu_count": 8,
  "precisions": ["fp16", "fp32"],
  "results": {
    "fcn_gpus8_fp16": {
      "model": "fcn",
      "gpu_count": 8,
      "precision": "fp16",
      "log": "/workspace/logs/fcn_gpus8_fp16/<timestamp>/<timestamp>.log",
      "result_json": "/workspace/logs/fcn_gpus8_fp16_result.json",
      "avg_iter_time": 0.0,
      "data_time": 0.0,
      "op_time": 0.0
    },
    "fcn_gpus8_fp32": {
      "model": "fcn",
      "gpu_count": 8,
      "precision": "fp32",
      "log": "/workspace/logs/fcn_gpus8_fp32/<timestamp>/<timestamp>.log",
      "result_json": "/workspace/logs/fcn_gpus8_fp32_result.json",
      "avg_iter_time": 0.0,
      "data_time": 0.0,
      "op_time": 0.0
    }
  }
}
```

如果 `CV_SEG_PRECISIONS=fp16,fp32`，生成脚本必须分别解析 fp16 和 fp32 的 mmengine 日志，写两个单项结果文件，并把两个结果都写入同一个 `/workspace/logs/eval_result.json`。如果用户只指定 `fp32`，则 `eval_result.json` 只包含 `fp32` 的结果。

下面是结果汇总逻辑的参考实现，生成脚本应采用等价逻辑：

```bash
python3 - <<'PY'
import glob, json, os, re

model = os.environ.get("CV_SEG_MODELS", "fcn").split(",")[0].strip()
gpu = os.environ.get("CV_SEG_NGPU", "1")
log_dir = os.environ.get("CV_SEG_LOG_DIR", "/workspace/logs")
marker = os.environ.get("CV_SEG_RUN_MARKER")
precisions = [p.strip() for p in os.environ.get("CV_SEG_PRECISIONS", "fp16,fp32").split(",") if p.strip()]
marker_mtime = os.path.getmtime(marker) if marker and os.path.exists(marker) else 0

results = {}
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
    "task": "cv_segmentation",
    "model": model,
    "gpu_count": int(gpu),
    "precisions": precisions,
    "results": results,
}
with open(os.path.join(log_dir, "eval_result.json"), "w", encoding="utf-8") as f:
    json.dump(aggregate, f, ensure_ascii=False, indent=2)
print("eval result json written: /workspace/logs/eval_result.json")
PY
```

---

## 关键性能指标

训练日志中应包含 `CustomIterTimerHook` 输出，例如：

```text
=== AVG_ITER_TIME: 0.1032s | DATA: 0.0067s | OP: 0.0965s ===
```

关注指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `AVG_ITER_TIME` | 平均迭代耗时（秒），核心吞吐指标。 |
| 性能（辅助） | `DATA` | 数据加载耗时。 |
| 性能（辅助） | `OP` | 纯计算耗时。 |

采集命令：

```bash
# 优先使用运行阶段导出的 CV_SEG_LATEST_LOG；如果单独执行采集命令，必须提供本次运行的 marker。
if [ -n "${CV_SEG_LATEST_LOG:-}" ]; then
  LOG="$CV_SEG_LATEST_LOG"
else
  MODEL_NAME=${CV_SEG_MODELS%%,*}
  PRECISION=${CV_SEG_PRECISIONS%%,*}
  CV_SEG_WORK_DIR=${CV_SEG_WORK_DIR:-${CV_SEG_LOG_DIR:-/workspace/logs}/${MODEL_NAME:-fcn}_gpus${CV_SEG_NGPU:-1}_${PRECISION:-fp16}}
  test -n "${CV_SEG_RUN_MARKER:-}"
  LOG=$(find "$CV_SEG_WORK_DIR" -type f -name "*.log" -newer "$CV_SEG_RUN_MARKER" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
fi
test -n "$LOG"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "AVG_ITER_TIME: \K[0-9.]+"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "DATA: \K[0-9.]+"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "OP: \K[0-9.]+"
```

JSON 结果输出必须遵循上方「结果文件命名与汇总规则」，生成单项 `*_result.json` 并汇总到 `/workspace/logs/eval_result.json`。

---

## 常见问题

1. **找不到 `/workspace/code/batch_segmentation.sh`**
   - 检查 Docker 是否挂载了宿主机 `/workspace/code` 到容器 `/workspace/code`。
   - 或设置 `CV_SEG_SCRIPT` 指向实际脚本路径。

2. **找不到 `onedl-mmsegmentation` 或 `onedl-mmcv`**
   - 检查 `CV_SEG_MMSEG_DIR` 和 `CV_SEG_MMCV_DIR`。
   - 当前用户实现期望它们分别位于 `/workspace/code/onedl-mmsegmentation` 和 `/workspace/code/onedl-mmcv`。

3. **预训练权重加载失败**
   - 检查 `CV_SEG_WEIGHT_PATH=/workspace/weight/resnet50_v1c-2cccc1ad.pth` 是否存在。
   - 若权重在其他目录，修改 `CV_SEG_WEIGHT_PATH` 并保证 Docker 挂载可见。

4. **数据集路径错误**
   - 检查 mmseg config 中 Cityscapes 的 `data_root`。
   - 如需覆盖，先确认 config 是否支持环境变量或在命令中追加 `--cfg-options`。

5. **没有 `AVG_ITER_TIME`**
   - 确认 `custom_iter_timer_hook.py` 已被项目 config 引入，或已复制到 `onedl-mmsegmentation` 可导入路径。
   - 检查训练是否跑到 hook 的统计区间。
   - 不要直接按 `ls -t` 读取历史最新日志；应使用 `CV_SEG_RUN_MARKER`，只查找 marker 之后生成/更新的 `.log`。

6. **GPU 数不匹配**
   - 当前脚本默认 `NGPU=1`。若要跑 8 卡，需要同步修改脚本的 `NGPU` / `--nproc_per_node`，并确认容器可见 GPU 数。
