---
name: nvidia-cv-segmentation
description: NVIDIA GPU 上 CV 语义分割训练性能评测技能。基于 /workspace/code 中的 onedl-mmsegmentation 和 batch_segmentation.sh，用于指导 executor 完成容器启动、脚本执行、日志采集与 AVG_ITER_TIME 性能指标分析。默认适配 fcn + Cityscapes + ResNetV1c-50 backbone 权重，可通过模型参数扩展到 deeplabv3、pspnet、apcnet 等模型。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 CV 分割模型测试"
- "帮我测试 fcn / deeplabv3 / pspnet / apcnet 分割训练性能"
- "在 nvidia 上跑 mmsegmentation segmentation benchmark"
- "帮我采集 segmentation 模型 AVG_ITER_TIME"

---

## 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `CV_SEG_PROJECT_ROOT` | `/workspace/code` | 是 | 项目根目录，需外部提供，包含 `onedl-mmsegmentation` 和 `onedl-mmcv` 源码；`batch_segmentation.sh` 可由 skill 资源复制到该目录 |
| `CV_SEG_DATA_DIR` | `/workspace/datasets/cityscapes` | 是 | Cityscapes 数据集目录 |
| `CV_SEG_WEIGHT_DIR` | `/workspace/weight` | 是 | 预训练权重目录，存放 `resnet50_v1c-2cccc1ad.pth` |
| `CV_SEG_LOGS_DIR` | `/workspace/logs` | 否 | 训练日志、work-dir 和汇总结果 `eval_result.json` 输出目录 |

**说明**：
- **CV_SEG_PROJECT_ROOT** 需要外部提供，挂载到 `/workspace/code`，至少应包含 `onedl-mmsegmentation` 和 `onedl-mmcv`；`batch_segmentation.sh` 可由 agent 从 `/workspace/scripts/batch_segmentation.sh` 复制到 `/workspace/code/batch_segmentation.sh`
- **CV_SEG_DATA_DIR** 需要外部提供，指向 Cityscapes 数据集目录
- **CV_SEG_WEIGHT_DIR** 需要外部提供，存放 ResNetV1c-50 backbone 预训练权重
- **CV_SEG_LOGS_DIR** 用于保存训练日志、mmseg work-dir 和汇总结果 `eval_result.json`
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

---

## 目录结构约定

- `$CV_SEG_PROJECT_ROOT`: 项目根目录，默认结构如下：
  ```text
  $CV_SEG_PROJECT_ROOT/                 # = /workspace/code
  ├── batch_segmentation.sh             # 运行前需要存在；可由本 skill 的 scripts/batch_segmentation.sh 复制到此处
  ├── onedl-mmsegmentation/             # 可选；镜像内置源码时可不提供
  │   ├── configs/
  │   ├── tools/train.py
  │   └── ...
  └── onedl-mmcv/                       # 可选；pip install 后通常不需要源码目录
  ```

  `batch_segmentation.sh` 会自动查找 mmsegmentation 源码目录；该目录必须包含 `configs/` 和 `tools/train.py`。如果镜像只安装了 Python 包但没有保留源码配置文件，仍需通过 `$CV_SEG_PROJECT_ROOT/onedl-mmsegmentation` 或 `CV_SEG_MMSEG_DIR` 提供 mmsegmentation 源码目录。

- `$CV_SEG_WEIGHT_DIR`: 预训练权重目录，默认结构如下：
  ```text
  $CV_SEG_WEIGHT_DIR/                   # = /workspace/weight
  └── resnet50_v1c-2cccc1ad.pth
  ```

- `$CV_SEG_DATA_DIR`: Cityscapes 数据集目录，默认结构如下：
  ```text
  $CV_SEG_DATA_DIR/                     # = /workspace/datasets/cityscapes
  ├── leftImg8bit/
  └── gtFine/
  ```

- `$CV_SEG_LOGS_DIR`: 日志和结果目录，默认映射到 `/workspace/logs`。

---

## 支持的模型配置

默认运行：`fcn`，固定执行 `fp16,fp32` 两种精度，默认 GPU 数来自 task config。

支持模型：`fcn`、`deeplabv3`、`pspnet`、`apcnet`。

默认权重位于 `/workspace/weight/resnet50_v1c-2cccc1ad.pth`。当前镜像内的 `onedl-mmsegmentation/configs/_base_/datasets/cityscapes.py` 已配置为直接从 `/workspace/datasets/cityscapes` 读取数据集，不需要再创建 `data/cityscapes` 软链接。

---

## 依赖要求

Docker 镜像：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:cv
```

容器内需要具备 Python 3.10、PyTorch + CUDA、MMEngine、onedl-mmcv、onedl-mmsegmentation。

---

## 交互参数映射规则

| 用户表达 | 必须设置的变量 | 结果 |
|---------|----------------|------|
| `1卡`、`单卡`、`card_count=1` | `export CV_SEG_NGPU=1` | `--nproc_per_node=1`，日志目录包含 `_gpus1_`。 |
| `8卡`、`八卡`、`card_count=8` | `export CV_SEG_NGPU=8` | `--nproc_per_node=8`，日志目录包含 `_gpus8_`。 |
| `fcn` / `deeplabv3` / `pspnet` / `apcnet` | `export CV_SEG_MODELS=<model>` | 运行对应模型配置。 |

如果用户没有指定，默认值为：`CV_SEG_NGPU=1`、`CV_SEG_MODELS=fcn`。精度固定为 `fp16,fp32`，不支持通过用户输入或环境变量切换单独精度。

生成评测脚本时，如果 `/workspace/code/batch_segmentation.sh` 不存在，应先将 agent 预置的 `/workspace/scripts/batch_segmentation.sh` 复制到 `/workspace/code/batch_segmentation.sh`，再执行：

```bash
if [ ! -f /workspace/code/batch_segmentation.sh ] && [ -f /workspace/scripts/batch_segmentation.sh ]; then
  cp /workspace/scripts/batch_segmentation.sh /workspace/code/batch_segmentation.sh
  chmod +x /workspace/code/batch_segmentation.sh
fi
test -f /workspace/code/batch_segmentation.sh
bash /workspace/code/batch_segmentation.sh
```

---

## 第一阶段：容器启动

```bash
docker run --gpus all \
  --network host --ipc host --shm-size=16g \
  -it --name cv_seg_bench \
  -v $CV_SEG_PROJECT_ROOT:/workspace/code:rw \
  -v $CV_SEG_DATA_DIR:/workspace/datasets/cityscapes:ro \
  -v $CV_SEG_WEIGHT_DIR:/workspace/weight:ro \
  -v $CV_SEG_LOGS_DIR:/workspace/logs:rw \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:cv \
  bash
```

如果容器名已存在：`docker rm -f cv_seg_bench`。

**注意**：
- `CV_SEG_PROJECT_ROOT`、`CV_SEG_DATA_DIR`、`CV_SEG_WEIGHT_DIR` 必须在宿主机提供
- 容器启动后、执行评测前，如果 `/workspace/code/batch_segmentation.sh` 不存在，应将预置的 `/workspace/scripts/batch_segmentation.sh` 复制到 `/workspace/code/batch_segmentation.sh` 并添加执行权限
- 输出目录 `CV_SEG_LOGS_DIR` 不存在时需先在宿主机创建，或由 executor 创建后再挂载

---

## 第二阶段：容器中执行评测

### 步骤 1：选择模型与源码目录

容器内代码、数据、权重和日志路径已通过卷挂载固定（详见[环境变量定义](#环境变量定义)）。如果镜像已内置 mmsegmentation 源码，通常无需额外设置目录环境变量。

```bash
# 选择要测试的模型和 GPU 数
MODEL_NAME="fcn"              # 可选: fcn, deeplabv3, pspnet, apcnet
GPU_NUM="${CARD_COUNT:-1}"    # 默认使用 task config 的 card_count，未提供时为 1
# 精度固定执行 fp16,fp32，不需要设置精度参数

# 如 mmsegmentation 源码在镜像内其他路径，可显式指定
# export CV_SEG_MMSEG_DIR=/opt/onedl-mmsegmentation
```

### 配置文件说明

**关键路径**：
```text
/workspace/code/onedl-mmsegmentation   # mmsegmentation 项目目录；也可为镜像内置源码路径
/workspace/code/onedl-mmcv             # 可选；mmcv 已安装时不需要源码目录
/workspace/datasets/cityscapes         # Cityscapes 数据集目录
/workspace/weight                      # 预训练权重目录
/workspace/logs                        # 训练日志、work-dir 和结果 JSON 输出目录
/workspace/code/batch_segmentation.sh  # 批量评测脚本；代码/镜像没有时由 skill scripts/batch_segmentation.sh 提供
```

**注意**：
- `batch_segmentation.sh` 可直接放在 `/workspace/code/batch_segmentation.sh`；如果运行环境中不存在，则从 agent 预置的 `/workspace/scripts/batch_segmentation.sh` 复制到该路径
- 脚本会自动查找 mmsegmentation 源码目录；数据集路径由镜像内 `configs/_base_/datasets/cityscapes.py` 直接指向 `/workspace/datasets/cityscapes`
- `onedl-mmcv` 已通过 `pip install .` 安装到镜像环境时，不需要提供 `/workspace/code/onedl-mmcv`

### 步骤 2：执行训练评测

运行批量评测脚本，训练日志和结果文件将保存至 `/workspace/logs`：

```bash
mkdir -p /workspace/logs
if [ ! -f /workspace/code/batch_segmentation.sh ] && [ -f /workspace/scripts/batch_segmentation.sh ]; then
  cp /workspace/scripts/batch_segmentation.sh /workspace/code/batch_segmentation.sh
  chmod +x /workspace/code/batch_segmentation.sh
fi
test -f /workspace/code/batch_segmentation.sh

export CV_SEG_NGPU="$GPU_NUM"
export CV_SEG_MODELS="$MODEL_NAME"
export CV_SEG_RUN_MARKER=/tmp/cv_seg_run_marker.$(date +%s).$$
touch "$CV_SEG_RUN_MARKER"

# 如 mmsegmentation 源码在镜像内其他路径，可显式指定
# export CV_SEG_MMSEG_DIR=/opt/onedl-mmsegmentation
bash /workspace/code/batch_segmentation.sh
```

上述指令的默认行为：
- 运行 `fcn` 模型
- 固定分别执行 `fp16` 和 `fp32` 两种精度
- 使用 `GPU_NUM` 指定的 GPU 数启动分布式训练
- 训练输出目录为 `/workspace/logs/${MODEL_NAME}_gpus${GPU_NUM}_${PRECISION}`
- 所有模型和精度组合的结构化结果统一写入 `/workspace/logs/eval_result.json`

**验证执行结果**：
```bash
# 查看最新训练日志
PRECISION="fp16"
LOG_DIR=/workspace/logs/${MODEL_NAME}_gpus${GPU_NUM}_${PRECISION}
LATEST_LOG=$(find "$LOG_DIR" -type f -name "*.log" -newer "$CV_SEG_RUN_MARKER" -printf "%T@ %p
" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
test -n "$LATEST_LOG"
tail -50 "$LATEST_LOG"

# 检查结果文件
cat /workspace/logs/eval_result.json
```

---

## 评测输出产物

训练日志和汇总结果均输出到 `/workspace/logs`。

| 文件路径 | 描述 |
| :--- | :--- |
| `/workspace/logs/${MODEL_NAME}_gpus${GPU_NUM}_${PRECISION}/<timestamp>/<timestamp>.log` | MMEngine 训练日志，包含 `AVG_ITER_TIME`、`DATA`、`OP` 指标 |
| `/workspace/logs/eval_result.json` | 本次运行所有模型、GPU 数和精度组合的汇总结果 |

**注意**：评测固定执行 `fp16,fp32` 两种精度；当一次运行多个模型时，所有组合的结果都写入同一个 `eval_result.json`，不再生成额外的 `*_result.json` 文件。

**汇总结果示例**：

```json
{
  "status": "success",
  "task": "cv_segmentation",
  "model": "fcn,pspnet",
  "gpu_count": 1,
  "precisions": ["fp16", "fp32"],
  "results": {
    "fcn_gpus1_fp16": {
      "model": "fcn",
      "gpu_count": 1,
      "precision": "fp16",
      "log": "/workspace/logs/fcn_gpus1_fp16/<timestamp>/<timestamp>.log",
      "avg_iter_time": 0.1032,
      "data_time": 0.0067,
      "op_time": 0.0965
    },
    "pspnet_gpus1_fp32": {
      "model": "pspnet",
      "gpu_count": 1,
      "precision": "fp32",
      "log": "/workspace/logs/pspnet_gpus1_fp32/<timestamp>/<timestamp>.log",
      "avg_iter_time": 0.1190,
      "data_time": 0.0071,
      "op_time": 0.1119
    }
  }
}
```

**验证输出文件**：

```bash
cat /workspace/logs/eval_result.json
```

---

## 关键性能指标

训练日志中应包含 AVG_ITER_TIME 输出，例如：

```text
=== AVG_ITER_TIME: 0.1032s | DATA: 0.0067s | OP: 0.0965s ===
```

`eval_result.json` 会记录每个模型、GPU 数和精度组合的详细性能数据。

#### 指标说明

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `avg_iter_time` | 平均每轮训练迭代耗时，核心训练性能指标，数值越低越好 |
| 性能（辅助） | `data_time` | 平均数据读取和准备耗时 |
| 性能（辅助） | `op_time` | 平均算子和反向传播等计算耗时 |
| 元信息 | `model` / `gpu_count` / `precision` | 模型、GPU 数和精度配置，用于区分不同测试组合 |

#### 指标采集

优先读取汇总结果文件：

```bash
python3 -c "
import json
path='/workspace/logs/eval_result.json'
print(json.dumps(json.load(open(path, 'r', encoding='utf-8')), indent=2, ensure_ascii=False))
"
```

如需从训练日志临时提取，可使用：

```bash
MODEL_NAME=${MODEL_NAME:-fcn}
GPU_NUM=${GPU_NUM:-${CARD_COUNT:-1}}
PRECISION=${PRECISION:-fp16}
LOG_DIR=/workspace/logs/${MODEL_NAME}_gpus${GPU_NUM}_${PRECISION}
LOG=$(find "$LOG_DIR" -type f -name "*.log" -printf "%T@ %p
" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2-)
test -n "$LOG"
grep "AVG_ITER_TIME" "$LOG" | tail -1
```

---

## 常见问题

1. **找不到 `batch_segmentation.sh`**：检查 `/workspace/code/batch_segmentation.sh` 是否存在；如果不存在，应将 agent 预置的 `/workspace/scripts/batch_segmentation.sh` 复制到 `/workspace/code/batch_segmentation.sh` 并添加执行权限。
2. **找不到 mmsegmentation 源码目录**：确认镜像内或挂载目录中存在包含 `configs/` 和 `tools/train.py` 的 mmsegmentation 源码目录；必要时设置 `CV_SEG_MMSEG_DIR=/path/to/onedl-mmsegmentation`。`onedl-mmcv` 已安装时不需要源码目录。
3. **预训练权重加载失败**：检查 `/workspace/weight/resnet50_v1c-2cccc1ad.pth` 是否存在。
4. **数据集路径错误**：检查 `/workspace/datasets/cityscapes` 是否包含 Cityscapes 数据集；当前镜像内 `configs/_base_/datasets/cityscapes.py` 应直接从该路径读取。
5. **没有 `AVG_ITER_TIME`**：确认项目配置已启用 AVG_ITER_TIME 日志输出，且不要直接读取历史最新日志，应使用 `CV_SEG_RUN_MARKER`。
6. **GPU 数不匹配**：导出 `CV_SEG_NGPU=<card_count>`，并确认容器可见 GPU 数。
