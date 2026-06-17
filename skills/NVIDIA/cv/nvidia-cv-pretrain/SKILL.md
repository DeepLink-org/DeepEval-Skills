---
name: nvidia-cv-pretrain
description: NVIDIA GPU 上 CV 预训练/分类模型训练性能评测技能。基于 /workspace/code 中的 onedl-mmpretrain 和 batch_pretrain.sh，用于指导 executor 完成容器启动、脚本执行、日志采集与 AVG_ITER_TIME 性能指标分析。默认适配 resnet50 + ImageNet，可通过模型参数扩展到 inception_v3、seresnet50、mobilenet_v2、shufflenet_v2、densenet121、swin_large、efficientnet_b2 等模型。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 CV 预训练模型测试"
- "帮我测试 resnet50 / inception_v3 / swin-large 分类训练性能"
- "在 nvidia 上跑 mmpretrain 分类模型 benchmark"
- "帮我批量测试 CV pretrain 模型性能"
- "采集 pretrain 模型 AVG_ITER_TIME"

---

## 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `CV_PRE_PROJECT_ROOT` | `/workspace/code` | 是 | 项目根目录，需外部提供，包含 `onedl-mmpretrain` 和 `onedl-mmcv` 源码；`batch_pretrain.sh` 可由 skill 资源复制到该目录 |
| `CV_PRE_DATA_DIR` | `/workspace/datasets/imagenet` | 是 | ImageNet 数据集目录 |
| `CV_PRE_LOGS_DIR` | `/workspace/logs` | 否 | 训练日志、work-dir 和汇总结果 `eval_result.json` 输出目录 |

**说明**：
- **CV_PRE_PROJECT_ROOT** 需要外部提供，挂载到 `/workspace/code`，至少应包含 `onedl-mmpretrain` 和 `onedl-mmcv`；`batch_pretrain.sh` 可由 agent 从 `/workspace/scripts/batch_pretrain.sh` 复制到 `/workspace/code/batch_pretrain.sh`
- **CV_PRE_DATA_DIR** 需要外部提供，指向 ImageNet 数据集目录
- **CV_PRE_LOGS_DIR** 用于保存训练日志、mmpretrain work-dir 和汇总结果 `eval_result.json`
- 表格中的"映射目录"列指明了容器启动时 `-v` 参数的挂载路径，即宿主机路径映射到容器内的路径

---

## 目录结构约定

- `$CV_PRE_PROJECT_ROOT`: 项目根目录，默认结构如下：
  ```text
  $CV_PRE_PROJECT_ROOT/                 # = /workspace/code
  ├── batch_pretrain.sh                 # 运行前需要存在；可由本 skill 的 scripts/batch_pretrain.sh 复制到此处
  ├── onedl-mmpretrain/                 # 可选；镜像内置源码时可不提供
  │   ├── configs/
  │   ├── tools/train.py
  │   └── ...
  └── onedl-mmcv/                       # 可选；pip install 后通常不需要源码目录
  ```

  `batch_pretrain.sh` 会自动查找 mmpretrain 源码目录；该目录必须包含 `configs/` 和 `tools/train.py`。如果镜像只安装了 Python 包但没有保留源码配置文件，仍需通过 `$CV_PRE_PROJECT_ROOT/onedl-mmpretrain` 或 `CV_PRE_MMPRE_DIR` 提供 mmpretrain 源码目录。

- `$CV_PRE_DATA_DIR`: ImageNet 数据集目录，默认结构如下：
  ```text
  $CV_PRE_DATA_DIR/                     # = /workspace/datasets/imagenet
  ├── train/
  └── val/
  ```

- `$CV_PRE_LOGS_DIR`: 日志和结果目录，默认映射到 `/workspace/logs`。

---

## 支持的模型配置

默认运行：`resnet50`，固定执行 `fp16,fp32` 两种精度，默认 GPU 数来自 task config。

支持模型：`resnet50`、`inception_v3`、`seresnet50`、`mobilenet_v2`、`shufflenet_v2`、`densenet121`、`swin_large`、`efficientnet_b2`。

当前镜像内的 `onedl-mmpretrain/configs/_base_/datasets/imagenet_bs32.py` 已配置为直接从 `/workspace/datasets/imagenet` 读取数据集，不需要再创建 `data/imagenet` 软链接。

---

## 依赖要求

Docker 镜像：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-cv:latest
```

容器内需要具备 Python 3.10、PyTorch + CUDA、MMEngine、onedl-mmcv、onedl-mmpretrain。

---

## 交互参数映射规则

| 用户表达 | 必须设置的变量 | 结果 |
|---------|----------------|------|
| `1卡`、`单卡`、`card_count=1` | `export CV_PRE_NGPU=1` | `--nproc_per_node=1`，日志目录包含 `_gpus1_`。 |
| `8卡`、`八卡`、`card_count=8` | `export CV_PRE_NGPU=8` | `--nproc_per_node=8`，日志目录包含 `_gpus8_`。 |
| `resnet50` / `swin_large` / `efficientnet_b2` 等 | `export CV_PRE_MODELS=<model>` | 运行对应模型配置。 |

如果用户没有指定，默认值为：`CV_PRE_NGPU=1`、`CV_PRE_MODELS=resnet50`。精度固定为 `fp16,fp32`，不支持通过用户输入或环境变量切换单独精度。

生成评测脚本时，如果 `/workspace/code/batch_pretrain.sh` 不存在，应先将 agent 预置的 `/workspace/scripts/batch_pretrain.sh` 复制到 `/workspace/code/batch_pretrain.sh`，再执行：

```bash
if [ ! -f /workspace/code/batch_pretrain.sh ] && [ -f /workspace/scripts/batch_pretrain.sh ]; then
  cp /workspace/scripts/batch_pretrain.sh /workspace/code/batch_pretrain.sh
  chmod +x /workspace/code/batch_pretrain.sh
fi
test -f /workspace/code/batch_pretrain.sh
bash /workspace/code/batch_pretrain.sh
```

---

## 第一阶段：容器启动

```bash
docker run --gpus all \
  --network host --ipc host --shm-size=128g \
  -it --name cv_pretrain_bench \
  -v $CV_PRE_PROJECT_ROOT:/workspace/code:rw \
  -v $CV_PRE_DATA_DIR:/workspace/datasets/imagenet:ro \
  -v $CV_PRE_LOGS_DIR:/workspace/logs:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-cv:latest \
  bash
```

如果容器名已存在：`docker rm -f cv_pretrain_bench`。

**注意**：
- `CV_PRE_PROJECT_ROOT`、`CV_PRE_DATA_DIR` 必须在宿主机提供
- 容器启动后、执行评测前，如果 `/workspace/code/batch_pretrain.sh` 不存在，应将预置的 `/workspace/scripts/batch_pretrain.sh` 复制到 `/workspace/code/batch_pretrain.sh` 并添加执行权限
- 输出目录 `CV_PRE_LOGS_DIR` 不存在时需先在宿主机创建，或由 executor 创建后再挂载

---

## 第二阶段：容器中执行评测

### 步骤 1：选择模型与源码目录

容器内代码、数据和日志路径已通过卷挂载固定（详见[环境变量定义](#环境变量定义)）。如果镜像已内置 mmpretrain 源码，通常无需额外设置目录环境变量。

```bash
# 选择要测试的模型和 GPU 数
MODEL_NAME="resnet50"         # 可选: resnet50, inception_v3, seresnet50, mobilenet_v2, shufflenet_v2, densenet121, swin_large, efficientnet_b2
GPU_NUM="${CARD_COUNT:-1}"    # 默认使用 task config 的 card_count，未提供时为 1
# 精度固定执行 fp16,fp32，不需要设置精度参数

# 如 mmpretrain 源码在镜像内其他路径，可显式指定
# export CV_PRE_MMPRE_DIR=/opt/onedl-mmpretrain
```

### 配置文件说明

**关键路径**：
```text
/workspace/code/onedl-mmpretrain       # mmpretrain 项目目录；也可为镜像内置源码路径
/workspace/code/onedl-mmcv             # 可选；mmcv 已安装时不需要源码目录
/workspace/datasets/imagenet           # ImageNet 数据集目录
/workspace/logs                        # 训练日志、work-dir 和结果 JSON 输出目录
/workspace/code/batch_pretrain.sh      # 批量评测脚本；代码/镜像没有时由 skill scripts/batch_pretrain.sh 提供
```

**注意**：
- `batch_pretrain.sh` 可直接放在 `/workspace/code/batch_pretrain.sh`；如果运行环境中不存在，则从 agent 预置的 `/workspace/scripts/batch_pretrain.sh` 复制到该路径
- 脚本会自动查找 mmpretrain 源码目录；数据集路径由镜像内 `configs/_base_/datasets/imagenet_bs32.py` 直接指向 `/workspace/datasets/imagenet`
- `onedl-mmcv` 已通过 `pip install .` 安装到镜像环境时，不需要提供 `/workspace/code/onedl-mmcv`

### 步骤 2：执行训练评测

运行批量评测脚本，训练日志和结果文件将保存至 `/workspace/logs`：

```bash
mkdir -p /workspace/logs
if [ ! -f /workspace/code/batch_pretrain.sh ] && [ -f /workspace/scripts/batch_pretrain.sh ]; then
  cp /workspace/scripts/batch_pretrain.sh /workspace/code/batch_pretrain.sh
  chmod +x /workspace/code/batch_pretrain.sh
fi
test -f /workspace/code/batch_pretrain.sh

export CV_PRE_NGPU="$GPU_NUM"
export CV_PRE_MODELS="$MODEL_NAME"
export CV_PRE_RUN_MARKER=/tmp/cv_pre_run_marker.$(date +%s).$$
touch "$CV_PRE_RUN_MARKER"

# 如 mmpretrain 源码在镜像内其他路径，可显式指定
# export CV_PRE_MMPRE_DIR=/opt/onedl-mmpretrain
bash /workspace/code/batch_pretrain.sh
```

上述指令的默认行为：
- 运行 `resnet50` 模型
- 固定分别执行 `fp16` 和 `fp32` 两种精度
- 使用 `GPU_NUM` 指定的 GPU 数启动分布式训练
- 训练输出目录为 `/workspace/logs/${MODEL_NAME}_gpus${GPU_NUM}_${PRECISION}`
- 所有模型和精度组合的结构化结果统一写入 `/workspace/logs/eval_result.json`

**验证执行结果**：
```bash
# 查看最新训练日志
PRECISION="fp16"
LOG_DIR=/workspace/logs/${MODEL_NAME}_gpus${GPU_NUM}_${PRECISION}
LATEST_LOG=$(find "$LOG_DIR" -type f -name "*.log" -newer "$CV_PRE_RUN_MARKER" -printf "%T@ %p
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
  "task": "cv_pretrain",
  "model": "resnet50,swin_large",
  "gpu_count": 1,
  "precisions": ["fp16", "fp32"],
  "results": {
    "resnet50_gpus1_fp16": {
      "model": "resnet50",
      "gpu_count": 1,
      "precision": "fp16",
      "log": "/workspace/logs/resnet50_gpus1_fp16/<timestamp>/<timestamp>.log",
      "avg_iter_time": 0.0387,
      "data_time": 0.0005,
      "op_time": 0.0382
    },
    "swin_large_gpus1_fp32": {
      "model": "swin_large",
      "gpu_count": 1,
      "precision": "fp32",
      "log": "/workspace/logs/swin_large_gpus1_fp32/<timestamp>/<timestamp>.log",
      "avg_iter_time": 0.2210,
      "data_time": 0.0045,
      "op_time": 0.2165
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
=== AVG_ITER_TIME: 0.0387s | DATA: 0.0005s | OP: 0.0382s ===
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
MODEL_NAME=${MODEL_NAME:-resnet50}
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

1. **找不到 `batch_pretrain.sh`**：检查 `/workspace/code/batch_pretrain.sh` 是否存在；如果不存在，应将 agent 预置的 `/workspace/scripts/batch_pretrain.sh` 复制到 `/workspace/code/batch_pretrain.sh` 并添加执行权限。
2. **找不到 mmpretrain 源码目录**：确认镜像内或挂载目录中存在包含 `configs/` 和 `tools/train.py` 的 mmpretrain 源码目录；必要时设置 `CV_PRE_MMPRE_DIR=/path/to/onedl-mmpretrain`。`onedl-mmcv` 已安装时不需要源码目录。
3. **数据集路径错误**：检查 `/workspace/datasets/imagenet` 是否包含 ImageNet 数据集；当前镜像内 `configs/_base_/datasets/imagenet_bs32.py` 应直接从该路径读取。
4. **没有 `AVG_ITER_TIME`**：确认项目配置已启用 AVG_ITER_TIME 日志输出，且不要直接读取历史最新日志，应使用 `CV_PRE_RUN_MARKER`。
5. **GPU 数不匹配**：导出 `CV_PRE_NGPU=<card_count>`，并确认容器可见 GPU 数。
