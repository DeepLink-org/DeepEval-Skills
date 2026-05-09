---
name: nvidia-cv-detection
description: NVIDIA GPU 上目标检测模型训练性能评测技能。基于 onedl-mmdetection，用于指导 executor 完成容器启动、批量检测训练脚本执行、日志采集与性能指标分析。适用于 faster_rcnn、mask_rcnn、cascade_rcnn、retinanet、yolov3、fcos、ssd300、centernet、solo、swin_mask_rcnn 等模型。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 CV detection 模型测试"
- "帮我测试 faster rcnn / mask rcnn / yolov3 检测训练性能"
- "在 nvidia 上跑 mmdetection benchmark"
- "帮我批量测试 CV detection 模型 FP32/FP16 性能"
- "采集 detection 模型 AVG_ITER_TIME"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 数据集目录：`/data/datasets`
- 代码挂载目录：`/workspace/code`
- 训练日志输出目录：`/workspace/logs`

---

### 支持的模型配置

**当前支持模型**（共 10 个）：
- `faster_rcnn`
- `mask_rcnn`
- `cascade_rcnn`
- `retinanet`
- `yolov3`
- `fcos`
- `ssd300`
- `centernet`
- `solo`
- `swin_mask_rcnn`

**当前支持任务**：
- 基于 `onedl-mmdetection` 的检测模型训练性能测试
- 每个模型分别测试 FP32 和 FP16 精度
- 使用 `custom_iter_timer_hook.py` 采集训练迭代耗时

**硬件要求**：
- 1 节点，8 张 NVIDIA GPU

---

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:mm_all
```

容器内已预装 PyTorch 2.7.0+cu128、TorchVision 0.22.0、MMEngine 0.10.9、onedl-mmcv、onedl-mmdetection 等。

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型权重路径**：
```bash
/data/models/weight/   # backbone 预训练权重目录
```

依赖以下权重文件：
- `resnet50-0676ba61.pth`
- `resnet50_msra-5891d200.pth`
- `darknet53-a628ea1b.pth`
- `vgg16_caffe-292e1171.pth`
- `resnet18-f37072fd.pth`
- `swin_tiny_patch4_window7_224.pth`

**数据集路径**：
```bash
/data/datasets/coco/   # COCO 2017 数据集
```

如模型权重或数据集路径发生变化，应同步修改 `batch_detection.sh` 中的相关路径。

**自定义 Hook**：
- `custom_iter_timer_hook.py` 位于 skill 目录下的 `tools/custom_iter_timer_hook.py`，部署时拷贝到 `onedl-mmdetection/` 根目录：
  ```bash
  cp tools/custom_iter_timer_hook.py /workspace/code/onedl-mmdetection/
  ```
- 用于在指定迭代区间（begin_iter=200, end_iter=500）统计平均迭代耗时。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name mmdet_benchmark \
  --gpus all \
  --shm-size=128g \
  -v /data/models:/data/models \
  -v /data/datasets:/data/datasets \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:mm_all \
  bash
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行训练脚本；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=128g`**：避免大 batch 数据加载时共享内存不足。
- 若已存在同名容器，需先执行 `docker rm -f mmdet_benchmark` 或更换 `--name`。

---

### 训练脚本

批量测试脚本位于 skill 目录下的 `scripts/batch_detection.sh`，部署时拷贝到代码挂载目录：

```bash
cp scripts/batch_detection.sh /workspace/code/
```

执行方式：

```bash
cd /workspace/code/onedl-mmdetection
bash ../batch_detection.sh 2>&1 | tee /workspace/logs/detection.log
```

当前脚本默认行为：
- 自动遍历 10 个检测模型
- 每个模型分别执行 FP32 和 FP16 两轮测试
- 8 卡分布式训练（`torch.distributed.launch --nproc_per_node=8`）
- 通过 `--cfg-options` 动态切换 `optim_wrapper.type=AmpOptimWrapper`（FP16）或 `OptimWrapper`（FP32）
- 自动为不同模型注入对应的 backbone 预训练权重
- 输出目录位于 `work_dirs/${MODEL_NAME}_gpus8_${PRECISION}`
- **不要修改** `begin_iter` (=200) / `end_iter` (=500) 等 CustomIterTimerHook 参数，否则与基线指标不可比

---

### 关键性能指标

训练日志中包含 `CustomIterTimerHook` 输出的迭代耗时汇总行，例如：

```text
2026/03/18 11:09:16 - mmengine - INFO - === AVG_ITER_TIME: 0.0951s | DATA: 0.0036s | OP: 0.0915s ===
```

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `AVG_ITER_TIME` | 平均迭代耗时（秒），核心吞吐指标 |
| 性能（辅助） | `DATA` | 数据加载耗时 |
| 性能（辅助） | `OP` | 纯算子计算耗时 |

**采集命令**（将 `LOG` 替换为实际日志路径，如 `/workspace/logs/detection.log`）：

```bash
# 核心：AVG_ITER_TIME
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "AVG_ITER_TIME: \K[0-9.]+"

# 辅助：分别提取 DATA 和 OP
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "DATA: \K[0-9.]+"
grep "AVG_ITER_TIME" "$LOG" | tail -1 | grep -oP "OP: \K[0-9.]+"
```

---

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f mmdet_benchmark` 后重试，或改用新容器名。

2. **找不到 `batch_detection.sh`**
   - 检查 `/workspace/code` 挂载是否包含该脚本。

3. **预训练权重加载失败**
   - 检查 `/data/models/weight/` 下对应权重文件是否存在。
   - 检查 config 中 backbone 初始化方式是否允许从 checkpoint 加载。

4. **数据集报错或找不到数据**
   - 检查 `/data/datasets/coco/` 是否包含 COCO 2017 数据集（`annotations/`、`train2017/`、`val2017/`）。

5. **无法统计性能数据**
   - 检查 `custom_iter_timer_hook.py` 是否已放置到 `onedl-mmdetection/` 根目录下。
   - 检查配置文件中是否正确导入 `CustomIterTimerHook`。

6. **共享内存不足**
   - 已使用 `--shm-size=128g`；若仍报错，检查数据加载 `num_workers` 与 Docker `--shm-size`。

7. **`grep -P` 不可用**
   - 换用支持 Perl 正则的环境执行命令，或将日志行复制到本地用 `python -c` 解析。
