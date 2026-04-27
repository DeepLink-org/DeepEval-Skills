---
name: h200-cv-detection
description: "H200芯片-CV场景-目标检测任务的评测流程。用于指导executor在 H200 环境下完成 onedl-mmdetection 检测模型训练环境准备、批量脚本执行、性能采集和结果分析。适用于 faster_rcnn、mask_rcnn、cascade_rcnn、retinanet、yolov3、fcos、ssd300、centernet、solo、swin_mask_rcnn 等模型。"
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 H200 上跑 CV detection 模型测试"
- "帮我测试 faster rcnn / mask rcnn / yolov3 检测训练性能"
- "在 H200 上跑 mmdetection benchmark"
- "帮我批量测试 CV detection 模型 FP32/FP16 性能"

---

### 支持的模型配置

**当前支持模型**：
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
- 基于 `onedl-mmdetection` 的检测模型训练/性能测试
- 使用 `custom_iter_timer_hook.py` 采集训练迭代耗时
- 批量执行多模型基准测试
- 对不同 backbone 预训练权重进行自动注入

**硬件要求**：
- 2 节点，共 16 张 NVIDIA H200 GPU
- 当前脚本默认每节点 8 卡
- 适合 PyTorch 2.x + CUDA 11.8+/12.x 环境

---

### 代码与环境要求

该 skill 基于 PyTorch 2.x 适配方案，检测模型使用 `onedl-mmdetection`，需提前安装 `onedl-mmcv` 与 `onedl-mmdetection`。
#### 1. 安装 onedl-mmcv

```bash
git clone https://github.com/VBTI-development/onedl-mmcv.git
cd onedl-mmcv
git checkout 55264919c4651084882c2ba6f888834aee9a4627

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;9.0a"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
python -m pip install -e . -v --no-build-isolation

cd ..
```

#### 2. 安装 onedl-mmdetection

```bash
git clone https://github.com/VBTI-development/onedl-mmdetection.git
cd onedl-mmdetection
git checkout c43b35b7553db279de8609a321cfd7fa0b733492
pip install -e .
cd ..
```

#### 3. 设置环境变量

```bash
export MMDET_PATH=/path/to/onedl-mmdetection
export MMCV_PATH=/path/to/onedl-mmcv
export PYTHONPATH=${MMPRE_PATH}:${MMCV_PATH}:$PYTHONPATH

---

### 自定义性能统计 Hook

为了在 PyTorch 2.x / mmengine 环境下采集稳定的迭代耗时，需使用：

```bash
tools/custom_iter_timer_hook.py
```
**放置位置**：将 `custom_iter_timer_hook.py` 拷贝到 `onedl-mmdetection/` 根目录下，确保 Python 可导入。
其作用：
- 在指定迭代区间内统计平均迭代时间
- 分离 `data_time` 与 `op_time`
- 到达设定结束迭代后主动退出，便于 benchmark 自动化

通常需要在配置文件中加入：

```python
custom_imports = dict(imports=['custom_iter_timer_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='CustomIterTimerHook', begin_iter=200, end_iter=500)]
default_hooks = dict(timer=None, checkpoint=None)
```

---
### 模型与权重要求

当前批量检测脚本会为不同模型自动注入 backbone 初始化权重，权重目录默认位于：

```bash
./models/weight
```

当前脚本依赖以下权重文件：
- `resnet50-0676ba61.pth`
- `resnet50_msra-5891d200.pth`
- `darknet53-a628ea1b.pth`
- `vgg16_caffe-292e1171.pth`
- `resnet18-f37072fd.pth`
- `swin_tiny_patch4_window7_224.pth`

如果这些权重不存在，相关检测模型会启动失败。

---

### 配置文件与模型列表

检测模型配置列表位于：

```bash
scripts/configs_path
```

当前 detection 批量测试脚本内置以下配置：
- `configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`
- `configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py`
- `configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py`
- `configs/retinanet/retinanet_r50_fpn_1x_coco.py`
- `configs/yolo/yolov3_d53_8xb8-320-273e_coco.py`
- `configs/fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py`
- `configs/ssd/ssd300_coco.py`
- `configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py`
- `configs/solo/decoupled-solo_r50_fpn_1x_coco.py`
- `configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py`

---

### 批量测试脚本

标准批量测试脚本位置：

```bash
scripts/batch_detection.sh
```

执行方式：

```bash
bash scripts/batch_detection.sh
```

当前脚本默认行为：
- 进入 `./models/onedl-mmdetection`
- 设置 `MMDET_PATH`、`MMCV_PATH`、`PYTHONPATH`
- 设置 `WEIGHT_DIR=./models/weight`
- 使用 `torch.distributed.launch` 进行 2 节点 16 卡训练
- 默认每节点 8 卡
- 当前脚本默认跑 `fp32`，默认对每个模型执行 `fp32` 和 `fp16` 两轮测试
- 输出目录位于 `work_dirs/${MODEL_NAME}_gpus${NGPU}_${PRECISION}`
- 通过 `--cfg-options` 动态切换 `optim_wrapper.type=AmpOptimWrapper` 或 `OptimWrapper`
涉及关键变量：

```bash
NODE_COUNT
NODE_RANK
MASTER_ADDR
MASTER_PORT
```

执行前应确保这些分布式变量已正确设置。

---

### 数据集要求

检测模型通常依赖 COCO 数据集。

需要按照 `onedl-mmdetection` 的数据集要求准备数据，并保证仓库内存在正确的数据目录或软链接。可参考 README 中 OpenMMLab 数据准备方式。

如果数据路径未正确配置，训练脚本会直接报错。

---

### 性能监控

训练完成后，从日志文件中读取 `AVG_ITER_TIME` 作为关键性能指标。

**日志路径**：
```
./models/onedl-mmdetection/work_dirs/<MODEL_NAME>_gpus<NGPU>_<PRECISION>/<TIMESTAMP>/<TIMESTAMP>.log
```

**日志格式示例**：
```
2026/03/18 11:09:16 - mmengine - INFO - === AVG_ITER_TIME: 0.0951s | DATA: 0.0036s | OP: 0.0915s ===
```

**提取方式**：使用以下命令从日志中提取 AVG_ITER_TIME：
```bash
grep "AVG_ITER_TIME" <LOG_PATH> | tail -1 | grep -oP "AVG_ITER_TIME: \K[0-9.]+"
```

该值即为模型训练的平均每次迭代耗时（秒），数值越小性能越好。

---

### 常见问题

1. **脚本启动失败**
   - 检查 `onedl-mmdetection` 和 `onedl-mmcv` 是否已正确安装
   - 检查 `PYTHONPATH` 是否包含对应仓库路径

2. **分布式训练无法建立连接**
   - 检查 `NODE_COUNT`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 是否正确
   - 检查节点间网络互通和 NCCL 环境

3. **预训练权重加载失败**
   - 检查配置路径是否存在
   - 检查 `./models/weight` 下对应权重文件是否存在
   - 检查 config 中 backbone 初始化方式是否允许从 checkpoint 加载
   - 检查 `custom_iter_timer_hook.py` 是否已放置到`onedl-mmdetection/` 可导入位置

4. **无法统计性能数据**
   - 检查 config 中是否正确导入 `CustomIterTimerHook`
   - 检查是否关闭默认 timer hook 避免冲突

5. **数据集报错或找不到数据**
   - 检查 COCO2017 数据目录或软链接是否按 `mmdetection` 要求准备完成
