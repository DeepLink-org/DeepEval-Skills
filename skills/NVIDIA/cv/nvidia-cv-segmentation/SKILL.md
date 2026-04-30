---
name: nvidia-cv-segmentation
description: "nvidia芯片-CV场景-分割任务的评测流程。用于指导executor在 nvidia 环境下完成 onedl-mmsegmentation 分割模型训练环境准备、批量脚本执行、性能采集和结果分析。适用于 deeplabv3、fcn、pspnet、apcnet 等模型。"
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 CV 分割模型测试"
- "帮我测试 deeplabv3 / fcn / pspnet / apcnet 分割训练性能"
- "在 nvidia 上跑 mmsegmentation 分割模型 benchmark"
- "帮我批量测试 CV segmentation 模型 FP32/FP16 性能"

---

### 支持的模型配置

**当前支持模型**：
- `deeplabv3`
- `fcn`
- `pspnet`
- `apcnet`

**当前支持任务**：
- 基于 `onedl-mmsegmentation` 的分割模型训练/性能测试
- 使用 `custom_iter_timer_hook.py` 采集训练迭代耗时
- 批量执行多模型基准测试（FP32 / FP16）

**硬件要求**：
- 2 节点，共 16 张 NVIDIA GPU
- 当前脚本默认每节点 8 卡
- 适合 PyTorch 2.x + CUDA 11.8+/12.x 环境

---

### 代码与环境要求

该 skill 基于 PyTorch 2.x 适配方案，分割模型使用 `onedl-mmsegmentation`，需提前安装 `onedl-mmcv` 与 `onedl-mmsegmentation`。

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

#### 2. 安装 onedl-mmsegmentation

```bash
git clone https://github.com/VBTI-development/onedl-mmsegmentation.git
cd onedl-mmsegmentation
git checkout f2dc1d0758593eaec3b257ed185fea35c86e6d26
pip install -e .
cd ..
```

#### 3. 设置环境变量
```bash
export MMSEG_PATH=/path/to/onedl-mmsegmentation                                                                                                                                                          
export MMCV_PATH=/path/to/onedl-mmcv                                                                                                                                                                     
export PYTHONPATH=${MMSEG_PATH}:${MMCV_PATH}:$PYTHONPATH   
```

---

### 自定义性能统计 Hook

为了在 PyTorch 2.x / mmengine 环境下采集稳定的迭代耗时，需使用：

```bash
tools/custom_iter_timer_hook.py
```

**放置位置**：将 `custom_iter_timer_hook.py` 拷贝到 `onedl-mmsegmentation/` 根目录下，确保 Python 可导入。

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
- `resnet50_v1c-2cccc1ad.pth`

如果权重不存在，相关检测模型会启动失败。

---
### 配置文件列表

分割模型配置列表位于：

```bash
scripts/configs_path
```

当前包含：
- `configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py`
- `configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py`
- `configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py`
- `configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py`

---

### 批量测试脚本

标准批量测试脚本位置：

```bash
scripts/batch_segmentation.sh
```

执行方式：

```bash
bash scripts/batch_segmentation.sh
```

当前脚本默认行为：
- 进入 `./models/onedl-mmsegmentation`
- 设置 `MMSEG_PATH`、`MMCV_PATH`、`PYTHONPATH`
- 设置 `WEIGHT_DIR=./models/weight`
- 使用 `torch.distributed.launch` 进行 2 节点 16 卡训练
- 默认每节点 8 卡
- 当前脚本默认跑 `fp32`，每个模型分别测试 `fp32` 和 `fp16`
- 所有模型共用 ResNetV1c-50 backbone 权重：`./models/weight/resnet50_v1c-2cccc1ad.pth`
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

分割模型通常依赖 Cityscapes 数据集。

需要按照 `onedl-mmsegmentation` 的数据集要求准备数据，并保证仓库内存在正确的数据目录或软链接。可参考 README 中 OpenMMLab 数据准备方式。

如果数据路径未正确配置，训练脚本会直接报错。

---

### 性能监控

训练完成后，从日志文件中读取 `AVG_ITER_TIME` 作为关键性能指标。

**日志路径**：
```
./models/onedl-mmsegmentation/work_dirs/<MODEL_NAME>_gpus<NGPU>_<PRECISION>/<TIMESTAMP>/<TIMESTAMP>.log
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
   - 检查 `onedl-mmsegmentation` 和 `onedl-mmcv` 是否已正确安装
   - 检查 `PYTHONPATH` 是否包含对应仓库路径

2. **分布式训练无法建立连接**
   - 检查 `NODE_COUNT`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 是否正确
   - 检查节点间网络互通和 NCCL 环境

3. **配置文件导入失败**
   - 检查配置路径是否与 `scripts/configs_path` 一致
   - 检查 `custom_iter_timer_hook.py` 是否已放置到 `onedl-mmsegmentation/` 可导入位置

4. **无法统计性能数据**
   - 检查 config 中是否正确导入 `CustomIterTimerHook`
   - 检查是否关闭默认 timer hook 避免冲突

5. **数据集报错或找不到数据**
   - 检查数据 Cityscapes 目录或软链接是否按 `mmsegmentation` 要求准备完成

