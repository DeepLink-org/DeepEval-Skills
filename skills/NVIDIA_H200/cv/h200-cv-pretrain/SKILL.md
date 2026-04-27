---
name: h200-cv-pretrain
description: "H200芯片-CV场景-预训练/分类任务的评测流程。用于指导executor在 H200 环境下完成 onedl-mmpretrain 分类模型训练环境准备、批量脚本执行、性能采集和结果分析。适用于 resnet50、inceptionv3、seresnet50、mobilenetv2、shufflenetv2、densenet121、swin-large、efficientnet-b2 等模型。"
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 H200 上跑 CV 预训练模型测试"
- "帮我测试 resnet50 / inceptionv3 / swin-large 分类训练性能"
- "在 H200 上跑 mmpretrain 分类模型 benchmark"
- "帮我批量测试 CV pretrain 模型 FP32/FP16 性能"

---

### 支持的模型配置

**当前支持模型**：
- `resnet50`
- `inception_v3`
- `seresnet50`
- `mobilenet_v2`
- `shufflenet_v2`
- `densenet121`
- `swin_large`
- `efficientnet_b2`

**当前支持任务**：
- 基于 `onedl-mmpretrain` 的分类模型训练/性能测试
- 使用 `custom_iter_timer_hook.py` 采集训练迭代耗时
- 批量执行多模型基准测试 （FP32 / FP16）

**硬件要求**：
- 2 节点，共 16 张 NVIDIA H200 GPU
- 当前脚本默认每节点 8 卡
- 适合 PyTorch 2.x + CUDA 11.8+/12.x 环境

---
### 代码与环境要求

该 skill 基于 PyTorch 2.x 适配方案，分类模型使用 `onedl-mmpretrain`，需提前安装 `onedl-mmcv` 与 `onedl-mmpretrain`。
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

#### 2. 安装 onedl-mmpretrain

```bash
git clone https://github.com/VBTI-development/onedl-mmpretrain.git
cd onedl-mmpretrain
git checkout 128b6079ecc1d089577d1e99b1f786887f48a1c1
pip install -e .
cd ..
```

#### 3. 设置环境变量

```bash
export MMPRE_PATH=/path/to/onedl-mmpretrain
export MMCV_PATH=/path/to/onedl-mmcv
export PYTHONPATH=${MMPRE_PATH}:${MMCV_PATH}:$PYTHONPATH
```

---

### 自定义性能统计 Hook

为了在 PyTorch 2.x / mmengine 环境下采集稳定的迭代耗时，需使用：

```bash
tools/custom_iter_timer_hook.py
```
**放置位置**：将 `custom_iter_timer_hook.py` 拷贝到 `onedl-mmpretrain/` 根目录下，确保 Python 可导入。

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

分类模型不依赖初始化权重

### 配置文件与模型列表

分类模型配置列表位于：

```bash
scripts/configs_path
```
当前 pretrain 批量测试脚本内置以下配置：
- `configs/resnet/resnet50_8xb32_in1k.py`
- `configs/inception_v3/inception-v3_8xb32_in1k.py`
- `configs/seresnet/seresnet50_8xb32_in1k.py`
- `configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py`
- `configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py`
- `configs/densenet/densenet121_4xb256_in1k.py`
- `configs/swin_transformer/swin-large_16xb64_in1k.py`
- `configs/efficientnet/efficientnet-b2_8xb32_in1k.py`

---

### 批量测试脚本

标准批量测试脚本位置：

```bash
scripts/batch_pretrain.sh
```

执行方式：

```bash
bash scripts/batch_pretrain.sh
```

当前脚本默认行为：
- 进入 `./models/onedl-mmpretrain`
- 设置 `MMPRE_PATH`、`MMCV_PATH`、`PYTHONPATH`
- 使用 `torch.distributed.launch` 进行 2 节点 16 卡训练
- 默认每节点 8 卡
- 当前脚本默认跑 `fp32`，分别测试 `fp32` 和 `fp16`
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

分类模型通常依赖 ImageNet 训练数据。

需要按照 `onedl-mmpretrain` 的数据集要求准备数据，并保证仓库内存在正确的数据目录或软链接。可参考 README 中 OpenMMLab 数据准备方式。

如果数据路径未正确配置，训练脚本会直接报错。

---

### 性能监控

**关键指标**：
- 平均迭代耗时（`AVG_ITER_TIME`）
- 数据加载耗时（`DATA`）
- 算子执行耗时（`OP`）
- GPU 利用率
- 显存占用
- 不同模型在 FP32/FP16 下的吞吐差异


---

### 常见问题

1. **脚本启动失败**
   - 检查 `onedl-mmpretrain` 和 `onedl-mmcv` 是否已正确安装
   - 检查 `PYTHONPATH` 是否包含对应仓库路径

2. **分布式训练无法建立连接**
   - 检查 `NODE_COUNT`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 是否正确
   - 检查节点间网络互通和 NCCL 环境

3. **配置文件导入失败**
   - 检查配置路径是否存在
   - 检查 `custom_iter_timer_hook.py` 是否已放置到`onedl-mmpretrain/`可导入位置

4. **无法统计性能数据**
   - 检查 config 中是否正确导入 `CustomIterTimerHook`
   - 检查是否关闭默认 timer hook 避免冲突

5. **数据集报错或找不到数据**
   - 检查数据目录或软链接是否按 `mmpretrain` 要求准备完成
