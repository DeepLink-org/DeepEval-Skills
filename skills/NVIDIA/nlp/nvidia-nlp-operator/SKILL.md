---
name: nvidia-nlp-operator
description: NVIDIA GPU 上 CUDA 算子性能评测技能。支持 GEMM、Conv2d（FP16/FP32）、长尾算子、Transformer Block、通信算子等基准值生成与性能测试，用于指导 executor 完成容器启动、编译、基准值生成、测试验证与性能指标采集的完整流程。
---

### 触发条件

当用户说以下任意内容时启动：
- "帮我生成 GEMM 算子基准值"
- "在 NVIDIA 上生成 GEMM 算子基准值"
- "跑一下 Conv2d 算子基准"
- "生成 CUDA 算子 baseline"
- "帮我跑长尾算子基准测试"
- "测试 NVIDIA GPU 算子性能"
- "运行 operator benchmark"

---


### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `OPERATOR_PROJECT_ROOT` | `/workspace/operators` | 是 | 算子项目根目录，需外部提供，包含所有算子测试代码、CSV 测试参数文件和测试脚本 |
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | 是 | 评测结果输出目录，存放基准值和测试结果 CSV 文件 |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | 是 | 日志输出目录 |

**说明**：
- **OPERATOR_PROJECT_ROOT** 需要外部提供，包含 GEMM/Conv2d 源码（cuda_ops/）、长尾算子代码（LongTail-Bench/）、Transformer Block 代码（transformer_block/）、通信算子脚本（communication_bench/），以及 CSV 测试参数文件（gemm_f16.csv、conv_f16.csv、longtail_perf.csv 等）和批量测试脚本（test_gemm.py、test_conv.py）
- **OPERATOR_RESULTS_DIR** 存放评测过程中生成的基准值、测试结果 CSV 文件
- **OPERATOR_LOGS_DIR** 存放编译日志、测试日志

**目录结构说明**：

`OPERATOR_PROJECT_ROOT` 映射到容器内 `/workspace/operators`，默认结构如下：
```
/workspace/operators/
├── cuda_ops/                  # GEMM/Conv2d CUDA 源码
│   ├── CMakeLists.txt
│   ├── cuda_gemm.cpp
│   ├── cudnn_convforward.cpp
│   └── build/                 # 编译产物目录
├── LongTail-Bench/            # 长尾算子（GPU 版本）
│   └── long_tail_bench/
│       ├── api/
│       ├── common/
│       ├── core/
│       └── samples/
├── LongTail-Bench-fp16/       # 长尾算子（仅 FP16 版本，特殊芯片使用）
├── transformer_block/         # Transformer Block 测试
│   ├── blocks/
│   ├── layers/
│   └── test.py
├── communication_bench/       # 通信算子测试脚本
│   ├── test_all.sh
│   ├── test_nccl.sh
│   ├── comm_sbatch.sh
│   └── parse_comm_result.py
├── gemm_f16.csv               # GEMM FP16 测试参数与基准值
├── gemm_f32.csv               # GEMM FP32 测试参数与基准值
├── conv_f16.csv               # Conv2d FP16 测试参数与基准值
├── conv_f32.csv               # Conv2d FP32 测试参数与基准值
├── longtail_perf.csv          # 长尾算子测试参数与基准值
├── longtail_perf_gpu.csv      # 长尾算子 GPU 测试参数与基准值
├── longtail_perf_cpu.csv      # 长尾算子 CPU 测试参数与基准值
├── longtail_perf_gpu_fp16.csv # 长尾算子 FP16 测试参数与基准值
├── test_gemm.py               # GEMM 批量测试脚本
├── test_conv.py               # Conv2d 批量测试脚本
└── readme.md
```

**注意**：
- CSV 文件（gemm_f16.csv 等）包含测试参数定义和 baseline/time/score 列，需以读写方式挂载，因为基准值生成和测试结果会写回 CSV
- 容器内工作目录为 `/workspace`


---


### 支持的算子配置

**当前支持算子**（共 4 类）：
- **GEMM**: 矩阵乘法算子，支持 FP16（使用 tensor core）和 FP32，涵盖多种矩阵维度（M/N/K）和转置组合
- **Conv2d**: 二维卷积算子，支持 FP16 和 FP32，涵盖多种输入尺寸、卷积核大小、padding、stride 组合
- **长尾算子**: 基于 LongTail-Bench 的 PyTorch 实现，支持 GPU 和 CPU 模式，涵盖 bbox2delta、nms、l2_loss、roi_align 等 100+ 长尾算子
- **Transformer Block**: 基于 PyTorch 的 Encoder/Decoder Layer 性能测试
- **通信算子**: 基于 OSU Micro-Benchmarks 或 NCCL-Tests 的 All-Reduce、All-Gather 等集合通信算子带宽与延迟测试（需多节点环境）

**当前支持任务**：
- 基准值生成（baseline）：在参考 GPU（如 A100）上生成算子性能基准值
- 性能测试（test）：在目标 GPU 上运行算子并计算相对于基准值的得分

**硬件要求**：
- 至少 1 张 NVIDIA GPU（GEMM、Conv2d、长尾算子、Transformer Block）
- 通信算子测试需要多节点多卡环境

---


### 依赖要求

**Docker 镜像**：
```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-operator:latest
```

容器内已预装：
- PyTorch（Transformer Block、长尾算子依赖）
- CUDA Toolkit（GEMM/Conv2d 编译依赖）
- cuDNN（Conv2d 依赖）
- CMake、make、g++（编译依赖）
- pandas（批量测试脚本依赖）
- Python 3.x


---


## 第一阶段：容器启动

### 选择算子类型

启动容器前，先确定要测试的算子类型：

```bash
# 选择要测试的算子类型
export OP_TYPE="gemm"  # 可选: gemm, conv, longtail, transformer, communication, all
```

### 容器创建命令

**挂载权限约定**：
- `:ro` — 只读，用于输入数据，防止误修改
- `:rw` — 读写，用于需要写入的目录

**公共参数**：

| 参数 | 说明 |
|------|------|
| `--gpus all` | 挂载所有 NVIDIA GPU |
| `--shm-size=16G` | 共享内存大小，避免大数据加载时内存不足 |
| `--ipc=host` | 使用主机 IPC，优化进程间通信 |
| `-w /workspace` | 容器内工作目录 |

**公共卷挂载**：
```bash
-v $OPERATOR_PROJECT_ROOT:/workspace/operators:rw \
-v $OPERATOR_RESULTS_DIR:/workspace/results:rw \
-v $OPERATOR_LOGS_DIR:/workspace/logs:rw
```

**基础启动命令**：

```bash
docker run -dit \
  --name nvidia-ops-test \
  --gpus all \
  --shm-size=16G \
  --ipc=host \
  -w /workspace \
  -v $OPERATOR_PROJECT_ROOT:/workspace/operators:rw \
  -v $OPERATOR_RESULTS_DIR:/workspace/results:rw \
  -v $OPERATOR_LOGS_DIR:/workspace/logs:rw \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-operator:latest \
  /bin/bash
```

**注意**：
- 若已存在同名容器，先执行 `docker rm -f nvidia-ops-test`
- `OPERATOR_PROJECT_ROOT` 使用 `:rw` 挂载，因为基准值生成和测试结果会写回 CSV 文件
- 如需限制可见 GPU，添加 `-e NVIDIA_VISIBLE_DEVICES=0,1` 等环境变量

### 容器管理命令

**进入已创建的容器**：
```bash
# 如果容器已在运行
docker exec -it nvidia-ops-test /bin/bash

# 如果容器已停止，先启动再进入
docker start nvidia-ops-test
docker exec -it nvidia-ops-test /bin/bash
```

**验证容器环境**：
```bash
# 检查 GPU 设备
nvidia-smi

# 检查 CUDA 编译器
nvcc --version

# 检查挂载的目录
ls -lh /workspace/
ls -lh /workspace/operators/
ls -lh /workspace/results/
ls -lh /workspace/logs/
```


---


## 第二阶段：容器中执行评测

### GEMM、Conv2d 算子

#### 步骤 1：编译

```bash
cd /workspace/operators/cuda_ops
mkdir -p build && cd build && cmake .. && make 2>&1 | tee /workspace/logs/compile.log
```

**验证编译产物**：
```bash
ls -lh /workspace/operators/cuda_ops/build/gemm
ls -lh /workspace/operators/cuda_ops/build/conv
```

#### 步骤 2：生成基准值

基准值生成使用模式 `0`（第三个参数），从 CSV 文件读取测试参数并将结果写入 `baseline` 列。如已有基准值则跳过。

**单独运行**（调试单个用例）：

```bash
cd /workspace/operators/cuda_ops

# GEMM 算子：m k n trans1 trans2 datatype
./build/gemm 2048 1024 4096 0 0 16     # FP16 示例

# Conv2d 算子：n c h w c_out k_w k_h pad_w pad_h stride_w stride_h datatype
./build/conv 8 3 224 224 64 3 3 1 1 1 1 16   # FP16 示例
```

参数说明：
- `datatype`：`16` 表示 FP16（使用 tensor core），`32` 表示 FP32
- `trans1`、`trans2`：`0` 表示不转置，`1` 表示转置

**批量运行**（推荐）：

```bash
cd /workspace/operators

# 生成 GEMM 基准值
python test_gemm.py /workspace/operators/gemm_f16.csv 16 0 2>&1 | tee /workspace/logs/gemm_f16_baseline.log
python test_gemm.py /workspace/operators/gemm_f32.csv 32 0 2>&1 | tee /workspace/logs/gemm_f32_baseline.log

# 生成 Conv2d 基准值
python test_conv.py /workspace/operators/conv_f16.csv 16 0 2>&1 | tee /workspace/logs/conv_f16_baseline.log
python test_conv.py /workspace/operators/conv_f32.csv 32 0 2>&1 | tee /workspace/logs/conv_f32_baseline.log
```

参数说明：
- 第一个参数：CSV 文件路径（包含测试参数定义）
- 第二个参数：`16` 表示 FP16，`32` 表示 FP32
- 第三个参数：`0` 表示生成基准值模式

**GEMM CSV 示例**（`gemm_f16.csv`）：

| m | k | n | trans1 | trans2 | datatype | baseline | time | score |
|---|---|------|--------|--------|----------|----------|------|-------|
| 2048 | 1024 | 4096 | 0 | 0 | 16 | 0.123 | | |
| 4096 | 2048 | 8192 | 0 | 0 | 16 | 0.456 | | |
| 8192 | 4096 | 16384 | 0 | 0 | 16 | 1.234 | | |
| 1024 | 1024 | 1024 | 1 | 0 | 16 | 0.089 | | |

**Conv2d CSV 示例**（`conv_f16.csv`）：

| n | c | h | w | c_out | k_w | k_h | pad_w | pad_h | stride_w | stride_h | datatype | baseline | time | score |
|---|---|---|---|---|------|-------|-----|-----|-------|-------|----------|----------|----------|------|-------|
| 8 | 3 | 224 | 224 | 64 | 3 | 3 | 1 | 1 | 1 | 1 | 16 | 0.234 | | |
| 16 | 64 | 112 | 112 | 128 | 3 | 3 | 1 | 1 | 1 | 1 | 16 | 0.567 | | |
| 32 | 128 | 56 | 56 | 256 | 3 | 3 | 1 | 1 | 2 | 2 | 16 | 1.012 | | |

**验证基准值生成结果**：
```bash
head -5 /workspace/operators/gemm_f16.csv
# 应看到 baseline 列已填充数值
```

---

### 长尾算子

#### 步骤 1：环境准备

```bash
cd /workspace/operators/LongTail-Bench
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### 步骤 2：生成基准值

**GPU 基准**：

```bash
cd /workspace/operators

python ./LongTail-Bench/long_tail_bench/api/api.py \
  -f /workspace/operators/longtail_perf.csv \
  --outcsv /workspace/results/ltout_gpu.csv \
  2>&1 | tee /workspace/logs/longtail_gpu_baseline.log
```

**GPU 基准 FP16**：

```bash
cd /workspace/operators

python ./LongTail-Bench-fp16/long_tail_bench/api/api.py \
  -f /workspace/operators/longtail_perf_gpu_fp16.csv \
  --outcsv /workspace/results/ltout_fp16.csv \
  2>&1 | tee /workspace/logs/longtail_gpu_baseline_fp16.log
```

**长尾算子 CSV 示例**（`longtail_perf.csv`）：

| operator | input_shape | datatype | baseline | time | score |
|----------|-------------|----------|----------|------|-------|
| bbox2delta | (1024, 4) | fp32 | 0.012 | | |
| nms | (1024, 4) | fp32 | 0.034 | | |
| roi_align | (256, 256, 7, 7) | fp32 | 0.156 | | |
| l2_loss | (1024, 512) | fp32 | 0.008 | | |
| nms | (1024, 4) | fp16 | 0.022 | | |
| roi_align | (256, 256, 7, 7) | fp16 | 0.098 | | |

---

### Transformer Block

Transformer Block 测试基于 PyTorch 实现，评估 Encoder Layer 和 Decoder Layer 的推理耗时。

```bash
cd /workspace/operators/transformer_block

python test.py 2>&1 | tee /workspace/logs/transformer_block.log
```

**测试内容**：
- Encoder Layer：测试 self-attention + FFN 前向传播耗时
- Decoder Layer：测试 self-attention + cross-attention + FFN 前向传播耗时
- 默认参数：d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, seq_len=512

**自定义参数**（可选）：
编辑 `test.py` 末尾的调用参数：
```python
# 修改以下参数进行自定义测试
test_transformer_encoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, seq_len=512, num_iterations=1000)
test_transformer_decoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, tgt_len=512, memory_len=512, num_iterations=1000)
```

---

### 通信算子（多节点环境）

通信算子测试需要多节点多卡环境，采用 OSU Micro-Benchmarks 或 NCCL-Tests 工具。

**前置条件**：
- 多节点间已配置 SSH 免密登录
- 节点间通过 InfiniBand 或高速网络互联
- 已安装 OpenMPI / UCX

**工具选择**：

| 工具 | 说明 |
|------|------|
| OSU Micro-Benchmarks | 测试 All-Reduce、All-Gather 等集合通信算子带宽和延迟 |
| NCCL-Tests | NVIDIA 官方 NCCL 性能测试工具集 |

详细的多节点通信测试配置和运行步骤请参考 `communication_bench/readme.md`。

---

### 关键性能指标

#### GEMM、Conv2d 算子

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `baseline` | 算子执行耗时（ms），数值越低越好 |

#### 长尾算子

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `baseline` | 算子执行耗时，数值越低越好 |

#### Transformer Block

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `Time per iteration` | Encoder/Decoder Layer 单次迭代平均耗时（秒） |

#### 通信算子

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `bandwidth` | 通信带宽（GB/s），数值越高越好 |
| 性能（必采） | `latency` | 通信延迟（us），数值越低越好 |

---

### 指标采集

请严格使用下列代码进行指标采集

**GEMM / Conv2d 算子 — 从 CSV 文件全量采集**：

```bash
python -c "
import pandas as pd
import json

result = {}
for f in ['/workspace/operators/gemm_f16.csv', '/workspace/operators/gemm_f32.csv',
          '/workspace/operators/conv_f16.csv', '/workspace/operators/conv_f32.csv']:
    df = pd.read_csv(f)
    if 'baseline' not in df.columns or df['baseline'].isna().all():
        continue
    key = f.split('/')[-1].replace('.csv', '')
    # 区分数据类型
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    # 提取 baseline 列
    baseline_vals = df['baseline'].dropna().tolist()
    result[key] = {
        'dtypes': dtypes,
        'baseline': baseline_vals,
        'data': df.to_dict(orient='records')
    }

with open('/workspace/results/result.json', 'w') as fp:
    json.dump(result, fp, indent=2, default=str)
print('result.json written to /workspace/results/')
"
```

**长尾算子 — 从输出 CSV 全量采集**：

```bash
python -c "
import pandas as pd
import json

result = {}
for csv_file in ['/workspace/results/ltout_gpu.csv', '/workspace/results/ltout_fp16.csv']:
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        continue
    key = csv_file.split('/')[-1].replace('.csv', '')
    # 区分数据类型
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    # 提取 baseline 列（如存在），否则提取 time 列
    metric_col = 'baseline' if 'baseline' in df.columns else ('time' if 'time' in df.columns else None)
    metric_vals = df[metric_col].dropna().tolist() if metric_col else []
    result[key] = {
        'dtypes': dtypes,
        'metric_col': metric_col,
        'metric_values': metric_vals,
        'data': df.to_dict(orient='records')
    }

with open('/workspace/results/result.json', 'w') as fp:
    json.dump(result, fp, indent=2, default=str)
print('result.json written to /workspace/results/')
"
```

**Transformer Block — 从日志采集**：

```bash
grep "Time per iteration" /workspace/logs/transformer_block.log
```

---

## 常见问题

1. **容器启动失败**
   - **镜像不存在**：确认镜像 `swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-operator:latest` 已拉取
   - **GPU 不可用**：检查 `nvidia-smi` 是否能正常显示 GPU，确认 NVIDIA 驱动和 nvidia-docker 已安装
   - **路径不存在**：确认 `OPERATOR_PROJECT_ROOT`、`OPERATOR_RESULTS_DIR`、`OPERATOR_LOGS_DIR` 路径在宿主机上存在

2. **编译失败**
   - **CUDA 工具链缺失**：确认容器内 `nvcc --version` 可用，检查 CUDA Toolkit 版本
   - **cuDNN 未安装**：确认 `/usr/include/cudnn.h` 或类似路径存在，检查 cuDNN 版本
   - **CMake/make 缺失**：确认 `cmake --version` 和 `make --version` 可用

3. **基准值生成异常**
   - **CSV 文件不存在**：确认 `/workspace/operators/` 下 CSV 文件存在且格式正确（需包含 `baseline`、`time`、`score` 列）
   - **GPU 显存不足**：通过 `nvidia-smi` 检查显存占用，必要时减少 batch 或输入尺寸
   - **已有基准值跳过**：如需重新生成，先清空 CSV 中的 baseline 列

4. **Transformer Block 运行失败**
   - **PyTorch 未安装**：确认 `python -c "import torch; print(torch.__version__)"` 可用
   - **GPU 不可用**：确认 `torch.cuda.is_available()` 返回 True

5. **GPU 显存不足**
   - **现象**：推理或测试过程中报错 `CUDA out of memory`
   - **解决方案**：
     step 1. **查看 GPU 使用情况**：
        ```bash
        nvidia-smi
        ```
     step 2. **指定空闲 GPU**：
        ```bash
        export CUDA_VISIBLE_DEVICES=0  # 指定使用卡 0
        ```
     step 3. **重新执行测试**

6. **通信算子测试失败**
   - **多节点环境未配置**：确认 SSH 免密登录已配置，所有节点间可互通
   - **MPI 未安装**：确认 OpenMPI 已安装且 `mpirun` 可用
   - **InfiniBand 未配置**：检查 RDMA/InfiniBand 设备是否可用
