---
name: cuda-ops-perf-test
description: "CUDA算子基准值生成，包括GEMM、Conv2d算子(FP16/FP32)和长尾算子。用于指导executor完成docker容器启动、编译和基准值生成的完整流程。"
argument-hint: [test-type] (gemm|conv|longtail)
---

# CUDA 算子基准值生成

## 触发条件

当用户说以下任意内容时启动：
- "帮我生成 GEMM 算子基准值"
- "跑一下 Conv2d 算子基准"
- "生成 CUDA 算子 baseline"
- "帮我跑长尾算子基准测试"

---

### 支持的算子配置

**当前支持算子**：
- **GEMM**: 矩阵乘法算子，支持 FP16（使用 tensor core）和 FP32
- **Conv2d**: 二维卷积算子，支持 FP16 和 FP32
- **长尾算子**: 基于 LongTail-Bench 的 Pytorch 实现，支持 GPU 和 CPU


---

### 启动配置

**Docker 运行命令**：
```bash
docker run -dit \
  --name cuda-ops-test \
  --gpus all \
  --shm-size=16g \
  -v /workspace/results:/workspace/results \
  -v /workspace/logs:/workspace/logs \
  -w /workspace \
  pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  bash
```

后续所有命令均在容器内执行：
```bash
docker exec -it cuda-ops-test bash
```

---

## GEMM、Conv2d 算子

### 编译

```bash
cd cuda_ops
mkdir -p build && cd build && cmake .. && make
```

### 生成基准值

**单独运行**：
```bash
cd cuda_ops
./build/gemm m k n trans1 trans2 datatype
./build/conv n c h w c_out k_w k_h pad_w pad_h stride_w stride_h datatype
```

参数说明：
- `datatype`: `16` 表示 FP16，`32` 表示 FP32
- `trans1`、`trans2`: `0` 表示不转置，`1` 表示转置

**批量运行**：
```bash
python test_conv.py /workspace/results/conv_f16.csv 16 0
python test_conv.py /workspace/results/conv_f32.csv 32 0
python test_gemm.py /workspace/results/gemm_f16.csv 16 0
python test_gemm.py /workspace/results/gemm_f32.csv 32 0
```

从 `*.csv` 文件读取参数并写入基准值，`16/32` 表示数据类型，`0` 表示生成基准值。如已有基准值则跳过。

**结果输出**：
基准值写入 `/workspace/results/` 下对应的 `*.csv` 文件的 `baseline` 列中，例如 `gemm_f16.csv` 存放 GEMM 算子在 FP16 下的基准值，`conv_f32.csv` 存放 Conv2d 算子在 FP32 下的基准值。

---

## 长尾算子

### 环境准备

```bash
cd LongTail-Bench
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 生成基准值

**GPU 基准**：
```bash
python ./long_tail_bench/api/api.py -f ../longtail_perf.csv --outcsv /workspace/results/ltout_gpu.csv
```

**CPU 基准**：
```bash
# 执行转化脚本（生成 samples-bak 备份，新 samples 仅支持 cpu）
sh script_for_cpu.sh

# 生成基准
DEVICE_CPU=1 python ./long_tail_bench/api/api.py -f ../longtail_perf.csv --outcsv /workspace/results/ltout_cpu.csv
```

如需恢复 GPU 测试，将 `samples-bak` 还原为 `samples`。

**结果输出**：
基准值写入 `/workspace/results/` 下 `--outcsv` 指定的 `*.csv` 文件中，例如 `ltout_gpu.csv` 存放 GPU 基准值，`ltout_cpu.csv` 存放 CPU 基准值。

---

### 常见问题

1. **编译失败**
   - 检查 cuDNN 9 是否正确安装
   - 检查 CMake 和 CUDA 工具链版本

2. **容器启动失败**
   - 检查 Docker 镜像 `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` 是否已拉取
   - 检查 GPU 驱动和 `--gpus all` 是否可用

3. **基准值生成异常**
   - 检查 `*.csv` 文件是否存在且格式正确
   - 检查 GPU 显存是否充足

4. **长尾算子 CPU 模式失败**
   - 确认已执行 `script_for_cpu.sh` 转化脚本
   - 确认 `PYTHONPATH` 已正确设置