---
name: nvidia-nlp-operator
description: NVIDIA GPU 上 CUDA 算子性能评测技能。支持 GEMM、Conv2d、长尾算子、Transformer Block 与多节点通信算子的基准生成、验证和结果采集。
---

# nvidia-nlp-operator

必须通过本 skill 的 `scripts/` 执行 GEMM/Conv2d、长尾算子、Transformer Block 的编译、评测和结果采集；不要将这些稳定命令重新内嵌到任务脚本。通信算子依赖集群网络、MPI 和调度配置，仍按资源仓库的 `communication_bench/readme.md` 执行。

## 触发条件

- 在 NVIDIA GPU 上生成或验证 GEMM、Conv2d 的性能基准
- 运行 LongTail-Bench 或 Transformer Block 性能测试
- 收集 NVIDIA CUDA 算子性能结果

## 输入、输出与容器

| 宿主环境变量 | 容器路径 | 权限 | 是否必需 | 用途 |
|---|---|---|---|---|
| `OPERATOR_PROJECT_ROOT` | `/workspace/operators` | `rw` | 是 | 算子源码、CSV 参数和批测脚本 |
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | `rw` | 是 | 输出 CSV 与 `result.json` |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | `rw` | 是 | 编译与评测日志 |

Docker 镜像：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-operator:latest
```

Executor 将 skill 的 `scripts/` 预置到容器内 `/workspace/scripts/`。`OPERATOR_PROJECT_ROOT` 必须含有 `cuda_ops/`、`LongTail-Bench/`、`LongTail-Bench-fp16/`、`transformer_block/`、`communication_bench/`、四个 GEMM/Conv CSV，以及 `test_gemm.py` 和 `test_conv.py`。

```bash
docker run -it --name nvidia-ops-test --gpus all --shm-size=16G --ipc=host -w /workspace \
  -v "$OPERATOR_PROJECT_ROOT:/workspace/operators:rw" \
  -v "$OPERATOR_RESULTS_DIR:/workspace/results:rw" \
  -v "$OPERATOR_LOGS_DIR:/workspace/logs:rw" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/nvidia-nlp-operator:latest bash
```

进入容器后先检查：

```bash
nvidia-smi
nvcc --version
test -f /workspace/scripts/build_cuda_ops.sh
test -f /workspace/scripts/run_gemm_conv.sh
```

## 脚本流程

### GEMM / Conv2d

先编译一次，再选择算子、模式和精度。`baseline` 会写入 CSV 的 `baseline` 列；`test` 会写入 `time`、`score` 列并使用已有 baseline 比较。

GEMM 必须直接使用 `nvcc` 编译。不要运行本项目的顶层 CMake：该 CMake 同时配置
Conv2d 并强制查找 cuDNN，而 GEMM 镜像不保证包含 cuDNN，CMake 失败后继续使用旧
CSV 会产生“评测成功但没有执行 kernel”的假阳性。

```bash
/workspace/scripts/build_cuda_ops.sh
/workspace/scripts/run_gemm_conv.sh all baseline all
/workspace/scripts/run_gemm_conv.sh gemm test fp16
/workspace/scripts/collect_results.sh gemm-conv
```

`run_gemm_conv.sh` 参数依次为 `[gemm|conv|all] [baseline|test] [fp16|fp32|all]`。输出日志为 `/workspace/logs/<operator>_<precision>_<mode>.log`。

### 长尾算子

```bash
/workspace/scripts/run_longtail.sh fp32
/workspace/scripts/run_longtail.sh fp16
/workspace/scripts/collect_results.sh longtail
```

脚本分别使用 `LongTail-Bench` 和 `LongTail-Bench-fp16`，输出 `/workspace/results/ltout_gpu.csv` 或 `ltout_fp16.csv`。

### Transformer Block

```bash
/workspace/scripts/run_transformer.sh
/workspace/scripts/collect_results.sh all
```

该资源的 `test.py` 固定测试 Encoder/Decoder 的默认参数；需要改变形状或迭代次数时，修改资源仓库中的 `transformer_block/test.py` 后再运行。日志中的 `Time per iteration` 单位为秒。

### 通信算子

通信测试要求多节点、多卡、免密 SSH、MPI/UCX 与高速网络。按 `/workspace/operators/communication_bench/readme.md` 选择 OSU Micro-Benchmarks 或 NCCL-Tests；记录带宽（GB/s，越高越好）与延迟（us，越低越好）。不要把单机默认值当作多机基线。

## 结果

`collect_results.sh` 会将已存在 CSV 的全量数值和 Transformer 日志指标写入 `/workspace/results/result.json`。该脚本支持 `gemm-conv`、`longtail` 或 `all`；`all` 用于需要合并已完成项目时。编译、评测与采集日志均保留在 `OPERATOR_LOGS_DIR`，便于排查 CUDA、cuDNN、显存不足或 CSV 格式问题。

常见失败先检查 GPU/驱动（`nvidia-smi`）、CUDA 编译器（`nvcc --version`）、`cuda_ops/build/gemm` 与 `conv` 是否生成，以及 CSV 是否具有测试参数和 baseline 列。显存不足时以 `CUDA_VISIBLE_DEVICES` 选择空闲 GPU，或缩小资源仓库中定义的测试形状。
