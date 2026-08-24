---
name: hygon-nlp-operator
description: 海光 DCU 上 HIP PyTorch 算子性能评测技能。支持 GEMM、Conv2d、长尾算子和 Transformer Block 的性能测试、日志记录与结果采集。
---

# hygon-nlp-operator

必须通过本 skill 的 `scripts/` 执行 GEMM/Conv2d、长尾算子、Transformer Block 的评测和结果采集；不要将这些稳定命令重新内嵌到任务脚本。通信算子依赖集群网络、MPI 和调度配置，仍按资源仓库的 `communication_bench/readme.md` 执行。

## 触发条件

- 在海光 DCU 上运行 GEMM、Conv2d 的 FP16/FP32 性能测试
- 运行 LongTail-Bench 或 Transformer Block 性能测试
- 收集海光 DCU 算子性能结果

## 输入、输出与容器

| 宿主环境变量 | 容器路径 | 权限 | 是否必需 | 用途 |
|---|---|---|---|---|
| `OPERATOR_PROJECT_ROOT` | `/workspace/operators` | `rw` | 是 | 海光适配算子源码、CSV 参数和批测脚本 |
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | `rw` | 是 | 输出 CSV 与 `result.json` |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | `rw` | 是 | 评测日志 |

Docker 镜像：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-nlp-operator:latest
```

Executor 将 skill 的 `scripts/` 预置到容器内 `/workspace/scripts/`。`OPERATOR_PROJECT_ROOT` 必须含 `speed_test/`、`LongTail-Bench/`、`LongTail-Bench-fp16/`、`transformer_block/`、`communication_bench/`、四个 GEMM/Conv CSV，以及海光适配的 `test_gemm.py` 和 `test_conv.py`。

```bash
docker run -it --name hygon-ops-test --ipc=host --shm-size=16G -w /workspace \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/dtk:/opt/dtk:ro -v /opt/hyhal:/opt/hyhal:ro \
  -v "$OPERATOR_PROJECT_ROOT:/workspace/operators:rw" \
  -v "$OPERATOR_RESULTS_DIR:/workspace/results:rw" \
  -v "$OPERATOR_LOGS_DIR:/workspace/logs:rw" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-nlp-operator:latest bash
```

进入容器后先检查：

```bash
rocm-smi
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
test -f /workspace/scripts/run_gemm_conv.sh
```

## 脚本流程

### GEMM / CONV

不需要 CMake。每个测试 case 的 DCU 实测耗时直接写入输出 CSV 的 `baseline` 列；不生成 `time`、`score`，也不提供 NVIDIA 的 baseline/test 两种模式。GEMM 预热 10 次、计时 1000 次；Conv 前向预热 10 次，Forward、Backward Filter、Backward Data 分别计时 1000 次并求和。

```bash
/workspace/scripts/run_gemm_conv.sh all all
/workspace/scripts/run_gemm_conv.sh gemm fp16
/workspace/scripts/run_gemm_conv.sh gemm fp32
/workspace/scripts/run_gemm_conv.sh conv fp16
/workspace/scripts/run_gemm_conv.sh conv fp32
/workspace/scripts/collect_results.sh gemm-conv
```

`run_gemm_conv.sh` 参数依次为 `[gemm|conv|all] [fp16|fp32|all]`。GEMM 输出 `gemm_fp16.csv`、`gemm_fp32.csv`；Conv 输出 `conv_fp16.csv`、`conv_fp32.csv`。仅执行 GEMM 时必须运行 `collect_results.sh gemm`；只有 GEMM 与 Conv 都完成时运行 `gemm-conv`。

### 长尾算子

```bash
/workspace/scripts/run_longtail.sh fp32
/workspace/scripts/run_longtail.sh fp16
/workspace/scripts/collect_results.sh longtail
```

脚本分别使用 `LongTail-Bench` 和 `LongTail-Bench-fp16`，输出 `longtail_fp32.csv`、`longtail_fp16.csv`。

### Transformer Block

```bash
/workspace/scripts/run_transformer.sh
/workspace/scripts/collect_results.sh all
```

资源 `transformer_block/test.py` 使用固定参数；Encoder、Decoder 预热 20 次、计时 1000 次，并在计时边界同步 DCU。日志中的 `Time per iteration` 单位为秒。

### 通信算子

通信测试要求多节点、多卡、免密 SSH、MPI/UCX 与高速网络。按 `/workspace/operators/speed_test/communication_bench/readme.md` 执行；记录带宽（GB/s，越高越好）与延迟（us，越低越好）。不要把单机默认值当作多机基线。

## 结果

`collect_results.sh` 支持 `gemm`、`gemm-conv`、`longtail` 或 `all`。`all` 检查 Conv FP16/FP32 各 63 行、GEMM FP16/FP32 各 224 行、长尾 FP16/FP32 各 40 行且 `baseline` 全部有效，并从 Transformer 日志提取 Encoder、Decoder 两项结果，最后生成 `/workspace/results/result.json`。

评测和采集日志均保留在 `OPERATOR_LOGS_DIR`。失败时检查 `rocm-smi`、DTK/PyTorch 版本、可用显存、输入 CSV 与对应日志。

### 算子精度

精度验证必须使用原始 NVIDIA 项目在 A100 上生成的
`accuracy_test/a100_data`，通过同目录的 `cuda_op_validate.py` 验证全部
`op_config.py` 测例的前向、输入梯度和参数梯度。禁止以临时 CPU 对照的
sanity check 替代该结果。

```bash
export OPERATOR_ACCURACY_DATA_DIR=/workspace/operators/accuracy_test/a100_data
/workspace/scripts/run_accuracy.sh
```

结果写入 `/workspace/results/accuracy/cuda_val_result.json` 与
`/workspace/results/accuracy/cuda_val_result.csv`，日志为
`/workspace/logs/accuracy.log`。如果 A100 数据未被随资源预置，脚本会失败
并给出数据目录提示；这时不能报告精度测试通过。