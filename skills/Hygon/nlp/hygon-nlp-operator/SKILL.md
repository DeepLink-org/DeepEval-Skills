---
name: hygon-nlp-operator
description: 海光 DCU 算子性能评测技能：GEMM 使用 GPUfusion 编译的原始 CUDA 实现；Conv2d 使用 DCU 原生 MIOpen（经 HIP-PyTorch）实现。支持长尾算子和 Transformer Block 的性能测试、日志记录与结果采集。
---

# hygon-nlp-operator

skill 的 `scripts/` 只负责环境准备、参数分发、结果采集和结果契约；算子执行逻辑必须优先复用镜像 `/workspace/operators` 中的项目脚本，避免在 skill 复制 benchmark 实现。通信算子依赖集群网络、MPI 和调度配置，仍按资源仓库的 `communication_bench/readme.md` 执行。

## 触发条件

- 在海光 DCU 上运行 GEMM、Conv2d 的 FP16/FP32 性能测试
- 运行 LongTail-Bench 或 Transformer Block 性能测试
- 收集海光 DCU 算子性能结果

## 输入、输出与容器

| 宿主环境变量 | 容器路径 | 权限 | 是否必需 | 用途 |
|---|---|---|---|---|
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | `rw` | 是 | 输出 CSV 与 `result.json` |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | `rw` | 是 | 评测日志 |

Docker 镜像：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/hygon-nlp-operator:latest
```

Executor 将 skill 的 `scripts/` 预置到容器内 `/workspace/scripts/`。`OPERATOR_PROJECT_ROOT` 必须含 `speed_test/`、`LongTail-Bench/`、`LongTail-Bench-fp16/`、`transformer_block/`、`communication_bench/`、四个 GEMM/Conv CSV、已由 GPUfusion 原样编译的 `cuda_ops/build/gemm`，DCU Conv 运行器 `speed_test/test_conv_dcu.py`、GPUfusion GEMM CSV 驱动 `speed_test/run_native_gemm.py`。

```bash
docker run -it --name hygon-ops-test --ipc=host --shm-size=16G -w /workspace \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/dtk:/opt/dtk:ro -v /opt/hyhal:/opt/hyhal:ro \
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

镜像构建阶段通过 GPUfusion 编译原始 CUDA GEMM CMake 工程，运行阶段直接调用原始 C++ GEMM 可执行文件。GPUfusion 的 cuDNN 兼容层在 `cudnnSetConvolution2dDescriptor` 会 abort，因而 Conv2d 改由 DCU 原生 MIOpen 后端（HIP-PyTorch）执行等价的 Forward、Backward Filter、Backward Data 并求和。每个测试 case 的 DCU 实测耗时直接写入输出 CSV 的 `baseline` 列；不生成 `time`、`score`，也不提供 NVIDIA 的 baseline/test 两种模式。GEMM 预热 10 次、计时 1000 次；Conv 前向预热 10 次，三个阶段各计时 1000 次。

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

## 结果

`collect_results.sh` 支持 `gemm`、`gemm-conv`、`longtail` 或 `all`。`all` 检查 Conv FP16/FP32 各 63 行、GEMM FP16/FP32 各 224 行、长尾 FP16/FP32 各 40 行且 `baseline` 全部有效，并从 Transformer 日志提取 Encoder、Decoder 两项结果，最后生成 `/workspace/results/result.json`。

评测和采集日志均保留在 `OPERATOR_LOGS_DIR`。镜像项目的 `transformer_block/test.py` 会直接输出 `AIBENCH_TRANSFORMER_ENCODER_SECONDS` 与 `AIBENCH_TRANSFORMER_DECODER_SECONDS`，结果脚本必须优先读取这两个标记，而不能手写错误词序的正则。

AIBenchAgent 评测必须在 `collect_results.sh` 之后使用 `/workspace/scripts/write_result_contract.py` 写标准结果契约：固定 `schema_version` 为 `1.2`，并传入本次任务的真实 `task_id`、`workload_fingerprint` 与正数 `duration_seconds`；禁止直接手写 `result.json`。

失败时检查 `rocm-smi`、GPUfusion/DTK 运行时版本、可用显存、输入 CSV 与对应日志。

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
## 生成评测脚本的强制模板

生成的 agent 脚本不得重新实现评测、CSV 校验、JSON 统计、结果合同或 duration 计算。它必须少于 10 行，并只调用确定性入口：

```bash
#!/bin/bash
set -euo pipefail
/workspace/scripts/run_operator_task.sh gemm Hygon_nlp_operator "$WORKLOAD_FINGERPRINT"
test -s /workspace/results/result.json
```

入口会运行对应项目脚本、采集 artifacts、计算 duration，并以正确的 `--task-id`、`--workload-fingerprint`、`--duration-seconds`、`--target` 参数调用 `write_result_contract.py`。禁止使用 Python heredoc 生成或验证 `result.json`。
