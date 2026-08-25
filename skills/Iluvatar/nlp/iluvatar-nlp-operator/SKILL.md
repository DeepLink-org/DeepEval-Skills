---
name: iluvatar-nlp-operator
description: 天数智芯 BI-V150 GPU 上的 PyTorch 算子性能评测技能。支持 GEMM、Conv2d、长尾算子和 Transformer Block 的验证、日志记录与结果采集。
---

# iluvatar-nlp-operator

必须通过本 skill 的 `scripts/` 调度评测和采集；算子实现、计时逻辑与规范化指标应保留在镜像的 `/workspace/operators` 项目中，不要复制到临时脚本。Conv 使用 PyTorch 2.7.1/CoreX 后端，不要回退到性能异常的 cuDNN 7兼容反向接口。

## 触发条件

- 在 天数 GPU 上生成或验证 GEMM、Conv2d 的性能基准
- 运行 LongTail-Bench 或 Transformer Block 性能测试
- 收集 天数 CoreX 算子性能结果

## 输入、输出与容器

| 宿主环境变量 | 容器路径 | 权限 | 是否必需 | 用途 |
|---|---|---|---|---|
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | `rw` | 是 | 输出CSV与 `result.json` |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | `rw` | 是 | 编译与评测日志 |

Docker镜像：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/iluvatar-nlp-operator:latest
```

Executor 将 skill 的 `scripts/` 预置到容器内 `/workspace/scripts/`。

```bash
docker run -it --name iluvatar-ops-test --privileged --ipc=host --shm-size=16G -w /workspace \
  -v "$OPERATOR_RESULTS_DIR:/workspace/results:rw" \
  -v "$OPERATOR_LOGS_DIR:/workspace/logs:rw" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/iluvatar-nlp-operator:latest bash
```

进入容器后先检查：

```bash
ixsmi
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
test -f /workspace/scripts/build_cuda_ops.sh
test -f /workspace/scripts/run_gemm_conv.sh
```

## 脚本流程

### GEMM / CONV

先编译一次，再选择算子和精度。GEMM/Conv2d 的结果 CSV 只写入 `baseline` 列，不生成 `time`、`score` 列，也不提供基准生成/对比两种模式。

```bash
/workspace/scripts/build_cuda_ops.sh
/workspace/scripts/run_gemm_conv.sh gemm fp16
/workspace/scripts/run_gemm_conv.sh gemm fp32
/workspace/scripts/run_gemm_conv.sh conv fp16
/workspace/scripts/run_gemm_conv.sh conv fp32
/workspace/scripts/collect_results.sh gemm-conv
```

`run_gemm_conv.sh` 参数依次为 `[gemm|conv|all] [fp16|fp32|all]`。脚本调用 `test_gemm.py` 和 `test_conv.py` 时固定传入 `0`，不再存在参数 `1` 的对比路径。Conv通过资源目录中的 `test_conv.py ... 0 torch` 走PyTorch接口，内部对Forward、Backward Filter、Backward Data分别计时1000次并求和，CSV只输出总 `baseline`，不增加分项列。输出日志为 `/workspace/logs/<operator>_<precision>_<mode>.log`。

仅执行 GEMM 时必须运行 `/workspace/scripts/collect_results.sh gemm`；只有 GEMM 与 Conv 都已完成时才运行 `gemm-conv`。

### 长尾算子

```bash
/workspace/scripts/run_longtail.sh fp32
/workspace/scripts/run_longtail.sh fp16
/workspace/scripts/collect_results.sh longtail
```

脚本分别使用 `LongTail-Bench` 和 `LongTail-Bench-fp16` 。长尾输入 CSV 的 `baseline` 初始为空，运行后只填写该列，不生成 `time`、`score`。仅 `batched_nms` 因MMCV扩展与当前环境不兼容，改用 `torchvision.ops.batched_nms`。

### Transformer Block

```bash
/workspace/scripts/run_transformer.sh
/workspace/scripts/collect_results.sh all
```

镜像项目的 `transformer_block/test.py` 使用固定参数；Encoder、Decoder均预热20次、计时1000次，且在计时循环前后执行 DCU 同步。项目脚本直接输出 `AIBENCH_TRANSFORMER_ENCODER_SECONDS` 与 `AIBENCH_TRANSFORMER_DECODER_SECONDS`；runner 只执行并校验这两个规范化指标。

## 结果

`collect_results.sh` 支持 `gemm`、`gemm-conv`、`longtail` 或 `all`。`all` 检查Conv FP16/FP32各63行、GEMM各224行、长尾各40行且 `baseline` 全部有效，并从Transformer日志提取Encoder和Decoder两项结果。

AIBenchAgent 评测脚本在执行 `collect_results.sh` 后，必须使用 `write_result_contract.py` 覆盖生成标准结果；不得手写 JSON。该脚本固定写出 `schema_version: "1.2"`，并校验 duration 为正数。例如长尾任务：

```bash
DURATION_SECONDS=$(( $(date +%s) - START_TIME ))
(( DURATION_SECONDS > 0 )) || DURATION_SECONDS=1
/workspace/scripts/write_result_contract.py \
  --target longtail \
  --task-id Iluvatar_nlp_operator \
  --workload-fingerprint "$WORKLOAD_FINGERPRINT" \
  --duration-seconds "$DURATION_SECONDS"
```

其中 `WORKLOAD_FINGERPRINT` 必须为本次任务的真实 fingerprint。禁止传空值、非数字 duration，或其他 schema 版本。

编译、评测与采集日志保留在 `OPERATOR_LOGS_DIR`。失败时先检查 `ixsmi`、PyTorch/CoreX版本、可用显存、`cuda_ops/build/gemm`、输入CSV和对应日志。
