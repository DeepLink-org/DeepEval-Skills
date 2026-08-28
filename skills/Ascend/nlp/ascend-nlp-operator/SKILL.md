---
name: ascend-nlp-operator
description: 在华为 Ascend 910B NPU 上执行算子精度与性能评测。用于算子 FP32/FP16 精度验证、原生 CANN GEMM、CUDA 口径 Conv2d 前向与双反向、FP32/FP16 长尾算子和 Transformer Block 测试，以及 AIBenchAgent Result Contract 2.0 采集；当任务的 chip_type 为 Ascend、task_type 为 operator，或用户要求运行 ACLNN、torch_npu、CANN、910B operator benchmark 时使用。
metadata:
  benchmark_specs:
    accuracy: benchmark_specs/accuracy.yaml
    gemm: benchmark_specs/gemm.yaml
    conv: benchmark_specs/conv.yaml
    longtail: benchmark_specs/longtail.yaml
    transformer: benchmark_specs/transformer_block.yaml
    transformer_block: benchmark_specs/transformer_block.yaml
---

# Ascend NLP Operator

使用镜像内置的 `/workspace/operators` 工程和 Skill 预置到 `/workspace/scripts` 的确定性脚本。评测覆盖精度、GEMM、Conv2d、LongTail 和 Transformer Block。不要从宿主机覆盖挂载工程目录，也不要临时重写 CSV 解析、汇总指标、case identity 或 `result.json`。

## 环境与目录

| 宿主环境变量 | 容器路径 | 权限 | 必需 | 用途 |
|---|---|---|---|---|
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | `rw` | 是 | CSV、JSON 和中间结果 |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | `rw` | 是 | 编译和执行日志 |

工程代码、CMake/C++ 工具链、torch_npu、完整 CANN Toolkit 和匹配的 910B ops 包均由镜像提供，不再通过 Skill 环境变量或 bind mount 注入。镜像内项目目录必须至少包含：

```text
/workspace/operators/
├── accuracy_test/
├── accuracy_reference_cpu/          # accuracy 每次运行时强制覆盖
├── speed_test/
│   ├── npu_ops/
│   ├── LongTail-Bench/
│   ├── LongTail-Bench-fp16/
│   ├── transformer_block/
│   ├── test_gemm_native.py
│   ├── test_conv_native.py
│   ├── gemm_f16.csv
│   ├── gemm_f32.csv
│   ├── conv_f16.csv
│   ├── conv_f32.csv
│   ├── longtail_f16.csv
│   └── longtail_f32.csv
└── npu_utils.py
```

六张 canonical CSV 只描述 case，`baseline`、`time`、`score` 必须全部为空。runner 生成本轮 input manifest 和结果副本；GEMM、Conv、LongTail 的 910B 实测延迟统一写入结果 CSV 的 `baseline`，`time` 和 `score` 保持为空。不得在正式 contract 中引入历史 A100 baseline、设备间 score、`status` 或 `error` 列。

GEMM 和 Conv 的标准 contract 使用原生 CANN 路径。两者都让全部 CSV shape 在一个原生进程中执行，共享 ACL Runtime、Device 和 Stream；每个 shape 仍独立创建 tensor、workspace、executor 并使用 ACL Event 计时。Conv 固定计算 `forward + backward-filter + backward-data`，不得改成寒武纪三套 CSV 的测法。

## 创建容器

使用已验证的镜像 `swr.cn-north-1.myhuaweicloud.com/deeplink/ascend-nlp-operators:latest`。先创建宿主输出目录；只挂载 results、logs、NPU 设备和驱动，不得覆盖镜像内 `/workspace/operators`。以下命令展示单卡评测；需要其他卡时增加对应 `/dev/davinciN`：

```bash
mkdir -p "$OPERATOR_RESULTS_DIR" "$OPERATOR_LOGS_DIR" && docker run -itd --name ascend-ops-test --entrypoint /bin/bash --privileged=true --network=host --ipc=host --shm-size=128g --workdir /workspace/operators --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -v "$OPERATOR_RESULTS_DIR:/workspace/results:rw" -v "$OPERATOR_LOGS_DIR:/workspace/logs:rw" swr.cn-north-1.myhuaweicloud.com/deeplink/ascend-nlp-operators:latest -lc 'exec tail -f /dev/null'
```

同名容器存在时先确认用途，再决定是否删除或改名。进入容器后执行：

```bash
CANN_ENV=/opt/ascend-dev/cann-8.5.0/set_env.sh; [[ -f "$CANN_ENV" ]] || CANN_ENV=/usr/local/Ascend/cann-8.5.0/set_env.sh; source "$CANN_ENV" && npu-smi info && python3 -c 'import torch, torch_npu; assert torch.npu.is_available(); x=torch.ones(1).npu(); torch.npu.synchronize(); print(torch.__version__, torch_npu.__version__, x)'
```

原生任务还必须确认：

```bash
command -v cmake && command -v g++ && test -f /opt/ascend-dev/cann-8.5.0/include/acl/acl.h && test -f /opt/ascend-dev/cann-8.5.0/include/aclnnop/aclnn_mm.h && test -f /opt/ascend-dev/cann-8.5.0/lib64/libacl_op_compiler.so && test -f /opt/ascend-dev/cann-8.5.0/lib64/libopapi_nn.so
```

## 选择任务

一次 Result Contract 2.0 结果只能绑定一个 BenchmarkSpec。根据任务配置的 `test_case` 选择一项：

| `test_case` | 内容 | BenchmarkSpec |
|---|---|---|
| `accuracy` | CPU/A100 reference 对 910B FP32/FP16 精度 | `benchmark_specs/accuracy.yaml` |
| `gemm` | 原生 ACLNN FP16/FP32 GEMM | `benchmark_specs/gemm.yaml` |
| `conv` | CUDA 口径原生 Conv2d 三段总延迟 | `benchmark_specs/conv.yaml` |
| `longtail` | torch_npu FP32/FP16 长尾算子 | `benchmark_specs/longtail.yaml` |
| `transformer` / `transformer_block` | Encoder/Decoder FP32 inference | `benchmark_specs/transformer_block.yaml` |

不要把 `all` 作为单任务；全量评测时为五个 `test_case` 分别创建任务。

## 公共执行前置

每个任务以以下步骤开始：

```bash
set -euo pipefail; CANN_ENV=/opt/ascend-dev/cann-8.5.0/set_env.sh; [[ -f "$CANN_ENV" ]] || CANN_ENV=/usr/local/Ascend/cann-8.5.0/set_env.sh; source "$CANN_ENV"; export NPU_DEVICE_ID="${NPU_DEVICE_ID:-0}"; test -d /workspace/operators/speed_test; mkdir -p /workspace/results /workspace/logs; rm -f /workspace/results/result.json /workspace/results/result.json.tmp; python3 -c 'import torch, torch_npu, pandas; assert torch.npu.is_available(); torch.npu.set_device(int(__import__("os").environ["NPU_DEVICE_ID"])); torch.ones(1, device="npu"); torch.npu.synchronize()'; BENCHMARK_STARTED_AT_NS=$(date +%s%N)
```

AIBenchAgent 必须注入 `AIBENCH_TASK_ID`、`AIBENCH_WORKLOAD_FINGERPRINT` 和四个 `AIBENCH_BENCHMARK_*` 变量。任一环境检查、编译或执行失败时返回非零，不得从旧产物继续采集成功结果。

## Accuracy

设置 `COLLECTOR_TARGET=accuracy`。每次运行都调用镜像内 `cpu_ground_truth_gen.py`，先在 `/workspace/operators` 下生成临时完整 reference，再用它覆盖 `/workspace/operators/accuracy_reference_cpu`。生成失败时保留旧目录且任务失败；不得改用 `/workspace/results` 或宿主目录。CPU 不支持的 FP16 case 会缺失，collector 只采集非空 dtype 结论。

```bash
mkdir -p /workspace/results/accuracy; rm -f /workspace/results/accuracy/npu_val_result.json /workspace/results/accuracy/npu_val_result.csv /workspace/results/accuracy/aibench_workload_fingerprint.txt; python3 /workspace/scripts/run_accuracy.py --project-root /workspace/operators/accuracy_test --reference-dir /workspace/operators/accuracy_reference_cpu --regenerate-cpu-reference --output-dir /workspace/results/accuracy --device "$NPU_DEVICE_ID" 2>&1 | tee /workspace/logs/accuracy.log
```

runner 接受 validator 的退出码 `2`，因为精度不通过是有效测量而不是执行器故障；缺少结果文件、非法 JSON 或其他退出码仍视为任务失败。

## GEMM

设置 `COLLECTOR_TARGET=gemm`。runner 严格校验两张 224 行 canonical CSV，重新构建并检查 `aclnnMm` 原生二进制，以 `validate=0` 批量执行，最后验证 448 个 `baseline` 均为有限正数且 FP16/FP32 shape 完全一致。

```bash
python3 /workspace/scripts/run_native.py --benchmark gemm --operators-root /workspace/operators --output-dir /workspace/results/gemm --log-dir /workspace/logs/gemm --device "$NPU_DEVICE_ID" --warmup "${BENCH_WARMUP:-10}" --iterations "${NATIVE_GEMM_ITERATIONS:-1000}"
```

禁止用 torch_npu `test_gemm.py` 的结果填充这个 BenchmarkSpec，除非先为 framework 路径建立独立 spec/version。

## Conv2d

设置 `COLLECTOR_TARGET=conv`。runner 严格校验两张 63 行 canonical CSV，重新构建原生 binary，并保持 CUDA 版单入口语义。结果 `baseline` 必须等于 `forward_ms + backward_weight_ms + backward_data_ms`。

```bash
python3 /workspace/scripts/run_native.py --benchmark conv --operators-root /workspace/operators --output-dir /workspace/results/conv --log-dir /workspace/logs/conv --device "$NPU_DEVICE_ID" --warmup "${BENCH_WARMUP:-10}" --iterations "${NATIVE_CONV_ITERATIONS:-1000}"
```

在线编译、首次执行和 `fusion_result.json` 维护都在 ACL Event 计时外。不要通过删除每个 shape 的缓存来“清理”测试；批处理进程必须保留。

## LongTail

设置 `COLLECTOR_TARGET=longtail`，使用 Skill runner 同时执行 `longtail_f32.csv` 和 `longtail_f16.csv`。runner 为本轮生成随机 token、删除旧 `results/torch.json`、固定项目工作目录，并且不传 `--validate`，使本轮 NPU 延迟写入 `baseline`。

```bash
python3 /workspace/scripts/run_longtail.py --operators-root /workspace/operators --output-dir /workspace/results/longtail --log-dir /workspace/logs/longtail --device "$NPU_DEVICE_ID" --warmup "${BENCH_WARMUP:-10}" --iterations "${LONGTAIL_ITERATIONS:-100}"
```

80 个 dtype case 必须全部得到有限正 `baseline`。任何未注册、跳过或运行失败的样例都会令任务非零退出；不要把不完整结果标记为成功。

## Transformer Block

设置 `COLLECTOR_TARGET=transformer_block`。为与 NVIDIA BenchmarkSpec 对齐，contract 固定 FP32、`eval()`、无 backward 的 inference；其他 dtype 或 forward+backward 只能作为探索性测试，不能写入本 spec。

```bash
python3 /workspace/scripts/run_transformer_block.py --project-root /workspace/operators/speed_test/transformer_block --output /workspace/results/transformer/transformer_block_cases.csv --device "$NPU_DEVICE_ID" --d-model 512 --num-heads 8 --ffn-hidden-size 2048 --batch-size 32 --sequence-length 512 --warmup-iterations 20 --measurement-iterations 1000 2>&1 | tee /workspace/logs/transformer_block.log
```

runner 调用工程已有的 NPU 同步计时实现，验证 encoder/decoder 完整性，并把 workload fingerprint 原子写入标准 CSV。

## 生成 Result Contract 2.0

目标任务成功后执行：

```bash
BENCHMARK_FINISHED_AT_NS=$(date +%s%N); DURATION_SECONDS=$(python3 -c 'import sys; print(max((int(sys.argv[2])-int(sys.argv[1]))/1e9,1e-9))' "$BENCHMARK_STARTED_AT_NS" "$BENCHMARK_FINISHED_AT_NS"); python3 /workspace/scripts/collect_cases.py --benchmark "$COLLECTOR_TARGET" --input-dir /workspace/results --output /workspace/results/result.json --duration-seconds "$DURATION_SECONDS"; test -s /workspace/results/result.json
```

collector 会校验字段、dtype、shape、有限正延迟、Conv 三段求和、LongTail run token、Transformer fingerprint 和 accuracy 布尔结论，再原子发布 `result.json`。禁止用 heredoc、日志正则或临时 pandas 脚本手写结果。

## 故障检查

- `acl.h` 或 `aclnn_mm.h` 找不到：镜像缺少完整 Toolkit 或匹配的 910B ops 包；修复镜像后重跑，不要临时增加工程目录挂载。
- `ModuleNotFoundError: tbe`：source 的不是完整 Toolkit，或 `PYTHONPATH` 被覆盖；保留 `set_env.sh` 追加的环境。
- `cmake: command not found`：使用包含编译工具链的镜像，不要在断网容器中依赖临时 apt 安装。
- Conv 明显慢但 torch_npu 正常：确认使用批处理后的 `test_conv_native.py`，并检查 `baseline` 是否确为三段之和。
- LongTail 个别空结果：查看对应日志和 Ascend 日志；正式结果要求 40 个 shape 全部成功，不要填零或复用旧数据。
- CPU reference 生成失败：检查 `/workspace/operators` 是否可写、CPU 不支持算子日志和剩余磁盘空间；不要切换到 `/workspace/results`。
- 精度 FP16 为空：镜像内 CPU reference 不支持该 dtype；这不等价于 FP16 通过。
- 结果采集失败：保留 CSV、JSON 和日志，修复真实执行问题后整项重跑，不要放宽 collector 校验。
