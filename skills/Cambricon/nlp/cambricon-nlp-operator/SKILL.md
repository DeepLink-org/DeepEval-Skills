---
name: cambricon-nlp-operator
description: 在寒武纪 MLU 上执行算子精度与性能评测。用于 GEMM、Conv2d 前向及反向、长尾算子和 Transformer Block 的测试、基准维护、结果校验与 AIBenchAgent Result Contract 2.0 采集；当任务的 chip_type 为 Cambricon、task_type 为 operator，或用户要求运行 CNNL/torch_mlu/MLU590 operator benchmark 时使用。
metadata:
  benchmark_specs:
    accuracy: benchmark_specs/accuracy.yaml
    gemm: benchmark_specs/gemm.yaml
    conv: benchmark_specs/conv.yaml
    convbackdata: benchmark_specs/conv_component.yaml
    convbackfilter: benchmark_specs/conv_component.yaml
    longtail: benchmark_specs/longtail.yaml
    transformer: benchmark_specs/transformer_block.yaml
    transformer_block: benchmark_specs/transformer_block.yaml
---

# Cambricon NLP Operator

使用镜像内置的 `/workspace/operators` 和 Skill 预置到 `/workspace/scripts` 的确定性 Python 脚本。不要临时重写 CSV 解析、case identity、汇总指标或 `result.json`，也不要依赖 Skill 自带的 Shell 包装脚本。

## 环境与目录

要求宿主机配置以下目录：

| 环境变量 | 容器路径 | 用途 |
|---|---|---|
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | CSV、JSON 与逐用例结果 |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | 编译和执行日志 |

镜像 `swr.cn-north-1.myhuaweicloud.com/deeplink/cambricon-nlp-operator:latest` 已包含：

```text
/workspace/operators/
├── accuracy_test/
│   ├── cpu_ground_truth_gen.py
│   ├── mlu_op_validate.py
│   └── passop_config.py
└── speed_test/
    ├── mlu_ops/
    ├── LongTail-Bench_mlu/
    ├── transformer_block/
    ├── mlu_test_gemm.py
    ├── mlu_test_conv.py
    ├── mlu_test_convbackdata.py
    ├── mlu_test_convbackfilter.py
    ├── mlu_test_conv_total.py
    └── *.csv
```

不要挂载 `/workspace/operators`，否则会遮蔽镜像中的源码和内置 baseline。只把结果与日志目录挂成读写。

## 创建容器

先在宿主机创建 `OPERATOR_RESULTS_DIR` 和 `OPERATOR_LOGS_DIR`。Creator 只生成一条 `docker run` 命令，不要拼接 `mkdir`、`export`、管道或其他 Shell 命令。

```bash
docker run -d \
  --name cambricon-ops-test \
  --privileged \
  --shm-size=16g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --workdir /workspace/operators \
  --mount type=bind,source="$OPERATOR_RESULTS_DIR",target=/workspace/results \
  --mount type=bind,source="$OPERATOR_LOGS_DIR",target=/workspace/logs \
  swr.cn-north-1.myhuaweicloud.com/deeplink/cambricon-nlp-operator:latest \
  tail -f /dev/null
```

使用 `config/skills/Cambricon/nlp/operator.json` 的实际路径替换变量。不要再添加 `--device=/dev/cambricon_*`；本流程用 `--privileged` 暴露 MLU 设备。同名容器存在时先确认用途，再决定是否替换。

在宿主机用 `cnmon` 检查设备；在容器内执行：

```bash
python3 -c "import torch, torch_mlu; assert torch.mlu.is_available(); print(torch.mlu.device_count())"
test -f /usr/local/neuware/include/cnnl.h
test -f /usr/local/neuware/lib64/libcnnl.so
test -d /workspace/operators/speed_test
test -d /workspace/results
test -d /workspace/logs
```

## 选择任务

把任务配置中的 `test_case` 作为唯一入口参数：

| `test_case` | 语义 | BenchmarkSpec |
|---|---|---|
| `accuracy` | CPU ground truth 对 MLU FP32 精度 | `benchmark_specs/accuracy.yaml` |
| `gemm` | FP16/FP32 GEMM 实测延迟 | `benchmark_specs/gemm.yaml` |
| `conv` | Conv2d 前向、反向数据、反向权重总延迟 | `benchmark_specs/conv.yaml` |
| `convbackdata` | Conv2d 反向数据延迟 | `benchmark_specs/conv_component.yaml` |
| `convbackfilter` | Conv2d 反向权重延迟 | `benchmark_specs/conv_component.yaml` |
| `longtail` | 40 项 MLU 长尾算子 FP32 延迟 | `benchmark_specs/longtail.yaml` |
| `transformer` / `transformer_block` | Encoder/Decoder FP32 inference 延迟 | `benchmark_specs/transformer_block.yaml` |

一次 Result Contract 2.0 结果只能绑定一个 BenchmarkSpec。不要把 `all` 当成单任务；需要全量评测时，为上表各 `test_case` 分别创建任务。

## 标准执行流程

AIBenchAgent 会注入任务、workload 与 BenchmarkSpec 身份环境变量。根据 `test_case` 只执行下文对应任务的直接命令；Skill 不提供统一 Shell 入口。生成的评测命令必须以以下公共步骤开始：

```bash
set -euo pipefail
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -d /workspace/operators
test -d /workspace/operators/speed_test
mkdir -p /workspace/results /workspace/logs
rm -f /workspace/results/result.json /workspace/results/result.json.tmp
BENCHMARK_STARTED_AT_NS=$(date +%s%N)
```

任务命令成功后，使用实际 `test_case` 作为 `COLLECTOR_TARGET`；仅把 `transformer` 规范化为 `transformer_block`。最后直接调用 Python collector：

```bash
BENCHMARK_FINISHED_AT_NS=$(date +%s%N)
DURATION_SECONDS="$(python3 -c \
  'import sys; print(max((int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000_000, 1e-9))' \
  "$BENCHMARK_STARTED_AT_NS" "$BENCHMARK_FINISHED_AT_NS")"
python3 /workspace/scripts/collect_cases.py \
  --benchmark "$COLLECTOR_TARGET" \
  --input-dir /workspace/results \
  --output /workspace/results/result.json \
  --duration-seconds "$DURATION_SECONDS"
test -s /workspace/results/result.json
```

任一环境检查、编译、测试或校验失败时必须返回非零，并且不得从旧产物继续采集成功结果。

为与 NVIDIA operator Skill 的 Result Contract 保持一致，GEMM、Conv2d 和 LongTail 的输入表 `baseline` 必须为空，本轮 MLU 实测值直接写入结果表的该列并转换为 case `latency_ms`；汇总指标统一使用 `*_baseline_{avg,p50,p95,min,max}_ms` 命名。

## 不可变评测语义

### Accuracy

设置 `COLLECTOR_TARGET=accuracy`，然后直接调用：

```bash
python3 /workspace/scripts/run_accuracy.py \
  --project-root /workspace/operators/accuracy_test \
  --result-dir /workspace/results/accuracy \
  --log-dir /workspace/logs/accuracy
```

runner 在独立子进程中把 `passop_config.device` 固定为 CPU 生成 ground truth，再固定为 MLU 执行验证，避免因容器安装了 `torch_mlu` 而在 MLU 上生成“CPU”真值。

参考工程的 ground-truth generator 只生成 FP32 数据；它没有生成 FP16 数据时，validator 会把缺失的 FP16 目录当作通过。因此 BenchmarkSpec 和 collector 只接纳 `passed_fp32`，不得把该 `passed_fp16` 当成有效测量。需要 FP16 精度时先补齐独立 CPU FP16 ground truth，并升级规范版本。

### GEMM

设置 `COLLECTOR_TARGET=gemm`。必须直接强制重新编译 GEMM sample，再复制只包含 case 与空 `baseline` 的内置表，并对结果副本执行 `mlu_test_gemm.py ... 0`：

```bash
mkdir -p /workspace/logs/build /workspace/logs/gemm /workspace/results/gemm
make -B -C /workspace/operators/speed_test/mlu_ops/gemm_sample -j"$(nproc)" \
  2>&1 | tee /workspace/logs/build/gemm.log
test -x /workspace/operators/speed_test/mlu_ops/gemm_sample/gemm_sample
if ldd /workspace/operators/speed_test/mlu_ops/gemm_sample/gemm_sample | grep -q 'not found'; then
  ldd /workspace/operators/speed_test/mlu_ops/gemm_sample/gemm_sample >&2
  exit 1
fi
python3 -c 'import torch, torch_mlu; assert torch.mlu.is_available() and torch.mlu.device_count() > 0; torch.ones(1, device="mlu"); torch.mlu.synchronize()'

cd /workspace/operators/speed_test
for dtype in FP16 FP32; do
  source_csv="gemm_${dtype}.csv"
  result_csv="/workspace/results/gemm/gemm_${dtype}_result.csv"
  test -f "$source_csv"
  rm -f "$result_csv" "$result_csv.tmp"
  cp "$source_csv" "$result_csv"
  python3 mlu_test_gemm.py "$result_csv" 0 \
    2>&1 | tee "/workspace/logs/gemm/gemm_${dtype}.log"
  test -s "$result_csv"
done
```

FP16 的 `i_d/o_d` 必须为 `0`，FP32 必须为 `1`。本轮实际延迟必须写入结果表的 `baseline`，collector 再将它转换为 case `latency_ms`；不得复用镜像中已经填值的旧表。

### Conv2d

镜像只内置 `conv_FP16.csv` 和 `conv_FP32.csv` 两张统一 case 表，列固定为：

```text
W,H,C,N,OutC,kw,kh,pw,ph,sh,sw,baseline
```

输入 `baseline` 必须全部为空。`test_case=conv` 固定表示三项综合性能：先强制编译 `conv`、`convbackdata`、`convbackfilter`，再由 `mlu_test_conv_total.py` 在临时目录中从统一表派生各分项需要的 `id/wd/od` 列，并以模式 `0` 依次实测前向、反向数据和反向权重。按 `W,H,C,N,OutC,kw,kh,pw,ph,sh,sw` 一对一校验后生成：

```text
baseline = forward_baseline_ms + backward_data_baseline_ms + backward_filter_baseline_ms
```

主结果 `conv_total_<dtype>_result.csv` 只保存统一 case 列和本轮生成的 `baseline`；三项明细保存在 `conv_total_<dtype>_result_components.csv`，不进入 NVIDIA 对齐的 Conv case contract。任一分项缺失、重复、非有限或非正时失败。显式 `convbackdata` / `convbackfilter` 任务也从同一张统一表派生输入，只运行目标分项并把本轮延迟写入结果 `baseline`；不再依赖外部 `convbk_*.csv`。

先按任务设置 `COLLECTOR_TARGET=conv`、`convbackdata` 或 `convbackfilter`，然后直接重新编译对应 sample：`conv` 编译三项，单独反向任务只编译目标项。

```bash
mkdir -p /workspace/logs/build /workspace/logs/conv /workspace/results/conv
if [ "$COLLECTOR_TARGET" = conv ]; then
  CONV_SAMPLES="conv convbackdata convbackfilter"
else
  CONV_SAMPLES="$COLLECTOR_TARGET"
fi
for sample in $CONV_SAMPLES; do
  sample_dir="/workspace/operators/speed_test/mlu_ops/${sample}_sample"
  make -B -C "$sample_dir" -j"$(nproc)" \
    2>&1 | tee "/workspace/logs/build/${sample}.log"
  test -x "$sample_dir/${sample}_sample"
  if ldd "$sample_dir/${sample}_sample" | grep -q 'not found'; then
    ldd "$sample_dir/${sample}_sample" >&2
    exit 1
  fi
done
python3 -c 'import torch, torch_mlu; assert torch.mlu.is_available() and torch.mlu.device_count() > 0; torch.ones(1, device="mlu"); torch.mlu.synchronize()'

cd /workspace/operators/speed_test
if [ "$COLLECTOR_TARGET" = conv ]; then
  for dtype in FP16 FP32; do
    source_csv="conv_${dtype}.csv"
    result_csv="/workspace/results/conv/conv_total_${dtype}_result.csv"
    details_csv="/workspace/results/conv/conv_total_${dtype}_result_components.csv"
    test -f "$source_csv"
    rm -f "$result_csv" "$result_csv.tmp" "$details_csv" "$details_csv.tmp"
    python3 mlu_test_conv_total.py \
      --dtype "$dtype" --input "$source_csv" --component total --output "$result_csv" \
      2>&1 | tee "/workspace/logs/conv/conv_total_${dtype}.log"
    test -s "$result_csv"
    test -s "$details_csv"
  done
else
  if [ "$COLLECTOR_TARGET" = convbackdata ]; then
    CONV_STEM=convbk_data
    CONV_COMPONENT=backward_data
  else
    CONV_STEM=convbk_filter
    CONV_COMPONENT=backward_filter
  fi
  for dtype in FP16 FP32; do
    source_csv="conv_${dtype}.csv"
    result_csv="/workspace/results/conv/${CONV_STEM}_${dtype}_result.csv"
    test -f "$source_csv"
    rm -f "$result_csv" "$result_csv.tmp"
    python3 mlu_test_conv_total.py \
      --dtype "$dtype" --input "$source_csv" \
      --component "$CONV_COMPONENT" --output "$result_csv" \
      2>&1 | tee "/workspace/logs/conv/${CONV_STEM}_${dtype}.log"
    test -s "$result_csv"
  done
fi
```

### LongTail

镜像只内置一张三列表 `longtail_perf.csv`：

```text
NO,op,baseline
```

`NO` 和 `op` 必须分别唯一，输入 `baseline` 必须全部为空。设置 `COLLECTOR_TARGET=longtail`，然后直接调用：

```bash
python3 -c 'import torch, torch_mlu; assert torch.mlu.is_available() and torch.mlu.device_count() > 0; torch.ones(1, device="mlu"); torch.mlu.synchronize()'
python3 /workspace/scripts/run_longtail.py \
  --project-root /workspace/operators/speed_test/LongTail-Bench_mlu \
  --source /workspace/operators/speed_test/longtail_perf.csv \
  --output /workspace/results/longtail/longtail_result.csv \
  --manifest /workspace/results/longtail/longtail_cases_input.csv \
  --log /workspace/logs/longtail/longtail.log
```

runner 为本轮生成 `longtail_cases_input.csv`，补充 LongTail-Bench 兼容列和随机 `aibench_run_token`，删除旧 `results/torch.json`，以 `--store_input_shape` 执行但不传 `--validate`。LongTail-Bench 因此把本轮 MLU 延迟写入结果 `baseline`。runner 在原子发布 `longtail_result.csv` 前校验 token、case 顺序、覆盖范围及全部正数 baseline；collector 再使用 manifest 做相同的本轮归属校验。不得从旧 `time/score` 或历史 baseline 采集结果。

### Transformer Block

设置 `COLLECTOR_TARGET=transformer_block`，然后直接调用 `run_transformer_block.py`，不要执行镜像中的 `transformer_block/test.py`：

```bash
python3 -c 'import torch, torch_mlu; assert torch.mlu.is_available() and torch.mlu.device_count() > 0; torch.ones(1, device="mlu"); torch.mlu.synchronize()'
mkdir -p /workspace/results/transformer /workspace/logs/transformer
rm -f /workspace/results/transformer/transformer_block_cases.csv \
  /workspace/results/transformer/transformer_block_cases.csv.tmp
python3 /workspace/scripts/run_transformer_block.py \
  --project-root /workspace/operators/speed_test/transformer_block \
  --output /workspace/results/transformer/transformer_block_cases.csv \
  --d-model 512 --num-heads 8 --ffn-hidden-size 2048 \
  --batch-size 32 --sequence-length 512 \
  --warmup-iterations 20 --measurement-iterations 1000 \
  2>&1 | tee /workspace/logs/transformer/transformer_block.log
test -s /workspace/results/transformer/transformer_block_cases.csv
```

原脚本使用 `.train()`；确定性 runner 使用 `eval()`、`torch.inference_mode()`，并在 MLU 同步边界内测量。

默认参数为 `d_model=512`、`num_heads=8`、`ffn_hidden_size=2048`、`batch_size=32`、序列长度 `512`、warmup `20`、测量 `1000` 次。改变参数时必须通过 runner 的显式 CLI 参数；这些参数会写入 case dimensions 和 workload 绑定的 CSV。

## Result Contract 2.0

必须由 `/workspace/scripts/collect_cases.py` 生成结果。collector 会：

- 从本轮生成的 `baseline`、Transformer 实测 CSV 或 accuracy JSON 生成逐用例 cases；
- 校验 dtype 编码、shape 字段、正数延迟、Conv/LongTail case 覆盖、运行标识和 Transformer fingerprint；
- 生成规范声明的 summary metrics；
- 使用 Agent 注入的 BenchmarkSpec 四字段，不自行拼接 `case_key`；
- 先写 `result.json.tmp`，解析确认后再原子替换。

禁止使用 heredoc、临时 pandas 脚本或日志正则手写 `result.json`。禁止用占位、估算、模拟、全零或本轮之前的测量报告 `status=success`。

## 基准维护

GEMM、Conv2d 和 LongTail 标准任务都读取镜像内置的空 baseline 模板，在结果目录中生成本轮 MLU baseline；不得覆盖镜像内模板，也不得把历史结果重新作为输入。这三个任务不再使用参考 baseline 与 score 语义。

## 故障检查

- 编译失败：检查 `NEUWARE_HOME`、`libcnnl.so`、`libcnrt.so` 和 `ldd` 的 `not found`。
- MLU 不可用：在宿主机检查 `cnmon` 与设备节点，在容器内检查 `torch.mlu.is_available()`。
- GEMM/Conv 失败：检查新编译二进制、dtype 编码、统一 CSV 列名、输入空 baseline 和结果正数 baseline。
- 长尾失败：检查三列表输入、`longtail_cases_input.csv`、`aibench_run_token`、`LongTail-Bench_mlu/results/torch.json` 和日志中的 traceback。
- Transformer 失败：确认模型与所有输入位于 MLU，且 `d_model` 能被 `num_heads` 整除。
- 结果校验失败：保留 CSV/JSON 与日志，修复真实执行问题后重跑；不要放宽 collector。
