---
name: metax-nlp-operator
description: 沐曦 MetaX GPU 算子精度与性能评测技能。支持 accuracy、GEMM、Conv2d（FP16/FP32）、长尾算子、Transformer Block 测试任务，用于指导 executor 完成容器启动、MetaX/MACA 环境确认、编译、基准值生成、精度验证、结果持久化与性能指标采集的完整流程。
metadata:
  benchmark_specs:
    accuracy: benchmark_specs/accuracy.yaml
    gemm: benchmark_specs/gemm.yaml
    conv: benchmark_specs/conv.yaml
    longtail: benchmark_specs/longtail.yaml
    transformer_block: benchmark_specs/transformer_block.yaml
---

### 触发条件

当用户提出以下任一请求时启动本 Skill：

- 在 MetaX GPU 上生成 GEMM 或 Conv2d 算子基准值
- 执行 MetaX 长尾算子或 Transformer Block 性能测试
- 执行 MetaX 算子精度验证
- 运行 MetaX operator benchmark

---

### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---|---|---|---|
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | 是 | 宿主机绝对路径，挂载后存放性能 CSV、精度 CSV/JSON 和汇总 JSON |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | 是 | 宿主机绝对路径，挂载后存放编译、精度与性能日志 |

**说明**：

- 算子代码已打包进镜像的 `/workspace/operators`，不要从宿主机额外挂载源码。
- 所有可持久化结果写入 `/workspace/results`，所有日志写入 `/workspace/logs`。
- 性能测试前先把镜像内置 CSV 复制到结果目录，只对副本运行批量脚本，避免覆盖镜像内容。

---

### 支持的算子配置

**当前支持算子**：

- **GEMM**：FP16、FP32，多种 M/N/K 和转置组合。
- **Conv2d**：FP16、FP32，多种输入尺寸、卷积核、padding 和 stride。
- **长尾算子**：LongTail-Bench 普通版本和 FP16 版本。
- **Transformer Block**：PyTorch Encoder Layer 和 Decoder Layer 前向性能。
- **精度测试**：PyTorch 算子前向输出、输入梯度和参数梯度验证。

**当前支持任务**：

- 性能 baseline 生成：在目标 MetaX GPU 上运行全量 CSV，并把实测耗时写入结果副本的 `baseline`。
- 精度验证：使用镜像内置 CPU 真值验证 MetaX 前向输出、输入梯度和参数梯度。
- 单算子冒烟测试：仅用于排障，不可替代全量结果。
- 结果汇总：采集性能 CSV、精度 CSV/JSON、日志，并按 Generator 当前协议生成 `result.json`。

**硬件要求**：

- 至少 1 张 MetaX GPU。
- 宿主机已安装匹配的 MetaX 驱动，并存在 `/dev/mxcd` 和 `/dev/dri`。
- 容器内 MACA、MetaX PyTorch 与宿主机驱动版本兼容。

---

### 依赖要求

评测代码已经完整打包在镜像中，不需要从宿主机挂载、复制或下载源码。容器内固定目录如下：

```text
/workspace/operators/
├── accuracy_test/
│   ├── cpu_ground_truth/
│   ├── cuda_ground_truth_gen.py
│   ├── cuda_op_validate.py
│   ├── mx_accuracy_result/
│   ├── op_config.py
│   ├── op_conv_config.py
│   ├── readme.md
│   └── utils.py
└── speed_test/
    ├── cuda_ops/
    ├── LongTail-Bench/
    ├── LongTail-Bench-fp16/
    ├── transformer_block/
    ├── test_gemm.py
    ├── test_conv.py
    └── 各类性能测试 CSV
```

- 精度测试固定从 `/workspace/operators/accuracy_test` 运行。
- GEMM、Conv2d、长尾算子和 Transformer Block 性能测试固定从 `/workspace/operators/speed_test` 运行。
- 结果写入 `/workspace/results`，日志写入 `/workspace/logs`。

镜像需包含 `/opt/maca`、CUDA 兼容层、MetaX PyTorch/torchvision、mcBLAS、mcDNN、
MACA Runtime、CMake、make、g++ 和 pandas；启动后按下文命令验证，不要仅凭镜像标签推断环境可用。

---

## 第一阶段：容器启动

### 选择算子类型

以任务配置中的 `test_case` 为唯一选择依据。支持 `accuracy`、`gemm`、`conv`、
`longtail`、`transformer_block` 和 `all`。只执行指定类别；只有 `all` 才执行全部类别。

### 容器创建命令

创建容器前，由宿主机或 Agent 预先创建 `OPERATOR_RESULTS_DIR` 和
`OPERATOR_LOGS_DIR`。Creator 只生成一条 `docker run` 命令，不要把 `docker pull`、
`mkdir`、`export`、管道或其他 Shell 命令拼接进去。

```bash
docker run -d \
  --name metax-ops-test \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=16g \
  --workdir /workspace/operators \
  -v "$OPERATOR_RESULTS_DIR:/workspace/results" \
  -v "$OPERATOR_LOGS_DIR:/workspace/logs" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/metax-nlp-operator:latest \
  tail -f /dev/null
```

- 必须用 `config/skills/MetaX/nlp/operator.json` 中的实际值替换两个宿主机路径变量。
- 不要挂载 `/workspace/operators`，否则会遮蔽镜像内置评测代码。
- 同名容器存在时先检查其用途；只有确认可以替换后才执行 `docker rm -f metax-ops-test`。
- 镜像不存在时，由宿主机或 Agent 单独执行 `docker pull`，不要让 Creator 生成复合命令。

### 容器管理与环境确认

```bash
docker exec -it metax-ops-test /bin/bash

# 以下命令在容器内执行
test -d /opt/maca
test -d /opt/maca/tools/cu-bridge
```

确认输出挂载与镜像内容：

```bash
ls -lh /workspace/operators/accuracy_test
ls -lh /workspace/operators/speed_test
test -w /workspace/results
test -w /workspace/logs
```

---

## 第二阶段：容器中执行评测

先统一创建输出目录；后续命令不得把最终结果留在源码目录：

```bash
export RESULT_ROOT=/workspace/results
export LOG_ROOT=/workspace/logs
mkdir -p "$RESULT_ROOT"/{accuracy,gemm,conv,longtail,transformer} \
         "$LOG_ROOT"/{accuracy,gemm,conv,longtail,transformer}
set -o pipefail
```

### 精度测试

**使用镜像内 CPU 真值**

CPU 真值已经随镜像生成并存放在：

```text
/workspace/operators/accuracy_test/cpu_ground_truth
```

不要重新运行 `cuda_ground_truth_gen.py`。直接使用该目录中的真值执行单算子、普通算子全量和 Conv 全配置验证。

**MetaX 普通算子全量验证**

```bash
python3 /workspace/scripts/run_accuracy.py \
  --project-root /workspace/operators/accuracy_test \
  --reference-dir /workspace/operators/accuracy_reference_cpu \
  --regenerate-cpu-reference \
  --output-dir "$RESULT_ROOT/accuracy" \
  --device "${CUDA_DEVICE_ID:-0}" \
  2>&1 | tee "$LOG_ROOT/accuracy/accuracy.log"
```
**精度结果**

完整验证生成：

```text
$RESULT_ROOT/accuracy/mx_val_result.csv
$RESULT_ROOT/accuracy/mx_val_result.json
```

CSV 和 JSON 保存相同的聚合通过状态：

- CSV 用于 Excel、人工筛选和统计。
- JSON 用于程序读取。
- 详细前向和梯度误差只在运行日志中。

**精度结果限制**

- 最终 `passed` 合并 FP32 和 FP16，不分别输出状态。
- CPU FP16 支持不完整。
- 缺少 FP16 真值目录时，当前验证代码可能将 FP16 视为通过；必须检查真值目录和日志，不能直接宣称 FP16 已验证。
- SVD、特征值、排序索引、Pooling 索引、随机和非确定性算子可能存在合理后端差异。
- 必须记录参考设备、MetaX 型号、PyTorch 版本和 MACA 版本。

---

### GEMM、Conv2d 性能测试

#### 编译 GEMM 和 Conv2d

批量测试脚本固定使用 `cuda_ops/build/gemm` 和 `cuda_ops/build/conv`，因此构建到 `build`：

```bash
cd "/workspace/operators/speed_test"

/opt/conda/bin/cmake \
  -S cuda_ops \
  -B cuda_ops/build \
  -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tee "$LOG_ROOT/gemm/compile.log"

/opt/conda/bin/cmake \
  --build cuda_ops/build \
  --parallel 2 \
  2>&1 | tee -a "$LOG_ROOT/gemm/compile.log"
```

验证：

```bash
test -x "/workspace/operators/speed_test/cuda_ops/build/gemm"
test -x "/workspace/operators/speed_test/cuda_ops/build/conv"
ldd "/workspace/operators/speed_test/cuda_ops/build/gemm"
ldd "/workspace/operators/speed_test/cuda_ops/build/conv"
```

若 `ldd` 出现 `not found`，不得继续性能测试。

#### GEMM 批量测试

复制镜像内参数 CSV 后，只对结果副本生成 baseline：

```bash
cd "/workspace/operators/speed_test"

for dtype in f16 f32; do
  cp "gemm_${dtype}.csv" "$RESULT_ROOT/gemm/gemm_${dtype}.csv"
done

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python test_gemm.py \
  "$RESULT_ROOT/gemm/gemm_f16.csv" 16 1 \
  2>&1 | tee "$LOG_ROOT/gemm/gemm_f16.log"

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python test_gemm.py \
  "$RESULT_ROOT/gemm/gemm_f32.csv" 32 1 \
  2>&1 | tee "$LOG_ROOT/gemm/gemm_f32.log"
```

第三个参数沿用镜像内现有脚本接口；当前 MetaX 实测值写入 `baseline`，不计算
`time` 或 `score`。不得把同设备生成的 baseline 伪装成相对评分。

#### Conv2d 批量测试

```bash
cd "/workspace/operators/speed_test"

for dtype in f16 f32; do
  cp "conv_${dtype}.csv" "$RESULT_ROOT/conv/conv_${dtype}.csv"
done

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python test_conv.py \
  "$RESULT_ROOT/conv/conv_f16.csv" 16 1 \
  2>&1 | tee "$LOG_ROOT/conv/conv_f16.log"

CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python test_conv.py \
  "$RESULT_ROOT/conv/conv_f32.csv" 32 1 \
  2>&1 | tee "$LOG_ROOT/conv/conv_f32.log"
```

### 长尾算子

#### FP32/FP16 版本

```bash
/opt/conda/bin/python /workspace/scripts/run_longtail.py \
  --operators-root /workspace/operators \
  --output-dir "$RESULT_ROOT/longtail" \
  --log-dir "$LOG_ROOT/longtail" \
  --device "${CUDA_DEVICE_ID:-0}" \
  --warmup "${BENCH_WARMUP:-10}" \
  --iterations "${LONGTAIL_ITERATIONS:-100}"
```

runner 为每次运行生成 token，校验 FP32/FP16 case 集合，统一把原始秒转换为 Result Contract 要求的毫秒，并原子发布结果 CSV。
必须只调用该 runner，不要直接调用 `long_tail_bench.api.api`，也不要向其传入
`--exec_mode`、`--iterations` 或 `--warmup` 等不支持的参数。runner 会选择带
MetaX PyTorch 的解释器，并自动从 FP16 输入中排除 FP32-only 的 `batched_nms`。

### Transformer Block 测试

```bash
set -euo pipefail
: "${AIBENCH_WORKLOAD_FINGERPRINT:?AIBENCH_WORKLOAD_FINGERPRINT is required}"
/opt/conda/bin/python /workspace/scripts/run_transformer_block.py \
  --project-root /workspace/operators/speed_test/transformer_block \
  --output "$RESULT_ROOT/transformer/transformer_block_cases.csv" \
  --device "${CUDA_DEVICE_ID:-0}" \
  --warmup-iterations "${BENCH_WARMUP:-20}" \
  --measurement-iterations "${TRANSFORMER_ITERATIONS:-1000}" \
  2>&1 | tee "$LOG_ROOT/transformer/transformer_block.log"
```

默认配置为 `d_model=512`、`n_head=8`、`ffn_hidden=2048`、`batch_size=32`、
`seq_len=512`、预热 20 次、迭代 1000 次和 FP32。

当前代码执行 FP32 inference 前向，不执行 backward；不要描述为完整训练性能。

---

### 关键性能指标

| 类别 | 必采指标 | 单位与语义 |
|---|---|---|
| GEMM | `baseline` | MetaX 实测耗时，ms，越低越好 |
| Conv2d | `baseline` | 前向、权重反向和输入反向耗时之和，ms，越低越好 |
| 长尾算子 | `baseline` | 单次延迟，ms，越低越好 |
| Transformer Block | `latency_ms` | Encoder/Decoder 前向单次迭代耗时，ms |
| 精度 | `passed` 与用例计数 | 基于镜像内 CPU 真值；必须结合误差日志判断 |

不得混合毫秒与秒，不得根据单算子冒烟结果推断全量结果。

### 指标采集

#### GEMM、Conv2d

采集以下 CSV 的全部 `baseline` 和参数列：

```text
$RESULT_ROOT/gemm/gemm_f16.csv
$RESULT_ROOT/gemm/gemm_f32.csv
$RESULT_ROOT/conv/conv_f16.csv
$RESULT_ROOT/conv/conv_f32.csv
```

确认：

```bash
/opt/conda/bin/python - <<'PY'
import pandas as pd

paths = [
    "/workspace/results/gemm/gemm_f16.csv",
    "/workspace/results/gemm/gemm_f32.csv",
    "/workspace/results/conv/conv_f16.csv",
    "/workspace/results/conv/conv_f32.csv",
]
for path in paths:
    df = pd.read_csv(path)
    assert "baseline" in df.columns
    # test_gemm.py/test_conv.py may omit empty optional columns; if present,
    # they must remain empty and are not part of the Result Contract metrics.
    for optional in ("time", "score"):
        if optional in df.columns:
            assert df[optional].isna().all()
    values = pd.to_numeric(df["baseline"], errors="coerce")
    assert values.notna().all() and (values > 0).all()
    print(path, len(df), values.notna().sum())
PY
```

#### 长尾算子

采集（runner 输出的本轮副本）：

```text
$RESULT_ROOT/longtail/longtail_fp32.csv
$RESULT_ROOT/longtail/longtail_fp16.csv
```

读取每个算子的 `op` 和 `baseline`。

#### Transformer Block

采集 `$RESULT_ROOT/transformer/transformer_block_cases.csv`，其中必须包含 encoder 和
decoder 两个 case 及本轮 `aibench_workload_fingerprint`。

#### 精度测试

采集：

```text
$RESULT_ROOT/accuracy/mx_val_result.csv
$RESULT_ROOT/accuracy/mx_val_result.json
$RESULT_ROOT/accuracy/aibench_workload_fingerprint.txt
$LOG_ROOT/accuracy/accuracy.log
```

不得仅交付 CSV/JSON 而丢弃误差日志。

#### result.json 汇总约束

#### 执行契约（Result Contract 2.0）

所有由 Generator 生成的 benchmark shell 都必须以以下约束运行：

```bash
set -euo pipefail
: "${AIBENCH_TASK_ID:?AIBENCH_TASK_ID is required}"
: "${AIBENCH_WORKLOAD_FINGERPRINT:?AIBENCH_WORKLOAD_FINGERPRINT is required}"
: "${AIBENCH_BENCHMARK_SPEC_ID:?AIBENCH_BENCHMARK_SPEC_ID is required}"
: "${AIBENCH_BENCHMARK_SPEC_VERSION:?AIBENCH_BENCHMARK_SPEC_VERSION is required}"
: "${AIBENCH_BENCHMARK_CASE_SCHEMA_VERSION:?AIBENCH_BENCHMARK_CASE_SCHEMA_VERSION is required}"
: "${AIBENCH_BENCHMARK_SPEC_SHA256:?AIBENCH_BENCHMARK_SPEC_SHA256 is required}"
```

AIBenchAgent 会注入 `AIBENCH_TASK_ID`、`AIBENCH_WORKLOAD_FINGERPRINT` 和
`AIBENCH_BENCHMARK_*` 身份变量。外层 shell 必须在调用任何 runner 或 collector
时保留这些变量，不得重新生成、覆盖、清空或使用固定常量替代。若变量缺失，必须
立即退出，禁止继续执行并生成结果。

各 runner 只负责执行实际 benchmark、校验本轮产物并原子写入 CSV；不得由外层临时
脚本解析日志、拼接 `case_key`、计算 summary metrics 或手写 `result.json`。collector
只负责从本轮结果目录读取 runner 产物、校验 workload fingerprint 和 case identity，
并生成 Result Contract 2.0 的 `result.json`。

Transformer runner 必须直接导入并测量 `EncoderLayer` 和 `DecoderLayer`，不调用或
解析镜像内 `test.py` 的标准输出；runner 生成的 CSV 必须包含 encoder、decoder 各一条
记录及本轮 `aibench_workload_fingerprint`。

必须使用 `collect_cases.py` 生成符合 Result Contract 2.0 的结果，禁止手写或拼接
`result.json`。结果顶层结构必须包含 `schema_version`、`task_id`、`status`、
`benchmark`、`metrics`、`cases` 和 `metadata`；`task_id`、benchmark spec 身份和
`workload_fingerprint` 必须从本轮环境变量或实际产物读取，禁止固定常量。

- 只汇总 `test_case` 实际执行的类别；禁止为未执行类别填入零值并宣称成功。
- GEMM、Conv2d 和长尾算子从结果目录中的实际 CSV 汇总 `baseline`；不要从镜像源码目录读取被遗漏的旧文件。
- 精度测试汇总实际用例数、通过数和通过率，并保留 CPU 真值来源和详细误差日志作为证据。
- Transformer 的 `latency_ms` 与 GEMM/Conv/LongTail 使用相同的毫秒单位。
- `measurement_count` 使用实际纳入汇总的测量项数量，`source` 指向实际读取的 CSV、JSON 或日志，`duration_seconds` 使用真实测量时长。

CSV/JSON 保留逐用例明细，`result.json` 只保留有限数值汇总和 Generator 当前协议
要求的测量证据。禁止使用占位、估算、模拟、默认值或全零数据生成成功结果。

统一采集命令如下（`COLLECTOR_TARGET` 必须与配置中的 `test_case` 一致）：

```bash
BENCHMARK_FINISHED_AT_NS=$(date +%s%N)
DURATION_SECONDS=$(python3 -c 'import sys; print(max((int(sys.argv[2])-int(sys.argv[1]))/1e9,1e-9))' "$BENCHMARK_STARTED_AT_NS" "$BENCHMARK_FINISHED_AT_NS")
python3 /workspace/scripts/collect_cases.py \
  --benchmark "$COLLECTOR_TARGET" \
  --input-dir /workspace/results \
  --output /workspace/results/result.json \
  --duration-seconds "$DURATION_SECONDS"
test -s /workspace/results/result.json
```

collector 统一将性能 CSV 的 `baseline` 映射为 `cases[*].metrics.latency_ms`，并计算
`*_total_cases`、`*_success_cases`、`*_failed_cases`、平均值、P50、P95、最小值和最大值。
任一 case 缺失、baseline 非有限正数、case identity 不一致、fingerprint 过期或结果
文件缺失时必须返回非零退出码，不得继续报告成功。

Shell 变量与 heredoc 内的 Python 变量不共享作用域。使用 `<<'PY'` 时，把动态值
通过位置参数或环境变量显式传入 Python，例如：

```bash
START_NS=$(date +%s%N)
# 在此执行 test_case 指定的实际测试
END_NS=$(date +%s%N)

python3 - "$START_NS" "$END_NS" <<'PY'
import math
import sys

start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
duration_seconds = (end_ns - start_ns) / 1_000_000_000
if not math.isfinite(duration_seconds) or duration_seconds <= 0:
    raise ValueError(f"invalid duration_seconds: {duration_seconds}")

# 按 Generator 当前提供的协议构造 result.json，并使用 duration_seconds。
PY
```

写入成功结果后，重新读取 `/workspace/results/result.json`，并按 Generator 当前提供
的校验规则验证通过，再以退出码 0 结束。测试、汇总或协议校验失败时，写入失败
结果并返回非零退出码；不得吞掉 `PIPESTATUS` 或用兜底成功结果掩盖失败。

---
