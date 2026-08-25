---
name: cambricon-nlp-operator
description: 寒武纪 MLU 算子精度与性能评测技能。支持 GEMM、Conv2d（FP16/FP32）、长尾算子、Transformer Block等基准值生成与性能测试，用于指导 executor 完成容器启动、编译、基准值生成、测试验证与性能指标采集的完整流程。
---

### 触发条件

当用户说以下任意内容时启动：

- "帮我生成 GEMM 算子基准值"
- "在寒武纪 MLU 上生成 GEMM 算子基准值"
- "跑一下 Conv2d 算子基准"
- "生成 CNNL 算子 baseline"
- "帮我跑 MLU 长尾算子基准测试"
- "验证 MLU 算子精度"
- "测试 MLU 算子性能"
- "运行 operator benchmark"

### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | 是 | 宿主机绝对路径，挂载后存放基准值和测试结果 CSV/JSON |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | 是 | 宿主机绝对路径，挂载后存放编译与测试日志 |

**说明**：

- 算子代码已打包进镜像的 `/workspace/operators`。
- **OPERATOR_RESULTS_DIR** 存放评测过程中生成的基准值、测试结果 CSV 和汇总 JSON。
- **OPERATOR_LOGS_DIR** 存放编译日志和测试日志。

**目录结构说明**：

镜像内默认结构如下：

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
    ├── gemm_FP16.csv / gemm_FP32.csv
    ├── conv_FP16.csv / conv_FP32.csv
    ├── convbk_data_FP16.csv / convbk_data_FP32.csv
    ├── convbk_filter_FP16.csv / convbk_filter_FP32.csv
    ├── conv_total_FP16.csv / conv_total_FP32.csv
    └── longtail_perf.csv
```

**注意**：

- 镜像内置 CSV 已包含可信的 `baseline` 或 `Forward_basetime`，默认直接用于目标 MLU 测试；不要在同一次评测中先重算 baseline 再计算 score。
- `conv_total_FP16.csv` 和 `conv_total_FP32.csv` 的 `baseline`/`basetime` 是外部提供的综合基准；只读取它，不要用三项分测 baseline 重算或覆盖。
- 测试前必须将内置 CSV 复制到 `OPERATOR_RESULTS_DIR`，只对副本运行测试模式，避免覆盖镜像内的基准文件。
- 所有可持久化结果写入 `/workspace/results`，所有日志写入 `/workspace/logs`。

### 支持的算子配置

**当前支持算子**（共 4 类）：

- **GEMM**：矩阵乘法算子，支持 FP16 和 FP32、多种 M/N/K 维度与转置组合。
- **Conv2d**：二维卷积前向、反向数据和反向权重，支持 FP16 和 FP32、多种输入尺寸、卷积核、padding 与 stride 组合。
- **长尾算子**：基于 MLU 版 LongTail-Bench 的 40 项 PyTorch 算子，包括 bbox2delta、bbox_overlaps、l2_loss 等。
- **Transformer Block**：基于 `torch_mlu` 的 Encoder/Decoder Layer 性能测试。

**当前支持任务**：

- 性能测试（默认）：读取镜像内置 baseline，在目标 MLU 上填写 `time` 并计算 `baseline/time` 得分。
- 基准值生成（仅显式请求）：只在用户明确要求更新参考基准时使用模式 `0`，将结果保存为独立基准文件；不得随后用同一设备的该文件给本次测试打分。

**硬件要求**：

- 至少 1 张寒武纪 MLU。
- 宿主机已安装匹配的寒武纪驱动并存在 `/dev/cambricon_ctl`、`/dev/cambricon_dev*`。
- 容器内 Neuware、PyTorch 和 `torch_mlu` 版本相互兼容。

### 依赖要求

**Docker 镜像**：`aibench-mlu590-operator:v1.1`。该镜像已经包含
`/workspace/operators` 下的全部精度、原生算子、长尾算子和 Transformer Block
测试代码，不要额外挂载源码。容器内已预装：

- PyTorch 与 `torch_mlu`（Transformer Block、长尾算子依赖）
- Neuware CNNL/CNRT（原生算子编译与运行依赖）
- make、g++（编译依赖）
- pandas、NumPy（批量测试脚本依赖）
- Python 3.x

---

## 第一阶段：容器启动

### 选择算子类型

以任务配置中的 `test_case` 为唯一选择依据。支持
`accuracy`、`gemm`、`conv`、`convbackdata`、`convbackfilter`、`longtail`、
`transformer` 和 `all`；其中 `conv` 固定表示前向、反向数据和反向权重的综合测试。

### 容器创建命令

**挂载权限约定**：

- `readonly`：只读，用于不需要修改的输入目录。
- 默认读写：用于结果、日志及需要写回 CSV 的项目目录。

**公共参数**：

| 参数 | 说明 |
|------|------|
| `--privileged` | 允许容器访问寒武纪设备和驱动能力 |
| `--shm-size=16g` | 设置共享内存大小，避免大输入导致空间不足 |
| `--ipc=host` | 使用宿主机 IPC 命名空间 |
| `--ulimit memlock=-1` | 取消锁页内存限制 |
| `--workdir /workspace/operators` | 设置容器内算子代码工作目录 |

**公共卷挂载**：

```bash
--mount type=bind,source="$OPERATOR_RESULTS_DIR",target=/workspace/results \
--mount type=bind,source="$OPERATOR_LOGS_DIR",target=/workspace/logs
```

创建容器前，由宿主机或 Agent 预先创建 `OPERATOR_RESULTS_DIR` 和 `OPERATOR_LOGS_DIR`。Creator 只生成一条 `docker run` 命令，不要把 `mkdir`、`export`、管道或其他 Shell 命令拼接进去。

**基础启动命令**：

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
  aibench-mlu590-operator:v1.1 \
  tail -f /dev/null
```

**注意**：

- 同名容器存在时先检查其用途；只有确认可以替换后才执行 `docker rm -f cambricon-ops-test`。
- 不要挂载 `/workspace/operators`，否则会遮蔽镜像中自带的精度和性能测试代码。
- `--privileged` 已开放 MLU 设备；本流程不要再增加 `--device=/dev/cambricon_*`。
- 必须用 `config/skills/Cambricon/nlp/operator.json` 中的实际值替换两个路径变量；

### 容器管理命令

**进入已创建的容器**：

```bash
# 如果容器已在运行
docker exec -it cambricon-ops-test /bin/bash

# 如果容器已停止，先启动再进入
docker start cambricon-ops-test
docker exec -it cambricon-ops-test /bin/bash
```

**验证容器环境**：

```bash
# 检查宿主机 MLU 设备
cnmon
ls -l /dev/cambricon_ctl /dev/cambricon_dev* 2>/dev/null

# 以下命令在容器内执行
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ls "$NEUWARE_HOME/include/cnnl.h"
ls "$NEUWARE_HOME/lib64/libcnnl.so"
python3 -c "import torch, torch_mlu; print(torch.__version__); print(torch.mlu.is_available()); print(torch.ones(1).to('mlu'))"

# 检查挂载目录
ls -lh /workspace/operators/
ls -lh /workspace/operators/accuracy_test/
ls -lh /workspace/operators/speed_test/
ls -lh /workspace/results/
ls -lh /workspace/logs/
```

不要要求容器内必须存在 `cnmon`；设备监控优先在宿主机执行。

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

### 算子精度测试

CPU ground truth 写入 `results/accuracy/ground_truth`，MLU 验证结果写入 `results/accuracy`，日志写入 `logs/accuracy`：

```bash
cd /workspace/operators/accuracy_test
OP_TEST_DEVICE=cpu python3 -c "from passop_config import device; assert str(device) == 'cpu', device"
OP_TEST_DEVICE=cpu PYLOGLEVEL=INFO \
python3 cpu_ground_truth_gen.py "$RESULT_ROOT/accuracy/ground_truth" \
  2>&1 | tee "$LOG_ROOT/accuracy/cpu_ground_truth.log"

OP_TEST_DEVICE=mlu python3 -c "from passop_config import device; assert str(device).startswith('mlu'), device"
OP_TEST_DEVICE=mlu PYLOGLEVEL=INFO \
python3 mlu_op_validate.py "$RESULT_ROOT/accuracy/ground_truth" "$RESULT_ROOT/accuracy" \
  2>&1 | tee "$LOG_ROOT/accuracy/mlu_validate.log"
```

当前代码若忽略 `OP_TEST_DEVICE` 导致 CPU 断言失败，应先适配 `passop_config.py`，或在 CPU-only PyTorch 环境中生成 ground truth。若使用已有可信 `tmp_data/`，可跳过 CPU 生成并将验证输入替换为该目录。不得用 MLU 输出冒充 CPU ground truth。

### GEMM、Conv2d 算子
#### 步骤 1：编译

```bash
cd /workspace/operators/speed_test
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for sample in gemm conv convbackdata convbackfilter; do
  category=conv; [ "$sample" = gemm ] && category=gemm
  make -B -C "mlu_ops/${sample}_sample" -j"$(nproc)" \
    2>&1 | tee "$LOG_ROOT/$category/compile_${sample}.log"
done
```

**验证编译产物**：

```bash
for bin in \
  mlu_ops/gemm_sample/gemm_sample \
  mlu_ops/conv_sample/conv_sample \
  mlu_ops/convbackdata_sample/convbackdata_sample \
  mlu_ops/convbackfilter_sample/convbackfilter_sample; do
  test -x "$bin"
  ldd "$bin" | grep -E 'cnnl|cnrt|cndrv|not found'
done
```

若出现 `libcnnl.so.* => not found`，先使用当前 Neuware 重新链接，并确认 Makefile 使用 `-L$(NEUWARE_HOME)/lib64 -Wl,-rpath,$(NEUWARE_HOME)/lib64 -lcnnl -lcnrt`。不要创建错误 SONAME 的软链接来掩盖 CNNL ABI 不匹配。

#### 步骤 2：使用镜像内置基准执行测试

只执行 `test_case` 指定的类别；`all` 才执行所有类别。禁止先运行模式 `0` 覆盖基准。

**GEMM**：复制内置基准，只对副本运行模式 `1`。

```bash
cd /workspace/operators/speed_test
for dtype in FP16 FP32; do
  result="$RESULT_ROOT/gemm/gemm_${dtype}_result.csv"
  cp "gemm_${dtype}.csv" "$result"
  python3 mlu_test_gemm.py "$result" 1 2>&1 | tee "$LOG_ROOT/gemm/gemm_${dtype}_test.log"
done
```

**Conv2d 综合性能**：`test_case=conv` 必须同时运行前向、反向数据和反向权重；
三项分测 CSV 只存于临时目录，最终每种 dtype 只输出一张综合表。

```bash
cd /workspace/operators/speed_test
for dtype in FP16 FP32; do
  python3 mlu_test_conv_total.py \
    --dtype "$dtype" \
    --baseline "conv_total_${dtype}.csv" \
    --output "$RESULT_ROOT/conv/conv_total_${dtype}_result.csv" \
    2>&1 | tee "$LOG_ROOT/conv/conv_total_${dtype}_test.log"
done
```

综合脚本只读取外部提供的 `baseline`，实测 `time = forward_time + backward_data_time + backward_filter_time`，再计算 `score = baseline / time`。按 `W,H,C,N,OutC,kw,kh,pw,ph,sh,sw` 一对一合并，任一分项缺失、重复、非有限或非正数都必须失败。输出统一使用 `sw`/`baseline`。

显式请求 `convbackdata` 或 `convbackfilter` 时，才单独复制相应 `convbk_*` CSV 并用原脚本模式 `1` 测试；不要把单项结果冒充 `conv` 综合结果。

如果用户明确要求重新生成参考 baseline，复制内置 CSV 为独立文件后运行模式 `0`，
例如：

```bash
cp gemm_FP16.csv "$RESULT_ROOT/gemm/gemm_FP16_generated_baseline.csv"
python3 mlu_test_gemm.py "$RESULT_ROOT/gemm/gemm_FP16_generated_baseline.csv" 0 2>&1 \
  | tee "$LOG_ROOT/gemm/gemm_FP16_generated_baseline.log"
```
生成的新基准只能作为独立产物归档；不要在本次运行中继续对它执行模式 `1`，否则
参考值与测试值来自同一设备、同一环境，score 会失去比较意义并趋近于 1。

**验证性能测试结果**：

```bash
head -5 "$RESULT_ROOT/gemm/gemm_FP16_result.csv"
head -5 "$RESULT_ROOT/conv/conv_total_FP16_result.csv"
```

---

### 长尾算子

#### 步骤 1：环境准备

```bash
cd /workspace/operators/speed_test/LongTail-Bench_mlu
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset DEVICE_CPU

python3 -c "import pandas as pd; df=pd.read_csv('../longtail_perf.csv'); print('rows:', len(df), 'unique:', df['op'].nunique()); assert len(df)==40 and df['op'].nunique()==40"
```

MLU 是异步设备。确认性能计时循环前后调用同步接口，且 profiler/trace 写盘不在计时区间内。

#### 步骤 2：使用内置基准执行性能测试

```bash
cp ../longtail_perf.csv "$RESULT_ROOT/longtail/longtail_result.csv"

python3 - "$RESULT_ROOT/longtail/longtail_result.csv" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
assert "baseline" in df.columns, "missing baseline column"
assert df["baseline"].notna().all(), "empty long-tail baseline values"
assert (df["baseline"].astype(float) > 0).all(), "non-positive long-tail baseline values"
PY

python3 -m long_tail_bench.api.api \
  -f "$RESULT_ROOT/longtail/longtail_result.csv" \
  --outcsv "$RESULT_ROOT/longtail/longtail_result.csv" \
  --validate \
  --store_input_shape \
  2>&1 | tee "$LOG_ROOT/longtail/longtail_test.log"
cp results/torch.json "$RESULT_ROOT/longtail/torch_test.json"
```

不要先省略 `--validate` 运行同一 CSV；省略该参数会把当前设备耗时写入
`baseline`，再测试就会使 score 趋近于 1。仅在用户明确要求更新长尾参考基准时，
才把结果写到独立的 `longtail_generated_baseline.csv`，且不用于本次设备评分。

结果中的 `time` 单位为 ms/iter。`inputshapes` 为空但 `time` 有值时，通常表示输入包含嵌套 Tensor、自定义对象或标量元数据，不等同于执行失败。

---

### Transformer Block

Transformer Block 测试基于 `torch_mlu`，评估 Encoder Layer 和 Decoder Layer 的前向耗时。

确认 `transformer_block/test.py` 导入 `torch_mlu`，模型和输入位于 MLU，并在计时前后调用 `torch.mlu.synchronize()`。

```bash
cd /workspace/operators/speed_test/transformer_block
python3 test.py 2>&1 | tee "$LOG_ROOT/transformer/transformer_block.log"
grep "Time per iteration" "$LOG_ROOT/transformer/transformer_block.log" \
  > "$RESULT_ROOT/transformer/transformer_block_result.txt"
```

**测试内容**：

- Encoder Layer：测试 self-attention + FFN 前向传播耗时。
- Decoder Layer：测试 self-attention + cross-attention + FFN 前向传播耗时。
- 默认参数：d_model=512、n_head=8、ffn_hidden=2048、batch_size=32、seq_len=512。

---

### 关键性能指标

精度测试必须采集 `mlu_val_result.csv` 与 `mlu_val_result.json`；每项 `validationResult` 应为通过。

#### GEMM、Conv2d 算子

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `baseline` | GEMM 参考耗时，或外部提供的 Conv 三项综合基准（ms） |
| 性能（必采） | `time` | GEMM 实测耗时；Conv 为前向、反向数据、反向权重实测耗时之和（ms） |
| 相对得分 | `score` | `baseline/time`，数值越高越好 |

#### 长尾算子

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `baseline` | 参考 MLU 单次迭代平均耗时（ms/iter） |
| 性能（必采） | `time` | 目标 MLU 单次迭代平均耗时（ms/iter） |
| 相对得分 | `score` | `baseline/time`，数值越高越好 |

#### Transformer Block

| 类型 | 指标 | 说明 |
|------|------|------|
| 性能（必采） | `Time per iteration` | Encoder/Decoder Layer 单次迭代平均耗时（秒） |

---

### 指标采集

按照 Generator 当前提供的 `result.json` 协议生成结果，禁止在 Skill 中固定
`schema_version`、`task_id` 或 `workload_fingerprint`。

- GEMM、Conv 和长尾算子：从实际结果 CSV 汇总 `baseline`、`time`、`score`；
  Conv 综合表中的 `time` 已是前向与两项反向之和，不得再次累加。
- 精度测试：汇总实际用例数、通过数和通过率。
- Transformer：单独汇总秒级 `Time per iteration`，不得与毫秒指标混算。
- `measurement_count` 使用实际纳入汇总的测量项数量，`source` 指向实际读取的
  CSV、JSON 或日志文件，`duration_seconds` 使用真实测量时长。

CSV/JSON 保留逐用例明细，`result.json` 只保留有限数值汇总和 Generator 当前协议
要求的测量证据；禁止使用占位、估算、模拟或全零数据生成成功结果。

#### 结果汇总脚本约束

Shell 变量与 heredoc 内的 Python 变量不共享作用域。使用 `<<'PY'` 时，必须把
`duration_seconds`、结果路径等动态值通过位置参数或环境变量显式传入 Python；
禁止在 Python 中直接引用 `DURATION_SECONDS`、`RESULT_ROOT` 等 Shell 变量名。
推荐使用以下模式：

```bash
START_NS=$(date +%s%N)
# 在此执行实际测试
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

若还需传入路径，继续追加引号包裹的位置参数并通过 `sys.argv` 读取。生成脚本前检查
Python 代码不存在未定义的裸 Shell 变量；写入成功结果后，必须重新读取
`/workspace/results/result.json`，并按 Generator 当前提供的校验规则验证通过，再以
退出码 0 结束任务。汇总或协议校验失败时必须写入失败结果并返回非零退出码。

---

## 常见问题

1. **容器启动失败**
   - **镜像不存在**：确认 `aibench-mlu590-operator:v1.1` 已存在于当前 Docker daemon。
   - **MLU 不可用**：在宿主机检查 `cnmon` 和 `/dev/cambricon_dev*`，在容器内检查 `torch.mlu.is_available()`。
   - **路径不存在**：确认 `OPERATOR_RESULTS_DIR`、`OPERATOR_LOGS_DIR` 在宿主机存在。
   - **挂载目录为空**：检查是否误用受路径访问限制的 Snap Docker daemon。

2. **编译失败**
   - **Neuware 缺失**：确认 `$NEUWARE_HOME/include/cnnl.h` 和 `$NEUWARE_HOME/lib64/libcnnl.so` 存在。
   - **动态库未找到**：设置 `LD_LIBRARY_PATH`，并用当前 Neuware 重新编译和链接。
   - **旧接口不兼容**：出现 `CNRT_FLOAT32`、`cnrtCastDataType`、`cnnlMatMul` 或 `cnnlGetConvolutionBackward*Algorithm` 未声明时，更新到当前 CNNL/CNRT 接口，不要使用 `-fpermissive` 忽略错误。

3. **基准值生成异常**
   - **CSV 文件不存在**：确认文件名、路径和字段与脚本一致。
   - **正常测试**：从镜像内置 CSV 重新复制一份到结果目录，只执行模式 `1` 或
     长尾命令的 `--validate`，不要执行基准生成模式。
   - **显式更新基准**：只将模式 `0` 的输出保存为独立产物，不要在同一次运行中
     用它给当前设备计算 score。
   - **pandas 类型警告**：确保写入 `time`、`baseline` 和 `score` 的值保持浮点类型。

4. **长尾算子运行失败**
   - **NumPy 旧别名**：将 `np.asscalar(x)`、`np.int`、`np.float` 分别替换为 `x.item()`、`np.int64`、`np.float64`。
   - **时间异常偏大**：检查 profiler 或 trace 写盘是否被计入计时区间。
   - **时间异常偏小**：检查计时前后是否调用 MLU 同步接口；否则可能只测到 CPU 提交时间。
   - **结果不完整**：检查日志中的 `ERROR:root`、`Traceback`、`AttributeError` 和 `RuntimeError`。

5. **Transformer Block 运行失败**
   - **PyTorch MLU 不可用**：确认 `import torch_mlu` 成功且 `torch.mlu.is_available()` 返回 True。
   - **设备适配不完整**：确认模型、输入和中间 Tensor 均位于 MLU，并使用 `torch.mlu.synchronize()`。
   - **MLU 显存不足**：通过 `cnmon` 查看占用，必要时减少 batch、序列长度或迭代次数。
