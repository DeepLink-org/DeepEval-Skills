---
name: iluvatar-nlp-operator
description: 天数智芯 BI-V150 GPU 上 NLP 算子性能评测技能。支持 GEMM、Conv2d（FP16/FP32）、长尾算子、Transformer Block 和通信算子，用于指导 executor 完成容器启动、编译、基准运行、结果验证与性能指标采集的完整流程。
metadata:
  benchmark_specs:
    gemm: benchmark_specs/gemm.yaml
    conv: benchmark_specs/conv.yaml
    longtail: benchmark_specs/longtail.yaml
    transformer_block: benchmark_specs/transformer_block.yaml
---

### 触发条件

当用户说以下任意内容时启动：
- “在天数智芯 GPU 上生成 GEMM 算子基准值”
- “跑一下 Conv2d 算子基准”
- “生成 CoreX 算子 baseline”
- “帮我跑长尾算子基准测试”
- “测试 BI-V150 Transformer Block 性能”
- “运行 Iluvatar operator benchmark”

---

### 环境变量定义

| 环境变量 | 映射目录 | 是否必需 | 说明 |
|---------|----------|----------|------|
| `OPERATOR_PROJECT_ROOT` | `/workspace/operators` | 是 | 天数智芯算子项目根目录，包含算子代码、CSV 和项目 runner |
| `OPERATOR_RESULTS_DIR` | `/workspace/results` | 是 | 本轮评测产物目录，存放 CSV 和 `result.json` |
| `OPERATOR_LOGS_DIR` | `/workspace/logs` | 是 | 编译和运行日志目录 |

**说明**：
- `OPERATOR_PROJECT_ROOT` 必须包含 CoreX CUDA GEMM、PyTorch/CoreX Conv2d、LongTail-Bench、Transformer Block 和输入 CSV。
- executor 会把本 skill 的 `scripts/` 预置到容器内 `/workspace/scripts/`，把当前任务的 BenchmarkSpec 预置到 `/workspace/benchmark_spec.yaml`。
- 输入 CSV 只作为 case 清单；本轮结果必须写到 `/workspace/results`，禁止用资源目录里的历史 baseline 直接生成结果。

**目录结构说明**：

```text
/workspace/operators/
├── cuda_ops/
│   ├── CMakeLists.txt
│   └── build/                  # CoreX 编译产物
├── test_gemm.py
├── test_conv.py
├── gemm_f16.csv
├── gemm_f32.csv
├── conv_f16.csv
├── conv_f32.csv
├── longtail_perf_gpu.csv
├── LongTail-Bench/
├── LongTail-Bench-fp16/
├── transformer_block/
└── communication_bench/
```

---

### 支持的算子配置

**当前支持算子**（共 5 类）：
- **GEMM**：使用 CoreX 工具链编译的 CUDA/CUBLAS 实现，覆盖 FP16/FP32、多种 M/N/K 和转置组合。
- **Conv2d**：使用 PyTorch 2.7.1/CoreX 后端，测量 Forward、Backward Filter 和 Backward Data 总延迟。
- **长尾算子**：基于 LongTail-Bench，分别运行 FP32 和 FP16 项目。
- **Transformer Block**：基于 PyTorch Encoder/Decoder Layer 的 FP32 inference 延迟测试。
- **通信算子**：按资源项目 `communication_bench/readme.md` 使用 ixCCL/MPI 工具执行，多节点任务不由本 skill 的四个 BenchmarkSpec 覆盖。

**当前支持任务**：
- 在目标天数智芯 GPU 上生成本轮实测 baseline。
- 把每个 case 的实际延迟和统一 summary metrics 写入 Result Contract 2.0。

**硬件要求**：
- GEMM、Conv2d、长尾算子、Transformer Block 至少需要 1 张可用 BI-V150 GPU。
- 通信算子需要多节点多卡及对应网络、MPI、ixCCL 环境。

---

### 依赖要求

**Docker 镜像**：

```bash
swr.cn-north-1.myhuaweicloud.com/deeplink/iluvatar-nlp-operator:latest
```

容器内已预装：
- CoreX 4.4.0、CUDA 兼容工具链和 cuDNN
- PyTorch 2.7.1/CoreX 后端
- CMake、make、Python 3.x、pandas
- LongTail-Bench 和 Transformer Block 所需 Python 依赖

---

## 第一阶段：容器启动

### 选择算子类型

```bash
export OP_TYPE="gemm"  # 可选: gemm, conv, longtail, transformer_block, communication
```

### 容器创建命令

**挂载权限约定**：
- `:ro` 用于宿主机运行时和只读输入。
- `:rw` 用于结果、日志以及执行时必须访问的项目目录。

**公共参数**：

| 参数 | 说明 |
|------|------|
| `--privileged` | 允许容器访问天数智芯 GPU 设备与运行时 |
| `--ipc=host` | 使用主机 IPC |
| `--shm-size=16G` | 提供足够共享内存 |
| `-w /workspace` | 设置容器工作目录 |

**基础启动命令**：

```bash
docker run -dit \
  --name iluvatar-ops-test \
  --privileged \
  --ipc=host \
  --shm-size=16G \
  -w /workspace \
  -v "$OPERATOR_PROJECT_ROOT:/workspace/operators:rw" \
  -v "$OPERATOR_RESULTS_DIR:/workspace/results:rw" \
  -v "$OPERATOR_LOGS_DIR:/workspace/logs:rw" \
  swr.cn-north-1.myhuaweicloud.com/deeplink/iluvatar-nlp-operator:latest \
  /bin/bash
```

**注意**：
- 若已存在同名容器，先确认无须保留后再删除 `iluvatar-ops-test`。
- 如需限制可见设备，使用 CoreX 镜像支持的 `CUDA_VISIBLE_DEVICES`。

### 容器管理命令

```bash
docker exec -it iluvatar-ops-test /bin/bash

docker start iluvatar-ops-test
docker exec -it iluvatar-ops-test /bin/bash
```

**验证容器环境**：

```bash
ixsmi
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
cmake --version
test -f /workspace/operators/cuda_ops/CMakeLists.txt
test -f /workspace/operators/test_gemm.py
test -f /workspace/operators/test_conv.py
ls -lh /workspace/scripts /workspace/results /workspace/logs
```

---

## 第二阶段：容器中执行评测

### GEMM、Conv2d 算子

#### 步骤 1：编译

GEMM 和 Conv2d 必须在本轮使用 CoreX 4.4.0 的头文件和动态库重新编译。配置或编译失败时
立即退出，禁止继续使用 `cuda_ops/build/` 中的旧二进制。

```bash
set -euo pipefail
BENCHMARK_STARTED_AT=$(date +%s)
cd /workspace/operators
mkdir -p cuda_ops/build
rm -f cuda_ops/build/gemm cuda_ops/build/conv
cmake --fresh -S cuda_ops -B cuda_ops/build \
  -DCUDNN_INCLUDE_DIR=/usr/local/corex-4.4.0/include \
  -DCUDNN_LIBRARIES=/usr/local/corex-4.4.0/lib64/libcudnn.so \
  2>&1 | tee /workspace/logs/compile.log
cmake --build cuda_ops/build -j"$(nproc)" \
  2>&1 | tee -a /workspace/logs/compile.log
test -x cuda_ops/build/gemm
test -x cuda_ops/build/conv
```

#### 步骤 2：生成基准值

每次从输入 case 清单复制出全新的临时 CSV，执行真实 kernel 后再原子发布到
`/workspace/results`。GEMM 由 `test_gemm.py` 调用新编译的二进制；Conv2d 必须通过
`test_conv.py ... 0 torch` 使用 PyTorch 2.7.1/CoreX 后端，不要回退到旧 cuDNN 7 兼容反向
接口。输出只使用 `baseline` 作为 `latency_ms` 来源。

```bash
set -euo pipefail
cd /workspace/operators

rm -f /workspace/results/gemm_f16.csv{,.tmp}
cp gemm_f16.csv /workspace/results/gemm_f16.csv.tmp
python3 test_gemm.py /workspace/results/gemm_f16.csv.tmp 16 0 \
  2>&1 | tee /workspace/logs/gemm_f16_baseline.log
test -s /workspace/results/gemm_f16.csv.tmp
mv /workspace/results/gemm_f16.csv.tmp /workspace/results/gemm_f16.csv

rm -f /workspace/results/gemm_f32.csv{,.tmp}
cp gemm_f32.csv /workspace/results/gemm_f32.csv.tmp
python3 test_gemm.py /workspace/results/gemm_f32.csv.tmp 32 0 \
  2>&1 | tee /workspace/logs/gemm_f32_baseline.log
test -s /workspace/results/gemm_f32.csv.tmp
mv /workspace/results/gemm_f32.csv.tmp /workspace/results/gemm_f32.csv

rm -f /workspace/results/conv_f16.csv{,.tmp}
cp conv_f16.csv /workspace/results/conv_f16.csv.tmp
python3 test_conv.py /workspace/results/conv_f16.csv.tmp 16 0 torch \
  2>&1 | tee /workspace/logs/conv_f16_baseline.log
test -s /workspace/results/conv_f16.csv.tmp
mv /workspace/results/conv_f16.csv.tmp /workspace/results/conv_f16.csv

rm -f /workspace/results/conv_f32.csv{,.tmp}
cp conv_f32.csv /workspace/results/conv_f32.csv.tmp
python3 test_conv.py /workspace/results/conv_f32.csv.tmp 32 0 torch \
  2>&1 | tee /workspace/logs/conv_f32_baseline.log
test -s /workspace/results/conv_f32.csv.tmp
mv /workspace/results/conv_f32.csv.tmp /workspace/results/conv_f32.csv
```

**GEMM CSV 格式**（`gemm_f16.csv`）：

| NO | M | N | K | transA | transB | baseline | time | score |
|---|---|---|---|---|---|---|---|---|
| 0 | 2048 | 4096 | 1024 | 0 | 0 | 0.123 | | |

**Conv2d CSV 格式**（`conv_f16.csv`）：

| W | H | C | N | OutC | kw | kh | pw | ph | sh | sv | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 224 | 224 | 3 | 8 | 64 | 3 | 3 | 1 | 1 | 1 | 1 | 0.234 |

---

### 长尾算子

#### 步骤 1：准备本轮输入

```bash
set -euo pipefail
BENCHMARK_STARTED_AT=$(date +%s)
python3 /workspace/scripts/prepare_longtail_input.py \
  --source /workspace/operators/longtail_perf_gpu.csv \
  --output /workspace/results/longtail_cases_input.csv
test -s /workspace/results/longtail_cases_input.csv
```

#### 步骤 2：生成基准值

```bash
python3 /workspace/scripts/run_longtail.py \
  --manifest /workspace/results/longtail_cases_input.csv \
  --f32-project-root /workspace/operators/LongTail-Bench \
  --f16-project-root /workspace/operators/LongTail-Bench-fp16 \
  --output-dir /workspace/results \
  --log-dir /workspace/logs
```

输出 `longtail_perf_gpu.csv` 和 `longtail_perf_gpu_fp16.csv`，每行必须携带本轮
`aibench_run_token`。不支持的 case 必须令本轮失败，不能静默删除后仍报告成功。

---

### Transformer Block

Transformer Block 必须使用 skill 的确定性 runner，不要直接执行项目 `test.py`。runner 统一
使用 FP32、`eval()`、`torch.inference_mode()`，并在 warmup 后用
`torch.cuda.synchronize(device)` 包围计时区间；CoreX 对外提供 CUDA 兼容的 PyTorch API。

```bash
set -euo pipefail
BENCHMARK_STARTED_AT=$(date +%s)
rm -f /workspace/results/transformer_block_cases.csv \
  /workspace/results/transformer_block_cases.csv.tmp
python3 /workspace/scripts/run_transformer_block.py \
  --project-root /workspace/operators/transformer_block \
  --output /workspace/results/transformer_block_cases.csv \
  --d-model 512 \
  --num-heads 8 \
  --ffn-hidden-size 2048 \
  --batch-size 32 \
  --sequence-length 512 \
  --warmup-iterations 20 \
  --measurement-iterations 1000 \
  2>&1 | tee /workspace/logs/transformer_block.log
test -s /workspace/results/transformer_block_cases.csv
```

runner 原子写入 encoder、decoder 各一条 case，并把实际参数与
`AIBENCH_WORKLOAD_FINGERPRINT` 写入 CSV。任何参数变化必须通过 runner 的显式 CLI 传入。

---

### 通信算子（多节点环境）

通信算子依赖资源项目、MPI、ixCCL 和集群网络配置。仅在资源包提供
`communication_bench/readme.md` 且用户明确要求通信测试时按该文档执行；不要把通信结果交给
本 skill 的 GEMM、Conv2d、LongTail 或 Transformer Block collector。

---

### 关键性能指标

| 算子 | 指标 | 说明 |
|------|------|------|
| GEMM | `latency_ms` | CSV `baseline`，越低越好 |
| Conv2d | `latency_ms` | Forward、Backward Filter、Backward Data 总延迟，越低越好 |
| 长尾算子 | `latency_ms` | CSV `baseline`，越低越好 |
| Transformer Block | `latency_ms` | Encoder/Decoder 单次 inference 平均延迟，越低越好 |

---

### 指标采集

请严格使用下列 collector 生成 Result Contract 2.0。AIBenchAgent 会注入
`AIBENCH_TASK_ID`、`AIBENCH_WORKLOAD_FINGERPRINT` 和 `AIBENCH_BENCHMARK_*` 环境变量。

**GEMM**：

```bash
BENCHMARK_FINISHED_AT=$(date +%s)
BENCHMARK_DURATION=$((BENCHMARK_FINISHED_AT - BENCHMARK_STARTED_AT))
if [ "$BENCHMARK_DURATION" -le 0 ]; then BENCHMARK_DURATION=1; fi
python3 /workspace/scripts/collect_cases.py \
  --benchmark gemm \
  --input-dir /workspace/results \
  --output /workspace/results/result.json \
  --duration-seconds "$BENCHMARK_DURATION"
```

只有 FP16/FP32 两次真实 GEMM 运行都成功并原子发布 `gemm_f16.csv`、`gemm_f32.csv` 后，
才允许调用 collector。collector 会生成全部 cases 和八个 summary metrics。

**Conv2d**：

```bash
python3 /workspace/scripts/collect_cases.py \
  --benchmark conv \
  --input-dir /workspace/results \
  --output /workspace/results/result.json \
  --duration-seconds "$BENCHMARK_DURATION"
```

Conv2d case identity 包含 dtype、batch、输入 H/W/C、输出通道、kernel、padding 和水平/垂直
stride。两份 CSV 都必须来自本轮 PyTorch/CoreX 实测。

**长尾算子**：

```bash
test -s /workspace/results/longtail_cases_input.csv
test -s /workspace/results/longtail_perf_gpu.csv
test -s /workspace/results/longtail_perf_gpu_fp16.csv
python3 /workspace/scripts/collect_cases.py \
  --benchmark longtail \
  --input-dir /workspace/results \
  --output /workspace/results/result.json \
  --duration-seconds "$BENCHMARK_DURATION"
```

必须直接调用上述 `run_longtail.py`。runner 分别以两份 LongTail 项目根目录作为工作目录，并在
每次运行前删除旧 `results/torch.json`，防止失败 case 复用历史结果。collector 会逐行核对
run token，并要求 f32/f16 与 run-scoped manifest 的算子集合和顺序完全一致。

**Transformer Block**：

```bash
test -s /workspace/results/transformer_block_cases.csv
python3 /workspace/scripts/collect_cases.py \
  --benchmark transformer_block \
  --input-dir /workspace/results \
  --output /workspace/results/result.json \
  --duration-seconds "$BENCHMARK_DURATION"
```

runner 负责 inference 模式、同步计时和原子 CSV；collector 核对 workload fingerprint、
encoder/decoder 完整性、参数和有限延迟，生成 2 条 cases、八个 summary metrics 以及标准
`result.json`。禁止临时脚本自行解析 CSV、拼接 `case_key`、计算分位数或手写 JSON。

---

## 常见问题

1. **容器启动失败**
   - 检查镜像是否存在，宿主机 `ixsmi` 和 CoreX 驱动是否可用。
   - 检查三个 `OPERATOR_*` 目录是否存在并按要求挂载。
2. **GPU 不可用**
   - 宿主机运行 `ixsmi`；容器内检查 `torch.cuda.is_available()` 和设备数。
   - 检查 CoreX 运行时、PyTorch 版本和 `CUDA_VISIBLE_DEVICES`。
3. **GEMM/Conv2d 失败**
   - 检查 CoreX 编译日志、`cuda_ops/build/gemm`、PyTorch/CoreX、输入 CSV 和对应日志。
   - 不得在失败后复用旧 CSV 调用 collector。
4. **长尾或 Transformer Block 失败**
   - 检查项目布局、PyTorch 依赖、可用显存和 `/workspace/logs`。
   - 不得从旧的 `torch.json`、日志或 CSV 恢复并报告成功。
