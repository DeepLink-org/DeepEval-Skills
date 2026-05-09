---
name: nvidia-nlp-inference
description: NVIDIA GPU 上基于 sglang 的 DeepSeek 文本推理评测技能。用于指导 executor 完成容器启动、模型服务启动、压测脚本执行、日志采集与性能结果分析。
---

## 触发条件

当用户说以下任意内容时启动：
- "我要在 nvidia 上跑 DeepSeek 推理"
- "帮我测试 sglang 推理性能"
- "在 nvidia 上压测 DeepSeek-R1"
- "帮我启动 sglang 服务并跑 bench_serving"
- "采集 DeepSeek-R1 推理吞吐"

---

**基础目录配置**：
- 模型权重目录：`/data/models`
- 数据集目录：`/data/datasets`
- 代码挂载目录：`/workspace/code`
- 推理日志输出目录：`/workspace/logs`
- 压测结果输出目录：`/workspace/results`

---

### 支持的模型配置

**当前支持模型**：
- **DeepSeek-R1**

**当前支持任务**：
- **模型服务启动**：使用 `sglang.launch_server`
- **离线压测**：使用 `sglang.bench_serving`

**硬件要求**：
- 1 节点，8 张 NVIDIA GPU（当前 `serve.sh` 中 `--tp 8`）
- 足够显存支撑 DeepSeek-R1 服务化推理与压测

---

### 依赖要求

依赖通过指定 Docker 镜像提供，不需要在宿主机额外安装：

```bash
registry.h.pjlab.org.cn/ailab-sys/sglang:nightly-dev-20251208-5e2cda61
```

容器内已预装 sglang 及相关依赖，可直接调用：

```bash
python3 -m sglang.launch_server
python3 -m sglang.bench_serving
```

---

### 模型与数据路径

当前脚本默认使用以下资源：

**模型路径**：
```bash
/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
```

**数据集路径**：
```bash
/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

如模型版本或数据集路径发生变化，应同步修改 `scripts/serve.sh` 和 `scripts/test.sh` 中的路径。

---

### 容器启动脚本

**Docker 运行命令**：
```bash
docker run -it \
  --name sglang_inference \
  --gpus all \
  --shm-size=128g \
  -v /data/models:/data/models \
  -v /data/datasets:/data/datasets \
  -v /workspace/code:/workspace/code \
  -v /workspace/logs:/workspace/logs \
  -v /workspace/results:/workspace/results \
  registry.h.pjlab.org.cn/ailab-sys/sglang:nightly-dev-20251208-5e2cda61 \
  bash
```

说明：
- 使用 **交互式** `-it` 进入 `bash`，便于在同一终端内执行脚本；如需后台常驻可改为 `-d` 并配合 `docker exec`。
- **`--shm-size=128g`**：避免大吞吐推理时共享内存不足。
- 若已存在同名容器，需先执行 `docker rm -f sglang_inference` 或更换 `--name`。

---

### 服务启动脚本

服务启动脚本位于 skill 目录下的 `scripts/serve.sh`，部署时拷贝到 `/workspace/code/`：

```bash
cp scripts/serve.sh /workspace/code/
```

执行方式：

```bash
cd /workspace/code
bash serve.sh 2>&1 | tee /workspace/logs/serve.log
```

当前默认行为：
- 创建 `logs/` 目录
- 启动 `sglang.launch_server`
- 使用 8 卡张量并行：`--tp 8`
- 默认监听端口：`30000`
- 输出日志到：`./logs/serve.log`

---

### 压测脚本

压测脚本位于 skill 目录下的 `scripts/test.sh`，部署时拷贝到 `/workspace/code/`：

```bash
cp scripts/test.sh /workspace/code/
```

执行方式：

```bash
cd /workspace/code
bash test.sh 2>&1 | tee /workspace/logs/bench.log
```

当前默认行为：
- 使用 `sglang.bench_serving` 对已启动服务进行压测
- 默认 HOST：`127.0.0.1`，可通过环境变量 `HOST` 覆盖
- 默认 PORT：`30000`，可通过环境变量 `PORT` 覆盖
- 默认输入长度：`2048`（`INPUT_LEN`），输出长度：`2048`（`OUTPUT_LEN`）
- 默认请求数：`2000`（`NUM_PROMPTS`）
- 结果输出到 `/workspace/logs/bench.csv`，日志输出到 `/workspace/logs/bench.log`
- **不要修改** `INPUT_LEN`、`OUTPUT_LEN`、`NUM_PROMPTS` 等核心参数，否则与基线指标不可比

---

### 关键性能指标

压测日志末尾会输出性能汇总，例如：

```text
Total token throughput (tok/s):          8324.38
Concurrency:                             1189.77
Mean E2E Latency (ms):                   585426.75
Mean TTFT (ms):                          376538.75
Mean TPOT (ms):                          102.05
Mean ITL (ms):                           102.05
```

关注以下指标：

| 类型 | 指标 | 说明 |
|---|---|---|
| 性能（必采） | `Output token throughput (tok/s)` | 输出 token 吞吐，核心吞吐指标 |
| 性能（辅助） | `Mean TTFT (ms)` | 首 token 平均延迟 |
| 性能（辅助） | `Mean TPOT (ms)` | 每输出 token 平均延迟（不含首 token） |
| 性能（辅助） | `Mean ITL (ms)` | 平均 token 间延迟 |
| 性能（辅助） | `Mean E2E Latency (ms)` | 端到端平均延迟 |
| 性能（辅助） | `Concurrency` | 并发数 |

**采集命令**（将 `LOG` 替换为实际日志路径，如 `/workspace/logs/bench.log`）：

```bash
# 核心：Output token throughput
grep "Output token throughput" "$LOG" | awk '{print $5, $6}'

# 辅助：TTFT
grep "Mean TTFT" "$LOG" | awk '{print $4, $5}'

# 辅助：TPOT
grep "Mean TPOT" "$LOG" | awk '{print $4, $5}'

# 辅助：ITL
grep "Mean ITL" "$LOG" | awk '{print $4, $5}'
```

---

### 常见问题

1. **容器名已存在**
   - 执行 `docker rm -f sglang_inference` 后重试，或改用新容器名。

2. **服务无法启动**
   - 检查模型路径 `/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/` 下对应权重是否存在。
   - 检查 `sglang.launch_server` 是否可用。
   - 检查 GPU 数量是否满足 `--tp 8`。

3. **压测连接失败**
   - 检查 `serve.sh` 是否已成功启动服务。
   - 检查 `test.sh` 中 `HOST` 和 `PORT` 是否与服务实际监听地址一致。
   - 如果服务运行在其他节点上，根据实际 IP 地址调整 `HOST` 变量。

4. **找不到模型或数据集**
   - 检查 `/data/models/` 和 `/data/datasets/` 挂载是否正确。

5. **共享内存不足**
   - 已使用 `--shm-size=128g`；若仍报错，可适当增大。

6. **日志或结果文件未生成**
   - 检查 `/workspace/logs` 写权限。
   - 检查 `tee` 重定向是否生效。
   - 检查当天日期子目录是否已创建。
