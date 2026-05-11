---
name: h200-nlp-inference
description: "NVIDIA H200 GPU 上 NLP 大语言模型(LLM,纯文本输入输出)基于 sglang 的服务化推理评测技能。用于指导 executor 完成模型服务启动、压测脚本执行、日志采集和性能结果分析。支持多模型(DeepSeek-R1 / ChatGLM3-6B-32K …),通过 MODEL_NAME 环境变量切换,新增模型只需增加一个 config 文件。"
---

# NVIDIA H200 NLP 推理专家

## 适用范围

- **硬件**:NVIDIA H200 GPU(默认 8 卡张量并行,可覆盖)
- **模型类别**:**NLP 大语言模型**,纯文本输入输出(decoder-only / encoder-decoder 的文本生成模型)
- **推理栈**:sglang 服务化推理(`sglang.launch_server` + `sglang.bench_serving`)
- **典型模型**:DeepSeek-R1、ChatGLM3、Qwen、Llama 等

**不适用**:多模态(VL/audio)、CV 分类/检测/分割、embedding/rerank 等非文本生成场景 —— 请分别走 `h200-cv-*` 等姊妹 skill。

## 触发条件

当用户说以下任意内容时启动(关键信号:**H200 硬件** + **NLP 大语言模型** + **sglang / 推理评测** 意图):
- "在 H200 上跑 `<NLP 模型>` 的推理评测"(`<NLP 模型>` 指 LLM,如 DeepSeek-R1、ChatGLM3、Qwen,下同)
- "在 H200 上用 sglang 启动 `<NLP 模型>` 推理服务并压测"
- "用 sglang.bench_serving 在 H200 上测 `<NLP 模型>` 的吞吐/时延"
- "在 H200 上评测 `<NLP 模型>` 的服务化推理性能"

如果用户请求的模型不在 `scripts/configs/` 已有列表中,先按 "如何添加新的 NLP 模型" 一节建立对应 config,再进入执行流程。如果用户请求的是非 NLP 场景(例如图像模型),不要启动本 skill,提示切换到对应的 CV/多模态 skill。

---

**基础目录配置**:
- 模型权重目录:`/data/models`(部分模型例外,见下方模型表)
- 数据集目录:`/data/datasets`
- 日志与压测结果默认输出到脚本执行目录下的日期子目录或 `logs/` 目录

---

### 支持的 NLP 模型

本 skill 通过 `scripts/configs/<model_name>.sh` 描述每个 NLP 模型的参数,由 `serve.sh` / `test.sh` 在运行时根据 `MODEL_NAME` 环境变量加载。

| MODEL_NAME            | Docker 镜像                                  | 模型路径                                                                                             | 默认 TP |
|-----------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------|---------|
| `deepseek-r1-0528`    | `sglang:nightly-dev-20251208-5e2cda61`       | `/data/models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52` | 8       |
| `chatglm3-6b-32k`     | `chatglm3_image`                             | `glm/models/chatglm3-6b-32k`                                                                         | 8       |

**默认模型**:`deepseek-r1-0528`(省略 `MODEL_NAME` 时使用)。

**当前支持任务**:
- **NLP 模型服务启动**:使用 `sglang.launch_server` 拉起文本生成服务(OpenAI 兼容 HTTP API)
- **推理性能压测**:使用 `sglang.bench_serving` 发起并发文本生成请求,测吞吐与时延

**硬件要求**:
- 8 张 NVIDIA H200 GPU(默认 `--tp 8`,可通过 `TP` 环境变量覆盖)
- 足够显存支撑所选 NLP 模型的服务化推理与压测

---

### 依赖要求

需要安装并可直接调用以下 Python 模块:

```bash
python3 -m sglang.launch_server
python3 -m sglang.bench_serving
```

压测脚本默认启用:

```bash
TRANSFORMERS_OFFLINE=1
```

各模型的 `--trust-remote-code` 等专属 flag 由 config 的 `EXTRA_SERVE_ARGS` 承载。

---

### 使用方式

**服务启动**:
```bash
# DeepSeek-R1(默认,等价于 MODEL_NAME=deepseek-r1-0528)
bash scripts/serve.sh

# ChatGLM3-6B-32K
MODEL_NAME=chatglm3-6b-32k bash scripts/serve.sh
```

**压测**:
```bash
# 压测默认模型
bash scripts/test.sh

# 压测 ChatGLM3,并指定 HOST
MODEL_NAME=chatglm3-6b-32k HOST=10.0.0.5 bash scripts/test.sh
```

**常用覆盖项**(环境变量,优先级高于 config):
- `TP` / `PORT` — 改并行度和端口
- `MODEL_PATH` — 临时指向另一份权重
- `HOST` — 压测时指定推理服务暴露的 IP(默认 `127.0.0.1`)
- `RANDOM_INPUT_LEN` / `RANDOM_OUTPUT_LEN` / `NUM_PROMPTS` / `SEED` — 压测负载参数

---

### Docker 运行命令

按所选模型选择对应镜像。通用模板:

```bash
docker run -d --gpus all --shm-size=100g \
  -v /data/models:/data/models \
  -v /data/datasets:/data/datasets \
  -v /workspace/results:/workspace/results \
  -v /workspace/tmp:/workspace/tmp \
  -v /workspace/logs:/workspace/logs \
  <IMAGE>
```

- **DeepSeek-R1**:`IMAGE=sglang:nightly-dev-20251208-5e2cda61`
- **ChatGLM3-6B-32K**:`IMAGE=chatglm3_image`(该镜像的模型路径不在 `/data/models` 下,需要确保 `glm/models/chatglm3-6b-32k` 在容器内可访问)

---

### 服务启动脚本

脚本位置:`scripts/serve.sh`

当前默认行为:
- 根据 `MODEL_NAME` 加载 `scripts/configs/${MODEL_NAME}.sh`
- 创建 `logs/` 目录
- 启动 `sglang.launch_server --model <config> --tp <config> --port <config> <EXTRA_SERVE_ARGS>`
- 输出日志到:`./logs/serve_${MODEL_NAME}.log`
- `MODEL_NAME` 对应 config 不存在时会打印可用列表并非零退出

---

### 压测脚本

脚本位置:`scripts/test.sh`

当前默认行为:
- 根据 `MODEL_NAME` 加载相同 config(保证和 serve 一致的 `MODEL_PATH`、`PORT`)
- 默认 host `127.0.0.1`、input/output 各 2048、2000 个 prompts
- 结果输出到 `${YYYYMMDD}/speed_in<IN>_out<OUT>_n<N>_${MODEL_NAME}.csv` 和同名 `.log`
- `HOST` 若推理服务在其他节点或容器网络上,需要根据实际暴露 IP 调整

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

1. **服务无法启动**
   - 检查 `MODEL_NAME` 对应 config 中的 `MODEL_PATH` 是否存在
   - 检查 `sglang` 是否已正确安装(所选镜像应自带)
   - 检查 GPU 数量是否满足 `--tp <TP>`(默认 8)

2. **压测连接失败**
   - 检查 `serve.sh` 是否已成功启动服务(对应端口是否在监听)
   - 检查 `test.sh` 的 `HOST` / `PORT` 是否与服务实际地址一致
   - 如服务运行在其他节点或容器网络上,通过 `HOST=...` 覆盖

3. **`Unknown MODEL_NAME=xxx`**
   - 脚本会列出 `scripts/configs/` 下可用的配置名,核对拼写
   - 如果是新模型,按 "如何添加新的 NLP 模型" 一节增加 config 文件

4. **找不到模型或数据集**
   - 确认 docker 挂载和 config 中的路径一致
   - ChatGLM3 的模型路径不在 `/data/models` 下,注意容器内可访问性

5. **日志或结果文件未生成**
   - 检查当前目录写权限
   - 检查日期目录和 `logs/` 目录是否创建成功

6. **性能异常波动**
   - 检查请求长度、并发设置和服务负载是否稳定
